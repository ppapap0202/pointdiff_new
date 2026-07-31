import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataset.dataset import ImageDataset
from models import build_model
from models.proposal_prior import build_mixed_x0_prior, select_density_points
from validate_diagnostics import (
    collate_points_padded_900,
    coverage_and_duplicates,
    dedup_points_xy,
    get_image_key_from_meta,
    load_checkpoint_into_model,
)

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--eval_split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--dataset_root", default="")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_n", type=int, default=900)
    parser.add_argument("--cover_radius", type=float, default=6.0)
    parser.add_argument("--random_trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg if cfg is not None else {}


def resolve_dataset_root(args, cfg):
    if args.dataset_root:
        return args.dataset_root, "custom"
    split = str(args.eval_split).strip().lower()
    if split == "train":
        return cfg["data_root"], "train"
    if split == "val":
        return cfg.get("val_root") or cfg["test_root"], "val"
    return cfg["test_root"], "test"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def m11_to_global_xy(points_m11, meta, tile_h, tile_w):
    if points_m11.numel() == 0:
        return points_m11.new_empty((0, 2)).cpu()
    h_full, w_full = meta["orig_size"]
    x0 = int(meta["tile_left"])
    y0 = int(meta["tile_top"])
    xs = (points_m11[:, 0] + 1.0) * 0.5 * float(tile_w - 1)
    ys = (points_m11[:, 1] + 1.0) * 0.5 * float(tile_h - 1)
    xs = (xs + x0).clamp(0, int(w_full) - 1)
    ys = (ys + y0).clamp(0, int(h_full) - 1)
    return torch.stack([xs, ys], dim=1).detach().cpu()


def add_points(store, method, img_key, points_m11, meta, tile_h, tile_w):
    store[method][img_key].append(m11_to_global_xy(points_m11, meta, tile_h, tile_w))


def aggregate_method(method_name, per_image_pred, per_image_gt, cover_radius):
    rows = []
    for img_key, gt_parts in per_image_gt.items():
        gt = torch.cat(gt_parts, dim=0).numpy().astype(np.float32) if gt_parts else np.zeros((0, 2), dtype=np.float32)
        gt = dedup_points_xy(gt, decimals=3)
        pred_parts = per_image_pred.get(img_key, [])
        pred = torch.cat(pred_parts, dim=0).numpy().astype(np.float32) if pred_parts else np.zeros((0, 2), dtype=np.float32)
        stats = coverage_and_duplicates(gt, pred, radius=float(cover_radius))
        rows.append({
            "method": method_name,
            "image_path": img_key,
            "gt_count": float(gt.shape[0]),
            "proposal_count": float(pred.shape[0]),
            "proposal_cover_ratio": stats["coverage_ratio"],
            "proposal_dup_per_gt_mean": stats["dup_per_gt_mean"],
            "proposal_dup_per_covered_gt_mean": stats["dup_per_covered_gt_mean"],
            "proposal_gt_with_multi_ratio": stats["gt_with_multi_ratio"],
        })
    return rows


def mean(rows, key):
    return float(np.mean([float(r[key]) for r in rows])) if rows else 0.0


def summarize_rows(method, rows, cover_radius):
    return {
        "method": method,
        "num_images": len(rows),
        "mean_gt_count": mean(rows, "gt_count"),
        "mean_proposal_count": mean(rows, "proposal_count"),
        "proposal_points_per_gt": mean(rows, "proposal_count") / max(mean(rows, "gt_count"), 1e-12),
        "proposal_cover_ratio_mean": mean(rows, "proposal_cover_ratio"),
        "proposal_dup_per_gt_mean": mean(rows, "proposal_dup_per_gt_mean"),
        "proposal_dup_per_covered_gt_mean": mean(rows, "proposal_dup_per_covered_gt_mean"),
        "proposal_gt_with_multi_ratio_mean": mean(rows, "proposal_gt_with_multi_ratio"),
        "cover_radius": float(cover_radius),
    }


def base_method(method):
    for suffix in ["_r0", "_r1", "_r2", "_r3", "_r4", "_r5", "_r6", "_r7", "_r8", "_r9"]:
        if method.endswith(suffix):
            return method[: -len(suffix)]
    return method


def group_summaries(trial_summaries):
    grouped = defaultdict(list)
    for row in trial_summaries:
        grouped[base_method(row["method"])].append(row)

    out = []
    metric_keys = [
        "mean_gt_count",
        "mean_proposal_count",
        "proposal_points_per_gt",
        "proposal_cover_ratio_mean",
        "proposal_dup_per_gt_mean",
        "proposal_dup_per_covered_gt_mean",
        "proposal_gt_with_multi_ratio_mean",
    ]
    for method, rows in grouped.items():
        item = {
            "method": method,
            "num_trials": len(rows),
            "num_images": rows[0]["num_images"] if rows else 0,
            "cover_radius": rows[0]["cover_radius"] if rows else 0.0,
        }
        for key in metric_keys:
            values = np.array([float(r[key]) for r in rows], dtype=np.float64)
            item[key] = float(values.mean()) if values.size else 0.0
            item[key + "_std"] = float(values.std(ddof=0)) if values.size else 0.0
        out.append(item)
    return sorted(out, key=lambda x: x["proposal_cover_ratio_mean"], reverse=True)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 7113064165))
    set_seed(seed)

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    model_args = SimpleNamespace(**cfg)
    model = build_model(model_args, training=False).to(device)
    model = load_checkpoint_into_model(model, args.ckpt_path, device)
    model.eval()

    dataset_root, resolved_split = resolve_dataset_root(args, cfg)
    batch_size = int(args.batch_size if args.batch_size is not None else cfg.get("batch_size", 8))
    num_workers = int(args.num_workers if args.num_workers is not None else cfg.get("num_workers", 0))

    dataset = ImageDataset(
        root=dataset_root,
        mode="points",
        tile_size=(256, 256),
        stride=(256, 256),
        gray=False,
        pad_if_needed=True,
        image_exts=(".jpg", ".png"),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_points_padded_900,
    )

    max_n = int(args.max_n)
    clamp_eps = float(cfg.get("eps_clip", 1e-3))
    mode = str(cfg.get("proposal_prior_mode", "density_only"))
    cell_capacity = int(cfg.get("proposal_prior_cell_capacity", 2))
    random_trials = max(1, int(args.random_trials))

    per_image_gt = defaultdict(list)
    pred_by_method = defaultdict(lambda: defaultdict(list))
    count_rows = []

    print(f"[INFO] checkpoint={args.ckpt_path}")
    print(f"[INFO] dataset={dataset_root} split={resolved_split} tiles={len(dataset)}")
    print(f"[INFO] mode={mode} capacity={cell_capacity} random_trials={random_trials}")
    print("[INFO] This is x0-prior-only cover; no DDIM is run.")

    with torch.no_grad():
        for images, points_pad, mask, metas in tqdm(loader, total=len(loader), desc="prior-x0 tiles", unit="batch"):
            images = images.to(device, non_blocking=True)
            points_device = points_pad.to(device, non_blocking=True)
            mask_device = mask.to(device, non_blocking=True)
            batch_size_cur, _, tile_h, tile_w = images.shape

            feats = model.encode(images)
            occupancy_logits, density = model.predict_proposal_prior(feats)

            mixed_trials = []
            for _ in range(random_trials):
                mixed, _ = build_mixed_x0_prior(
                    occupancy_logits,
                    density,
                    num_slots=max_n,
                    clamp_eps=clamp_eps,
                    mode=mode,
                    density_cell_capacity=cell_capacity,
                )
                mixed_trials.append(mixed)

            for b in range(batch_size_cur):
                meta = metas[b]
                img_key = get_image_key_from_meta(meta)
                h_full, w_full = meta["orig_size"]
                x0 = int(meta["tile_left"])
                y0 = int(meta["tile_top"])

                mask_b = mask[b].bool()
                gt_local = points_pad[b][mask_b]
                if gt_local.numel() > 0:
                    gt_x = (gt_local[:, 0] + x0).round().long().clamp(0, int(w_full) - 1)
                    gt_y = (gt_local[:, 1] + y0).round().long().clamp(0, int(h_full) - 1)
                    per_image_gt[img_key].append(torch.stack([gt_x, gt_y], dim=1))

                density_b = density[b]
                rounded_count = int(torch.round(density_b.sum()).clamp(0, max_n).item())
                density_pts = select_density_points(
                    density_b,
                    rounded_count,
                    cell_capacity=cell_capacity,
                )
                density_count = int(density_pts.size(0))
                gt_tile_count = int(mask_device[b].sum().item())

                add_points(
                    pred_by_method,
                    "density_guided_only",
                    img_key,
                    density_pts,
                    meta,
                    tile_h,
                    tile_w,
                )

                count_rows.append({
                    "image_path": img_key,
                    "tile_left": x0,
                    "tile_top": y0,
                    "gt_tile_count": gt_tile_count,
                    "density_sum": float(density_b.sum().item()),
                    "density_rounded_count": rounded_count,
                    "density_guided_count": density_count,
                    "density_abs_count_error": abs(float(density_count) - float(gt_tile_count)),
                })

                for r in range(random_trials):
                    random_same = occupancy_logits.new_empty((density_count, 2)).uniform_(
                        -1.0 + clamp_eps,
                        1.0 - clamp_eps,
                    )
                    random_full = occupancy_logits.new_empty((max_n, 2)).uniform_(
                        -1.0 + clamp_eps,
                        1.0 - clamp_eps,
                    )
                    add_points(
                        pred_by_method,
                        f"random_same_density_count_r{r}",
                        img_key,
                        random_same,
                        meta,
                        tile_h,
                        tile_w,
                    )
                    add_points(
                        pred_by_method,
                        f"random_900_r{r}",
                        img_key,
                        random_full,
                        meta,
                        tile_h,
                        tile_w,
                    )
                    add_points(
                        pred_by_method,
                        f"density_plus_random_fill_900_r{r}",
                        img_key,
                        mixed_trials[r][b],
                        meta,
                        tile_h,
                        tile_w,
                    )

    trial_rows = []
    per_image_rows = []
    for method, per_image_pred in sorted(pred_by_method.items()):
        rows = aggregate_method(method, per_image_pred, per_image_gt, args.cover_radius)
        per_image_rows.extend(rows)
        trial_rows.append(summarize_rows(method, rows, args.cover_radius))

    grouped = group_summaries(trial_rows)
    count_summary = {
        "mean_tile_gt_count": float(np.mean([r["gt_tile_count"] for r in count_rows])) if count_rows else 0.0,
        "mean_density_sum": float(np.mean([r["density_sum"] for r in count_rows])) if count_rows else 0.0,
        "mean_density_guided_count": float(np.mean([r["density_guided_count"] for r in count_rows])) if count_rows else 0.0,
        "mean_density_abs_tile_count_error": float(np.mean([r["density_abs_count_error"] for r in count_rows])) if count_rows else 0.0,
    }
    output = {
        "checkpoint": args.ckpt_path,
        "config": args.config,
        "dataset_root": dataset_root,
        "eval_split": resolved_split,
        "seed": seed,
        "random_trials": random_trials,
        "max_n": max_n,
        "cover_radius": float(args.cover_radius),
        "proposal_prior_mode": mode,
        "proposal_prior_cell_capacity": cell_capacity,
        "note": "x0-prior-only cover before forward noising and DDIM.",
        "count_summary": count_summary,
        "grouped_summary": grouped,
        "trial_summary": sorted(trial_rows, key=lambda x: x["proposal_cover_ratio_mean"], reverse=True),
    }

    summary_path = os.path.join(args.save_dir, "summary.json")
    grouped_path = os.path.join(args.save_dir, "grouped_summary.csv")
    trial_path = os.path.join(args.save_dir, "trial_summary.csv")
    per_image_path = os.path.join(args.save_dir, "per_image_prior_x0_cover.csv")
    count_path = os.path.join(args.save_dir, "tile_density_count.csv")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    with open(grouped_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(grouped[0].keys()) if grouped else ["method"])
        writer.writeheader()
        writer.writerows(grouped)

    with open(trial_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(trial_rows[0].keys()) if trial_rows else ["method"])
        writer.writeheader()
        writer.writerows(sorted(trial_rows, key=lambda x: x["proposal_cover_ratio_mean"], reverse=True))

    with open(per_image_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()) if per_image_rows else ["method"])
        writer.writeheader()
        writer.writerows(per_image_rows)

    with open(count_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(count_rows[0].keys()) if count_rows else ["image_path"])
        writer.writeheader()
        writer.writerows(count_rows)

    print("\n[COUNT]")
    print(json.dumps(count_summary, indent=2, ensure_ascii=False))
    print("\n[GROUPED]")
    print(json.dumps(grouped, indent=2, ensure_ascii=False))
    print(f"\n[SAVED] {summary_path}")
    print(f"[SAVED] {grouped_path}")
    print(f"[SAVED] {trial_path}")
    print(f"[SAVED] {per_image_path}")
    print(f"[SAVED] {count_path}")


if __name__ == "__main__":
    main()
