import argparse
import csv
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset.dataset import ImageDataset
from models import build_model
from models.diffusion_utils import CosineAbarSchedule


def set_seed(seed: int = 7113064165):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    def load_config(yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    base = argparse.ArgumentParser()
    base.add_argument("--config", default=r"config/train.yaml", type=str)
    args0, _ = base.parse_known_args()

    cfg = load_config(args0.config)
    parser = argparse.ArgumentParser(parents=[base], add_help=False)
    for k, v in cfg.items():
        parser.add_argument(f"--{k}", type=type(v), default=v)

    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument("--dataset_root", type=str, default="")
    parser.add_argument("--cover_radius", type=float, default=6.0)
    parser.add_argument("--dup_radius", type=float, default=6.0)
    parser.add_argument("--nms_radius", type=float, default=5.0)
    parser.add_argument("--top_k_worst", type=int, default=20)
    parser.add_argument("--max_n", type=int, default=900)
    return parser.parse_args()


def collate_points_padded(batch, max_n=900):
    imgs, pts, metas = zip(*batch)
    imgs = torch.stack(imgs, 0)

    B = len(pts)
    padded = torch.full((B, max_n, 2), fill_value=-10.0, dtype=torch.float32)
    mask = torch.zeros((B, max_n), dtype=torch.bool)

    for i, p in enumerate(pts):
        n = int(p.size(0))
        m = min(n, max_n)
        if m > 0:
            padded[i, :m] = p[:m]
            mask[i, :m] = True

    return imgs, padded, mask, list(metas)


def collate_points_padded_900(batch):
    return collate_points_padded(batch, max_n=900)


def load_checkpoint_into_model(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict):
        if "model_state" in state:
            model.load_state_dict(state["model_state"])
        elif "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            model.load_state_dict(state)
    else:
        model.load_state_dict(state)
    return model


def get_image_key_from_meta(meta: dict):
    for k in ["img_path", "image_path", "img_name", "image_name", "image_id"]:
        if k in meta:
            return meta[k]
    return str(meta)


def dedup_points_xy(xy_np: np.ndarray, decimals: int = 3) -> np.ndarray:
    if xy_np.shape[0] == 0:
        return xy_np
    key = np.round(xy_np, decimals=decimals)
    _, idx = np.unique(key, axis=0, return_index=True)
    return xy_np[np.sort(idx)]


def make_ddim_steps(T=1000, steps=20, device="cpu"):
    return torch.linspace(T - 1, 0, steps, device=device).round().long()


@torch.no_grad()
def ddim_reverse_step_eta(p_t, eps_pred, abar_t, abar_prev, eta=0.0):
    tiny = torch.finfo(p_t.dtype).eps
    abar_t = abar_t.clamp(0, 1)
    abar_prev = abar_prev.clamp(0, 1)

    sqrt_abar_t = (abar_t + tiny).sqrt()
    sqrt_one_mt = (1.0 - abar_t).clamp_min(0).sqrt()
    x0_pred = (p_t - sqrt_one_mt * eps_pred) / sqrt_abar_t

    alpha_t = (abar_t / (abar_prev + tiny)).clamp_min(tiny)
    sigma2 = (eta ** 2) * ((1.0 - abar_prev).clamp_min(0) / (1.0 - abar_t).clamp_min(tiny)) * (1.0 - alpha_t).clamp_min(0)
    sigma = sigma2.clamp_min(0).sqrt()
    c_eps = (1.0 - abar_prev - sigma2).clamp_min(0).sqrt()

    mean = (abar_prev + tiny).sqrt() * x0_pred + c_eps * eps_pred
    if eta > 0:
        mean = mean + sigma * torch.randn_like(p_t)
    return mean


@torch.no_grad()
def radius_nms_xy(pts_xy: torch.Tensor, scores: torch.Tensor, r: float):
    if pts_xy.numel() == 0:
        return torch.empty((0,), dtype=torch.long)

    order = scores.argsort(descending=True)
    keep = []
    taken = torch.zeros((pts_xy.size(0),), dtype=torch.bool, device=pts_xy.device)
    r2 = float(r) * float(r)

    for idx in order:
        idx = int(idx.item())
        if taken[idx]:
            continue
        keep.append(idx)
        d2 = ((pts_xy - pts_xy[idx]).pow(2).sum(dim=1))
        taken |= (d2 <= r2)

    return torch.tensor(keep, dtype=torch.long)


def coverage_and_duplicates(gt_xy: np.ndarray, pred_xy: np.ndarray, radius: float):
    if gt_xy.shape[0] == 0:
        return {
            "gt_count": 0,
            "covered_count": 0,
            "coverage_ratio": 0.0,
            "dup_per_gt_mean": 0.0,
            "dup_per_covered_gt_mean": 0.0,
            "gt_with_multi_ratio": 0.0,
        }

    if pred_xy.shape[0] == 0:
        return {
            "gt_count": int(gt_xy.shape[0]),
            "covered_count": 0,
            "coverage_ratio": 0.0,
            "dup_per_gt_mean": 0.0,
            "dup_per_covered_gt_mean": 0.0,
            "gt_with_multi_ratio": 0.0,
        }

    diff = gt_xy[:, None, :] - pred_xy[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    hit_counts = (d2 <= (radius * radius)).sum(axis=1)

    covered = hit_counts > 0
    multi = hit_counts > 1
    covered_count = int(covered.sum())

    return {
        "gt_count": int(gt_xy.shape[0]),
        "covered_count": covered_count,
        "coverage_ratio": float(covered.mean()),
        "dup_per_gt_mean": float(hit_counts.mean()),
        "dup_per_covered_gt_mean": float(hit_counts[covered].mean()) if covered_count > 0 else 0.0,
        "gt_with_multi_ratio": float(multi.mean()),
    }


def aggregate_metric(rows, key):
    if not rows:
        return 0.0
    return float(np.mean([row[key] for row in rows]))


def resolve_dataset_root(args):
    split = str(getattr(args, "eval_split", "test")).strip().lower()
    explicit_root = str(getattr(args, "dataset_root", "")).strip()
    if explicit_root:
        return explicit_root, "custom"
    if split == "train":
        return args.data_root, "train"
    if split == "test":
        return args.test_root, "test"
    if split == "val":
        # Current project config does not expose a full validation root; fall back to test_root
        return args.test_root, "val->test_root"
    raise ValueError(f"Unsupported eval_split={args.eval_split!r}. Use train, test, val, or set --dataset_root.")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(getattr(args, "seed", 7113064165)))

    assert args.ckpt_path, "Please set --ckpt_path"
    dataset_root, resolved_split = resolve_dataset_root(args)
    assert dataset_root, "Please set --dataset_root or ensure the selected split root exists in config."

    os.makedirs(args.save_dir, exist_ok=True)

    model = build_model(args, training=False)
    model = load_checkpoint_into_model(model, args.ckpt_path, device)
    model.to(device).eval()

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
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_points_padded_900 if int(args.max_n) == 900 else collate_points_padded_900,
    )

    T = int(args.diffusion_T)
    steps = int(args.ddim_steps)
    sched = CosineAbarSchedule(T=T)
    abar = sched.abar.to(device=device)
    t_seq = torch.unique_consecutive(make_ddim_steps(T=T, steps=steps, device=device))
    eps_clip = float(getattr(args, "eps_clip", 1e-3))
    hard_thresh = float(getattr(args, "hard_thresh", 0.0))
    ddim_eta = float(getattr(args, "ddim_eta", 0.0))

    per_image_proposals_xy = defaultdict(list)
    per_image_proposals_prob = defaultdict(list)
    per_image_candidates_xy = defaultdict(list)
    per_image_candidates_prob = defaultdict(list)
    per_image_gt_points_xy = defaultdict(list)
    per_image_size = {}

    print(f"[INFO] Loaded checkpoint: {args.ckpt_path}")
    print(f"[INFO] Eval split: {resolved_split}")
    print(f"[INFO] Dataset root: {dataset_root}")
    print(f"[INFO] Dataset tiles: {len(dataset)}")
    print(f"[INFO] Save dir: {args.save_dir}")

    for images, points_pad, mask, metas in loader:
        images = images.to(device)
        B, _, H, W = images.shape
        N = points_pad.shape[1]
        feats = model.encode(images)

        p0_list = []
        prob_list = []
        posmask_list = []
        R = int(getattr(args, "num_realizations", 1))

        for _ in range(R):
            p_t = torch.empty((B, N, 2), device=device).uniform_(-1.0 + eps_clip, 1.0 - eps_clip)
            exist_prob_last = None
            pos_mask_last = None

            for i, t_int in enumerate(t_seq.tolist()):
                t_tensor = torch.full((B, 1), int(t_int), device=device, dtype=torch.long)
                abar_t = abar[int(t_int)].view(1, 1, 1).expand(B, 1, 1)
                need_exist = (i == len(t_seq) - 1)

                eps_pred, exist_logit, _, _, _ = model.denoise(
                    feats, p_t, t_tensor,
                    abar_t=abar_t,
                    clamp_eps=1e-6,
                    need_exist=need_exist,
                )

                if need_exist:
                    if exist_logit is None:
                        raise RuntimeError("need_exist=True but denoise returned None exist_logit")
                    if exist_logit.dim() == 3 and exist_logit.size(-1) == 2:
                        exist_prob_last = torch.softmax(exist_logit, dim=-1)[..., 1]
                        gate_mode = getattr(args, "test_gate_mode", "argmax_or_prob")
                        if gate_mode == "prob_only":
                            pos_mask_last = exist_prob_last > hard_thresh
                        elif gate_mode == "argmax_only":
                            pos_mask_last = exist_logit.argmax(-1) == 1
                        else:
                            pos_mask_last = (exist_logit.argmax(-1) == 1) | (exist_prob_last > hard_thresh)
                    elif exist_logit.dim() == 3 and exist_logit.size(-1) == 1:
                        exist_prob_last = torch.sigmoid(exist_logit.squeeze(-1))
                        pos_mask_last = exist_prob_last > hard_thresh
                    else:
                        exist_prob_last = torch.sigmoid(exist_logit)
                        pos_mask_last = exist_prob_last > hard_thresh

                if i + 1 < len(t_seq):
                    abar_prev = abar[int(t_seq[i + 1].item())].view(1, 1, 1).expand(B, 1, 1)
                    eta_step = ddim_eta
                else:
                    abar_prev = torch.ones((B, 1, 1), device=device)
                    eta_step = 0.0

                p_t = ddim_reverse_step_eta(p_t, eps_pred, abar_t, abar_prev, eta=eta_step)
                p_t = p_t.clamp(-1.0 + eps_clip, 1.0 - eps_clip)

            if exist_prob_last is None or pos_mask_last is None:
                raise RuntimeError("DDIM loop finished without final probabilities.")

            p0_list.append(p_t.detach())
            prob_list.append(exist_prob_last.detach())
            posmask_list.append(pos_mask_last.detach())

        for b in range(B):
            meta = metas[b]
            img_key = get_image_key_from_meta(meta)
            H_full, W_full = meta["orig_size"]
            x0 = int(meta["tile_left"])
            y0 = int(meta["tile_top"])
            per_image_size[img_key] = (int(H_full), int(W_full))

            mask_b = mask[b].cpu()
            gt_local = points_pad[b][mask_b]
            if gt_local.numel() > 0:
                gt_x = (gt_local[:, 0] + x0).round().long().clamp(0, W_full - 1)
                gt_y = (gt_local[:, 1] + y0).round().long().clamp(0, H_full - 1)
                per_image_gt_points_xy[img_key].append(torch.stack([gt_x, gt_y], dim=1))

            for r in range(R):
                pts_norm_all = p0_list[r][b]
                sc_all = prob_list[r][b]
                pm = posmask_list[r][b]

                xs_prop = (pts_norm_all[:, 0] + 1) * 0.5 * (W - 1)
                ys_prop = (pts_norm_all[:, 1] + 1) * 0.5 * (H - 1)
                xs_prop_g = (xs_prop + x0).clamp(0, W_full - 1)
                ys_prop_g = (ys_prop + y0).clamp(0, H_full - 1)

                per_image_proposals_xy[img_key].append(torch.stack([xs_prop_g, ys_prop_g], dim=1).detach().cpu())
                per_image_proposals_prob[img_key].append(sc_all.detach().cpu())

                pts_norm = pts_norm_all[pm]
                sc = sc_all[pm]
                if sc.numel() == 0:
                    continue

                xs = (pts_norm[:, 0] + 1) * 0.5 * (W - 1)
                ys = (pts_norm[:, 1] + 1) * 0.5 * (H - 1)
                xs_g = (xs + x0).clamp(0, W_full - 1)
                ys_g = (ys + y0).clamp(0, H_full - 1)
                per_image_candidates_xy[img_key].append(torch.stack([xs_g, ys_g], dim=1).detach().cpu())
                per_image_candidates_prob[img_key].append(sc.detach().cpu())

    rows = []
    for img_key in sorted(per_image_size.keys()):
        gt_all = np.zeros((0, 2), dtype=np.float32)
        if img_key in per_image_gt_points_xy and len(per_image_gt_points_xy[img_key]) > 0:
            gt_all = torch.cat(per_image_gt_points_xy[img_key], dim=0).detach().cpu().numpy().astype(np.float32)
            gt_all = dedup_points_xy(gt_all, decimals=3)

        prop_xy = np.zeros((0, 2), dtype=np.float32)
        prop_sc = np.zeros((0,), dtype=np.float32)
        if img_key in per_image_proposals_xy and len(per_image_proposals_xy[img_key]) > 0:
            prop_xy = torch.cat(per_image_proposals_xy[img_key], dim=0).detach().cpu().numpy().astype(np.float32)
            prop_sc = torch.cat(per_image_proposals_prob[img_key], dim=0).detach().cpu().numpy().astype(np.float32)

        cand_xy = np.zeros((0, 2), dtype=np.float32)
        cand_sc = np.zeros((0,), dtype=np.float32)
        if img_key in per_image_candidates_xy and len(per_image_candidates_xy[img_key]) > 0:
            cand_xy = torch.cat(per_image_candidates_xy[img_key], dim=0).detach().cpu().numpy().astype(np.float32)
            cand_sc = torch.cat(per_image_candidates_prob[img_key], dim=0).detach().cpu().numpy().astype(np.float32)

        if cand_xy.shape[0] > 0:
            keep = radius_nms_xy(torch.from_numpy(cand_xy), torch.from_numpy(cand_sc), r=float(args.nms_radius))
            final_xy = cand_xy[keep.numpy()]
            final_sc = cand_sc[keep.numpy()]
        else:
            final_xy = np.zeros((0, 2), dtype=np.float32)
            final_sc = np.zeros((0,), dtype=np.float32)

        proposal_stats = coverage_and_duplicates(gt_all, prop_xy, radius=float(args.cover_radius))
        final_stats = coverage_and_duplicates(gt_all, final_xy, radius=float(args.cover_radius))

        gt_count = float(gt_all.shape[0])
        pred_count = float(final_xy.shape[0])
        err = pred_count - gt_count

        rows.append({
            "image_path": img_key,
            "gt_count": gt_count,
            "pred_count": pred_count,
            "abs_error": float(abs(err)),
            "signed_error": float(err),
            "proposal_count": int(prop_xy.shape[0]),
            "candidate_count_pre_nms": int(cand_xy.shape[0]),
            "proposal_cover_ratio": proposal_stats["coverage_ratio"],
            "proposal_dup_per_gt_mean": proposal_stats["dup_per_gt_mean"],
            "proposal_dup_per_covered_gt_mean": proposal_stats["dup_per_covered_gt_mean"],
            "proposal_gt_with_multi_ratio": proposal_stats["gt_with_multi_ratio"],
            "final_cover_ratio": final_stats["coverage_ratio"],
            "final_dup_per_gt_mean": final_stats["dup_per_gt_mean"],
            "final_dup_per_covered_gt_mean": final_stats["dup_per_covered_gt_mean"],
            "final_gt_with_multi_ratio": final_stats["gt_with_multi_ratio"],
        })

    mae = aggregate_metric(rows, "abs_error")
    rmse = float(np.sqrt(np.mean([row["signed_error"] ** 2 for row in rows]))) if rows else 0.0
    summary = {
        "num_images": len(rows),
        "mae": mae,
        "rmse": rmse,
        "mean_gt_count": aggregate_metric(rows, "gt_count"),
        "mean_pred_count": aggregate_metric(rows, "pred_count"),
        "mean_proposal_count": aggregate_metric(rows, "proposal_count"),
        "mean_candidate_count_pre_nms": aggregate_metric(rows, "candidate_count_pre_nms"),
        "proposal_cover_ratio_mean": aggregate_metric(rows, "proposal_cover_ratio"),
        "final_cover_ratio_mean": aggregate_metric(rows, "final_cover_ratio"),
        "proposal_dup_per_gt_mean": aggregate_metric(rows, "proposal_dup_per_gt_mean"),
        "final_dup_per_gt_mean": aggregate_metric(rows, "final_dup_per_gt_mean"),
        "proposal_dup_per_covered_gt_mean": aggregate_metric(rows, "proposal_dup_per_covered_gt_mean"),
        "final_dup_per_covered_gt_mean": aggregate_metric(rows, "final_dup_per_covered_gt_mean"),
        "proposal_gt_with_multi_ratio_mean": aggregate_metric(rows, "proposal_gt_with_multi_ratio"),
        "final_gt_with_multi_ratio_mean": aggregate_metric(rows, "final_gt_with_multi_ratio"),
        "cover_radius": float(args.cover_radius),
        "dup_radius_note": "duplicate metrics are counted inside the same cover radius window around each GT",
        "nms_radius": float(args.nms_radius),
        "hard_thresh": hard_thresh,
        "ddim_steps": steps,
        "num_realizations": int(getattr(args, "num_realizations", 1)),
        "ckpt_path": args.ckpt_path,
        "eval_split": resolved_split,
        "dataset_root": dataset_root,
        "config_data_root": args.data_root,
        "config_test_root": args.test_root,
    }

    rows_sorted = sorted(rows, key=lambda x: x["abs_error"], reverse=True)
    worst_rows = rows_sorted[: int(args.top_k_worst)]

    print("\n[SUMMARY]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n[WORST-{len(worst_rows)}]")
    for row in worst_rows:
        print(
            f"{os.path.basename(str(row['image_path']))} | "
            f"gt={row['gt_count']:.0f} pred={row['pred_count']:.0f} abs_err={row['abs_error']:.0f} | "
            f"prop_cover={row['proposal_cover_ratio']:.3f} final_cover={row['final_cover_ratio']:.3f} | "
            f"prop_dup={row['proposal_dup_per_gt_mean']:.2f} final_dup={row['final_dup_per_gt_mean']:.2f}"
        )

    csv_path = os.path.join(args.save_dir, "per_image_diagnostics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["image_path"])
        writer.writeheader()
        if rows:
            writer.writerows(rows)

    summary_path = os.path.join(args.save_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    worst_path = os.path.join(args.save_dir, "worst_images.csv")
    with open(worst_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(worst_rows[0].keys()) if worst_rows else ["image_path"])
        writer.writeheader()
        if worst_rows:
            writer.writerows(worst_rows)

    print(f"\n[SAVED] {csv_path}")
    print(f"[SAVED] {summary_path}")
    print(f"[SAVED] {worst_path}")


if __name__ == "__main__":
    main()
