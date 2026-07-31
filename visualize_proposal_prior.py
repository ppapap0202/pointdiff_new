import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from dataset import build_dataset
from models import build_model
from models.proposal_prior import build_proposal_prior_targets, select_guided_prior_points


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model_state(model, state_dict):
    if isinstance(state_dict, dict) and "model_state" in state_dict:
        state_dict = state_dict["model_state"]
    incompatible = model.load_state_dict(state_dict, strict=False)
    return incompatible


def tensor_image_to_numpy(image):
    image = image.detach().cpu().clamp(0, 1)
    if image.size(0) == 1:
        arr = image[0].numpy()
        return np.stack([arr, arr, arr], axis=-1)
    return image.permute(1, 2, 0).numpy()


def norm_xy_to_pixel(points_m11, width, height):
    if points_m11.numel() == 0:
        return np.zeros((0, 2), dtype=np.float32)
    points = points_m11.detach().float().cpu()
    xy = (points + 1.0) * 0.5
    xy[:, 0] *= float(width)
    xy[:, 1] *= float(height)
    return xy.numpy()


def density_to_image(density, height, width):
    d = density.detach().float().cpu().squeeze().numpy()
    if d.size == 0:
        return np.zeros((height, width), dtype=np.float32)
    d_t = torch.from_numpy(d).view(1, 1, d.shape[0], d.shape[1])
    up = torch.nn.functional.interpolate(
        d_t, size=(height, width), mode="bilinear", align_corners=False
    )
    up = up.squeeze().numpy()
    if float(up.max()) > float(up.min()):
        up = (up - up.min()) / (up.max() - up.min())
    return up


def pick_indices(dataset, count, mode, explicit):
    if explicit:
        return [int(x) for x in explicit.split(",") if x.strip()]
    if mode == "first":
        return list(range(min(count, len(dataset))))

    scored = []
    for idx in range(len(dataset)):
        _, pts, meta = dataset[idx]
        scored.append((int(pts.size(0)), idx, meta))
    scored.sort(reverse=True)
    return [idx for _, idx, _ in scored[:count]]


def draw_tile(
    image_np,
    gt_xy,
    density,
    target_density,
    occupancy_prob,
    proposal_xy,
    meta,
    out_path,
    title,
):
    height, width = image_np.shape[:2]
    density_img = density_to_image(density, height, width)
    target_img = density_to_image(target_density, height, width)
    occupancy_img = density_to_image(occupancy_prob, height, width)

    fig, axes = plt.subplots(1, 5, figsize=(22, 5), dpi=140)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].imshow(image_np)
    if gt_xy.size:
        axes[0].scatter(gt_xy[:, 0], gt_xy[:, 1], s=14, c="#ff3b30", linewidths=0.4, edgecolors="white")
    axes[0].set_title(f"image + GT ({gt_xy.shape[0]})")

    axes[1].imshow(image_np)
    axes[1].imshow(target_img, cmap="magma", alpha=0.58)
    if gt_xy.size:
        axes[1].scatter(gt_xy[:, 0], gt_xy[:, 1], s=8, c="cyan", linewidths=0.0)
    axes[1].set_title(f"target density sum={float(target_density.sum()):.1f}")

    axes[2].imshow(image_np)
    axes[2].imshow(density_img, cmap="magma", alpha=0.58)
    axes[2].set_title(f"pred density sum={float(density.sum()):.1f}")

    axes[3].imshow(image_np)
    axes[3].imshow(occupancy_img, cmap="viridis", alpha=0.62)
    axes[3].set_title(f"occupancy prob max={float(occupancy_prob.max()):.2f}")

    axes[4].imshow(image_np)
    if proposal_xy.size:
        axes[4].scatter(
            proposal_xy[:, 0],
            proposal_xy[:, 1],
            s=18,
            c="#ffd60a",
            linewidths=0.45,
            edgecolors="black",
            label="proposal",
        )
    if gt_xy.size:
        axes[4].scatter(
            gt_xy[:, 0],
            gt_xy[:, 1],
            s=12,
            c="#ff3b30",
            linewidths=0.4,
            edgecolors="white",
            label="GT",
        )
    axes[4].set_title(f"proposal ({proposal_xy.shape[0]}) + GT")
    axes[4].legend(loc="lower right", fontsize=7, framealpha=0.75)

    image_name = Path(str(meta.get("image_path", "tile"))).name
    fig.suptitle(
        f"{title} | {image_name} tile={meta.get('tile_index_in_img')} "
        f"top={meta.get('tile_top')} left={meta.get('tile_left')}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/train_density_only_prior_from_0197_stage1.yaml")
    parser.add_argument(
        "--ckpt",
        default=r"D:\output\density_only_prior_from_0197_stage1\best_cover_epoch0006_cover0.6828.pth",
    )
    parser.add_argument("--save_dir", default="vis_results/proposal_prior_density_viz")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--num_tiles", type=int, default=8)
    parser.add_argument("--indices", default="")
    parser.add_argument("--pick", choices=["crowded", "first"], default="crowded")
    parser.add_argument("--mode", default="")
    parser.add_argument("--capacity", type=int, default=-1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg_dict = load_config(args.config)
    cfg = SimpleNamespace(**cfg_dict)
    mode = args.mode or str(getattr(cfg, "proposal_prior_mode", "density_only"))
    capacity = args.capacity if args.capacity > 0 else int(getattr(cfg, "proposal_prior_cell_capacity", 2))

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    train_data, val_data = build_dataset(cfg)
    dataset = train_data if args.split == "train" else val_data
    indices = pick_indices(dataset, args.num_tiles, args.pick, args.indices)

    model = build_model(cfg, training=False).to(device)
    checkpoint = torch.load(args.ckpt, map_location=device)
    incompatible = load_model_state(model, checkpoint)
    model.eval()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": args.ckpt,
        "split": args.split,
        "mode": mode,
        "capacity": capacity,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "tiles": [],
    }

    with torch.no_grad():
        for rank, idx in enumerate(indices):
            image, pts, meta = dataset[idx]
            _, height, width = image.shape
            image_b = image.unsqueeze(0).to(device)
            feats = model.encode(image_b)
            occupancy_logits, density = model.predict_proposal_prior(feats)
            k_guided = int(torch.round(density[0].sum()).clamp(0, 900).item())
            guided = select_guided_prior_points(
                occupancy_logits[0],
                density[0],
                k_guided,
                mode=mode,
                density_cell_capacity=capacity,
            )
            occ_target, density_target, gt_count = build_proposal_prior_targets(
                pts.unsqueeze(0).to(device),
                torch.ones((1, pts.size(0)), dtype=torch.bool, device=device),
                image_h=height,
                image_w=width,
                map_h=density.size(-2),
                map_w=density.size(-1),
                sigma=float(getattr(cfg, "proposal_prior_sigma", 1.0)),
            )

            image_np = tensor_image_to_numpy(image)
            gt_xy = pts.detach().cpu().numpy()
            proposal_xy = norm_xy_to_pixel(guided, width=width, height=height)
            stem = f"{rank:02d}_idx{idx:05d}_gt{pts.size(0):03d}_k{k_guided:03d}"
            out_path = save_dir / f"{stem}.png"
            draw_tile(
                image_np=image_np,
                gt_xy=gt_xy,
                density=density[0, 0],
                target_density=density_target[0, 0],
                occupancy_prob=occupancy_logits[0, 0].sigmoid(),
                proposal_xy=proposal_xy,
                meta=meta,
                out_path=out_path,
                title=f"proposal prior {mode} cap={capacity}",
            )

            flat = density[0, 0].flatten()
            topk = min(20, flat.numel())
            top_vals, top_idx = torch.topk(flat, topk)
            map_w = density.size(-1)
            top_cells = [
                {
                    "cell_x": int((int(i) % map_w)),
                    "cell_y": int((int(i) // map_w)),
                    "density": float(v),
                }
                for v, i in zip(top_vals.detach().cpu(), top_idx.detach().cpu())
            ]
            summary["tiles"].append(
                {
                    "rank": rank,
                    "dataset_index": int(idx),
                    "output": str(out_path.resolve()),
                    "image_path": str(meta.get("image_path")),
                    "tile_top": int(meta.get("tile_top", 0)),
                    "tile_left": int(meta.get("tile_left", 0)),
                    "tile_index_in_img": int(meta.get("tile_index_in_img", 0)),
                    "gt_count": int(pts.size(0)),
                    "pred_density_sum": float(density[0].sum()),
                    "k_guided": int(k_guided),
                    "proposal_count": int(guided.size(0)),
                    "target_density_sum": float(density_target[0].sum()),
                    "top_density_cells": top_cells,
                }
            )
            print(f"[SAVE] {out_path} gt={pts.size(0)} k={k_guided}")

    summary_path = save_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[DONE] summary={summary_path.resolve()}")


if __name__ == "__main__":
    main()
