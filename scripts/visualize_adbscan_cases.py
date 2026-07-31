import argparse
import csv
import os
import sys

import cv2
import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scan_adaptive_dbscan import cluster_from_precomputed, compute_local_count_features


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_path", required=True)
    parser.add_argument("--grid_csv", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--candidate_sample", type=int, default=3000)
    return parser.parse_args()


def parse_bool(text):
    return str(text).strip().lower() in {"1", "true", "yes", "y", "on"}


def load_best_params(grid_csv):
    with open(grid_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows in {grid_csv}")
    best = sorted(rows, key=lambda r: (float(r["MAE"]), abs(float(r["bias"]))))[0]
    return {
        "param_id": best["param_id"],
        "k": int(best["k"]),
        "dense_kdist_thresh": float(best["dense_kdist_thresh"]),
        "alpha": float(best["alpha"]),
        "min_eps": float(best["min_eps"]),
        "dense_min_eps": float(best["dense_min_eps"]),
        "max_eps": float(best["max_eps"]),
        "min_eps_mode": str(best["min_eps_mode"]),
        "min_samples": int(best["min_samples"]),
        "connect_rule": str(best["connect_rule"]),
        "score_power": float(best["score_power"]),
        "keep_noise": parse_bool(best["keep_noise"]),
        "count_gate_c4_dup": float(best["count_gate_c4_dup"]),
        "count_gate_c6_dup": float(best["count_gate_c6_dup"]),
        "count_gate_growth84_dense": float(best["count_gate_growth84_dense"]),
        "count_gate_growth124_dense": float(best["count_gate_growth124_dense"]),
        "count_gate_dup_min_eps": float(best["count_gate_dup_min_eps"]),
        "count_gate_dense_alpha": float(best["count_gate_dense_alpha"]),
        "count_gate_dense_max_eps": float(best["count_gate_dense_max_eps"]),
    }


def nearest_stats(pred_xy, gt_xy):
    if pred_xy.shape[0] == 0 or gt_xy.shape[0] == 0:
        return 0.0, 0.0
    diff = pred_xy[:, None, :] - gt_xy[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    d = np.sqrt(np.min(d2, axis=1))
    return float(np.percentile(d, 50)), float(np.percentile(d, 90))


def cluster_image(item, params):
    xy = item["xy"].float().cpu()
    score = item["score"].float().cpu()
    if xy.numel() == 0:
        return torch.zeros((0, 2), dtype=torch.float32), torch.zeros((0,), dtype=torch.float32)

    dist = torch.cdist(xy, xy)
    k_eff = min(max(1, int(params["k"])), max(1, xy.size(0)))
    kth = torch.topk(dist, k=k_eff, largest=False).values[:, -1]
    local_counts = compute_local_count_features(dist)
    centers, center_scores, _, _, cluster_sizes, _ = cluster_from_precomputed(
        xy,
        score,
        dist,
        kth,
        params,
        local_counts=local_counts,
    )
    return centers, cluster_sizes


def draw_points(img, points, color, radius, thickness=-1, limit=None, seed=0):
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        return
    if limit is not None and pts.shape[0] > int(limit):
        rng = np.random.default_rng(seed)
        idx = rng.choice(pts.shape[0], size=int(limit), replace=False)
        pts = pts[idx]
    h, w = img.shape[:2]
    for x, y in pts:
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if 0 <= xi < w and 0 <= yi < h:
            cv2.circle(img, (xi, yi), int(radius), color, int(thickness), lineType=cv2.LINE_AA)


def add_label(img, lines):
    pad = 8
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thick = 2
    line_h = 24
    width = 0
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, font, scale, thick)
        width = max(width, tw)
    height = pad * 2 + line_h * len(lines)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (width + pad * 2, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, dst=img)
    for i, line in enumerate(lines):
        y = pad + 18 + i * line_h
        cv2.putText(img, line, (pad, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def load_image(path, size):
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        h, w = int(size[0]), int(size[1])
        return np.zeros((h, w, 3), dtype=np.uint8)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = int(size[0]), int(size[1])
    if img.shape[0] != h or img.shape[1] != w:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return img


def save_overlay(out_path, img_key, item, centers, cluster_sizes, rank_name, candidate_sample):
    img = load_image(img_key, item.get("size", (0, 0)))
    gt_xy = item["gt_xy"].float().cpu().numpy()
    cand_xy = item["xy"].float().cpu().numpy()
    center_xy = centers.float().cpu().numpy()
    pred = int(center_xy.shape[0])
    gt = int(round(float(item["gt_count"])))
    err = pred - gt
    p50, p90 = nearest_stats(center_xy, gt_xy)

    cand_layer = img.copy()
    draw_points(cand_layer, cand_xy, (255, 210, 0), 1, -1, limit=candidate_sample, seed=abs(hash(img_key)) % (2**32))
    img = cv2.addWeighted(cand_layer, 0.35, img, 0.65, 0)
    draw_points(img, gt_xy, (0, 255, 80), 3, -1)
    draw_points(img, center_xy, (255, 40, 40), 3, -1)
    draw_points(img, center_xy, (255, 255, 255), 5, 1)

    lines = [
        f"{rank_name} | {os.path.basename(img_key)}",
        f"GT {gt}  Pred {pred}  Err {err:+d}  Abs {abs(err)}",
        f"Cand {cand_xy.shape[0]}  Clusters {pred}  p50/p90 {p50:.1f}/{p90:.1f}px",
        "green=GT  yellow=candidates(sample)  red=DBSCAN centers",
    ]
    add_label(img, lines)
    cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    cache = torch.load(args.cache_path, map_location="cpu")
    params = load_best_params(args.grid_csv)

    rows = []
    clustered = {}
    for img_key, item in cache["images"].items():
        centers, sizes = cluster_image(item, params)
        pred = int(centers.size(0))
        gt = int(round(float(item["gt_count"])))
        row = {
            "image_path": img_key,
            "gt": gt,
            "pred": pred,
            "signed_error": pred - gt,
            "abs_error": abs(pred - gt),
            "candidate_count": int(item["candidate_count"]),
        }
        rows.append(row)
        clustered[img_key] = (centers, sizes)

    rows_sorted = sorted(rows, key=lambda r: (r["abs_error"], os.path.basename(r["image_path"])))
    selected = []
    selected.extend(("best", i, row) for i, row in enumerate(rows_sorted[: args.top_k], start=1))
    worst_rows = list(reversed(rows_sorted[-args.top_k:]))
    selected.extend(("worst", i, row) for i, row in enumerate(worst_rows, start=1))

    out_rows = []
    for kind, rank, row in selected:
        img_key = row["image_path"]
        centers, sizes = clustered[img_key]
        name = f"{kind}{rank}_{os.path.splitext(os.path.basename(img_key))[0]}_gt{row['gt']}_pred{row['pred']}_err{row['signed_error']:+d}.png"
        out_path = os.path.join(args.save_dir, name)
        save_overlay(
            out_path,
            img_key,
            cache["images"][img_key],
            centers,
            sizes,
            f"{kind.upper()} {rank}",
            args.candidate_sample,
        )
        out_row = dict(row)
        out_row["kind"] = kind
        out_row["rank"] = rank
        out_row["output_path"] = out_path
        out_rows.append(out_row)
        print(f"[SAVE] {out_path}")

    csv_path = os.path.join(args.save_dir, "selected_cases.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"[SAVE] {csv_path}")
    print(f"[PARAM] {params['param_id']}")


if __name__ == "__main__":
    main()
