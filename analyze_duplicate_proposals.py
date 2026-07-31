import argparse
import csv
import os

import numpy as np
import torch
import yaml

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []


def parse_csv_floats(text):
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def radius_tag(radius):
    text = f"{float(radius):g}"
    return text.replace(".", "p")


def parse_args():
    def load_config(yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    base = argparse.ArgumentParser()
    base.add_argument("--config", default=r"config/train.yaml", type=str)
    args0, _ = base.parse_known_args()

    cfg = load_config(args0.config)
    parser = argparse.ArgumentParser(parents=[base], add_help=False)
    for key, value in cfg.items():
        parser.add_argument(f"--{key}", type=type(value), default=value)

    parser.add_argument("--dup_cache_path", type=str, default="")
    parser.add_argument("--dup_save_dir", type=str, default="")
    parser.add_argument("--dup_radii", type=str, default="1,2,4,6,8,10,12,16")
    parser.add_argument("--dense_gt_dist", type=float, default=8.0)
    parser.add_argument("--sparse_gt_dist", type=float, default=16.0)
    parser.add_argument("--watch_images", type=str, default="IMG_165.jpg,IMG_36.jpg,IMG_104.jpg,IMG_50.jpg")

    return parser.parse_args()


def load_torch(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def find_latest_cache(args):
    if args.dup_cache_path:
        return args.dup_cache_path

    default = os.path.join(f"{args.save_dir}_ADBSCAN_SCAN", "adaptive_dbscan_candidate_cache.pt")
    if os.path.exists(default):
        return default

    search_root = os.path.join(os.getcwd(), "vis_results")
    matches = []
    for root, _, files in os.walk(search_root):
        if "adaptive_dbscan_candidate_cache.pt" in files:
            path = os.path.join(root, "adaptive_dbscan_candidate_cache.pt")
            matches.append((os.path.getmtime(path), path))
    if not matches:
        raise FileNotFoundError(
            "Cannot find adaptive_dbscan_candidate_cache.pt. "
            "Pass --dup_cache_path or run scan_adaptive_dbscan.py once with --refresh_cache."
        )
    matches.sort(reverse=True)
    return matches[0][1]


def percentile(values, p, default=0.0):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return float(default)
    return float(np.percentile(values, p))


def finite_percentile(values, p, default=-1.0):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(default)
    return float(np.percentile(values, p))


def nearest_gt_distances(gt_xy):
    n = int(gt_xy.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.float32)
    if n == 1:
        return np.full((1,), np.inf, dtype=np.float32)

    diff = gt_xy[:, None, :] - gt_xy[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    return np.min(dist, axis=1).astype(np.float32)


def group_name(nearest_gt_dist, dense_gt_dist, sparse_gt_dist):
    if not np.isfinite(nearest_gt_dist):
        return "isolated"
    if nearest_gt_dist <= float(dense_gt_dist):
        return "dense_gt"
    if nearest_gt_dist >= float(sparse_gt_dist):
        return "sparse_gt"
    return "mid_gt"


def score_stats(scores):
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return 0.0, 0.0, 0.0
    return float(np.max(scores)), float(np.mean(scores)), float(np.percentile(scores, 90))


def compute_gt_rows(cache, radii, args):
    rows = []
    image_rows = []
    watch = {x.strip() for x in str(args.watch_images).split(",") if x.strip()}
    watch_rows = []

    for image_path, item in tqdm(cache["images"].items(), total=len(cache["images"]), desc="GT duplicate stats", dynamic_ncols=True):
        xy = item["xy"].detach().cpu().numpy().astype(np.float32)
        score = item["score"].detach().cpu().numpy().astype(np.float32)
        gt_xy = item["gt_xy"].detach().cpu().numpy().astype(np.float32)
        nearest_gt = nearest_gt_distances(gt_xy)

        per_image_counts = {radius_tag(r): [] for r in radii}
        per_image_nearest_prop = []

        if gt_xy.shape[0] > 0 and xy.shape[0] > 0:
            diff = gt_xy[:, None, :] - xy[None, :, :]
            dist = np.sqrt(np.sum(diff * diff, axis=2)).astype(np.float32)
        else:
            dist = np.zeros((gt_xy.shape[0], xy.shape[0]), dtype=np.float32)

        for gt_idx in range(gt_xy.shape[0]):
            row = {
                "image_path": image_path,
                "image_name": os.path.basename(str(image_path)),
                "gt_idx": int(gt_idx),
                "gt_x": float(gt_xy[gt_idx, 0]),
                "gt_y": float(gt_xy[gt_idx, 1]),
                "nearest_gt_dist": float(nearest_gt[gt_idx]) if np.isfinite(nearest_gt[gt_idx]) else -1.0,
                "gt_group": group_name(nearest_gt[gt_idx], args.dense_gt_dist, args.sparse_gt_dist),
                "candidate_count_image": int(item["candidate_count"]),
                "proposal_count_image": int(item["proposal_count"]),
                "raw_slot_count_image": int(item.get("raw_slot_count", -1)),
                "merged_valid_count_image": int(item.get("merged_valid_count", item["proposal_count"])),
                "merge_drop_count_image": int(item.get("merge_drop_count", -1)),
                "threshold_count_image": int(item.get("threshold_count", item["candidate_count"])),
            }

            if xy.shape[0] > 0:
                d_gt = dist[gt_idx]
                nearest_prop_dist = float(np.min(d_gt))
            else:
                d_gt = np.zeros((0,), dtype=np.float32)
                nearest_prop_dist = -1.0
            row["nearest_prop_dist"] = nearest_prop_dist
            per_image_nearest_prop.append(nearest_prop_dist)

            for radius in radii:
                tag = radius_tag(radius)
                inside = d_gt <= float(radius)
                count = int(np.sum(inside))
                smax, smean, sp90 = score_stats(score[inside] if score.size else [])
                row[f"count_r{tag}"] = count
                row[f"score_max_r{tag}"] = smax
                row[f"score_mean_r{tag}"] = smean
                row[f"score_p90_r{tag}"] = sp90
                per_image_counts[tag].append(count)

            rows.append(row)
            if os.path.basename(str(image_path)) in watch:
                watch_rows.append(row)

        raw_slot_count = int(item.get("raw_slot_count", 0))
        merged_valid_count = int(item.get("merged_valid_count", item["proposal_count"]))
        merge_drop_count = int(item.get("merge_drop_count", max(0, raw_slot_count - merged_valid_count)))
        threshold_count = int(item.get("threshold_count", item["candidate_count"]))
        candidate_count = int(item["candidate_count"])
        image_row = {
            "image_path": image_path,
            "image_name": os.path.basename(str(image_path)),
            "gt_count": int(gt_xy.shape[0]),
            "raw_slot_count": raw_slot_count,
            "merged_valid_count": merged_valid_count,
            "merge_drop_count": merge_drop_count,
            "threshold_count": threshold_count,
            "candidate_count": candidate_count,
            "proposal_count": int(item["proposal_count"]),
            "merge_keep_ratio": float(merged_valid_count / raw_slot_count) if raw_slot_count > 0 else -1.0,
            "threshold_keep_ratio": float(threshold_count / merged_valid_count) if merged_valid_count > 0 else -1.0,
            "candidate_keep_ratio": float(candidate_count / merged_valid_count) if merged_valid_count > 0 else -1.0,
            "nearest_gt_p50": finite_percentile(nearest_gt, 50),
            "nearest_gt_p90": finite_percentile(nearest_gt, 90),
            "nearest_prop_p50": finite_percentile(per_image_nearest_prop, 50),
            "nearest_prop_p90": finite_percentile(per_image_nearest_prop, 90),
        }
        for radius in radii:
            tag = radius_tag(radius)
            counts = per_image_counts[tag]
            image_row[f"count_r{tag}_mean"] = float(np.mean(counts)) if counts else 0.0
            image_row[f"count_r{tag}_p50"] = percentile(counts, 50)
            image_row[f"count_r{tag}_p75"] = percentile(counts, 75)
            image_row[f"count_r{tag}_p90"] = percentile(counts, 90)
            image_row[f"count_r{tag}_p95"] = percentile(counts, 95)
            image_row[f"count_r{tag}_max"] = float(np.max(counts)) if counts else 0.0
        image_rows.append(image_row)

    return rows, image_rows, watch_rows


def summarize_gt_rows(rows, radii):
    groups = ["all", "dense_gt", "mid_gt", "sparse_gt", "isolated"]
    out = []

    for group in groups:
        if group == "all":
            group_rows = rows
        else:
            group_rows = [r for r in rows if r["gt_group"] == group]

        summary = {
            "group": group,
            "n_gt": len(group_rows),
            "nearest_gt_p50": finite_percentile([r["nearest_gt_dist"] for r in group_rows if r["nearest_gt_dist"] >= 0], 50),
            "nearest_gt_p75": finite_percentile([r["nearest_gt_dist"] for r in group_rows if r["nearest_gt_dist"] >= 0], 75),
            "nearest_gt_p90": finite_percentile([r["nearest_gt_dist"] for r in group_rows if r["nearest_gt_dist"] >= 0], 90),
            "nearest_gt_p95": finite_percentile([r["nearest_gt_dist"] for r in group_rows if r["nearest_gt_dist"] >= 0], 95),
            "nearest_prop_p50": finite_percentile([r["nearest_prop_dist"] for r in group_rows if r["nearest_prop_dist"] >= 0], 50),
            "nearest_prop_p90": finite_percentile([r["nearest_prop_dist"] for r in group_rows if r["nearest_prop_dist"] >= 0], 90),
        }

        for radius in radii:
            tag = radius_tag(radius)
            counts = [r[f"count_r{tag}"] for r in group_rows]
            summary[f"count_r{tag}_mean"] = float(np.mean(counts)) if counts else 0.0
            summary[f"count_r{tag}_p50"] = percentile(counts, 50)
            summary[f"count_r{tag}_p75"] = percentile(counts, 75)
            summary[f"count_r{tag}_p90"] = percentile(counts, 90)
            summary[f"count_r{tag}_p95"] = percentile(counts, 95)
            summary[f"count_r{tag}_max"] = float(np.max(counts)) if counts else 0.0
            summary[f"recommended_k_gt_r{tag}_p90_plus1"] = int(np.ceil(summary[f"count_r{tag}_p90"] + 1))
            summary[f"recommended_k_gt_r{tag}_p95_plus1"] = int(np.ceil(summary[f"count_r{tag}_p95"] + 1))
        out.append(summary)

    return out


def write_csv(path, rows, fallback_fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else fallback_fields
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    radii = parse_csv_floats(args.dup_radii)
    if not radii:
        raise ValueError("--dup_radii must contain at least one radius.")

    cache_path = find_latest_cache(args)
    save_dir = args.dup_save_dir or os.path.join(os.path.dirname(cache_path), "DUPLICATE_STATS")
    os.makedirs(save_dir, exist_ok=True)

    print(f"[LOAD-CACHE] {cache_path}")
    print("[NOTE] This cache stores gated candidate proposal coordinates, not all raw proposals.")
    cache = load_torch(cache_path)

    gt_rows, image_rows, watch_rows = compute_gt_rows(cache, radii, args)
    summary_rows = summarize_gt_rows(gt_rows, radii)

    per_gt_path = os.path.join(save_dir, "gt_duplicate_stats.csv")
    per_image_path = os.path.join(save_dir, "image_duplicate_stats.csv")
    summary_path = os.path.join(save_dir, "duplicate_summary.csv")
    watch_path = os.path.join(save_dir, "watch_gt_duplicate_stats.csv")

    write_csv(per_gt_path, gt_rows, fallback_fields=["image_path", "gt_idx"])
    write_csv(per_image_path, image_rows, fallback_fields=["image_path"])
    write_csv(summary_path, summary_rows, fallback_fields=["group", "n_gt"])
    write_csv(watch_path, watch_rows, fallback_fields=["image_path", "gt_idx"])

    print(f"[SAVE-CSV] {per_gt_path}")
    print(f"[SAVE-CSV] {per_image_path}")
    print(f"[SAVE-CSV] {summary_path}")
    print(f"[SAVE-CSV] {watch_path}")

    print("\n[Summary]")
    if image_rows and any(int(r.get("raw_slot_count", 0)) > 0 for r in image_rows):
        raw_total = sum(int(r["raw_slot_count"]) for r in image_rows)
        merged_total = sum(int(r["merged_valid_count"]) for r in image_rows)
        threshold_total = sum(int(r["threshold_count"]) for r in image_rows)
        candidate_total = sum(int(r["candidate_count"]) for r in image_rows)
        print("  merge/cache funnel:")
        print(f"    raw slots total = {raw_total}")
        print(f"    merged valid total = {merged_total}  keep={merged_total / max(raw_total, 1):.4f}")
        print(f"    threshold total = {threshold_total}  keep={threshold_total / max(merged_total, 1):.4f}")
        print(f"    candidate total = {candidate_total}  keep={candidate_total / max(merged_total, 1):.4f}")
    for row in summary_rows:
        if row["group"] == "all":
            print(f"  all GT: n={row['n_gt']}")
            for radius in radii:
                tag = radius_tag(radius)
                print(
                    f"    r={radius:g}: "
                    f"count p50={row[f'count_r{tag}_p50']:.1f}, "
                    f"p90={row[f'count_r{tag}_p90']:.1f}, "
                    f"p95={row[f'count_r{tag}_p95']:.1f}, "
                    f"k_p90+1={row[f'recommended_k_gt_r{tag}_p90_plus1']}, "
                    f"k_p95+1={row[f'recommended_k_gt_r{tag}_p95_plus1']}"
                )


if __name__ == "__main__":
    main()
