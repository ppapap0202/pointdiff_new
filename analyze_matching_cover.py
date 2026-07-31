import argparse
import csv
import os

import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []


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

    parser.add_argument("--cache_path", type=str, default="")
    parser.add_argument("--cover_save_dir", type=str, default="")
    parser.add_argument("--cover_radius", type=float, default=6.0)
    parser.add_argument("--watch_images", type=str, default="IMG_165.jpg,IMG_36.jpg,IMG_104.jpg,IMG_50.jpg,IMG_90.jpg")
    return parser.parse_args()


def load_torch(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def find_latest_cache(args):
    if args.cache_path:
        return args.cache_path

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
        raise FileNotFoundError("Cannot find adaptive_dbscan_candidate_cache.pt. Pass --cache_path.")
    matches.sort(reverse=True)
    return matches[0][1]


def as_numpy_xy(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return arr.reshape(0, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected xy array with shape [N,2], got {arr.shape}")
    return arr


def first_existing_xy(item, keys):
    for key in keys:
        if key in item:
            return key, as_numpy_xy(item[key])
    return "", None


def matching_cover_stats(gt_xy, prop_xy, radius):
    gt_xy = as_numpy_xy(gt_xy)
    prop_xy = as_numpy_xy(prop_xy)
    n_gt = int(gt_xy.shape[0])
    n_prop = int(prop_xy.shape[0])
    if n_gt == 0:
        return {
            "gt_count": 0,
            "point_count": n_prop,
            "matched_gt": 0,
            "cover_ratio": 0.0,
            "uncovered_gt": 0,
        }
    if n_prop == 0:
        return {
            "gt_count": n_gt,
            "point_count": 0,
            "matched_gt": 0,
            "cover_ratio": 0.0,
            "uncovered_gt": n_gt,
        }

    tree = cKDTree(prop_xy)
    indptr = [0]
    indices = []
    for gt in gt_xy:
        cols = tree.query_ball_point(gt, float(radius))
        if cols:
            indices.extend(cols)
        indptr.append(len(indices))

    data = np.ones(len(indices), dtype=np.bool_)
    graph = csr_matrix((data, np.asarray(indices, dtype=np.int32), np.asarray(indptr, dtype=np.int32)),
                       shape=(n_gt, n_prop))
    if graph.nnz == 0:
        matched_gt = 0
    else:
        # For a [gt, proposal] graph, perm_type="column" returns one matched proposal
        # index per GT row, with -1 for unmatched GTs.
        match = maximum_bipartite_matching(graph, perm_type="column")
        matched_gt = int(np.sum(match >= 0))

    return {
        "gt_count": n_gt,
        "point_count": n_prop,
        "matched_gt": matched_gt,
        "cover_ratio": float(matched_gt / max(n_gt, 1)),
        "uncovered_gt": int(n_gt - matched_gt),
    }


def write_csv(path, rows, fallback_fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else fallback_fields
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(stage_rows):
    by_stage = {}
    for row in stage_rows:
        stage = row["stage"]
        by_stage.setdefault(stage, []).append(row)

    summary = []
    for stage, rows in by_stage.items():
        available_rows = [r for r in rows if int(r["available"]) == 1]
        total_gt = sum(int(r["gt_count"]) for r in available_rows)
        total_points = sum(int(r["point_count"]) for r in available_rows)
        total_matched = sum(int(r["matched_gt"]) for r in available_rows)
        total_uncovered = sum(int(r["uncovered_gt"]) for r in available_rows)
        image_ratios = [float(r["cover_ratio"]) for r in available_rows if int(r["gt_count"]) > 0]
        summary.append({
            "stage": stage,
            "available": 1 if available_rows else 0,
            "source_key": available_rows[0]["source_key"] if available_rows else "",
            "num_images": len(available_rows),
            "total_gt": total_gt,
            "total_points": total_points,
            "matched_gt": total_matched,
            "uncovered_gt": total_uncovered,
            "global_matching_cover_ratio": float(total_matched / max(total_gt, 1)) if available_rows else -1.0,
            "image_cover_ratio_mean": float(np.mean(image_ratios)) if image_ratios else -1.0,
            "image_cover_ratio_p50": float(np.percentile(image_ratios, 50)) if image_ratios else -1.0,
            "image_cover_ratio_p90": float(np.percentile(image_ratios, 90)) if image_ratios else -1.0,
            "mean_points_per_image": float(total_points / max(len(available_rows), 1)) if available_rows else -1.0,
        })
    return summary


def main():
    args = parse_args()
    cache_path = find_latest_cache(args)
    save_dir = args.cover_save_dir or os.path.join(os.path.dirname(cache_path), "MATCHING_COVER")
    os.makedirs(save_dir, exist_ok=True)

    print(f"[LOAD-CACHE] {cache_path}")
    cache = load_torch(cache_path)
    images = cache.get("images", {})
    if not images:
        raise ValueError("Cache does not contain an 'images' dictionary.")

    stage_specs = [
        ("raw", ["raw_xy", "raw_points_xy", "raw_slot_xy", "raw_slots_xy"]),
        ("merged", ["merged_xy", "merged_points_xy", "merged_valid_xy", "proposal_xy", "proposal_points_xy"]),
        ("candidate", ["xy", "candidate_xy", "candidate_points_xy"]),
    ]
    watch = {x.strip() for x in str(args.watch_images).split(",") if x.strip()}
    stage_rows = []
    watch_rows = []

    for image_path, item in tqdm(images.items(), total=len(images), desc="Matching cover", dynamic_ncols=True):
        gt_xy = as_numpy_xy(item["gt_xy"])
        image_name = os.path.basename(str(image_path))
        for stage, keys in stage_specs:
            source_key, prop_xy = first_existing_xy(item, keys)
            if prop_xy is None:
                row = {
                    "image_path": image_path,
                    "image_name": image_name,
                    "stage": stage,
                    "available": 0,
                    "source_key": "",
                    "cover_radius": float(args.cover_radius),
                    "gt_count": int(gt_xy.shape[0]),
                    "point_count": -1,
                    "matched_gt": -1,
                    "uncovered_gt": -1,
                    "cover_ratio": -1.0,
                    "candidate_count": int(item.get("candidate_count", -1)),
                    "proposal_count": int(item.get("proposal_count", -1)),
                    "raw_slot_count": int(item.get("raw_slot_count", -1)),
                    "merged_valid_count": int(item.get("merged_valid_count", -1)),
                }
            else:
                stats = matching_cover_stats(gt_xy, prop_xy, args.cover_radius)
                row = {
                    "image_path": image_path,
                    "image_name": image_name,
                    "stage": stage,
                    "available": 1,
                    "source_key": source_key,
                    "cover_radius": float(args.cover_radius),
                    "gt_count": int(stats["gt_count"]),
                    "point_count": int(stats["point_count"]),
                    "matched_gt": int(stats["matched_gt"]),
                    "uncovered_gt": int(stats["uncovered_gt"]),
                    "cover_ratio": float(stats["cover_ratio"]),
                    "candidate_count": int(item.get("candidate_count", -1)),
                    "proposal_count": int(item.get("proposal_count", -1)),
                    "raw_slot_count": int(item.get("raw_slot_count", -1)),
                    "merged_valid_count": int(item.get("merged_valid_count", -1)),
                }
            stage_rows.append(row)
            if image_name in watch:
                watch_rows.append(row)

    summary_rows = summarize(stage_rows)
    per_image_path = os.path.join(save_dir, "matching_cover_per_image.csv")
    summary_path = os.path.join(save_dir, "matching_cover_summary.csv")
    watch_path = os.path.join(save_dir, "matching_cover_watch_images.csv")
    write_csv(per_image_path, stage_rows, fallback_fields=["image_path", "stage"])
    write_csv(summary_path, summary_rows, fallback_fields=["stage", "available"])
    write_csv(watch_path, watch_rows, fallback_fields=["image_path", "stage"])

    print(f"[SAVE-CSV] {per_image_path}")
    print(f"[SAVE-CSV] {summary_path}")
    print(f"[SAVE-CSV] {watch_path}")

    print("\n[Summary]")
    print(f"  cover_radius = {float(args.cover_radius):g}")
    for row in summary_rows:
        stage = row["stage"]
        if int(row["available"]) == 0:
            print(f"  {stage}: unavailable in this cache (missing coordinate key)")
            continue
        print(
            f"  {stage}: source={row['source_key']} "
            f"global={row['global_matching_cover_ratio']:.4f} "
            f"image_mean={row['image_cover_ratio_mean']:.4f} "
            f"matched={row['matched_gt']}/{row['total_gt']} "
            f"mean_points={row['mean_points_per_image']:.1f}"
        )

    missing = [r["stage"] for r in summary_rows if int(r["available"]) == 0]
    if missing:
        print("\n[NOTE]")
        print(
            "  Exact raw/merged matching cover needs raw_xy/merged_xy coordinates in the cache. "
            "This cache only has coordinate data for stages marked available."
        )


if __name__ == "__main__":
    main()
