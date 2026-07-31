import argparse
import csv
import os
from collections import deque

import numpy as np
import torch
import yaml

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

FEATURE_NAMES = [
    "log_d1",
    "log_d4",
    "log_d8",
    "log_d15",
    "log_ratio_4_1",
    "log_ratio_8_4",
    "log_ratio_15_8",
    "log_count1",
    "log_count2",
    "log_count4",
    "log_count8",
    "score",
]


def nearest_gt_distance_stats(pred_xy, gt_xy):
    out = {
        "count": int(pred_xy.shape[0]),
        "mean": 0.0,
        "p50": 0.0,
        "p90": 0.0,
        "p95": 0.0,
    }
    if pred_xy.shape[0] == 0 or gt_xy.shape[0] == 0:
        return out

    diff = pred_xy[:, None, :] - gt_xy[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    dmin = np.sqrt(np.min(d2, axis=1))

    out["mean"] = float(np.mean(dmin))
    out["p50"] = float(np.percentile(dmin, 50))
    out["p90"] = float(np.percentile(dmin, 90))
    out["p95"] = float(np.percentile(dmin, 95))
    return out


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

    parser.add_argument("--gmm_cache_path", type=str, default="")
    parser.add_argument("--gmm_save_dir", type=str, default="")
    parser.add_argument("--gmm_fit_max_points", type=int, default=200000)
    parser.add_argument("--gmm_components", type=int, default=6)
    parser.add_argument("--gmm_iters", type=int, default=60)
    parser.add_argument("--gmm_reg_covar", type=float, default=1e-4)
    parser.add_argument("--gmm_seed", type=int, default=7113064165)

    parser.add_argument("--gmm_dense_alpha", type=float, default=0.35)
    parser.add_argument("--gmm_dense_min_eps", type=float, default=1.0)
    parser.add_argument("--gmm_dense_max_eps", type=float, default=2.0)
    parser.add_argument("--gmm_dup_alpha", type=float, default=1.1)
    parser.add_argument("--gmm_dup_min_eps", type=float, default=4.0)
    parser.add_argument("--gmm_dup_max_eps", type=float, default=8.0)
    parser.add_argument("--gmm_sparse_alpha", type=float, default=0.9)
    parser.add_argument("--gmm_sparse_min_eps", type=float, default=3.0)
    parser.add_argument("--gmm_sparse_max_eps", type=float, default=8.0)
    parser.add_argument("--gmm_eps_min", type=float, default=1.0)
    parser.add_argument("--gmm_eps_max", type=float, default=8.0)

    parser.add_argument("--gmm_min_samples", type=int, default=1)
    parser.add_argument("--gmm_connect_rule", type=str, default="mutual", choices=["mutual", "either", "mean"])
    parser.add_argument("--gmm_score_power", type=float, default=2.0)
    parser.add_argument("--gmm_drop_noise", action="store_true")
    parser.add_argument("--watch_images", type=str, default="IMG_165.jpg,IMG_36.jpg,IMG_104.jpg,IMG_50.jpg")

    return parser.parse_args()


def load_torch(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def find_latest_cache(args):
    if args.gmm_cache_path:
        return args.gmm_cache_path

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
            "Pass --gmm_cache_path or run scan_adaptive_dbscan.py once with --refresh_cache."
        )
    matches.sort(reverse=True)
    return matches[0][1]


def safe_kth(dist_for_knn, k):
    n = int(dist_for_knn.size(0))
    kth = max(1, min(int(k), max(1, n - 1)))
    return torch.kthvalue(dist_for_knn, kth, dim=1).values


def compute_features(xy, score):
    xy = xy.float().cpu()
    score = score.float().cpu().clamp(0.0, 1.0)
    n = int(xy.size(0))

    if n == 0:
        empty = np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
        d_map = {
            "d1": torch.zeros((0,), dtype=torch.float32),
            "d4": torch.zeros((0,), dtype=torch.float32),
            "d8": torch.zeros((0,), dtype=torch.float32),
            "d15": torch.zeros((0,), dtype=torch.float32),
        }
        return empty, d_map

    if n == 1:
        raw = np.zeros((1, len(FEATURE_NAMES)), dtype=np.float32)
        raw[0, FEATURE_NAMES.index("score")] = float(score[0].item())
        one = torch.ones((1,), dtype=torch.float32)
        d_map = {"d1": one, "d4": one, "d8": one, "d15": one}
        return raw, d_map

    dist = torch.cdist(xy, xy)
    dist_for_knn = dist.clone()
    dist_for_knn.fill_diagonal_(float("inf"))

    d1 = safe_kth(dist_for_knn, 1)
    d4 = safe_kth(dist_for_knn, 4)
    d8 = safe_kth(dist_for_knn, 8)
    d15 = safe_kth(dist_for_knn, 15)

    counts = {}
    for r in [1.0, 2.0, 4.0, 8.0]:
        counts[r] = (dist_for_knn <= r).sum(dim=1).float()

    eps = 1e-3
    raw = np.stack([
        np.log1p(d1.numpy()),
        np.log1p(d4.numpy()),
        np.log1p(d8.numpy()),
        np.log1p(d15.numpy()),
        np.log((d4.numpy() + eps) / (d1.numpy() + eps)),
        np.log((d8.numpy() + eps) / (d4.numpy() + eps)),
        np.log((d15.numpy() + eps) / (d8.numpy() + eps)),
        np.log1p(counts[1.0].numpy()),
        np.log1p(counts[2.0].numpy()),
        np.log1p(counts[4.0].numpy()),
        np.log1p(counts[8.0].numpy()),
        score.numpy(),
    ], axis=1).astype(np.float32)

    d_map = {"d1": d1, "d4": d4, "d8": d8, "d15": d15}
    return raw, d_map


def standardize_fit(x):
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def standardize_apply(x, mean, std):
    return ((x - mean) / std).astype(np.float32)


def select_unique_indices(scores_list):
    selected = []
    used = set()
    for scores in scores_list:
        order = np.argsort(-scores)
        pick = None
        for idx in order:
            idx = int(idx)
            if idx not in used:
                pick = idx
                break
        if pick is None:
            pick = int(order[0])
        used.add(pick)
        selected.append(pick)
    return selected


def init_gmm_means(x_std, x_raw, n_components):
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    dup_score = (
        x_raw[:, idx["log_count1"]]
        + 0.8 * x_raw[:, idx["log_count2"]]
        + 0.7 * x_raw[:, idx["log_ratio_4_1"]]
        + 0.5 * x_raw[:, idx["log_ratio_15_8"]]
        - 0.2 * x_raw[:, idx["log_d4"]]
    )
    dense_score = (
        x_raw[:, idx["log_count4"]]
        + x_raw[:, idx["log_count8"]]
        - 0.8 * x_raw[:, idx["log_count1"]]
        - 0.5 * x_raw[:, idx["log_d15"]]
    )
    sparse_score = (
        x_raw[:, idx["log_d8"]]
        + x_raw[:, idx["log_d15"]]
        - x_raw[:, idx["log_count4"]]
    )
    picks = select_unique_indices([dense_score, dup_score, sparse_score])
    n_components = max(3, min(int(n_components), int(x_std.shape[0])))

    while len(picks) < n_components:
        current = x_std[picks]
        d2 = ((x_std[:, None, :] - current[None, :, :]) ** 2).sum(axis=2).min(axis=1)
        d2[picks] = -np.inf
        picks.append(int(np.argmax(d2)))

    return x_std[picks].copy()


def logsumexp(a, axis=1, keepdims=True):
    m = np.max(a, axis=axis, keepdims=True)
    return m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=keepdims) + 1e-12)


def fit_diag_gmm(x_std, x_raw, n_components=6, n_iter=60, reg_covar=1e-4):
    n, d = x_std.shape
    k = max(3, min(int(n_components), int(n)))
    means = init_gmm_means(x_std, x_raw, n_components=k)
    variances = np.ones((k, d), dtype=np.float32)
    weights = np.full((k,), 1.0 / k, dtype=np.float32)

    for _ in range(int(n_iter)):
        log_prob = []
        for c in range(k):
            var = np.maximum(variances[c], reg_covar)
            lp = -0.5 * (
                np.sum(np.log(2.0 * np.pi * var))
                + np.sum(((x_std - means[c]) ** 2) / var, axis=1)
            )
            log_prob.append(np.log(weights[c] + 1e-12) + lp)
        log_prob = np.stack(log_prob, axis=1)
        log_norm = logsumexp(log_prob, axis=1, keepdims=True)
        resp = np.exp(log_prob - log_norm).astype(np.float32)

        nk = resp.sum(axis=0) + 1e-6
        weights = (nk / n).astype(np.float32)
        means = ((resp.T @ x_std) / nk[:, None]).astype(np.float32)
        for c in range(k):
            diff = x_std - means[c]
            variances[c] = ((resp[:, c][:, None] * diff * diff).sum(axis=0) / nk[c]).astype(np.float32)
        variances = np.maximum(variances, reg_covar)

    return {"weights": weights, "means": means, "variances": variances}


def gmm_predict_proba(model, x_std):
    weights = model["weights"]
    means = model["means"]
    variances = model["variances"]
    k = int(weights.shape[0])
    log_prob = []
    for c in range(k):
        var = np.maximum(variances[c], 1e-8)
        lp = -0.5 * (
            np.sum(np.log(2.0 * np.pi * var))
            + np.sum(((x_std - means[c]) ** 2) / var, axis=1)
        )
        log_prob.append(np.log(weights[c] + 1e-12) + lp)
    log_prob = np.stack(log_prob, axis=1)
    log_norm = logsumexp(log_prob, axis=1, keepdims=True)
    return np.exp(log_prob - log_norm).astype(np.float32)


def zscore(values):
    values = np.asarray(values, dtype=np.float32)
    return (values - values.mean()) / (values.std() + 1e-6)


def assign_roles(component_raw_means):
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    sparse_score = (
        component_raw_means[:, idx["log_d8"]]
        + component_raw_means[:, idx["log_d15"]]
        - component_raw_means[:, idx["log_count4"]]
    )
    dup_score = (
        component_raw_means[:, idx["log_count1"]]
        + 0.8 * component_raw_means[:, idx["log_count2"]]
        + 0.7 * component_raw_means[:, idx["log_ratio_4_1"]]
        + 0.5 * component_raw_means[:, idx["log_ratio_15_8"]]
        - 0.2 * component_raw_means[:, idx["log_d4"]]
    )
    dense_score = (
        component_raw_means[:, idx["log_count4"]]
        + component_raw_means[:, idx["log_count8"]]
        - 0.8 * component_raw_means[:, idx["log_count1"]]
        - 0.5 * component_raw_means[:, idx["log_d15"]]
    )

    k = int(component_raw_means.shape[0])
    role_names = ["dense", "duplicate", "sparse"]
    role_scores = np.stack([
        zscore(dense_score),
        zscore(dup_score),
        zscore(sparse_score),
    ], axis=1)

    sparse = int(np.argmax(role_scores[:, 2]))
    remaining = [i for i in range(k) if i != sparse]
    duplicate = max(remaining, key=lambda c: float(role_scores[c, 1]))
    remaining = [i for i in remaining if i != duplicate]
    dense = max(remaining, key=lambda c: float(role_scores[c, 0])) if remaining else sparse

    labels = np.full((k,), -1, dtype=np.int64)
    labels[dense] = 0
    labels[duplicate] = 1
    labels[sparse] = 2

    for comp in range(k):
        if labels[comp] < 0:
            labels[comp] = int(np.argmax(role_scores[comp]))

    roles = {
        name: np.where(labels == rid)[0].astype(np.int64).tolist()
        for rid, name in enumerate(role_names)
    }
    return roles, labels, role_scores


def compute_soft_radii(args, d_map, proba_by_role):
    d8 = d_map["d8"].numpy().astype(np.float32)
    d15 = d_map["d15"].numpy().astype(np.float32)

    eps_dense = np.clip(
        float(args.gmm_dense_alpha) * d8,
        float(args.gmm_dense_min_eps),
        float(args.gmm_dense_max_eps),
    )
    eps_dup = np.clip(
        float(args.gmm_dup_alpha) * d15,
        float(args.gmm_dup_min_eps),
        float(args.gmm_dup_max_eps),
    )
    eps_sparse = np.clip(
        float(args.gmm_sparse_alpha) * d8,
        float(args.gmm_sparse_min_eps),
        float(args.gmm_sparse_max_eps),
    )

    radii = (
        proba_by_role["dense"] * eps_dense
        + proba_by_role["duplicate"] * eps_dup
        + proba_by_role["sparse"] * eps_sparse
    )
    radii = np.clip(radii, float(args.gmm_eps_min), float(args.gmm_eps_max))
    return torch.from_numpy(radii.astype(np.float32))


def cluster_with_radii(pts, scores, dist, radii, args):
    pts = pts.float().cpu()
    scores = scores.float().cpu().clamp_min(0.0)
    n = int(pts.size(0))
    if n == 0:
        empty_xy = torch.zeros((0, 2), dtype=torch.float32)
        empty_1d = torch.zeros((0,), dtype=torch.float32)
        empty_long = torch.zeros((0,), dtype=torch.long)
        return empty_xy, empty_1d, empty_long, 0
    if n == 1:
        return pts.clone(), scores.clone(), torch.ones((1,), dtype=torch.long), 0

    ri = radii[:, None]
    rj = radii[None, :]
    if args.gmm_connect_rule == "either":
        eps_pair = torch.maximum(ri, rj)
    elif args.gmm_connect_rule == "mean":
        eps_pair = 0.5 * (ri + rj)
    else:
        eps_pair = torch.minimum(ri, rj)

    adjacency = dist <= eps_pair
    adjacency.fill_diagonal_(True)
    core = adjacency.sum(dim=1) >= max(1, int(args.gmm_min_samples))

    labels = torch.full((n,), -1, dtype=torch.long)
    visited = torch.zeros((n,), dtype=torch.bool)
    adjacency_np = adjacency.numpy()
    core_np = core.numpy()
    labels_np = labels.numpy()
    visited_np = visited.numpy()

    cluster_id = 0
    for start in range(n):
        if visited_np[start]:
            continue
        visited_np[start] = True
        if not core_np[start]:
            continue
        labels_np[start] = cluster_id
        q = deque(np.flatnonzero(adjacency_np[start]).tolist())
        while q:
            j = q.popleft()
            if not visited_np[j]:
                visited_np[j] = True
                if core_np[j]:
                    q.extend(np.flatnonzero(adjacency_np[j]).tolist())
            if labels_np[j] < 0:
                labels_np[j] = cluster_id
        cluster_id += 1

    labels = torch.from_numpy(labels_np.copy()).long()
    noise_mask = labels < 0
    num_noise = int(noise_mask.sum().item())

    centers = []
    center_scores = []
    sizes = []
    for cid in range(cluster_id):
        idx = torch.nonzero(labels == cid, as_tuple=False).squeeze(1)
        weights = scores[idx].clamp_min(1e-6).pow(float(args.gmm_score_power))
        centers.append((pts[idx] * weights[:, None]).sum(dim=0) / weights.sum())
        center_scores.append(scores[idx].max())
        sizes.append(torch.tensor(idx.numel(), dtype=torch.long))

    if not bool(args.gmm_drop_noise) and num_noise > 0:
        noise_idx = torch.nonzero(noise_mask, as_tuple=False).squeeze(1)
        for idx in noise_idx.tolist():
            centers.append(pts[idx])
            center_scores.append(scores[idx])
            sizes.append(torch.tensor(1, dtype=torch.long))

    if not centers:
        empty_xy = torch.zeros((0, 2), dtype=torch.float32)
        empty_1d = torch.zeros((0,), dtype=torch.float32)
        empty_long = torch.zeros((0,), dtype=torch.long)
        return empty_xy, empty_1d, empty_long, num_noise

    centers = torch.stack(centers, dim=0).float()
    center_scores = torch.stack(center_scores, dim=0).float()
    sizes = torch.stack(sizes, dim=0).long()
    order = center_scores.argsort(descending=True)
    return centers[order], center_scores[order], sizes[order], num_noise


def write_csv(path, rows, fallback_fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else fallback_fields
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    n = max(1, len(rows))
    signed = np.array([float(r["pred_gmm_dbscan"]) - float(r["gt"]) for r in rows], dtype=np.float32)
    abs_err = np.abs(signed)
    return {
        "images": len(rows),
        "MAE": float(abs_err.mean()) if rows else 0.0,
        "RMSE": float(np.sqrt(np.mean(signed ** 2))) if rows else 0.0,
        "bias": float(signed.mean()) if rows else 0.0,
        "avg_pred": float(np.mean([float(r["pred_gmm_dbscan"]) for r in rows])) if rows else 0.0,
        "avg_gt": float(np.mean([float(r["gt"]) for r in rows])) if rows else 0.0,
        "over": int(np.sum(signed > 0)),
        "under": int(np.sum(signed < 0)),
        "dense_prob_mean": float(np.mean([float(r["p_dense_mean"]) for r in rows])) if rows else 0.0,
        "duplicate_prob_mean": float(np.mean([float(r["p_duplicate_mean"]) for r in rows])) if rows else 0.0,
        "sparse_prob_mean": float(np.mean([float(r["p_sparse_mean"]) for r in rows])) if rows else 0.0,
        "radius_mean_avg": float(np.mean([float(r["radius_mean"]) for r in rows])) if rows else 0.0,
        "radius_p90_avg": float(np.mean([float(r["radius_p90"]) for r in rows])) if rows else 0.0,
    }


def main():
    args = parse_args()
    cache_path = find_latest_cache(args)
    save_dir = args.gmm_save_dir or os.path.join(os.path.dirname(cache_path), "GMM_DBSCAN")
    os.makedirs(save_dir, exist_ok=True)

    print(f"[LOAD-CACHE] {cache_path}")
    cache = load_torch(cache_path)
    images = cache["images"]

    rng = np.random.default_rng(int(args.gmm_seed))
    features_by_image = {}
    dmaps_by_image = {}
    all_features = []

    for img_key, item in tqdm(images.items(), total=len(images), desc="Computing GMM features", dynamic_ncols=True):
        raw, d_map = compute_features(item["xy"], item["score"])
        features_by_image[img_key] = raw
        dmaps_by_image[img_key] = d_map
        if raw.shape[0] > 0:
            all_features.append(raw)

    if not all_features:
        raise RuntimeError("No candidate features found in cache.")

    all_features = np.concatenate(all_features, axis=0)
    if all_features.shape[0] > int(args.gmm_fit_max_points):
        idx = rng.choice(all_features.shape[0], size=int(args.gmm_fit_max_points), replace=False)
        fit_raw = all_features[idx]
    else:
        fit_raw = all_features

    mean, std = standardize_fit(fit_raw)
    fit_std = standardize_apply(fit_raw, mean, std)
    model = fit_diag_gmm(
        fit_std,
        fit_raw,
        n_components=int(args.gmm_components),
        n_iter=int(args.gmm_iters),
        reg_covar=float(args.gmm_reg_covar),
    )
    component_raw_means = model["means"] * std + mean
    component_raw_means = component_raw_means.astype(np.float32)
    roles, role_labels, role_scores = assign_roles(component_raw_means)

    component_rows = []
    role_names = ["dense", "duplicate", "sparse"]
    for comp in range(int(model["weights"].shape[0])):
        row = {
            "component": comp,
            "role": role_names[int(role_labels[comp])],
            "weight": float(model["weights"][comp]),
            "role_score_dense": float(role_scores[comp, 0]),
            "role_score_duplicate": float(role_scores[comp, 1]),
            "role_score_sparse": float(role_scores[comp, 2]),
        }
        for i, name in enumerate(FEATURE_NAMES):
            row[f"mean_{name}"] = float(component_raw_means[comp, i])
        component_rows.append(row)

    rows = []
    watch = {x.strip() for x in str(args.watch_images).split(",") if x.strip()}
    watch_rows = []

    for img_key, item in tqdm(images.items(), total=len(images), desc="GMM DBSCAN", dynamic_ncols=True):
        xy = item["xy"].float().cpu()
        score = item["score"].float().cpu()
        gt_xy = item["gt_xy"].float().cpu()
        gt_np = gt_xy.numpy().astype(np.float32)

        raw = features_by_image[img_key]
        d_map = dmaps_by_image[img_key]
        raw_std = standardize_apply(raw, mean, std)
        n_components = int(model["weights"].shape[0])
        proba = (
            gmm_predict_proba(model, raw_std)
            if raw.shape[0] > 0
            else np.zeros((0, n_components), dtype=np.float32)
        )
        proba_by_role = {
            "dense": proba[:, roles["dense"]].sum(axis=1) if proba.shape[0] else np.zeros((0,), dtype=np.float32),
            "duplicate": proba[:, roles["duplicate"]].sum(axis=1) if proba.shape[0] else np.zeros((0,), dtype=np.float32),
            "sparse": proba[:, roles["sparse"]].sum(axis=1) if proba.shape[0] else np.zeros((0,), dtype=np.float32),
        }

        radii = compute_soft_radii(args, d_map, proba_by_role)
        dist = torch.cdist(xy, xy) if int(xy.size(0)) > 0 else torch.zeros((0, 0), dtype=torch.float32)
        centers, _, cluster_sizes, num_noise = cluster_with_radii(xy, score, dist, radii, args)
        centers_np = centers.numpy().astype(np.float32)
        stats_cluster = nearest_gt_distance_stats(centers_np, gt_np)

        pred = float(centers.size(0))
        gt = float(item["gt_count"])
        radii_np = radii.numpy().astype(np.float32)
        hard_component = np.argmax(proba, axis=1) if proba.shape[0] else np.zeros((0,), dtype=np.int64)
        hard_role = role_labels[hard_component] if hard_component.size else np.zeros((0,), dtype=np.int64)

        row = {
            "image_path": img_key,
            "pred_gmm_dbscan": pred,
            "gt": gt,
            "abs_err": abs(pred - gt),
            "proposal_count": int(item["proposal_count"]),
            "candidate_count": int(item["candidate_count"]),
            "cluster_count": int(centers.size(0)),
            "noise_count": int(num_noise),
            "radius_mean": float(np.mean(radii_np)) if radii_np.size else 0.0,
            "radius_p50": float(np.percentile(radii_np, 50)) if radii_np.size else 0.0,
            "radius_p90": float(np.percentile(radii_np, 90)) if radii_np.size else 0.0,
            "p_dense_mean": float(np.mean(proba_by_role["dense"])) if proba.shape[0] else 0.0,
            "p_duplicate_mean": float(np.mean(proba_by_role["duplicate"])) if proba.shape[0] else 0.0,
            "p_sparse_mean": float(np.mean(proba_by_role["sparse"])) if proba.shape[0] else 0.0,
            "hard_dense_ratio": float(np.mean(hard_role == 0)) if hard_role.size else 0.0,
            "hard_duplicate_ratio": float(np.mean(hard_role == 1)) if hard_role.size else 0.0,
            "hard_sparse_ratio": float(np.mean(hard_role == 2)) if hard_role.size else 0.0,
            "cluster_size_mean": float(cluster_sizes.float().mean().item()) if cluster_sizes.numel() else 0.0,
            "cluster_size_p90": float(np.percentile(cluster_sizes.numpy(), 90)) if cluster_sizes.numel() else 0.0,
            "cluster_p50_gt_dist": stats_cluster["p50"],
            "cluster_p90_gt_dist": stats_cluster["p90"],
        }
        rows.append(row)
        if os.path.basename(str(img_key)) in watch:
            watch_rows.append(row)

    rows.sort(key=lambda r: float(r["abs_err"]), reverse=True)
    summary = summarize(rows)

    summary_path = os.path.join(save_dir, "gmm_dbscan_summary.csv")
    component_path = os.path.join(save_dir, "gmm_components.csv")
    global_path = os.path.join(save_dir, "gmm_global_summary.csv")
    watch_path = os.path.join(save_dir, "gmm_watch_images.csv")

    write_csv(summary_path, rows, fallback_fields=["image_path"])
    write_csv(component_path, component_rows, fallback_fields=["component", "role"])
    write_csv(global_path, [summary], fallback_fields=["MAE"])
    write_csv(watch_path, watch_rows, fallback_fields=["image_path"])

    print(f"[SAVE-CSV] {summary_path}")
    print(f"[SAVE-CSV] {component_path}")
    print(f"[SAVE-CSV] {global_path}")
    print(f"[SAVE-CSV] {watch_path}")

    print("\n[GMM roles]")
    for row in component_rows:
        print(f"  component {row['component']}: role={row['role']} weight={row['weight']:.3f}")

    print("\n[GMM DBSCAN]")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key} = {value:.4f}")
        else:
            print(f"  {key} = {value}")


if __name__ == "__main__":
    main()
