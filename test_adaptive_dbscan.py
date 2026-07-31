import argparse
import csv
import os
from collections import defaultdict, deque

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

from dataset.dataset import ImageDataset
from models import build_model
from models.diffusion_utils import CosineAbarSchedule
from test import (
    collate_points_padded,
    dedup_points_xy,
    ddim_reverse_step_eta,
    get_image_key_from_meta,
    load_checkpoint_into_model,
    make_ddim_steps,
    nearest_gt_distance_stats,
    set_seed,
)


@torch.no_grad()
def adaptive_dbscan_xy(
        pts_xy,
        scores=None,
        k=4,
        alpha=0.65,
        min_eps=2.0,
        max_eps=8.0,
        min_eps_mode="fixed",
        dense_min_eps=1.0,
        dense_kdist_thresh=3.0,
        min_samples=1,
        connect_rule="mutual",
        score_power=2.0,
        keep_noise=True,
):
    """
    Adaptive DBSCAN for 2D proposal points.

    Each point gets its own eps from local kNN distance:
        eps_i = clip(alpha * kth_neighbor_distance_i, min_eps_i, max_eps)

    min_eps_mode:
        fixed:    min_eps_i = min_eps
        adaptive: min_eps_i = dense_min_eps when kth distance is small,
                  otherwise min_eps. This keeps extremely dense images from
                  chaining nearby heads while preserving stronger merging in
                  normal/sparse regions.

    connect_rule:
        mutual: d(i,j) <= min(eps_i, eps_j), conservative in dense crowds
        either: d(i,j) <= max(eps_i, eps_j), more aggressive merging
        mean:   d(i,j) <= (eps_i + eps_j) / 2
    """
    pts = torch.as_tensor(pts_xy, dtype=torch.float32).detach().cpu()
    if pts.numel() == 0:
        empty_xy = torch.zeros((0, 2), dtype=torch.float32)
        empty_1d = torch.zeros((0,), dtype=torch.float32)
        empty_long = torch.zeros((0,), dtype=torch.long)
        return {
            "centers": empty_xy,
            "scores": empty_1d,
            "labels": empty_long,
            "radii": empty_1d,
            "min_eps_used": empty_1d,
            "dense_mask": torch.zeros((0,), dtype=torch.bool),
            "cluster_sizes": empty_long,
            "num_noise": 0,
        }

    if scores is None:
        sc = torch.ones((pts.size(0),), dtype=torch.float32)
    else:
        sc = torch.as_tensor(scores, dtype=torch.float32).detach().cpu()
    sc = sc.clamp_min(0.0)

    n = int(pts.size(0))
    if n == 1:
        return {
            "centers": pts.clone(),
            "scores": sc.clone(),
            "labels": torch.zeros((1,), dtype=torch.long),
            "radii": torch.full((1,), float(min_eps), dtype=torch.float32),
            "min_eps_used": torch.full((1,), float(min_eps), dtype=torch.float32),
            "dense_mask": torch.zeros((1,), dtype=torch.bool),
            "cluster_sizes": torch.ones((1,), dtype=torch.long),
            "num_noise": 0,
        }

    dist = torch.cdist(pts, pts)
    dist_for_knn = dist.clone()
    dist_for_knn.fill_diagonal_(float("inf"))

    kth = max(1, min(int(k), n - 1))
    kth_dist = torch.kthvalue(dist_for_knn, kth, dim=1).values
    if min_eps_mode == "adaptive":
        dense_mask = kth_dist <= float(dense_kdist_thresh)
        min_eps_used = torch.where(
            dense_mask,
            torch.full_like(kth_dist, float(dense_min_eps)),
            torch.full_like(kth_dist, float(min_eps)),
        )
    else:
        dense_mask = torch.zeros_like(kth_dist, dtype=torch.bool)
        min_eps_used = torch.full_like(kth_dist, float(min_eps))
    radii = torch.maximum(float(alpha) * kth_dist, min_eps_used).clamp(max=float(max_eps))

    ri = radii[:, None]
    rj = radii[None, :]
    if connect_rule == "either":
        eps_pair = torch.maximum(ri, rj)
    elif connect_rule == "mean":
        eps_pair = 0.5 * (ri + rj)
    else:
        eps_pair = torch.minimum(ri, rj)

    adjacency = (dist <= eps_pair)
    adjacency.fill_diagonal_(True)

    min_samples = max(1, int(min_samples))
    core = adjacency.sum(dim=1) >= min_samples
    labels = torch.full((n,), -1, dtype=torch.long)
    visited = torch.zeros((n,), dtype=torch.bool)

    cluster_id = 0
    adjacency_np = adjacency.numpy()
    core_np = core.numpy()
    visited_np = visited.numpy()
    labels_np = labels.numpy()

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
    cluster_scores = []
    cluster_sizes = []

    for cid in range(cluster_id):
        idx = torch.nonzero(labels == cid, as_tuple=False).squeeze(1)
        weights = sc[idx].clamp_min(1e-6).pow(float(score_power))
        center = (pts[idx] * weights[:, None]).sum(dim=0) / weights.sum()
        centers.append(center)
        cluster_scores.append(sc[idx].max())
        cluster_sizes.append(torch.tensor(idx.numel(), dtype=torch.long))

    if keep_noise and num_noise > 0:
        noise_idx = torch.nonzero(noise_mask, as_tuple=False).squeeze(1)
        for idx in noise_idx.tolist():
            labels[idx] = cluster_id
            centers.append(pts[idx])
            cluster_scores.append(sc[idx])
            cluster_sizes.append(torch.tensor(1, dtype=torch.long))
            cluster_id += 1

    if len(centers) == 0:
        empty_xy = torch.zeros((0, 2), dtype=torch.float32)
        empty_1d = torch.zeros((0,), dtype=torch.float32)
        empty_long = torch.zeros((0,), dtype=torch.long)
        return {
            "centers": empty_xy,
            "scores": empty_1d,
            "labels": labels,
            "radii": radii,
            "min_eps_used": min_eps_used,
            "dense_mask": dense_mask,
            "cluster_sizes": empty_long,
            "num_noise": num_noise,
        }

    centers = torch.stack(centers, dim=0).float()
    cluster_scores = torch.stack(cluster_scores, dim=0).float()
    cluster_sizes = torch.stack(cluster_sizes, dim=0).long()
    order = cluster_scores.argsort(descending=True)

    return {
        "centers": centers[order],
        "scores": cluster_scores[order],
        "labels": labels,
        "radii": radii,
        "min_eps_used": min_eps_used,
        "dense_mask": dense_mask,
        "cluster_sizes": cluster_sizes[order],
        "num_noise": num_noise,
    }


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

    parser.add_argument("--adbscan_save_dir", type=str, default="")
    parser.add_argument("--adbscan_k", type=int, default=20)
    parser.add_argument("--adbscan_alpha", type=float, default=0.65)
    parser.add_argument("--adbscan_min_eps", type=float, default=2.0)
    parser.add_argument("--adbscan_max_eps", type=float, default=8.0)
    parser.add_argument(
        "--adbscan_min_eps_mode",
        type=str,
        default="fixed",
        choices=["fixed", "adaptive"],
    )
    parser.add_argument("--adbscan_dense_min_eps", type=float, default=1.0)
    parser.add_argument("--adbscan_dense_kdist_thresh", type=float, default=3.0)
    parser.add_argument("--adbscan_min_samples", type=int, default=1)
    parser.add_argument(
        "--adbscan_connect_rule",
        type=str,
        default="mutual",
        choices=["mutual", "either", "mean"],
    )
    parser.add_argument("--adbscan_score_power", type=float, default=2.0)
    parser.add_argument("--adbscan_drop_noise", action="store_true")
    parser.add_argument("--save_cluster_csv", action="store_true")

    return parser.parse_args()


def draw_points(img_rgb, xy, color, radius):
    out = img_rgb
    h, w = out.shape[:2]
    for x, y in np.asarray(xy, dtype=np.float32):
        x_i = int(np.clip(round(float(x)), 0, w - 1))
        y_i = int(np.clip(round(float(y)), 0, h - 1))
        cv2.circle(out, (x_i, y_i), int(radius), color, -1)
    return out


def main():
    args = parse_args()
    save_dir = args.adbscan_save_dir or f"{args.save_dir}_ADBSCAN"
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(getattr(args, "seed", 7113064165))
    set_seed(seed)

    assert args.ckpt_path, "Please set --ckpt_path"
    assert args.test_root, "Please set --test_root"

    model = build_model(args, training=False)
    model = load_checkpoint_into_model(model, args.ckpt_path, device)
    model.to(device).eval()

    print(f"[INFO] Loaded checkpoint: {args.ckpt_path}")
    print(f"[INFO] Save dir: {save_dir}")
    print("[Adaptive DBSCAN]")
    for key in [
        "adbscan_k",
        "adbscan_alpha",
        "adbscan_min_eps",
        "adbscan_max_eps",
        "adbscan_min_eps_mode",
        "adbscan_dense_min_eps",
        "adbscan_dense_kdist_thresh",
        "adbscan_min_samples",
        "adbscan_connect_rule",
        "adbscan_score_power",
        "adbscan_drop_noise",
        "hard_thresh",
        "ddim_steps",
        "num_realizations",
    ]:
        print(f"  {key} = {getattr(args, key, None)}")

    dataset = ImageDataset(
        root=args.test_root,
        mode="points",
        tile_size=(256, 256),
        stride=(256, 256),
        gray=False,
        pad_if_needed=True,
        image_exts=(".jpg", ".png"),
    )
    print(f"[INFO] Dataset tiles: {len(dataset)}", flush=True)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_points_padded,
    )

    t_total = int(args.diffusion_T)
    steps = int(args.ddim_steps)
    sched = CosineAbarSchedule(T=t_total)
    abar = sched.abar.to(device=device)
    t_seq = make_ddim_steps(T=t_total, steps=steps, device=device)
    eps = float(args.eps_clip)

    per_image_proposals_xy = defaultdict(list)
    per_image_proposals_prob = defaultdict(list)
    per_image_points_xy = defaultdict(list)
    per_image_points_prob = defaultdict(list)
    per_image_gt_points_xy = defaultdict(list)
    per_image_size = {}
    per_image_vis = {}

    with torch.no_grad():
        for images, points_pad, mask, metas in tqdm(
                loader,
                total=len(loader),
                desc="DDIM inference",
                dynamic_ncols=True,
        ):
            images = images.to(device)
            mask = mask.to(device)
            batch_size, _, h_tile, w_tile = images.shape
            num_points = points_pad.shape[1]

            feats = model.encode(images)
            p0_list = []
            prob_list = []
            posmask_list = []
            validmask_list = []

            runs = int(getattr(args, "num_realizations", 1))
            for _ in range(runs):
                p_t = torch.empty((batch_size, num_points, 2), device=device).uniform_(
                    -1.0 + eps,
                    1.0 - eps,
                )
                exist_prob_last = None
                pos_mask_last = None
                pred_points_last = None
                pred_valid_last = None

                for i, t_int in enumerate(t_seq.tolist()):
                    t_int = int(t_int)
                    t_tensor = torch.full((batch_size, 1), t_int, device=device, dtype=torch.long)
                    abar_t = abar[t_int].view(1, 1, 1).expand(batch_size, 1, 1)
                    need_exist = (i == len(t_seq) - 1)

                    eps_pred, exist_logit, _, pred_points_for_cls, pred_valid_mask = model.denoise(
                        feats,
                        p_t,
                        t_tensor,
                        abar_t=abar_t,
                        clamp_eps=1e-6,
                        need_exist=need_exist,
                    )

                    if need_exist:
                        if exist_logit is None:
                            raise RuntimeError("need_exist=True but denoise returned None exist_logit")
                        if pred_points_for_cls is None:
                            raise RuntimeError("need_exist=True but denoise returned no prediction points")

                        pred_points_last = pred_points_for_cls
                        pred_valid_last = pred_valid_mask

                        if exist_logit.dim() == 3 and exist_logit.size(-1) == 1:
                            exist_logit = exist_logit.squeeze(-1)

                        if exist_logit.dim() == 3 and exist_logit.size(-1) == 2:
                            prob_pos = torch.softmax(exist_logit, dim=-1)[..., 1]
                            exist_prob_last = prob_pos
                            gate_mode = getattr(args, "test_gate_mode", "argmax_or_prob")
                            if gate_mode == "prob_only":
                                pos_mask_last = prob_pos > float(args.hard_thresh)
                            elif gate_mode == "argmax_only":
                                pos_mask_last = exist_logit.argmax(-1) == 1
                            else:
                                pos_mask_last = (exist_logit.argmax(-1) == 1) | (
                                    prob_pos > float(args.hard_thresh)
                                )
                        elif exist_logit.dim() == 2:
                            exist_prob_last = torch.sigmoid(exist_logit)
                            pos_mask_last = exist_prob_last > float(args.hard_thresh)
                        else:
                            raise RuntimeError(f"Unexpected exist_logit shape: {tuple(exist_logit.shape)}")

                        if pred_valid_last is not None:
                            pos_mask_last = pos_mask_last & pred_valid_last.bool()

                    if i + 1 < len(t_seq):
                        abar_prev = abar[int(t_seq[i + 1].item())].view(1, 1, 1).expand(batch_size, 1, 1)
                        eta_step = float(getattr(args, "ddim_eta", 0.3))
                    else:
                        abar_prev = torch.ones((batch_size, 1, 1), device=device)
                        eta_step = 0.0

                    p_t = ddim_reverse_step_eta(p_t, eps_pred, abar_t, abar_prev, eta=eta_step)
                    p_t = p_t.clamp(-1.0 + eps, 1.0 - eps)

                if exist_prob_last is None or pos_mask_last is None or pred_points_last is None:
                    raise RuntimeError("DDIM loop finished without final probabilities or points")

                p0_list.append(pred_points_last.detach())
                prob_list.append(exist_prob_last.detach())
                posmask_list.append(pos_mask_last.detach())
                if pred_valid_last is None:
                    validmask_list.append(torch.ones_like(pos_mask_last, dtype=torch.bool).detach())
                else:
                    validmask_list.append(pred_valid_last.bool().detach())

            for b in range(batch_size):
                meta = metas[b]
                img_key = get_image_key_from_meta(meta)
                h_full, w_full = meta["orig_size"]
                left = int(meta["tile_left"])
                top = int(meta["tile_top"])

                if img_key not in per_image_size:
                    per_image_size[img_key] = (int(h_full), int(w_full))

                points_b = points_pad[b]
                mask_b = mask[b].detach().cpu()
                gt_b = points_b[mask_b]
                if gt_b.numel() > 0:
                    gt_x = (gt_b[:, 0] + left).round().long().clamp(0, w_full - 1)
                    gt_y = (gt_b[:, 1] + top).round().long().clamp(0, h_full - 1)
                    per_image_gt_points_xy[img_key].append(torch.stack([gt_x, gt_y], dim=1))

                if img_key not in per_image_vis:
                    img_bgr = cv2.imread(img_key, cv2.IMREAD_COLOR)
                    if img_bgr is not None:
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        if (img_rgb.shape[0], img_rgb.shape[1]) != (h_full, w_full):
                            img_rgb = cv2.resize(img_rgb, (w_full, h_full))
                        per_image_vis[img_key] = {
                            "img_np": np.ascontiguousarray(img_rgb),
                            "out_name": os.path.basename(img_key),
                            "H": int(h_full),
                            "W": int(w_full),
                        }

                for run_idx in range(runs):
                    pts_norm_all = p0_list[run_idx][b]
                    sc_all = prob_list[run_idx][b]
                    pm = posmask_list[run_idx][b]
                    vm = validmask_list[run_idx][b]

                    pts_prop = pts_norm_all[vm]
                    sc_prop = sc_all[vm]
                    xs_prop = (pts_prop[:, 0] + 1) * 0.5 * (w_tile - 1)
                    ys_prop = (pts_prop[:, 1] + 1) * 0.5 * (h_tile - 1)
                    xs_prop_g = (xs_prop + left).clamp(0, w_full - 1)
                    ys_prop_g = (ys_prop + top).clamp(0, h_full - 1)

                    per_image_proposals_xy[img_key].append(
                        torch.stack([xs_prop_g, ys_prop_g], dim=1).detach().cpu()
                    )
                    per_image_proposals_prob[img_key].append(sc_prop.detach().cpu())

                    pts_norm = pts_norm_all[pm]
                    sc = sc_all[pm]
                    if sc.numel() == 0:
                        continue

                    keep_low = sc > 0.01
                    pts_norm = pts_norm[keep_low]
                    sc = sc[keep_low]
                    if sc.numel() == 0:
                        continue

                    xs = (pts_norm[:, 0] + 1) * 0.5 * (w_tile - 1)
                    ys = (pts_norm[:, 1] + 1) * 0.5 * (h_tile - 1)
                    xs_g = (xs + left).clamp(0, w_full - 1)
                    ys_g = (ys + top).clamp(0, h_full - 1)

                    per_image_points_xy[img_key].append(
                        torch.stack([xs_g, ys_g], dim=1).detach().cpu()
                    )
                    per_image_points_prob[img_key].append(sc.detach().cpu())

    per_image_gt_cnt = {}
    for img_key in sorted(per_image_size.keys()):
        if img_key in per_image_gt_points_xy and len(per_image_gt_points_xy[img_key]) > 0:
            gt_all = torch.cat(per_image_gt_points_xy[img_key], dim=0).numpy().astype(np.float32)
            gt_all = dedup_points_xy(gt_all, decimals=3)
            per_image_gt_cnt[img_key] = float(gt_all.shape[0])
        else:
            per_image_gt_cnt[img_key] = 0.0

    per_image_pred = {}
    per_image_error = {}
    per_image_cluster_rows = []
    summary_rows = []

    for img_key in tqdm(
            sorted(per_image_gt_cnt.keys()),
            desc="Adaptive DBSCAN",
            dynamic_ncols=True,
    ):
        if img_key in per_image_proposals_xy and len(per_image_proposals_xy[img_key]) > 0:
            xy_prop = torch.cat(per_image_proposals_xy[img_key], dim=0).float()
            xy_prop_np = xy_prop.numpy().astype(np.float32)
        else:
            xy_prop_np = np.zeros((0, 2), dtype=np.float32)

        if img_key in per_image_proposals_prob and len(per_image_proposals_prob[img_key]) > 0:
            sc_prop = torch.cat(per_image_proposals_prob[img_key], dim=0).float()
            sc_prop_np = sc_prop.numpy().astype(np.float32)
        else:
            sc_prop_np = np.zeros((0,), dtype=np.float32)

        if img_key in per_image_gt_points_xy and len(per_image_gt_points_xy[img_key]) > 0:
            gt_all = torch.cat(per_image_gt_points_xy[img_key], dim=0).numpy().astype(np.float32)
            gt_all = dedup_points_xy(gt_all, decimals=3).astype(np.float32)
        else:
            gt_all = np.zeros((0, 2), dtype=np.float32)

        if img_key in per_image_points_xy and len(per_image_points_xy[img_key]) > 0:
            xy = torch.cat(per_image_points_xy[img_key], dim=0).float()
            sc = torch.cat(per_image_points_prob[img_key], dim=0).float()
        else:
            xy = torch.zeros((0, 2), dtype=torch.float32)
            sc = torch.zeros((0,), dtype=torch.float32)

        clustered = adaptive_dbscan_xy(
            xy,
            sc,
            k=int(args.adbscan_k),
            alpha=float(args.adbscan_alpha),
            min_eps=float(args.adbscan_min_eps),
            max_eps=float(args.adbscan_max_eps),
            min_eps_mode=str(args.adbscan_min_eps_mode),
            dense_min_eps=float(args.adbscan_dense_min_eps),
            dense_kdist_thresh=float(args.adbscan_dense_kdist_thresh),
            min_samples=int(args.adbscan_min_samples),
            connect_rule=str(args.adbscan_connect_rule),
            score_power=float(args.adbscan_score_power),
            keep_noise=not bool(args.adbscan_drop_noise),
        )
        centers = clustered["centers"]
        center_scores = clustered["scores"]
        cluster_sizes = clustered["cluster_sizes"]
        centers_np = centers.numpy().astype(np.float32)

        pred = float(centers.size(0))
        gt = float(per_image_gt_cnt[img_key])
        err = abs(pred - gt)
        per_image_pred[img_key] = pred
        per_image_error[img_key] = err

        stats_all = nearest_gt_distance_stats(xy.numpy().astype(np.float32), gt_all)
        stats_cluster = nearest_gt_distance_stats(centers_np, gt_all)

        radii = clustered["radii"].numpy().astype(np.float32)
        min_eps_used = clustered["min_eps_used"].numpy().astype(np.float32)
        dense_mask = clustered["dense_mask"].numpy().astype(bool)
        row = {
            "image_path": img_key,
            "pred_adbscan": pred,
            "gt": gt,
            "abs_err": err,
            "proposal_count": int(xy_prop_np.shape[0]),
            "candidate_count": int(xy.size(0)),
            "cluster_count": int(centers.size(0)),
            "noise_count": int(clustered["num_noise"]),
            "radius_mean": float(np.mean(radii)) if radii.size else 0.0,
            "radius_p50": float(np.percentile(radii, 50)) if radii.size else 0.0,
            "radius_p90": float(np.percentile(radii, 90)) if radii.size else 0.0,
            "min_eps_mean": float(np.mean(min_eps_used)) if min_eps_used.size else 0.0,
            "dense_point_ratio": float(np.mean(dense_mask)) if dense_mask.size else 0.0,
            "cand_p50_gt_dist": stats_all["p50"],
            "cand_p90_gt_dist": stats_all["p90"],
            "cluster_p50_gt_dist": stats_cluster["p50"],
            "cluster_p90_gt_dist": stats_cluster["p90"],
        }
        summary_rows.append(row)

        for cid in range(centers.size(0)):
            per_image_cluster_rows.append({
                "image_path": img_key,
                "cluster_idx": cid,
                "x": float(centers[cid, 0].item()),
                "y": float(centers[cid, 1].item()),
                "score": float(center_scores[cid].item()),
                "size": int(cluster_sizes[cid].item()),
            })

        if img_key in per_image_vis:
            per_image_vis[img_key].update({
                "proposal_xy_all": xy_prop_np,
                "candidate_xy_all": xy.numpy().astype(np.float32),
                "pred_xy_adbscan": centers_np,
            })

        print(
            f"[ADBSCAN] {os.path.basename(str(img_key))} "
            f"cand={xy.size(0)} clusters={centers.size(0)} gt={gt:.0f} err={err:.0f} "
            f"r50={row['radius_p50']:.2f} r90={row['radius_p90']:.2f}"
        )

    img_keys = sorted(per_image_gt_cnt.keys())
    abs_err = [per_image_error[k] for k in img_keys]
    sq_err = [(per_image_pred[k] - per_image_gt_cnt[k]) ** 2 for k in img_keys]
    mae = float(np.mean(abs_err)) if abs_err else 0.0
    rmse = float(np.sqrt(np.mean(sq_err))) if sq_err else 0.0
    print(f"\n[Adaptive DBSCAN] Per-image MAE={mae:.2f}, RMSE={rmse:.2f}")

    summary_csv = os.path.join(save_dir, "adaptive_dbscan_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(summary_rows[0].keys()) if summary_rows else ["image_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[SAVE-CSV] {summary_csv}")

    if bool(args.save_cluster_csv):
        cluster_csv = os.path.join(save_dir, "adaptive_dbscan_clusters.csv")
        with open(cluster_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = list(per_image_cluster_rows[0].keys()) if per_image_cluster_rows else ["image_path"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_image_cluster_rows)
        print(f"[SAVE-CSV] {cluster_csv}")

    ranked = sorted(per_image_error.items(), key=lambda x: x[1], reverse=bool(args.pick_worst))[
        : int(args.top_k_vis)
    ]
    print(f"\n[Save {'worst' if args.pick_worst else 'best'}-{int(args.top_k_vis)} Visualizations]")

    for rank, (img_key, err) in enumerate(
            tqdm(ranked, desc="Saving visualizations", dynamic_ncols=True),
            start=1,
    ):
        if img_key not in per_image_vis:
            print(f"  (skip) {img_key} no cached image.")
            continue

        vis = per_image_vis[img_key]
        h_img, w_img = vis["H"], vis["W"]
        out_name = vis["out_name"]
        pred_total = per_image_pred[img_key]
        gt_total = per_image_gt_cnt[img_key]

        if img_key in per_image_gt_points_xy and len(per_image_gt_points_xy[img_key]) > 0:
            gt_xy = torch.cat(per_image_gt_points_xy[img_key], dim=0).numpy().astype(np.float32)
            gt_xy = dedup_points_xy(gt_xy, decimals=3)
        else:
            gt_xy = np.zeros((0, 2), dtype=np.float32)

        prop_xy = np.asarray(vis.get("proposal_xy_all", np.zeros((0, 2), np.float32)), dtype=np.float32)
        cand_xy = np.asarray(vis.get("candidate_xy_all", np.zeros((0, 2), np.float32)), dtype=np.float32)
        pred_xy = np.asarray(vis.get("pred_xy_adbscan", np.zeros((0, 2), np.float32)), dtype=np.float32)

        img_prop = vis["img_np"].copy()
        draw_points(img_prop, gt_xy, (0, 255, 0), 2)
        draw_points(img_prop, prop_xy, (255, 255, 255), 1)
        out_prop = os.path.join(
            save_dir,
            f"top{rank:02d}_PROPOSALS_pred{pred_total:.2f}_gt{gt_total:.2f}_err{err:.2f}_{out_name}",
        )
        cv2.imwrite(out_prop, cv2.cvtColor(img_prop, cv2.COLOR_RGB2BGR))
        print(f"  [SAVE-PROPOSALS] {out_prop}")

        img_cand = vis["img_np"].copy()
        draw_points(img_cand, gt_xy, (0, 255, 0), 2)
        draw_points(img_cand, cand_xy, (0, 0, 255), 2)
        out_cand = os.path.join(
            save_dir,
            f"top{rank:02d}_CANDIDATES_pred{pred_total:.2f}_gt{gt_total:.2f}_err{err:.2f}_{out_name}",
        )
        cv2.imwrite(out_cand, cv2.cvtColor(img_cand, cv2.COLOR_RGB2BGR))
        print(f"  [SAVE-CANDIDATES] {out_cand}")

        img_cluster = vis["img_np"].copy()
        draw_points(img_cluster, gt_xy, (0, 255, 0), 2)
        draw_points(img_cluster, pred_xy, (255, 0, 0), 2)
        out_cluster = os.path.join(
            save_dir,
            f"top{rank:02d}_AFTER_ADBSCAN_pred{pred_total:.2f}_gt{gt_total:.2f}_err{err:.2f}_{out_name}",
        )
        cv2.imwrite(out_cluster, cv2.cvtColor(img_cluster, cv2.COLOR_RGB2BGR))
        print(f"  [SAVE-AFTER-ADBSCAN] {out_cluster}")

        _ = h_img, w_img


if __name__ == "__main__":
    main()
