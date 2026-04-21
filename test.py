import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import cv2
import numpy as np
from collections import defaultdict
import csv

from models import build_model
from models.diffusion_utils import CosineAbarSchedule
from dataset.dataset import ImageDataset

import random


def set_seed(seed: int = 7113064165):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism flags (can reduce performance; comment out if undesired)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------
# Utils
# ----------------------------
def make_ddim_steps(T=1000, steps=20, device="cpu"):
    # long tensor descending
    return torch.linspace(T - 1, 0, steps, device=device, dtype=torch.long)


@torch.no_grad()
def ddim_reverse_step_eta(p_t, eps_pred, abar_t, abar_prev, eta=0.0):
    """
    DDIM reverse step with optional stochasticity (eta).
    p_t, eps_pred: [B,N,2]
    abar_t, abar_prev: [B,1,1]
    eta=0 -> deterministic DDIM (same as your current version)
    """
    tiny = torch.finfo(p_t.dtype).eps

    abar_t = abar_t.clamp(0, 1)
    abar_prev = abar_prev.clamp(0, 1)

    sqrt_abar_t = (abar_t + tiny).sqrt()
    sqrt_one_mt = (1.0 - abar_t).clamp_min(0).sqrt()

    # x0 prediction
    x0_pred = (p_t - sqrt_one_mt * eps_pred) / sqrt_abar_t

    # deterministic base coefficients
    alpha_t = (abar_t / (abar_prev + tiny)).clamp_min(tiny)  # [B,1,1]
    # DDIM sigma_t (common form)
    # sigma^2 = eta^2 * ((1-abar_prev)/(1-abar_t)) * (1 - alpha_t)
    sigma2 = (eta ** 2) * ((1.0 - abar_prev).clamp_min(0) / (1.0 - abar_t).clamp_min(tiny)) * (1.0 - alpha_t).clamp_min(
        0)
    sigma = sigma2.clamp_min(0).sqrt()

    # coefficient for eps_pred in mean term
    c_eps2 = (1.0 - abar_prev - sigma2).clamp_min(0)
    c_eps = c_eps2.sqrt()

    mean = (abar_prev + tiny).sqrt() * x0_pred + c_eps * eps_pred

    if eta > 0:
        z = torch.randn_like(p_t)
        p_prev = mean + sigma * z
    else:
        p_prev = mean

    return p_prev


def nearest_gt_distance_stats(pred_xy: np.ndarray, gt_xy: np.ndarray):
    """
    pred_xy: [M,2] float32
    gt_xy  : [G,2] float32 (global pixel)
    returns dict with count/mean/p50/p90/p95
    """
    out = {
        "count": int(pred_xy.shape[0]),
        "mean": 0.0,
        "p50": 0.0,
        "p90": 0.0,
        "p95": 0.0,
    }
    if pred_xy.shape[0] == 0 or gt_xy.shape[0] == 0:
        return out

    # pairwise distance: [M,G]
    # 注意：M 很大時會吃記憶體，但 top-k 可視化圖通常可接受
    diff = pred_xy[:, None, :] - gt_xy[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    dmin = np.sqrt(np.min(d2, axis=1))

    out["mean"] = float(np.mean(dmin))
    out["p50"] = float(np.percentile(dmin, 50))
    out["p90"] = float(np.percentile(dmin, 90))
    out["p95"] = float(np.percentile(dmin, 95))
    return out


def gt_radius_topk_conf_stats(
    gt_xy: np.ndarray,         # [G,2]
    pred_xy: np.ndarray,       # [M,2]
    pred_sc: np.ndarray,       # [M]
    radius: float = 6.0,
    topk: int = 2,
):
    """
    對每個 GT，統計半徑 radius 內有幾個 prediction/proposal，
    並取 confidence top-1 / top-2。

    回傳:
      rows: list[dict]，每個 GT 一列
      summary: 整體摘要
    """
    rows = []
    summary = {
        "num_gt": int(gt_xy.shape[0]),
        "num_pred": int(pred_xy.shape[0]),
        "radius": float(radius),
        "gt_with_0": 0,
        "gt_with_1": 0,
        "gt_with_2p": 0,
        "top1_mean": 0.0,
        "top2_mean": 0.0,
        "top1_median": 0.0,
        "top2_median": 0.0,
    }

    if gt_xy.shape[0] == 0:
        return rows, summary
    if pred_xy.shape[0] == 0 or pred_sc.shape[0] == 0:
        summary["gt_with_0"] = int(gt_xy.shape[0])
        return rows, summary

    diff = gt_xy[:, None, :] - pred_xy[None, :, :]   # [G,M,2]
    d2 = np.sum(diff * diff, axis=2)                 # [G,M]
    r2 = float(radius) * float(radius)

    top1_list = []
    top2_list = []

    for gi in range(gt_xy.shape[0]):
        inside = d2[gi] <= r2
        idx = np.where(inside)[0]
        n_hit = int(idx.shape[0])

        if n_hit == 0:
            summary["gt_with_0"] += 1
            rows.append({
                "gt_idx": gi,
                "gt_x": float(gt_xy[gi, 0]),
                "gt_y": float(gt_xy[gi, 1]),
                "n_in_r": 0,
                "top1": -1.0,
                "top2": -1.0,
                "top1_dist": -1.0,
                "top2_dist": -1.0,
            })
            continue

        if n_hit == 1:
            summary["gt_with_1"] += 1
        else:
            summary["gt_with_2p"] += 1

        sc_local = pred_sc[idx]
        d_local = np.sqrt(d2[gi, idx])

        order = np.argsort(-sc_local)  # high to low
        sc_sorted = sc_local[order]
        d_sorted = d_local[order]

        top1 = float(sc_sorted[0]) if sc_sorted.shape[0] >= 1 else -1.0
        top2 = float(sc_sorted[1]) if sc_sorted.shape[0] >= 2 and topk >= 2 else -1.0
        top1_dist = float(d_sorted[0]) if d_sorted.shape[0] >= 1 else -1.0
        top2_dist = float(d_sorted[1]) if d_sorted.shape[0] >= 2 and topk >= 2 else -1.0

        if top1 >= 0:
            top1_list.append(top1)
        if top2 >= 0:
            top2_list.append(top2)

        rows.append({
            "gt_idx": gi,
            "gt_x": float(gt_xy[gi, 0]),
            "gt_y": float(gt_xy[gi, 1]),
            "n_in_r": n_hit,
            "top1": top1,
            "top2": top2,
            "top1_dist": top1_dist,
            "top2_dist": top2_dist,
        })

    if len(top1_list) > 0:
        summary["top1_mean"] = float(np.mean(top1_list))
        summary["top1_median"] = float(np.median(top1_list))
    if len(top2_list) > 0:
        summary["top2_mean"] = float(np.mean(top2_list))
        summary["top2_median"] = float(np.median(top2_list))

    return rows, summary


@torch.no_grad()
def ddim_reverse_step(p_t, eps_pred, abar_t, abar_prev):
    """
    p_t, eps_pred: [B,N,2]
    abar_t, abar_prev: [B,1,1] or [1,1,1]
    """
    tiny = torch.finfo(p_t.dtype).eps

    abar_t = abar_t.clamp(0, 1)
    abar_prev = abar_prev.clamp(0, 1)

    sqrt_abar_t = (abar_t + tiny).sqrt()
    sqrt_one_mt = (1.0 - abar_t).clamp_min(0).sqrt()

    x0_pred = (p_t - sqrt_one_mt * eps_pred) / sqrt_abar_t

    sqrt_abar_prev = (abar_prev + tiny).sqrt()
    sqrt_one_mp = (1.0 - abar_prev).clamp_min(0).sqrt()

    p_prev = sqrt_abar_prev * x0_pred + sqrt_one_mp * eps_pred
    return p_prev


def collate_points_padded(batch, max_n=900):
    """
    dataset returns: (img_tensor, points_tensor[pixels], meta_dict)
    """
    imgs, pts, metas = zip(*batch)
    imgs = torch.stack(imgs, 0)  # [B,C,H,W]

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
    # 你的 meta 看起來會有 image_path
    for k in ["img_path", "image_path", "img_name", "image_name", "image_id"]:
        if k in meta:
            return meta[k]
    return str(meta)


@torch.no_grad()
def pixel_max_merge(flat_idx: torch.Tensor, scores: torch.Tensor):
    """
    PyTorch 1.13.1 compatible unique-pixel max merge.
    flat_idx: [N] int64 (pixel_id = y*W + x)
    scores  : [N] float
    return:
      uniq_flat: [M] unique pixel ids (sorted)
      max_score: [M] max score in each pixel
    """
    if flat_idx.numel() == 0:
        return flat_idx, scores

    order = torch.argsort(flat_idx)
    flat_s = flat_idx[order]
    sc_s = scores[order]

    change = torch.ones_like(flat_s, dtype=torch.bool)
    change[1:] = flat_s[1:] != flat_s[:-1]
    starts = torch.nonzero(change, as_tuple=False).squeeze(1)

    uniq_flat = flat_s[starts]

    max_score = torch.empty((starts.numel(),), device=scores.device, dtype=scores.dtype)
    for i in range(starts.numel()):
        s = int(starts[i].item())
        e = int(starts[i + 1].item()) if i + 1 < starts.numel() else int(flat_s.numel())
        max_score[i] = sc_s[s:e].max()

    return uniq_flat, max_score


@torch.no_grad()
def radius_nms_xy(pts_xy: torch.Tensor, scores: torch.Tensor, r: float):
    """
    greedy NMS with radius (pixel)
    pts_xy: [M,2] float
    scores: [M] float
    return keep indices (on cpu)
    """
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


@torch.no_grad()
def point_nms_count(xy_pix: torch.Tensor, score: torch.Tensor, r: float) -> int:
    """
    VAL-style greedy radius NMS count.
    xy_pix: [M,2] float (pixel coords)
    score : [M] float
    r     : radius in pixels
    return: kept count
    """
    M = xy_pix.size(0)
    if M == 0:
        return 0
    order = score.argsort(descending=True)
    xy = xy_pix[order]
    keep = []
    r2 = float(r) * float(r)
    for i in range(M):
        if not keep:
            keep.append(i)
            continue
        prev = xy[keep]  # [K,2]
        d2 = ((prev - xy[i]).pow(2).sum(dim=1))
        if (d2 >= r2).all():
            keep.append(i)
    return len(keep)


def dedup_points_xy(xy_np: np.ndarray, decimals: int = 3) -> np.ndarray:
    """Deduplicate XY points by rounding."""
    if xy_np.shape[0] == 0:
        return xy_np
    key = np.round(xy_np, decimals=decimals)
    _, idx = np.unique(key, axis=0, return_index=True)
    return xy_np[np.sort(idx)]


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

    # debug params (yaml 沒寫也給預設)
    parser.add_argument("--gt_debug_radius", type=float, default=6.0)
    parser.add_argument("--gt_debug_topk", type=int, default=2)
    parser.add_argument("--save_gt_debug_csv", action="store_true")

    return parser.parse_args()


# ----------------------------
# Main
# Main
# ----------------------------
if __name__ == "__main__":

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- VAL-style deterministic seed ---
    seed = int(getattr(args, 'seed', 7113064165))
    set_seed(seed)

    assert args.ckpt_path, "Please set --ckpt_path"
    assert args.test_root, "Please set --test_root"

    os.makedirs(args.save_dir, exist_ok=True)

    model = build_model(args, training=False)
    model = load_checkpoint_into_model(model, args.ckpt_path, device)
    model.to(device).eval()
    print(f"[INFO] Loaded checkpoint: {args.ckpt_path}")
    print("[DEBUG PARAMS]")
    for k in [
        "ckpt_path", "test_root", "hard_thresh", "ddim_steps", "diffusion_T",
        "eps_clip", "seed", "batch_size", "num_workers", "gt_debug_radius", "gt_debug_topk"
    ]:
        print(f"  {k} =", getattr(args, k, None))
    print("  test_gate_mode =", getattr(args, "test_gate_mode", "argmax_or_prob(default)"))
    print("  ddim_eta =", getattr(args, "ddim_eta", None))

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

    # diffusion schedule
    T = int(args.diffusion_T)
    steps = int(args.ddim_steps)
    sched = CosineAbarSchedule(T=T)
    abar = sched.abar.to(device=device)  # [T]
    t_seq = make_ddim_steps(T=T, steps=steps, device=device)  # [steps]
    eps = float(args.eps_clip)

    # per-image aggregation (global)
    per_image_proposals_xy = defaultdict(list)      # img_key -> [tensor(K,2) cpu]  # all proposals before gate
    per_image_proposals_prob = defaultdict(list)    # img_key -> [tensor(K,) cpu]    # all proposal prob
    per_image_points_xy = defaultdict(list)         # img_key -> [tensor(K,2) cpu]  # candidates after gate
    per_image_points_prob = defaultdict(list)       # img_key -> [tensor(K,) cpu]
    per_image_gt_sum = defaultdict(float)           # (deprecated; kept for compatibility)
    per_image_diag_stats = {}                       # img_key -> dict(distance stats)
    per_image_gt_debug = {}                         # img_key -> gt-topk debug summary
    per_image_gt_points_xy = defaultdict(list)
    per_image_size = {}                             # img_key -> (H_full, W_full)
    per_image_vis_sample = {}                       # img_key -> dict for visualization
    csv_rows_all = []

    # ----------------------------
    # Iterate patches
    # ----------------------------
    for images, points_pad, mask, metas in loader:
        images = images.to(device)
        mask = mask.to(device)  # [B,N] bool
        B, C, H, W = images.shape
        N = points_pad.shape[1]  # usually 900

        # encode once
        feats = model.encode(images)

        # ----------------------------
        # DDIM sampling R times
        # ----------------------------
        p0_list = []
        prob_list = []
        posmask_list = []

        t0 = int(t_seq[0].item())  # most noisy
        abar_t0 = abar[t0].view(1, 1, 1)  # [1,1,1]

        R = int(getattr(args, "num_realizations", 1))
        for r in range(R):
            # VAL-style init: uniform in [-1,1]
            p_t = torch.empty((B, N, 2), device=device).uniform_(-1.0 + eps, 1.0 - eps)

            exist_prob_last = None

            for i, t_int in enumerate(t_seq.tolist()):
                t_int = int(t_int)
                t_tensor = torch.full((B, 1), t_int, device=device, dtype=torch.long)
                abar_t = abar[t_int].view(1, 1, 1).expand(B, 1, 1)

                need_exist = (i == len(t_seq) - 1)  # ✅ last step only
                eps_pred, exist_logit, pro,_ ,_ = model.denoise(
                    feats, p_t, t_tensor,
                    abar_t=abar_t,
                    clamp_eps=1e-6,
                    need_exist=need_exist
                )

                if need_exist:
                    if exist_logit is None:
                        raise RuntimeError("need_exist=True but denoise returned None exist_logit")

                    # --- Case A: BCE 1-logit: [B,N] or [B,N,1] ---
                    if exist_logit.dim() == 3 and exist_logit.size(-1) == 1:
                        exist_logit = exist_logit.squeeze(-1)  # [B,N]
                        exist_prob_last = torch.sigmoid(exist_logit)  # [B,N]
                        pos_mask_last = (exist_prob_last > float(args.hard_thresh))

                    elif exist_logit.dim() == 2:
                        exist_prob_last = torch.sigmoid(exist_logit)  # [B,N]
                        pos_mask_last = (exist_prob_last > float(args.hard_thresh))

                    # --- Case B: CE 2-class logits: [B,N,2] ---
                    elif exist_logit.dim() == 3 and exist_logit.size(-1) == 2:
                        # prob 用來排序 / merge / nms
                        prob_pos = torch.softmax(exist_logit, dim=-1)[..., 1]  # [B,N]
                        exist_prob_last = prob_pos

                        gate_mode = getattr(args, "test_gate_mode", "argmax_or_prob")
                        if gate_mode == "prob_only":
                            pos_mask_last = (prob_pos > float(args.hard_thresh))
                        elif gate_mode == "argmax_only":
                            pos_mask_last = (exist_logit.argmax(-1) == 1)
                        else:  # "argmax_or_prob"
                            pos_mask_last = (exist_logit.argmax(-1) == 1) | (prob_pos > float(args.hard_thresh))

                    else:
                        raise RuntimeError(f"Unexpected exist_logit shape: {tuple(exist_logit.shape)}")

                if i + 1 < len(t_seq):
                    abar_prev = abar[int(t_seq[i + 1].item())].view(1, 1, 1).expand(B, 1, 1)
                    eta_step = float(getattr(args, "ddim_eta", 0.3))
                else:
                    abar_prev = torch.ones((B, 1, 1), device=device)
                    eta_step = 0.0  # final step no noise

                p_t = ddim_reverse_step_eta(p_t, eps_pred, abar_t, abar_prev, eta=eta_step)
                p_t = p_t.clamp(-1.0 + eps, 1.0 - eps)

            if exist_prob_last is None:
                raise RuntimeError("DDIM loop finished but exist_prob_last is None. Check need_exist logic.")

            p0_list.append(p_t.detach())           # [B,N,2]
            prob_list.append(exist_prob_last.detach())  # [B,N]
            posmask_list.append(pos_mask_last.detach())

        # ----------------------------
        # For each patch in batch
        # ----------------------------
        for b in range(B):
            meta = metas[b]
            img_key = get_image_key_from_meta(meta)

            H_full, W_full = meta["orig_size"]
            x0 = int(meta["tile_left"])
            y0 = int(meta["tile_top"])

            if img_key not in per_image_size:
                per_image_size[img_key] = (int(H_full), int(W_full))

            # GT points (global) for visualization later
            points_pad_b = points_pad[b]  # CPU tensor [N,2]
            mask_b_cpu = mask[b].detach().cpu()
            gt_points_b = points_pad_b[mask_b_cpu]  # [n_gt,2] in patch pixels
            if gt_points_b.numel() > 0:
                gt_xs = (gt_points_b[:, 0] + x0).round().long().clamp(0, W_full - 1)
                gt_ys = (gt_points_b[:, 1] + y0).round().long().clamp(0, H_full - 1)
                per_image_gt_points_xy[img_key].append(torch.stack([gt_xs, gt_ys], dim=1))

            # For visualization caching (read full image once)
            if img_key not in per_image_vis_sample:
                img_bgr_full = cv2.imread(img_key, cv2.IMREAD_COLOR)
                if img_bgr_full is not None:
                    img_rgb_full = cv2.cvtColor(img_bgr_full, cv2.COLOR_BGR2RGB)
                    if (img_rgb_full.shape[0], img_rgb_full.shape[1]) != (H_full, W_full):
                        img_rgb_full = cv2.resize(img_rgb_full, (W_full, H_full))
                    per_image_vis_sample[img_key] = {
                        "img_np": np.ascontiguousarray(img_rgb_full),
                        "out_name": os.path.basename(img_key),
                        "H": int(H_full),
                        "W": int(W_full),
                    }

            # ----------------------------
            # Accumulate predictions (global) VAL-style: keep prob>thr candidates, no pixel-merge here
            # ----------------------------
            for r in range(R):
                pts_norm_all = p0_list[r][b]  # [N,2] on GPU, raw proposals in [-1,1]
                sc_all = prob_list[r][b]      # [N]   on GPU, P(pos)
                pm = posmask_list[r][b]       # [N] bool

                # -------------------------------------------------
                # A. save ALL proposals (before gate)
                # -------------------------------------------------
                xs_prop = (pts_norm_all[:, 0] + 1) * 0.5 * (W - 1)
                ys_prop = (pts_norm_all[:, 1] + 1) * 0.5 * (H - 1)

                xs_prop_g = (xs_prop + x0).clamp(0, W_full - 1)
                ys_prop_g = (ys_prop + y0).clamp(0, H_full - 1)

                per_image_proposals_xy[img_key].append(
                    torch.stack([xs_prop_g, ys_prop_g], dim=1).detach().cpu()
                )
                per_image_proposals_prob[img_key].append(sc_all.detach().cpu())

                # -------------------------------------------------
                # B. gate -> candidates
                # -------------------------------------------------
                pts_norm = pts_norm_all[pm]
                sc = sc_all[pm]
                if sc.numel() == 0:
                    continue

                # optional very low threshold
                low_thr = 0.01
                keep2 = (sc > low_thr)
                pts_norm = pts_norm[keep2]
                sc = sc[keep2]
                if sc.numel() == 0:
                    continue

                # [-1,1] -> patch pixel (float, not rounded)
                xs = (pts_norm[:, 0] + 1) * 0.5 * (W - 1)
                ys = (pts_norm[:, 1] + 1) * 0.5 * (H - 1)

                # to global pixel (float)
                xs_g = (xs + x0).clamp(0, W_full - 1)
                ys_g = (ys + y0).clamp(0, H_full - 1)

                per_image_points_xy[img_key].append(
                    torch.stack([xs_g, ys_g], dim=1).detach().cpu()
                )
                per_image_points_prob[img_key].append(sc.detach().cpu())

    # ----------------------------
    # Merge per image and compute metrics (VAL-style)
    # ----------------------------
    per_image_pred_hard_sum = {}
    per_image_error_hard = {}

    nms_r = 0.0

    # compute per-image GT count from deduplicated global GT points
    per_image_gt_cnt = {}
    for img_key in sorted(per_image_size.keys()):
        if img_key in per_image_gt_points_xy and len(per_image_gt_points_xy[img_key]) > 0:
            gt_all = torch.cat(per_image_gt_points_xy[img_key], dim=0).numpy().astype(np.float32)  # [M,2] int-ish
            gt_all = dedup_points_xy(gt_all, decimals=3)
            per_image_gt_cnt[img_key] = float(gt_all.shape[0])
        else:
            per_image_gt_cnt[img_key] = 0.0

    for img_key in sorted(per_image_gt_cnt.keys()):
        # all proposals before gate
        if img_key in per_image_proposals_xy and len(per_image_proposals_xy[img_key]) > 0:
            xy_prop = torch.cat(per_image_proposals_xy[img_key], dim=0).float()  # [Mp,2]
            xy_prop_np = xy_prop.numpy().astype(np.float32)
        else:
            xy_prop_np = np.zeros((0, 2), dtype=np.float32)

        if img_key in per_image_proposals_prob and len(per_image_proposals_prob[img_key]) > 0:
            sc_prop = torch.cat(per_image_proposals_prob[img_key], dim=0).float()  # [Mp]
            sc_prop_np = sc_prop.numpy().astype(np.float32)
        else:
            sc_prop_np = np.zeros((0,), dtype=np.float32)

        # candidates after gate
        if img_key not in per_image_points_xy or len(per_image_points_xy[img_key]) == 0:
            per_image_pred_hard_sum[img_key] = 0.0

            if img_key in per_image_vis_sample:
                per_image_vis_sample[img_key].update({
                    "proposal_xy_all": xy_prop_np.astype(np.float32),
                    "proposal_prob_all": sc_prop_np.astype(np.float32),
                    "candidate_xy_all": np.zeros((0, 2), dtype=np.float32),
                    "candidate_prob_all": np.zeros((0,), dtype=np.float32),
                    "pred_xy_hard_nms": np.zeros((0, 2), dtype=np.float32),
                })
            continue

        xy = torch.cat(per_image_points_xy[img_key], dim=0).float()   # [M,2] cpu
        sc = torch.cat(per_image_points_prob[img_key], dim=0).float()  # [M] cpu
        diag_hi_thr = float(getattr(args, "diag_prob_hi", 0.50))
        diag_lo_thr = float(getattr(args, "diag_prob_lo", 0.15))

        xy_np = xy.numpy().astype(np.float32)
        sc_np = sc.numpy().astype(np.float32)

        hi_mask = (sc_np >= diag_hi_thr)
        lo_mask = (sc_np <= diag_lo_thr)

        # GT global points for this image (deduped)
        if img_key in per_image_gt_points_xy and len(per_image_gt_points_xy[img_key]) > 0:
            gt_all_np = torch.cat(per_image_gt_points_xy[img_key], dim=0).numpy().astype(np.float32)
            gt_all_np = dedup_points_xy(gt_all_np, decimals=3).astype(np.float32)
        else:
            gt_all_np = np.zeros((0, 2), dtype=np.float32)

        # ----------------------------
        # GT radius top-k confidence debug
        # ----------------------------
        dbg_r = float(getattr(args, "gt_debug_radius", 6.0))
        dbg_topk = int(getattr(args, "gt_debug_topk", 2))

        prop_rows, prop_summary = gt_radius_topk_conf_stats(
            gt_xy=gt_all_np,
            pred_xy=xy_prop_np,
            pred_sc=sc_prop_np,
            radius=dbg_r,
            topk=dbg_topk,
        )

        cand_rows, cand_summary = gt_radius_topk_conf_stats(
            gt_xy=gt_all_np,
            pred_xy=xy_np,
            pred_sc=sc_np,
            radius=dbg_r,
            topk=dbg_topk,
        )

        per_image_gt_debug[img_key] = {
            "proposal": prop_summary,
            "candidate": cand_summary,
        }

        print(
            f"[GT-TOPK][PROP] {os.path.basename(str(img_key))} "
            f"r={dbg_r:.1f} | "
            f"GT0={prop_summary['gt_with_0']} "
            f"GT1={prop_summary['gt_with_1']} "
            f"GT2+={prop_summary['gt_with_2p']} | "
            f"top1_mean={prop_summary['top1_mean']:.3f} "
            f"top2_mean={prop_summary['top2_mean']:.3f}"
        )
        print(
            f"[GT-TOPK][CAND] {os.path.basename(str(img_key))} "
            f"r={dbg_r:.1f} | "
            f"GT0={cand_summary['gt_with_0']} "
            f"GT1={cand_summary['gt_with_1']} "
            f"GT2+={cand_summary['gt_with_2p']} | "
            f"top1_mean={cand_summary['top1_mean']:.3f} "
            f"top2_mean={cand_summary['top2_mean']:.3f}"
        )

        if bool(getattr(args, "save_gt_debug_csv", False)):
            for row in prop_rows:
                row_out = dict(row)
                row_out["img_key"] = img_key
                row_out["stage"] = "proposal"
                row_out["radius"] = dbg_r
                csv_rows_all.append(row_out)
            for row in cand_rows:
                row_out = dict(row)
                row_out["img_key"] = img_key
                row_out["stage"] = "candidate"
                row_out["radius"] = dbg_r
                csv_rows_all.append(row_out)

        key = np.round(xy_np, 1)
        unique = np.unique(key, axis=0).shape[0]
        print("N=", xy.shape[0], "unique~", unique)

        # --- NMS: get kept indices & kept points (for both count and visualization) ---
        keep = radius_nms_xy(xy, sc, r=nms_r)  # [K] cpu long
        xy_keep = xy[keep]                     # [K,2] cpu float
        sc_keep = sc[keep]                     # [K]   cpu float (optional)
        xy_keep_np = xy_keep.numpy().astype(np.float32)

        stats_all = nearest_gt_distance_stats(xy_np, gt_all_np)
        stats_hi = nearest_gt_distance_stats(xy_np[hi_mask], gt_all_np)
        stats_lo = nearest_gt_distance_stats(xy_np[lo_mask], gt_all_np)
        stats_nms = nearest_gt_distance_stats(xy_keep_np, gt_all_np)

        per_image_diag_stats[img_key] = {
            "all": stats_all,
            "hi": stats_hi,
            "lo": stats_lo,
            "nms": stats_nms,
            "diag_hi_thr": diag_hi_thr,
            "diag_lo_thr": diag_lo_thr,
        }

        print(
            f"[DIAG] {os.path.basename(str(img_key))} "
            f"ALL(n={stats_all['count']}, p50={stats_all['p50']:.1f}, p90={stats_all['p90']:.1f}) | "
            f"HI@>={diag_hi_thr:.2f}(n={stats_hi['count']}, p50={stats_hi['p50']:.1f}, p90={stats_hi['p90']:.1f}) | "
            f"LO@<={diag_lo_thr:.2f}(n={stats_lo['count']}, p50={stats_lo['p50']:.1f}, p90={stats_lo['p90']:.1f}) | "
            f"NMS(n={stats_nms['count']}, p50={stats_nms['p50']:.1f}, p90={stats_nms['p90']:.1f})"
        )
        pred_hard = float(keep.numel())
        per_image_pred_hard_sum[img_key] = pred_hard

        # cache for visualization
        if img_key in per_image_vis_sample:
            # split raw candidates by confidence for diagnosis visualization
            xy_hi = xy_np[hi_mask]
            xy_lo = xy_np[lo_mask]

            per_image_vis_sample[img_key].update({
                # 1) all proposals before gate
                "proposal_xy_all": xy_prop_np.astype(np.float32),
                "proposal_prob_all": sc_prop_np.astype(np.float32),

                # 2) all candidates after gate + low_thr, before NMS
                "candidate_xy_all": xy_np.astype(np.float32),
                "candidate_prob_all": sc_np.astype(np.float32),

                # split candidates by confidence
                "raw_xy_hi": xy_hi.astype(np.float32),  # yellow
                "raw_xy_lo": xy_lo.astype(np.float32),  # red

                # 3) HARD kept points after NMS
                "pred_xy_hard_nms": xy_keep_np.astype(np.float32),

                # diagnostics
                "diag_stats": per_image_diag_stats.get(img_key, None),
                "gt_debug": per_image_gt_debug.get(img_key, None),
                "r_pix": float(nms_r),
            })

    # ----------------------------
    # Per-image MAE/RMSE (HARD only, VAL-style)
    # ----------------------------
    img_keys = sorted(per_image_gt_cnt.keys())
    abs_hard, sq_hard = [], []

    print("\n[Per-Image Results | VAL-style HARD]")
    for k in img_keys:
        pred_hard = float(per_image_pred_hard_sum.get(k, 0.0))
        gt = float(per_image_gt_cnt.get(k, 0.0))

        err_hard = abs(pred_hard - gt)
        abs_hard.append(err_hard)
        sq_hard.append((pred_hard - gt) ** 2)

        per_image_error_hard[k] = err_hard
        print(f"- {k}: hard={pred_hard:.2f}, gt={gt:.2f}, |err_hard|={err_hard:.2f}")

    MAE_hard = float(np.mean(abs_hard)) if abs_hard else 0.0
    RMSE_hard = float(np.sqrt(np.mean(sq_hard))) if sq_hard else 0.0

    print(f"\n[VAL-style HARD] Per-image MAE={MAE_hard:.2f}, RMSE={RMSE_hard:.2f}")
    per_image_pred_soft_sum = {}
    per_image_error_soft = {}

    # ----------------------------
    # Save GT debug CSV
    # ----------------------------
    if bool(getattr(args, "save_gt_debug_csv", False)) and len(csv_rows_all) > 0:
        csv_path = os.path.join(args.save_dir, "gt_topk_debug.csv")
        fieldnames = [
            "img_key", "stage", "radius",
            "gt_idx", "gt_x", "gt_y",
            "n_in_r", "top1", "top2",
            "top1_dist", "top2_dist"
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows_all)
        print(f"[SAVE-CSV] {csv_path}")

    # ----------------------------
    # Save Top-K visualizations
    # ----------------------------
    metric = "hard"
    error_dict = per_image_error_hard

    top_k = int(args.top_k_vis)
    pick_worst = bool(args.pick_worst)

    ranked = sorted(error_dict.items(), key=lambda x: x[1], reverse=pick_worst)[:top_k]
    print(f"\n[Save {'bottom' if pick_worst else 'top'}-{top_k} Visualizations]")

    for rank, (k, err) in enumerate(ranked, start=1):
        if k not in per_image_vis_sample:
            print(f"  (skip) {k} no cached image.")
            continue

        vis = per_image_vis_sample[k]
        img_np = vis["img_np"].copy()
        H_img, W_img = vis["H"], vis["W"]

        # GT points
        if k in per_image_gt_points_xy:
            gt_all_np = torch.cat(per_image_gt_points_xy[k], dim=0).numpy().astype(np.float32)
            gt_all_np = dedup_points_xy(gt_all_np, decimals=3)
            gt_xs = gt_all_np[:, 0].astype(np.int32)
            gt_ys = gt_all_np[:, 1].astype(np.int32)
        else:
            gt_xs = np.zeros((0,), dtype=np.int32)
            gt_ys = np.zeros((0,), dtype=np.int32)

        for gx, gy in zip(gt_xs, gt_ys):
            gx = int(np.clip(gx, 0, W_img - 1))
            gy = int(np.clip(gy, 0, H_img - 1))
            cv2.circle(img_np, (gx, gy), 2, (0, 255, 0), -1)  # green

        # hard nms predictions
        pred_xy = np.asarray(vis.get("pred_xy_hard_nms", np.zeros((0, 2), np.float32)), dtype=np.float32)
        for x, y in pred_xy:
            x_i = int(np.clip(round(float(x)), 0, W_img - 1))
            y_i = int(np.clip(round(float(y)), 0, H_img - 1))
            cv2.circle(img_np, (x_i, y_i), 2, (255, 0, 0), -1)  # blue

        out_name = vis["out_name"]
        pred_total = per_image_pred_hard_sum[k]
        gt_total = per_image_gt_cnt[k]

        # -------------------------------------------------
        # A. Save ALL PROPOSALS (before gate)
        #   GT = green, proposals = white
        # -------------------------------------------------
        img_prop = vis["img_np"].copy()

        for gx, gy in zip(gt_xs, gt_ys):
            gx = int(np.clip(gx, 0, W_img - 1))
            gy = int(np.clip(gy, 0, H_img - 1))
            cv2.circle(img_prop, (gx, gy), 2, (0, 255, 0), -1)  # green

        proposal_xy = np.asarray(
            vis.get("proposal_xy_all", np.zeros((0, 2), np.float32)),
            dtype=np.float32
        )
        for x, y in proposal_xy:
            x_i = int(np.clip(round(float(x)), 0, W_img - 1))
            y_i = int(np.clip(round(float(y)), 0, H_img - 1))
            cv2.circle(img_prop, (x_i, y_i), 1, (255, 255, 255), -1)  # white

        img_prop_bgr = cv2.cvtColor(img_prop, cv2.COLOR_RGB2BGR)
        out_path_prop = os.path.join(
            args.save_dir,
            f"top{rank:02d}_PROPOSALS_pred{pred_total:.2f}_gt{gt_total:.2f}_err{err:.2f}_{out_name}"
        )
        cv2.imwrite(out_path_prop, img_prop_bgr)
        print(f"  [SAVE-PROPOSALS] {out_path_prop}")

        # -------------------------------------------------
        # B. Save ALL CANDIDATES (after gate, before NMS)
        #   GT = green, candidates = red
        # -------------------------------------------------
        img_cand = vis["img_np"].copy()

        for gx, gy in zip(gt_xs, gt_ys):
            gx = int(np.clip(gx, 0, W_img - 1))
            gy = int(np.clip(gy, 0, H_img - 1))
            cv2.circle(img_cand, (gx, gy), 2, (0, 255, 0), -1)  # green

        candidate_xy = np.asarray(
            vis.get("candidate_xy_all", np.zeros((0, 2), np.float32)),
            dtype=np.float32
        )
        for x, y in candidate_xy:
            x_i = int(np.clip(round(float(x)), 0, W_img - 1))
            y_i = int(np.clip(round(float(y)), 0, H_img - 1))
            cv2.circle(img_cand, (x_i, y_i), 2, (0, 0, 255), -1)  # red

        img_cand_bgr = cv2.cvtColor(img_cand, cv2.COLOR_RGB2BGR)
        out_path_cand = os.path.join(
            args.save_dir,
            f"top{rank:02d}_CANDIDATES_pred{pred_total:.2f}_gt{gt_total:.2f}_err{err:.2f}_{out_name}"
        )
        cv2.imwrite(out_path_cand, img_cand_bgr)
        print(f"  [SAVE-CANDIDATES] {out_path_cand}")

        # -------------------------------------------------
        # C. Save AFTER NMS
        #   GT = green, kept NMS points = blue
        # -------------------------------------------------
        img_nms = vis["img_np"].copy()

        for gx, gy in zip(gt_xs, gt_ys):
            gx = int(np.clip(gx, 0, W_img - 1))
            gy = int(np.clip(gy, 0, H_img - 1))
            cv2.circle(img_nms, (gx, gy), 2, (0, 255, 0), -1)  # green

        pred_xy = np.asarray(
            vis.get("pred_xy_hard_nms", np.zeros((0, 2), np.float32)),
            dtype=np.float32
        )
        for x, y in pred_xy:
            x_i = int(np.clip(round(float(x)), 0, W_img - 1))
            y_i = int(np.clip(round(float(y)), 0, H_img - 1))
            cv2.circle(img_nms, (x_i, y_i), 2, (255, 0, 0), -1)  # blue

        img_nms_bgr = cv2.cvtColor(img_nms, cv2.COLOR_RGB2BGR)
        out_path_nms = os.path.join(
            args.save_dir,
            f"top{rank:02d}_AFTER_NMS_pred{pred_total:.2f}_gt{gt_total:.2f}_err{err:.2f}_{out_name}"
        )
        cv2.imwrite(out_path_nms, img_nms_bgr)
        print(f"  [SAVE-AFTER-NMS] {out_path_nms}")
