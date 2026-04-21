import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import cv2
import numpy as np
from collections import defaultdict

from models import build_model
from models.diffusion_utils import CosineAbarSchedule
from dataset.dataset import ImageDataset

import random

def set_seed(seed: int = 7113064165):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
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

    # 你 test 必備參數（若 yaml 沒寫也能跑）
    # parser.add_argument("--ckpt_path", type=str, default=getattr(cfg, "ckpt_path", ""))
    # parser.add_argument("--test_root", type=str, default=getattr(cfg, "test_root", ""))
    # parser.add_argument("--save_dir", type=str, default=getattr(cfg, "save_dir", "./vis_test"))
    # parser.add_argument("--batch_size", type=int, default=int(getattr(cfg, "batch_size", 8)))
    # parser.add_argument("--num_workers", type=int, default=int(getattr(cfg, "num_workers", 4)))
    # parser.add_argument("--num_realizations", type=int, default=int(getattr(cfg, "num_realizations", 1)))
    # parser.add_argument("--hard_thresh", type=float, default=float(getattr(cfg, "hard_thresh", 0.45)))
    # parser.add_argument("--pick_worst", action="store_true", default=bool(getattr(cfg, "pick_worst", False)))
    # parser.add_argument("--ddim_steps", type=int, default=int(getattr(cfg, "ddim_steps", 50)))
    # parser.add_argument("--T", type=int, default=int(getattr(cfg, "T", 1000)))
    # parser.add_argument("--eps_clip", type=float, default=float(getattr(cfg, "eps_clip", 1e-3)))
    # parser.add_argument("--top_k_vis", type=int, default=int(getattr(cfg, "top_k_vis", 10)))
    # parser.add_argument("--nms_r_ratio", type=float, default=float(getattr(cfg, "nms_r_ratio", 0.005)))
    # parser.add_argument("--inner_margin_ratio", type=float, default=float(getattr(cfg, "inner_margin_ratio", 0.001)))
    # parser.add_argument("--seed", type=int, default=int(getattr(cfg, "seed", 7113064165)))
    return parser.parse_args()


# ----------------------------
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
    per_image_points_xy = defaultdict(list)   # img_key -> [tensor(K,2) cpu]
    per_image_points_prob = defaultdict(list) # img_key -> [tensor(K,) cpu]
    per_image_gt_sum = defaultdict(float)  # (deprecated; kept for compatibility)
    per_image_gt_points_xy = defaultdict(list)
    per_image_size = {}  # img_key -> (H_full, W_full)
    per_image_vis_sample = {}  # img_key -> dict for visualization

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

        t0 = int(t_seq[0].item())          # most noisy
        abar_t0 = abar[t0].view(1, 1, 1)   # [1,1,1]

        R = 1  # VAL-style: single realization
        for r in range(R):
            # VAL-style init: uniform in [-1,1]
            p_t = torch.empty((B, N, 2), device=device).uniform_(-1.0 + eps, 1.0 - eps)

            exist_prob_last = None

            for i, t_int in enumerate(t_seq.tolist()):
                t_int = int(t_int)
                t_tensor = torch.full((B, 1), t_int, device=device, dtype=torch.long)
                abar_t = abar[t_int].view(1, 1, 1).expand(B, 1, 1)

                need_exist = (i == len(t_seq) - 1)  # ✅ last step only
                eps_pred, exist_logit = model.denoise(
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
                        pos_mask_last = (exist_prob_last > float(args.hard_thresh))  # or None if you don't need mask

                    elif exist_logit.dim() == 2:
                        exist_prob_last = torch.sigmoid(exist_logit)  # [B,N]
                        pos_mask_last = (exist_prob_last > float(args.hard_thresh))

                    # --- Case B: CE 2-class logits: [B,N,2] ---
                    elif exist_logit.dim() == 3 and exist_logit.size(-1) == 2:
                        # prob 用來排序 / merge / nms
                        prob_pos = torch.softmax(exist_logit, dim=-1)[..., 1]  # [B,N]
                        exist_prob_last = prob_pos

                        # ✅ 你要的 ARGMAX gate（最快）
                        #pos_mask_last = (exist_logit.argmax(dim=-1) == 1)  # [B,N] bool

                        # 如果你想要「argmax + 低門檻」也可以混用：
                        pos_mask_last = (exist_logit.argmax(-1) == 1) | (prob_pos > float(args.hard_thresh))

                    else:
                        raise RuntimeError(f"Unexpected exist_logit shape: {tuple(exist_logit.shape)}")

                if i + 1 < len(t_seq):
                    abar_prev = abar[int(t_seq[i + 1].item())].view(1, 1, 1).expand(B, 1, 1)
                else:
                    abar_prev = torch.ones((B, 1, 1), device=device)

                p_t = ddim_reverse_step(p_t, eps_pred, abar_t, abar_prev)
                p_t = p_t.clamp(-1.0 + eps, 1.0 - eps)

            if exist_prob_last is None:
                raise RuntimeError("DDIM loop finished but exist_prob_last is None. Check need_exist logic.")

            p0_list.append(p_t.detach())            # [B,N,2]
            prob_list.append(exist_prob_last.detach())  # [B,N]

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

            # GT count will be computed from deduplicated global GT points at the end (VAL-style)

            # GT points (global) for visualization later
            # points_pad are pixels in patch coordinate on CPU right now
            # we need only valid ones
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
            thr = args.hard_thresh
            for r in range(R):
                pts_norm = p0_list[r][b]      # [N,2] on GPU, in [-1,1]
                sc = prob_list[r][b]          # [N]   on GPU

                cand = sc > thr
                if not cand.any():
                    continue

                pts_norm = pts_norm[cand]
                sc = sc[cand]

                # [-1,1] -> patch pixel (float, not rounded)
                xs = (pts_norm[:, 0] + 1) * 0.5 * (W - 1)
                ys = (pts_norm[:, 1] + 1) * 0.5 * (H - 1)

                # to global pixel (float)
                xs_g = (xs + x0).clamp(0, W_full - 1)
                ys_g = (ys + y0).clamp(0, H_full - 1)

                per_image_points_xy[img_key].append(torch.stack([xs_g, ys_g], dim=1).detach().cpu())
                per_image_points_prob[img_key].append(sc.detach().cpu())

    
    # ----------------------------
    # Merge per image and compute metrics (VAL-style)
    # ----------------------------
    per_image_pred_hard_sum = {}
    per_image_error_hard = {}

    thr = args.hard_thresh
    nms_r = 3.0

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
        if img_key not in per_image_points_xy or len(per_image_points_xy[img_key]) == 0:
            per_image_pred_hard_sum[img_key] = 0.0
            continue

        xy = torch.cat(per_image_points_xy[img_key], dim=0).float()   # [M,2] cpu
        sc = torch.cat(per_image_points_prob[img_key], dim=0).float() # [M]   cpu
        key = np.round(xy, 1)  # 0.1 pixel 粗略
        unique = np.unique(key, axis=0).shape[0]
        print("N=", xy.shape[0], "unique~", unique)
        # VAL-style full-image greedy NMS count
        # --- NMS: get kept indices & kept points (for both count and visualization) ---
        keep = radius_nms_xy(xy, sc, r=nms_r)  # [K] cpu long
        xy_keep = xy[keep]  # [K,2] cpu float
        sc_keep = sc[keep]  # [K]   cpu float (optional)

        pred_hard = float(keep.numel())  # use the SAME keep for count
        per_image_pred_hard_sum[img_key] = pred_hard

        # cache for visualization
        if img_key in per_image_vis_sample:
            per_image_vis_sample[img_key].update({
                # ALL points after (sc > thr) + merged patches
                "xs_all": xy[:, 0].numpy().astype(np.float32),
                "ys_all": xy[:, 1].numpy().astype(np.float32),
                "exist_prob_all": sc.numpy().astype(np.float32),

                # HARD kept points after NMS (blue)
                "pred_xy_hard_nms": xy_keep.numpy().astype(np.float32),

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
            gt_all = torch.cat(per_image_gt_points_xy[k], dim=0)  # [Mgt,2]
            gt_xs = gt_all[:, 0].numpy()
            gt_ys = gt_all[:, 1].numpy()
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

        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(
            args.save_dir,
            f"top{rank:02d}_{metric}_pred{pred_total:.2f}_gt{gt_total:.2f}_err{err:.2f}_{out_name}"
        )
        cv2.imwrite(out_path, img_bgr)
        print(f"  [SAVE] {out_path}")

        # Optional: save ALL raw merged pixels (red) on full image
        img_all = vis["img_np"].copy()
        xs_all = np.asarray(vis.get("xs_all", np.zeros((0,), np.float32)))
        ys_all = np.asarray(vis.get("ys_all", np.zeros((0,), np.float32)))

        # draw GT on ALL image too
        for gx, gy in zip(gt_xs, gt_ys):
            gx = int(np.clip(gx, 0, W_img - 1))
            gy = int(np.clip(gy, 0, H_img - 1))
            cv2.circle(img_all, (gx, gy), 2, (0, 255, 0), -1)

        for x, y in zip(xs_all, ys_all):
            x_i = int(np.clip(round(float(x)), 0, W_img - 1))
            y_i = int(np.clip(round(float(y)), 0, H_img - 1))
            cv2.circle(img_all, (x_i, y_i), 2, (0, 0, 255), -1)  # red

        img_all_bgr = cv2.cvtColor(img_all, cv2.COLOR_RGB2BGR)
        out_path_all = os.path.join(
            args.save_dir,
            f"top{rank:02d}_ALLPTS_{metric}_pred{pred_total:.2f}_gt{gt_total:.2f}_err{err:.2f}_{out_name}"
        )
        cv2.imwrite(out_path_all, img_all_bgr)
        print(f"  [SAVE-ALL] {out_path_all}")
