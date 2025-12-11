import os
import argparse
import yaml
import torch
import time
from torch.utils.data import DataLoader
from models import build_model
from models.diffusion_utils import CosineAbarSchedule
from dataset.dataset import ImageDataset
import torch.nn.functional as F
import cv2
import numpy as np
from collections import defaultdict

# ---------- DDIM utils ----------
def make_ddim_steps(T=1000, steps=20, device='cpu'):
    ts = torch.linspace(T-1, 0, steps, device=device, dtype=torch.long)
    return ts

@torch.no_grad()
def ddim_reverse_step(p_t, eps_pred, abar_t, abar_prev):
    # 期待形狀：
    # p_t, eps_pred: [B, N, 2]
    # abar_t, abar_prev: [B, 1, 1] 或 [1, 1, 1]
    # 值域：0 <= abar_* <= 1，且 abar_prev >= abar_t（越靠近資料端）
    tiny = torch.finfo(p_t.dtype).eps  # 比 1e-8 更隨 dtype

    abar_t      = abar_t.clamp(0, 1)
    abar_prev   = abar_prev.clamp(0, 1)

    sqrt_abar_t = (abar_t + tiny).sqrt()
    sqrt_one_mt = (1.0 - abar_t).clamp_min(0).sqrt()

    # \hat{x}_0
    x0_pred = (p_t - sqrt_one_mt * eps_pred) / sqrt_abar_t

    sqrt_abar_prev = abar_prev.sqrt()
    sqrt_one_mp    = (1.0 - abar_prev).clamp_min(0).sqrt()

    # η=0
    p_prev = sqrt_abar_prev * x0_pred + sqrt_one_mp * eps_pred
    return p_prev


# ---------- collate ----------
def collate_points_padded(batch, max_n=900):
    import torch
    imgs, pts, metas = zip(*batch)
    imgs = torch.stack(imgs, 0)      # [B,C,H,W]

    B = len(pts)
    padded = torch.full((B, max_n, 2), fill_value=-10.0, dtype=torch.float32)
    mask   = torch.zeros((B, max_n), dtype=torch.bool)

    for i, p in enumerate(pts):
        n = int(p.size(0))
        m = min(n, max_n)
        if m > 0:
            padded[i, :m] = p[:m]
            mask[i, :m]   = True

    return imgs, padded, mask, list(metas)

# ---------- config ----------
def parse_args():
    def load_config(yaml_path):
        with open(yaml_path, 'r', encoding="utf-8") as f:
            return yaml.safe_load(f)

    base = argparse.ArgumentParser()
    base.add_argument('--config', default=r'config/train.yaml', type=str)
    args0, _ = base.parse_known_args()

    cfg = load_config(args0.config)
    parser = argparse.ArgumentParser(parents=[base], add_help=False)
    for k, v in cfg.items():
        parser.add_argument(f'--{k}', type=type(v), default=v)
    return parser.parse_args()

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
    """
    取得能唯一識別「原圖」的 key。
    """
    for k in ['img_path', 'image_path', 'img_name', 'image_name', 'image_id']:
        if k in meta:
            return meta[k]
    return str(meta)

def radius_nms_xyxy(pts_xy, scores, r):
    # pts_xy: [N,2] 像素座標；scores: [N]
    keep = []
    order = scores.argsort(descending=True)
    taken = torch.zeros(len(pts_xy), dtype=torch.bool, device=pts_xy.device)
    for i in order:
        if taken[i]:
            continue
        keep.append(i.item())
        d2 = ((pts_xy - pts_xy[i])**2).sum(-1)
        taken |= (d2 <= r*r)
    return torch.tensor(keep, device=pts_xy.device)

# ---------- main ----------
if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = r"D:\output\Leps_new\last_epoch1079.pth"
    model = build_model(args, training=False)
    model = load_checkpoint_into_model(model, ckpt_path, device)
    model.to(device).eval()
    print(f"[INFO] Loaded checkpoint: {ckpt_path}")

    dataset = ImageDataset(
        root=args.test_root,
        mode='points',
        tile_size=(256, 256),
        stride=(256, 256),
        gray=False,
        pad_if_needed=True,
        image_exts=('.jpg', '.png'),
    )
    print(f"[INFO] Dataset tiles: {len(dataset)}", flush=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_points_padded,
    )

    T = 1000
    steps = 50
    sched = CosineAbarSchedule(T=T)
    abar = sched.abar.to(device=device)
    t_seq = make_ddim_steps(T=T, steps=steps, device=device)
    eps = 1e-3

    # 用來「以原圖為單位」聚合預測與 GT
    per_image_pred_soft_sum = defaultdict(float)
    per_image_pred_hard_sum = defaultdict(float)
    per_image_gt_sum = defaultdict(float)
    per_image_vis_sample = {}
    save_dir = r"C:\pycharm\pointdiff_new\vis_results\vis_results_Leps_1079_6"
    os.makedirs(save_dir, exist_ok=True)

    for images, points_pad, mask, metas in loader:
        images = images.to(device)
        mask = mask.to(device)

        B, C, H, W = images.shape

        # 影像轉 numpy（RGB 假設），供暫存代表 patch 用
        imgs_np_all = (
            images.detach().cpu().permute(0, 2, 3, 1).numpy() * 255
        ).clip(0, 255).astype(np.uint8)

        # 編碼
        feats = model.encode(images)
        feats_zero = [f * 0 for f in feats] if isinstance(feats, (list, tuple)) else feats * 0

        # ---------- 在 p_t 上做 R 次取樣 ----------
        N = points_pad.shape[1]
        t0 = t_seq[0].item()                     # 最髒的時間步
        abar_t0 = abar[t0].view(1, 1, 1)        # [1,1,1]

        p0_list = []  # 裝 R 組 p0，每個 [B, N, 2]
        R = getattr(args, "num_realizations", 1)  # 多重取樣次數，預設 1（等同原本）

        for r in range(R):
            p_t = torch.randn(B, N, 2, device=device)  # ~ N(0, I)
            p_t = p_t * torch.sqrt(1.0 - abar_t0)      # ~ N(0, 1 - abar_t0)
            p_t = p_t.clamp(-1.0 + eps, 1.0 - eps)

            # DDIM 多步反推
            for i, t_int in enumerate(t_seq.tolist()):
                t_tensor = torch.full((B, 1), t_int, device=device, dtype=torch.long)
                abar_t = abar[t_int].view(1, 1, 1)  # [1,1,1]

                eps_pred, _ = model.denoise(
                    feats, p_t, t_tensor, abar_t=abar_t, clamp_eps=1e-6
                )
                if i + 1 < len(t_seq):
                    abar_prev = abar[t_seq[i + 1]].view(1, 1, 1)
                else:
                    abar_prev = torch.tensor(1.0, device=device).view(1, 1, 1)

                p_t = ddim_reverse_step(p_t, eps_pred, abar_t, abar_prev)

            p0_list.append(p_t.detach())  # [B, N, 2]

        # 把 R 組 p0 接起來：[B, R*N, 2]
        p_all = torch.cat(p0_list, dim=1)

        hard_thresh = getattr(args, "hard_thresh", 0.5)

        pred_cnt_list = []
        pred_cnt_hard_list = []
        gt_cnt_list = []

        # 給可視化用的 list（每張 patch 對應一個 (M_b,2)/(M_b,)）
        p_merged_np_all = []
        exist_merged_np_all = []

        # 依「原圖 key」累加人數 & 暫存代表 patch（含 GT）
        for b in range(B):
            meta = metas[b]
            img_key = get_image_key_from_meta(meta)

            # ---------- 合併多次取樣的座標（以 pixel 去重） ----------
            pts_norm = p_all[b]  # [R*N, 2] in [-1,1]

            xs = (pts_norm[:, 0] + 1) * 0.5 * (W - 1)
            ys = (pts_norm[:, 1] + 1) * 0.5 * (H - 1)

            xs_int = xs.round().clamp(0, W - 1).long()
            ys_int = ys.round().clamp(0, H - 1).long()

            flat = ys_int * W + xs_int          # [R*N]
            uniq_flat = torch.unique(flat)      # [M_b]

            x_pix = (uniq_flat % W).float()
            y_pix = (uniq_flat // W).float()

            x_norm = (x_pix / (W - 1)) * 2.0 - 1.0
            y_norm = (y_pix / (H - 1)) * 2.0 - 1.0
            pts_merged_norm = torch.stack([x_norm, y_norm], dim=1).to(device)  # [M_b, 2]
            M_b = pts_merged_norm.shape[0]

            # ---------- 丟入 cond + conf_head ----------
            p_norm_b_1 = pts_merged_norm.unsqueeze(0)  # [1, M_b, 2]

            if isinstance(feats, (list, tuple)):
                feats_b = [f[b:b+1] for f in feats]          # 每個 [1, ...]
                pf_b = model.cond(*feats_b, p_norm_b_1)     # [1, M_b, C]
            else:
                feats_b = feats[b:b+1]
                pf_b = model.cond(feats_b, p_norm_b_1)      # [1, M_b, C]

            exist_logit_b = model.conf_head(pf_b)[0]        # [M_b]
            exist_prob_b = torch.sigmoid(exist_logit_b)     # [M_b]

            # ---------- 用 merged 的 exist_prob_b 算這張 patch 的 soft / hard count ----------
            pred_cnt_b = exist_prob_b.sum()
            pred_cnt_hard_b = (exist_prob_b >= hard_thresh).float().sum()
            gt_cnt_b = mask[b].sum().float()

            pred_cnt_list.append(pred_cnt_b.item())
            pred_cnt_hard_list.append(pred_cnt_hard_b.item())
            gt_cnt_list.append(gt_cnt_b.item())

            # 給後面 MAE / 統計用（如果你有 per_image_pred_soft_sum 之類，也可以在這裡加）
            per_image_pred_soft_sum[img_key] += float(pred_cnt_b.item())
            per_image_pred_hard_sum[img_key] += float(pred_cnt_hard_b.item())
            per_image_gt_sum[img_key] += float(gt_cnt_b.item())

            # 存 numpy 給可視化用（合併後的 M_b 點）
            pts_merged_np = pts_merged_norm.detach().cpu().numpy()   # (M_b, 2)
            exist_merged_np = exist_prob_b.detach().cpu().numpy()    # (M_b,)
            p_merged_np_all.append(pts_merged_np)
            exist_merged_np_all.append(exist_merged_np)

            # ---------- per_image_vis_sample：只挑一個代表 patch 來畫 ----------
            if img_key not in per_image_vis_sample:
                # 這個 patch 的影像（RGB 假設；灰階轉 BGR 方便畫色）
                img_np = imgs_np_all[b]
                if img_np.ndim == 2:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
                elif img_np.shape[2] == 1:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
                img_np = np.ascontiguousarray(img_np)
                H_patch, W_patch = img_np.shape[:2]

                # ★ 現在存的是「多次取樣合併後的 M_b 個點」
                pred_points_all = pts_merged_np          # (M_b, 2) in [-1,1]（其實剛轉過來是 pixel，但你也可以直接用 xs_all/ys_all）
                xs_all = (pred_points_all[:, 0] + 1) * 0.5 * (W_patch - 1)
                ys_all = (pred_points_all[:, 1] + 1) * 0.5 * (H_patch - 1)
                exist_prob_all = exist_merged_np         # (M_b,)

                # 原本內框用的浮點座標（從 xs_all / ys_all 複製）
                xs_f = xs_all.copy()
                ys_f = ys_all.copy()
                m_inner = int(0.001 * min(H_patch, W_patch))  # 跟下面保持一致

                exist_prob = exist_prob_all
                keep = (
                    (xs_f >= m_inner) & (xs_f < W_patch - m_inner) &
                    (ys_f >= m_inner) & (ys_f < H_patch - m_inner)
                )
                xs_f = xs_f[keep]
                ys_f = ys_f[keep]
                exist_prob = exist_prob[keep]

                # 這個 patch 的 GT（已是像素座標）
                gt_points_b = points_pad[b].detach().cpu().numpy()  # (N,2)
                gt_mask_b = mask[b].detach().cpu().numpy().astype(bool)  # (N,)
                gt_points_b = gt_points_b[gt_mask_b]
                if gt_points_b.size > 0:
                    gt_xs = gt_points_b[:, 0].astype(int)
                    gt_ys = gt_points_b[:, 1].astype(int)
                else:
                    gt_xs = np.zeros((0,), dtype=int)
                    gt_ys = np.zeros((0,), dtype=int)

                out_name = os.path.basename(meta['image_path']) if 'image_path' in meta else f"{img_key}.jpg"
                per_image_vis_sample[img_key] = {
                    'img_np': img_np,
                    'xs_f': xs_f.astype(np.float32),
                    'ys_f': ys_f.astype(np.float32),
                    'exist_prob': exist_prob,
                    'gt_xs': gt_xs,
                    'gt_ys': gt_ys,
                    'out_name': out_name,
                    'H': H_patch,
                    'W': W_patch,
                    # 這裡一樣存「所有合併後的點」，給 ALLPTS 用
                    'xs_all': xs_all.astype(np.float32),
                    'ys_all': ys_all.astype(np.float32),
                    'exist_prob_all': exist_prob_all.astype(np.float32),
                }

        # 這裡再把 patch-level 的 pred/gt 變成 tensor 方便之後用
        pred_cnt = torch.tensor(pred_cnt_list, device=device)          # [B]
        pred_cnt_hard = torch.tensor(pred_cnt_hard_list, device=device)# [B]
        gt_cnt = torch.tensor(gt_cnt_list, device=device)              # [B]

        # 如果你後面還有用到 p_t_np_all / exist_np_all，可以在這裡 assign
        p_t_np_all = p_merged_np_all
        exist_np_all = exist_merged_np_all


    # ---------------- 以「原圖」為單位計算 MAE / RMSE，並列出每張結果 ----------------
    img_keys = sorted(per_image_gt_sum.keys())
    abs_soft, sq_soft = [], []
    abs_hard, sq_hard = [], []
    per_image_error_soft = {}
    per_image_error_hard = {}

    print("\n[Per-Image Results]")
    for k in img_keys:
        pred_soft = per_image_pred_soft_sum[k]
        pred_hard = per_image_pred_hard_sum[k]
        gt = per_image_gt_sum[k]

        err_soft = abs(pred_soft - gt)
        err_hard = abs(pred_hard - gt)

        abs_soft.append(err_soft)
        sq_soft.append((pred_soft - gt) ** 2)

        abs_hard.append(err_hard)
        sq_hard.append((pred_hard - gt) ** 2)

        per_image_error_soft[k] = err_soft
        per_image_error_hard[k] = err_hard

        print(f"- {k}: soft={pred_soft:.2f}, hard={pred_hard:.2f}, gt={gt:.2f}, "
              f"|err_soft|={err_soft:.2f}, |err_hard|={err_hard:.2f}")

    MAE_soft = float(np.mean(abs_soft))
    RMSE_soft = float(np.sqrt(np.mean(sq_soft)))
    MAE_hard = float(np.mean(abs_hard))
    RMSE_hard = float(np.sqrt(np.mean(sq_hard)))

    print(f"[SOFT] Per-image MAE={MAE_soft:.2f}, RMSE={RMSE_soft:.2f}")
    print(f"[HARD] Per-image MAE={MAE_hard:.2f}, RMSE={RMSE_hard:.2f}")

    # ---------------- 只輸出 Top-10 誤差最小的圖片（以「原圖」為單位） ----------------
    metric = "hard"  # 或 "soft"

    if metric == "hard":
        error_dict = per_image_error_hard
    elif metric == "soft":
        error_dict = per_image_error_soft
    else:
        raise ValueError(f"Unknown metric: {metric}")

    top_k = 10
    # True = 看最慘的，False = 看最好的
    pick_worst = False

    ranked = sorted(
        error_dict.items(),
        key=lambda x: x[1],
        reverse=pick_worst
    )[:top_k]
    if pick_worst:
        print(f"\n[Save bottom-{top_k} Visualizations]")
    else:
        print(f"\n[Save top-{top_k} Visualizations]")
    for rank, (k, err) in enumerate(ranked, start=1):
        if k not in per_image_vis_sample:
            print(f"  (skip) {k} has no vis sample cached.")
            continue

        vis = per_image_vis_sample[k]
        img_np = vis['img_np'].copy()  # 代表 patch
        px = np.asarray(vis['xs_f'], dtype=np.float32)
        py = np.asarray(vis['ys_f'], dtype=np.float32)
        exist_prob = np.asarray(vis['exist_prob'], dtype=np.float32)
        H = vis['H']
        W = vis['W']

        # === 用模型估計的人數 n = round(sum(sigmoid))，挑 top-n 分數最高的點 ===
        n = int(max(0, round(float(exist_prob.sum()))))
        n = int(min(n, exist_prob.shape[0]))

        if n > 0:
            r_pix = max(3, int(0.02 * min(H, W)))

            pts_t = torch.from_numpy(np.stack([px, py], axis=1)).to(device).float()
            scr_t = torch.from_numpy(exist_prob).to(pts_t.device).float()

            keep_idx_t = radius_nms_xyxy(pts_t, scr_t, r=r_pix)

            # ---------- MOD: 在這裡重新定義 m，避免使用到外面 loop 的殘值 ----------
            m = int(0.001 * min(H, W))
            a = ((W - 2 * m) * (H - 2 * m)) / float(W * H)
            a = max(a, 1e-6)

            n_in = float(exist_prob.sum())
            n_hat = int(round(n_in / a))

            n_floor = int(np.ceil(0.8 * len(keep_idx_t)))
            n = max(n_hat, n_floor)
            n = min(n, len(keep_idx_t))

            top_idx = keep_idx_t[:n].cpu().numpy()

            if top_idx.size < n:
                all_idx = np.arange(exist_prob.shape[0])
                remain = np.setdiff1d(all_idx, top_idx, assume_unique=False)
                fill_k = min(n - top_idx.size, remain.size)
                if fill_k > 0:
                    fill = remain[np.argsort(-exist_prob[remain])[:fill_k]]
                    top_idx = np.concatenate([top_idx, fill], axis=0)

            print(f"[dbg] raw={exist_prob.shape[0]} after_inner={len(px)} "
                  f"after_nms={len(keep_idx_t)} n_soft={n} drawn={len(top_idx)} "
                  f"r={r_pix}")
        else:
            top_idx = np.array([], dtype=int)

        # 畫「GT 點」（綠色）
        if 'gt_xs' in vis and 'gt_ys' in vis:
            for (gx, gy) in zip(vis['gt_xs'], vis['gt_ys']):
                gx = int(np.clip(gx, 0, W - 1))
                gy = int(np.clip(gy, 0, H - 1))
                cv2.circle(img_np, (gx, gy), radius=3, color=(0, 255, 0), thickness=-1)

        xs_draw = np.clip(np.round(px[top_idx]).astype(int), 0, W - 1)
        ys_draw = np.clip(np.round(py[top_idx]).astype(int), 0, H - 1)
        for (x, y) in zip(xs_draw, ys_draw):
            cv2.circle(img_np, (int(x), int(y)), 3, (255, 0, 0), -1)

        try:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        except cv2.error:
            img_bgr = img_np

        pred_total = per_image_pred_hard_sum[k]
        gt_total = per_image_gt_sum[k]
        out_name = vis['out_name']
        out_path = os.path.join(
            save_dir,
            f"top{rank:02d}_pred{pred_total:.2f}_gt{gt_total:.2f}_err{err:.2f}_{out_name}"
        )
        print(
            f"  [SAVE] rank={rank:02d} pred={pred_total:.2f} gt={gt_total:.2f} |err|={err:.2f} (patch-n={n}) -> {out_path}")
        cv2.imwrite(out_path, img_bgr)
        print(f"  [SAVE] rank={rank:02d} |err|={err:.2f} (n={n}) -> {out_path}")

        # ---------- NEW: 額外輸出一張「所有 raw N 點」的圖 ----------
        img_all = vis['img_np'].copy()
        xs_all = np.asarray(vis['xs_all'], dtype=np.float32)
        ys_all = np.asarray(vis['ys_all'], dtype=np.float32)
        xs_int = np.clip(np.round(xs_all).astype(int), 0, W - 1)
        ys_int = np.clip(np.round(ys_all).astype(int), 0, H - 1)

        uniq = np.unique(np.stack([xs_int, ys_int], axis=1), axis=0)

        print(f"[dbg-ALL] total={len(xs_all)} unique_pixels={len(uniq)}")
        # 先畫 GT（綠色）
        if 'gt_xs' in vis and 'gt_ys' in vis:
            for (gx, gy) in zip(vis['gt_xs'], vis['gt_ys']):
                gx = int(np.clip(gx, 0, W - 1))
                gy = int(np.clip(gy, 0, H - 1))
                cv2.circle(img_all, (gx, gy), 3, (0, 255, 0), -1)

        # 再畫所有候選點（紅色，N 一般是 900）
        for x, y in zip(xs_all, ys_all):
            x_int = int(np.clip(round(x), 0, W - 1))
            y_int = int(np.clip(round(y), 0, H - 1))
            cv2.circle(img_all, (x_int, y_int), 2, (0, 0, 255), -1)

        try:
            img_all_bgr = cv2.cvtColor(img_all, cv2.COLOR_RGB2BGR)
        except cv2.error:
            img_all_bgr = img_all

        out_path_all = os.path.join(
            save_dir,
            f"top{rank:02d}_ALLPTS_pred{pred_total:.2f}_gt{gt_total:.2f}_err{err:.2f}_{out_name}"
        )
        cv2.imwrite(out_path_all, img_all_bgr)
        print(f"  [SAVE-ALL] rank={rank:02d} -> {out_path_all}")
