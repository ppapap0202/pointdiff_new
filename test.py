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

    ckpt_path = args.ckpt_path
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
    per_image_points_xy = defaultdict(list)  # img_key -> [tensor(K, 2), ...]
    per_image_points_prob = defaultdict(list)  # img_key -> [tensor(K,), ...]
    per_image_size = {}                        # img_key -> (H_full, W_full)
    per_image_gt_points_xy = defaultdict(list)
    save_dir = args.save_dir
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
            print(r)
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

        hard_thresh = args.hard_thresh

        pred_cnt_list = []
        pred_cnt_hard_list = []
        gt_cnt_list = []

        # 給可視化用的 list（每張 patch 對應一個 (M_b,2)/(M_b,)）
        p_merged_np_all = []
        exist_merged_np_all = []

        # 依「原圖 key」累加人數 & 暫存代表 patch（含 GT）
        for b in range(B):
            #print(b)
            meta = metas[b]
            #print(meta)
            img_key = get_image_key_from_meta(meta)
            #print(img_key)
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
            #print(pts_merged_norm.shape)
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
            # per_image_pred_soft_sum[img_key] += float(pred_cnt_b.item())
            # per_image_pred_hard_sum[img_key] += float(pred_cnt_hard_b.item())
            per_image_gt_sum[img_key] += float(gt_cnt_b.item())

            # 存 numpy 給可視化用（合併後的 M_b 點）
            pts_merged_np = pts_merged_norm.detach().cpu().numpy()   # (M_b, 2)
            exist_merged_np = exist_prob_b.detach().cpu().numpy()    # (M_b,)
            p_merged_np_all.append(pts_merged_np)
            exist_merged_np_all.append(exist_merged_np)
            # ---------- (A) 將這個 patch 的點轉成「原圖 global pixel 座標」 ----------
            # 先算 patch 內 pixel 座標
            xs_patch = (pts_merged_norm[:, 0] + 1) * 0.5 * (W - 1)  # [M_b]
            ys_patch = (pts_merged_norm[:, 1] + 1) * 0.5 * (H - 1)  # [M_b]

            # 2) 從 meta 取出這個 tile 在原圖的左上角 (tile_left, tile_top)
            #    orig_size = [H_full, W_full]
            H_full, W_full = meta['orig_size']
            x0 = meta['tile_left']   # global x offset
            y0 = meta['tile_top']    # global y offset
            if img_key not in per_image_size:
                per_image_size[img_key] = (H_full, W_full)
            xs_global = xs_patch + x0  # [M_b]
            ys_global = ys_patch + y0  # [M_b]

            # 1. 先算出 Global 座標（這部分維持不變）
            xs_global = xs_patch + x0  # [M_b]
            ys_global = ys_patch + y0  # [M_b]

            # 2. 轉成 Global 整數並 Clamp（這部分維持不變）
            xs_global_int = xs_global.round().long()
            ys_global_int = ys_global.round().long()
            xs_global_int = xs_global_int.clamp(0, W_full - 1)
            ys_global_int = ys_global_int.clamp(0, H_full - 1)

            # 3. 【關鍵修改】：改用 xs_patch 和 ys_patch 來製作遮罩
            #    將 Patch 座標轉整數，檢查是否落在 Patch 的原點 (0,0)
            xs_patch_int = xs_patch.round().long()
            ys_patch_int = ys_patch.round().long()

            #    過濾條件：只要在 Patch 內的 (0,0)，就視為 Padding/背景雜訊
            mask_valid = ~((xs_patch_int == 0) & (ys_patch_int == 0))

            # 4. 應用遮罩（這部分維持不變）
            xs_global_int = xs_global_int[mask_valid]
            ys_global_int = ys_global_int[mask_valid]
            exist_prob_valid = exist_prob_b[mask_valid]

            # 存到「以原圖為單位」的 dict 裡，等所有 patch 跑完再一起 merge
            if xs_global_int.numel() > 0:
                per_image_points_xy[img_key].append(
                    torch.stack([xs_global_int, ys_global_int], dim=1).cpu()
                )  # (K, 2)
                per_image_points_prob[img_key].append(
                    exist_prob_valid.detach().cpu()
                )  # (K,)
            mask_b_cpu = mask[b].detach().cpu()  # 把 boolean mask 拉回 CPU
            gt_points_b = points_pad[b][mask_b_cpu]  # (n_gt, 2) patch 座標（在 CPU）
            if gt_points_b.numel() > 0:
                gt_xs_g = gt_points_b[:, 0] + x0
                gt_ys_g = gt_points_b[:, 1] + y0
                gt_xs_g = gt_xs_g.round().long().clamp(0, W_full - 1)
                gt_ys_g = gt_ys_g.round().long().clamp(0, H_full - 1)
                per_image_gt_points_xy[img_key].append(
                    torch.stack([gt_xs_g, gt_ys_g], dim=1)
                )
            # ---------- per_image_vis_sample：只挑一個代表 patch 來畫 ----------
            # ---------- per_image_vis_sample：只挑一個代表「原圖」來畫 ----------
            # if img_key not in per_image_vis_sample:
            #     # 1) 讀整張原圖
            #     img_bgr_full = cv2.imread(meta['image_path'], cv2.IMREAD_COLOR)
            #     if img_bgr_full is None:
            #         print(f"[WARN] fail to read image: {meta['image_path']}")
            #         continue
            #     img_np = cv2.cvtColor(img_bgr_full, cv2.COLOR_BGR2RGB)
            #
            #     H_full, W_full = meta['orig_size']
            #     # 確保讀進來的尺寸跟 orig_size 一樣，不一樣就 resize 一下
            #     if (img_np.shape[0], img_np.shape[1]) != (H_full, W_full):
            #         img_np = cv2.resize(img_np, (W_full, H_full))
            #     img_np = np.ascontiguousarray(img_np)
            #
            #     # 2) 這個 patch 的「全部預測點」(global 座標)
            #     xs_all = xs_global_int.cpu().numpy().astype(np.float32)
            #     ys_all = ys_global_int.cpu().numpy().astype(np.float32)
            #     exist_prob_all = exist_prob_valid.detach().cpu().numpy().astype(np.float32)
            #
            #     # 3) 內框過濾（改用原圖的 H_full, W_full）
            #     xs_f = xs_all.copy()
            #     ys_f = ys_all.copy()
            #     m_inner = int(0.001 * min(H_full, W_full))
            #
            #     keep_inner = (
            #             (xs_f >= m_inner) & (xs_f < W_full - m_inner) &
            #             (ys_f >= m_inner) & (ys_f < H_full - m_inner)
            #     )
            #     xs_f = xs_f[keep_inner]
            #     ys_f = ys_f[keep_inner]
            #     exist_prob = exist_prob_all[keep_inner]
            #
            #     # 4) 這個 patch 的 GT 也轉成 global（patch 座標 + offset）
            #     gt_points_b = points_pad[b].detach().cpu().numpy()  # (N,2) patch 座標
            #     gt_mask_b = mask[b].detach().cpu().numpy().astype(bool)
            #     gt_points_b = gt_points_b[gt_mask_b]
            #     if gt_points_b.size > 0:
            #         gt_xs = gt_points_b[:, 0] + x0
            #         gt_ys = gt_points_b[:, 1] + y0
            #         gt_xs = np.clip(gt_xs, 0, W_full - 1).astype(int)
            #         gt_ys = np.clip(gt_ys, 0, H_full - 1).astype(int)
            #     else:
            #         gt_xs = np.zeros((0,), dtype=int)
            #         gt_ys = np.zeros((0,), dtype=int)
            #
            #     out_name = os.path.basename(meta['image_path'])
            #
            #     per_image_vis_sample[img_key] = {
            #         'img_np': img_np,  #  整張原圖
            #         'xs_f': xs_f.astype(np.float32),
            #         'ys_f': ys_f.astype(np.float32),
            #         'exist_prob': exist_prob,
            #         'gt_xs': gt_xs,
            #         'gt_ys': gt_ys,
            #         'out_name': out_name,
            #         'H': H_full,  #  原圖高度
            #         'W': W_full,  #  原圖寬度
            #         'xs_all': xs_all.astype(np.float32),
            #         'ys_all': ys_all.astype(np.float32),
            #         'exist_prob_all': exist_prob_all.astype(np.float32),
            #     }

        # 這裡再把 patch-level 的 pred/gt 變成 tensor 方便之後用
        pred_cnt = torch.tensor(pred_cnt_list, device=device)          # [B]
        pred_cnt_hard = torch.tensor(pred_cnt_hard_list, device=device)# [B]
        gt_cnt = torch.tensor(gt_cnt_list, device=device)              # [B]
        # 如果你後面還有用到 p_t_np_all / exist_np_all，可以在這裡 assign
        p_t_np_all = p_merged_np_all
        exist_np_all = exist_merged_np_all

    per_image_pred_soft_sum.clear()
    per_image_pred_hard_sum.clear()
    per_image_vis_sample.clear()
    for img_key in per_image_points_xy.keys():
        pts_list = per_image_points_xy[img_key]
        prob_list = per_image_points_prob[img_key]

        pts_all = torch.cat(pts_list, dim=0)  # [M_all, 2]
        prob_all = torch.cat(prob_list, dim=0)  # [M_all]
        H_img, W_img = per_image_size[img_key]
        xs_g = pts_all[:, 0].long().clamp(0, W_img - 1)
        ys_g = pts_all[:, 1].long().clamp(0, H_img - 1)



        flat = ys_g * W_img + xs_g
        uniq_flat, inv = torch.unique(flat, return_inverse=True)
        x_uniq = (uniq_flat % W_img).float()  # [M_img]
        y_uniq = (uniq_flat // W_img).float()  # [M_img]
        M_img = uniq_flat.shape[0]
        prob_merged = torch.zeros(M_img, dtype=prob_all.dtype)

        for i_pix in range(M_img):
            hit_mask = (inv == i_pix)
            if hit_mask.any():
                prob_merged[i_pix] = prob_all[hit_mask].max()

        valid_mask = (prob_merged > hard_thresh)  # 注意：這裡用 >，後面畫圖也用同一套
        x_valid = x_uniq[valid_mask]
        y_valid = y_uniq[valid_mask]
        s_valid = prob_merged[valid_mask]

        r_pix = max(3, int(0.005 * min(H_img, W_img)))

        pred_xy_hard_nms = np.zeros((0, 2), dtype=np.float32)  # (K,2) global pixel
        if x_valid.numel() > 0:
            pts_t = torch.stack([x_valid, y_valid], dim=1).to(device).float()  # [N,2]
            scr_t = s_valid.to(device).float()
            keep_idx = radius_nms_xyxy(pts_t, scr_t, r=r_pix).detach().cpu()

            x_keep = x_valid[keep_idx].detach().cpu().numpy().astype(np.float32)
            y_keep = y_valid[keep_idx].detach().cpu().numpy().astype(np.float32)
            pred_xy_hard_nms = np.stack([x_keep, y_keep], axis=1)  # (K,2)

        pred_hard_nms = float(pred_xy_hard_nms.shape[0])

        pred_soft = float(prob_merged.sum().item())
        pred_hard_raw = float((prob_merged >= hard_thresh).float().sum().item())
        pred_hard = pred_hard_nms
        per_image_pred_soft_sum[img_key] = pred_soft
        per_image_pred_hard_sum[img_key] = pred_hard
        # ---- 建立可視化用的資料（整張圖） ----
        img_bgr_full = cv2.imread(img_key, cv2.IMREAD_COLOR)
        if img_bgr_full is None:
            print(f"[WARN] fail to read image: {img_key}")
            continue
        img_rgb_full = cv2.cvtColor(img_bgr_full, cv2.COLOR_BGR2RGB)
        if (img_rgb_full.shape[0], img_rgb_full.shape[1]) != (H_img, W_img):
            img_rgb_full = cv2.resize(img_rgb_full, (W_img, H_img))
        img_rgb_full = np.ascontiguousarray(img_rgb_full)

        xs_all = x_uniq.cpu().numpy().astype(np.float32)
        ys_all = y_uniq.cpu().numpy().astype(np.float32)
        exist_prob_all = prob_merged.cpu().numpy().astype(np.float32)

        # 內框過濾
        xs_f = xs_all.copy()
        ys_f = ys_all.copy()
        m_inner = int(0.001 * min(H_img, W_img))
        keep_inner = (
                (xs_f >= m_inner) & (xs_f < W_img - m_inner) &
                (ys_f >= m_inner) & (ys_f < H_img - m_inner)
        )
        xs_f = xs_f[keep_inner]
        ys_f = ys_f[keep_inner]
        exist_prob = exist_prob_all[keep_inner]

        # GT 全圖座標
        if img_key in per_image_gt_points_xy:
            gt_all = torch.cat(per_image_gt_points_xy[img_key], dim=0)  # (M_gt, 2)
            gt_xs = gt_all[:, 0].cpu().numpy().clip(0, W_img - 1).astype(int)
            gt_ys = gt_all[:, 1].cpu().numpy().clip(0, H_img - 1).astype(int)
        else:
            gt_xs = np.zeros((0,), dtype=int)
            gt_ys = np.zeros((0,), dtype=int)

        per_image_vis_sample[img_key] = {
            'img_np': img_rgb_full,
            'xs_f': xs_f.astype(np.float32),
            'ys_f': ys_f.astype(np.float32),
            'exist_prob': exist_prob.astype(np.float32),
            'gt_xs': gt_xs,
            'gt_ys': gt_ys,
            'out_name': os.path.basename(img_key),
            'H': H_img,
            'W': W_img,
            'xs_all': xs_all,
            'ys_all': ys_all,
            'exist_prob_all': exist_prob_all,
            'pred_xy_hard_nms': pred_xy_hard_nms,  # <- 新增：硬預測最終點
            'r_pix': int(r_pix),  # <- 新增：用的 NMS 半徑
        }

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
    pick_worst = args.pick_worst

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
        img_np = vis['img_np'].copy()
        H = vis['H']
        W = vis['W']

        # 取出 hard+NMS 最終點（這批點數會對齊 pred_total）
        pred_xy = np.asarray(vis.get('pred_xy_hard_nms', np.zeros((0, 2), np.float32)), dtype=np.float32)
        r_pix = int(vis.get('r_pix', max(3, int(0.005 * min(H, W)))))

        # 畫 GT（綠色）
        if 'gt_xs' in vis and 'gt_ys' in vis:
            for (gx, gy) in zip(vis['gt_xs'], vis['gt_ys']):
                gx = int(np.clip(gx, 0, W - 1))
                gy = int(np.clip(gy, 0, H - 1))
                cv2.circle(img_np, (gx, gy), radius=3, color=(0, 255, 0), thickness=-1)

        # 畫 Hard 預測（藍色）
        for (x, y) in pred_xy:
            x_i = int(np.clip(round(float(x)), 0, W - 1))
            y_i = int(np.clip(round(float(y)), 0, H - 1))
            cv2.circle(img_np, (x_i, y_i), 3, (255, 0, 0), -1)

        n = int(pred_xy.shape[0])
        print(f"[dbg-hard] hard_draw={n} r={r_pix}")



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

        # 再畫所有候選點（紅色，N 一般是 900）
        for x, y in zip(xs_all, ys_all):
            x_int = int(np.clip(round(x), 0, W - 1))
            y_int = int(np.clip(round(y), 0, H - 1))
            cv2.circle(img_all, (x_int, y_int), 2, (0, 0, 255), -1)

        try:
            img_all_bgr = cv2.cvtColor(img_all, cv2.COLOR_RGB2BGR)
        except cv2.error:
            img_all_bgr = img_all
        # 畫「GT 點」（綠色）
        if 'gt_xs' in vis and 'gt_ys' in vis:
            for (gx, gy) in zip(vis['gt_xs'], vis['gt_ys']):
                gx = int(np.clip(gx, 0, W - 1))
                gy = int(np.clip(gy, 0, H - 1))
                cv2.circle(img_np, (gx, gy), radius=3, color=(0, 255, 0), thickness=-1)
        out_path_all = os.path.join(
            save_dir,
            f"top{rank:02d}_ALLPTS_pred{pred_total:.2f}_gt{gt_total:.2f}_err{err:.2f}_{out_name}"
        )
        cv2.imwrite(out_path_all, img_all_bgr)
        print(f"  [SAVE-ALL] rank={rank:02d} -> {out_path_all}")
