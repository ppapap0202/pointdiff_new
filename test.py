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
    eps = 1e-8
    sqrt_abar_t    = (abar_t + eps).sqrt()
    sqrt_one_mt    = (1.0 - abar_t).clamp_min(0).sqrt()
    x0_pred        = (p_t - sqrt_one_mt * eps_pred) / sqrt_abar_t

    sqrt_abar_prev = abar_prev.clamp(0, 1).sqrt()
    sqrt_one_mp    = (1.0 - abar_prev).clamp_min(0).sqrt()
    p_prev         = sqrt_abar_prev * x0_pred + sqrt_one_mp * eps_pred
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
    依你的 ImageDataset meta 欄位調整。常見有：
      - 'img_path' / 'image_path'
      - 'img_name' / 'image_name'
      - 'image_id'
    """
    for k in ['img_path', 'image_path', 'img_name', 'image_name', 'image_id']:
        if k in meta:
            return meta[k]
    # 最保險：整個 meta 序列化（不建議，除非真的沒有合適欄位）
    return str(meta)
# ---------- main ----------
if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = r"C:\pycharm\pointdiff_new\output3\last_epoch0069.pth"
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
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_points_padded,
    )

    T = 1000
    steps = 50
    sched = CosineAbarSchedule(T=T)
    abar = sched.abar.to(device=device)
    t_seq = make_ddim_steps(T=T, steps=steps, device=device)
    clamp_eps = 1e-3

    # 用來「以原圖為單位」聚合預測與 GT
    per_image_pred_sum = defaultdict(float)
    per_image_gt_sum   = defaultdict(float)
    per_image_vis_sample = {}
    save_dir = "vis_results"
    os.makedirs(save_dir, exist_ok=True)

    for images, points_pad, mask, metas in loader:
        images = images.to(device)
        mask = mask.to(device)

        B, C, H, W = images.shape

        # 影像轉 numpy（RGB 假設），供暫存代表 patch 用
        imgs_np_all = (images.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).clip(0, 255).astype(np.uint8)

        # 編碼
        feats = model.encode(images)
        feats_zero = [f * 0 for f in feats] if isinstance(feats, (list, tuple)) else feats * 0
        # 初始 p_t
        N = points_pad.shape[1]
        p_t = torch.empty((B, N, 2), device=device).uniform_(-1.0 + clamp_eps, 1.0 - clamp_eps)

        # DDIM 多步反推
        for i, t_int in enumerate(t_seq.tolist()):
            t_tensor = torch.full((B, 1), t_int, device=device, dtype=torch.long)
            eps_pred, exist_logit = model.denoise(feats, p_t, t_tensor)

            abar_t = abar[t_int]
            abar_prev = abar[t_seq[i + 1]] if i + 1 < len(t_seq) else torch.tensor(1.0, device=device)
            p_t = ddim_reverse_step(p_t, eps_pred, abar_t, abar_prev)
            p_t = p_t.clamp(min=-1.0 + clamp_eps, max=1.0 - clamp_eps)

        # 最後一次 exist_logit → 每點存在機率
        exist_prob_batch = torch.sigmoid(exist_logit)  # [B, N]
        pred_cnt = exist_prob_batch.sum(dim=1)  # [B]
        gt_cnt = mask.sum(dim=1).float()  # [B]

        # 便利暫存需要的 numpy
        p_t_np_all = p_t.detach().cpu().numpy()  # [B, N, 2] in [-1,1]
        exist_np_all = exist_prob_batch.detach().cpu().numpy()  # [B, N]

        # 依「原圖 key」累加人數 & 暫存代表 patch（含 GT）
        for b in range(B):
            meta = metas[b]
            img_key = get_image_key_from_meta(meta)

            per_image_pred_sum[img_key] += float(pred_cnt[b].item())
            per_image_gt_sum[img_key] += float(gt_cnt[b].item())

            if img_key not in per_image_vis_sample:
                # 這個 patch 的影像（RGB 假設；灰階轉 BGR 方便畫色）
                img_np = imgs_np_all[b]
                if img_np.ndim == 2:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
                elif img_np.shape[2] == 1:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
                img_np = np.ascontiguousarray(img_np)

                # 預測點：[-1,1] -> 像素
                pred_points = p_t_np_all[b]  # (N,2)
                xs = ((pred_points[:, 0] + 1) * 0.5 * W).astype(int)
                ys = ((pred_points[:, 1] + 1) * 0.5 * H).astype(int)

                # 這個 patch 的存在機率
                exist_prob = exist_np_all[b]  # (N,)

                # 這個 patch 的 GT（已是像素座標）
                gt_points_b = points_pad[b].detach().cpu().numpy()  # (N,2)
                gt_mask_b = mask[b].detach().cpu().numpy().astype(bool)  # (N,)
                gt_points_b = gt_points_b[gt_mask_b]
                gt_xs = gt_points_b[:, 0].astype(int) if gt_points_b.size else np.zeros((0,), dtype=int)
                gt_ys = gt_points_b[:, 1].astype(int) if gt_points_b.size else np.zeros((0,), dtype=int)

                out_name = os.path.basename(meta['image_path']) if 'image_path' in meta else f"{img_key}.jpg"

                per_image_vis_sample[img_key] = {
                    'img_np': img_np,  # 代表 patch 圖（暫為 RGB/BGR 混合，存檔前再轉）
                    'xs': xs, 'ys': ys,  # 預測點像素座標
                    'exist_prob': exist_prob,  # (N,)
                    'gt_xs': gt_xs, 'gt_ys': gt_ys,
                    'out_name': out_name,
                    'H': H, 'W': W,
                }

    # ---------------- 以「原圖」為單位計算 MAE / RMSE，並列出每張結果 ----------------
    img_keys = sorted(per_image_gt_sum.keys())
    abs_errors, sq_errors = [], []
    per_image_error = {}  # img_key -> |pred-gt|

    print("\n[Per-Image Results]")
    for k in img_keys:
        pred = per_image_pred_sum[k]
        gt = per_image_gt_sum[k]
        err = abs(pred - gt)
        abs_errors.append(err)
        sq_errors.append((pred - gt) ** 2)
        per_image_error[k] = err
        print(f"- {k}: pred={pred:.2f}, gt={gt:.2f}, |err|={err:.2f}")

    if len(abs_errors) > 0:
        mae = float(np.mean(abs_errors))
        rmse = float(np.sqrt(np.mean(sq_errors)))
    else:
        mae, rmse = float('nan'), float('nan')

    print("\n[Overall]")
    print(f"MAE  (per-image counting) = {mae:.4f}")
    print(f"RMSE (per-image counting) = {rmse:.4f}")

    # ---------------- 只輸出 Top-10 誤差最小的圖片（以「原圖」為單位） ----------------
    top_k = 10
    ranked = sorted(per_image_error.items(), key=lambda x: x[1])[:top_k]

    print(f"\n[Save Top-{top_k} Visualizations]")
    for rank, (k, err) in enumerate(ranked, start=1):
        if k not in per_image_vis_sample:
            print(f"  (skip) {k} has no vis sample cached.")
            continue

        vis = per_image_vis_sample[k]
        img_np = vis['img_np'].copy()  # 代表 patch
        xs, ys = vis['xs'], vis['ys']
        exist_prob = np.asarray(vis['exist_prob'], dtype=np.float32)
        H, W = vis['H'], vis['W']

        # === 用模型估計的人數 n = round(sum(sigmoid))，挑 top-n 分數最高的點 ===
        n = int(max(0, round(float(exist_prob.sum()))))
        n = int(min(n, exist_prob.shape[0]))

        if n > 0:
            top_idx = np.argpartition(-exist_prob, n - 1)[:n]
            top_idx = top_idx[np.argsort(-exist_prob[top_idx])]
        else:
            top_idx = np.array([], dtype=int)

        # 畫「預測點」（藍色；OpenCV 是 BGR → 藍色 (255,0,0)）
        for i in top_idx:
            x = int(np.clip(xs[i], 0, W - 1))
            y = int(np.clip(ys[i], 0, H - 1))
            cv2.circle(img_np, (x, y), radius=3, color=(255, 0, 0), thickness=-1)

        # 畫「GT 點」（綠色）
        if 'gt_xs' in vis and 'gt_ys' in vis:
            for (gx, gy) in zip(vis['gt_xs'], vis['gt_ys']):
                gx = int(np.clip(gx, 0, W - 1))
                gy = int(np.clip(gy, 0, H - 1))
                cv2.circle(img_np, (gx, gy), radius=3, color=(0, 255, 0), thickness=-1)

        # 存檔（若 img_np 是 RGB，轉 BGR 避免色偏）
        out_name = vis['out_name']
        out_path = os.path.join(save_dir, f"top{rank:02d}_err{err:.2f}_{out_name}")
        try:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        except cv2.error:
            # 若本來就是 BGR（或單通道），轉換會出錯；那就直接存
            img_bgr = img_np

        pred_total = per_image_pred_sum[k]
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

    # # === 可視化輸出 (Pred vs GT) ===
    # exist_prob = torch.sigmoid(5j/.detach())[0].cpu().numpy()
    # pred_points = p_t[0].cpu().numpy()  # (N,2)
    # xs = ((pred_points[:, 0] + 1) * 0.5 * W).astype(int)
    # ys = ((pred_points[:, 1] + 1) * 0.5 * H).astype(int)
    #
    # # 原始圖片
    # img_np = (images[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    # if img_np.shape[2] == 1:
    #     img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    # img_np = np.ascontiguousarray(img_np)
    #
    #     # --- 畫預測點（紅色） ---
    # for (x, y, p) in zip(xs, ys, exist_prob):
    #     if p > 0:  # threshold
    #         cv2.circle(img_np, (x, y), radius=3, color=(0, 0, 255), thickness=-1)
    #
    # # --- 畫 GT 標註點（綠色） ---
    # gt_points = points_pad[0].cpu().numpy()
    # gt_mask = mask[0].cpu().numpy()
    # gt_points = gt_points[gt_mask]  # 只取有效點
    # gt_xs = gt_points[:, 0].astype(int)
    # gt_ys = gt_points[:, 1].astype(int)
    #
    # for (x, y) in zip(gt_xs, gt_ys):
    #     cv2.circle(img_np, (x, y), radius=3, color=(0, 255, 0), thickness=-1)
    #
    # # 存圖
    # out_name = os.path.basename(metas[0]['image_path'])
    # out_path = os.path.join(save_dir, f"vis_{out_name}")
    # cv2.imwrite(out_path, img_np)
    # print(f"[SAVE] {out_path}")
