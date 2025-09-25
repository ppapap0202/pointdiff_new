import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from models import build_model
from models.diffusion_utils import CosineAbarSchedule
from dataset.dataset import ImageDataset
import cv2
import numpy as np

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

# ---------- main ----------
if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = r"D:\PyCharmMiscProject\pointdiff_new\output\best_epoch0069_val2.93.pth"
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
    loader = DataLoader(
        dataset,
        batch_size=1,
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

    save_dir = "vis_results"
    os.makedirs(save_dir, exist_ok=True)

    for images, points_pad, mask, metas in loader:
        images = images.to(device)
        mask = mask.to(device)
        B, C, H, W = images.shape

        feats = model.encode(images)

        N = points_pad.shape[1]
        p_t = torch.empty((B, N, 2), device=device).uniform_(
            -1.0 + clamp_eps, 1.0 - clamp_eps
        )

        last_exist_logit = None
        for i, t_int in enumerate(t_seq.tolist()):
            t_tensor = torch.full((B, 1), t_int, device=device, dtype=torch.long)
            eps_pred, exist_logit = model.denoise(feats, p_t, t_tensor)
            last_exist_logit = exist_logit
            pred_cnt = torch.sigmoid(exist_logit).sum(dim=1)
            gt_cnt = mask.sum(dim=1).float()
            L_cnt = F.l1_loss(pred_cnt, gt_cnt)

            abar_t = abar[t_int]
            abar_prev = abar[t_seq[i+1]] if i+1 < len(t_seq) else torch.tensor(1.0, device=device)
            p_t = ddim_reverse_step(p_t, eps_pred, abar_t, abar_prev)
            p_t = p_t.clamp(min=-1.0 + clamp_eps, max=1.0 - clamp_eps)

        # === 可視化輸出 (Pred vs GT) ===
        exist_prob = torch.sigmoid(last_exist_logit.detach())[0].cpu().numpy()
        pred_points = p_t[0].cpu().numpy()  # (N,2)
        xs = ((pred_points[:, 0] + 1) * 0.5 * W).astype(int)
        ys = ((pred_points[:, 1] + 1) * 0.5 * H).astype(int)

        # 原始圖片
        img_np = (images[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        if img_np.shape[2] == 1:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        img_np = np.ascontiguousarray(img_np)

        # --- 畫預測點（紅色） ---
        for (x, y, p) in zip(xs, ys, exist_prob):
            if p > 0:  # threshold
                cv2.circle(img_np, (x, y), radius=3, color=(0, 0, 255), thickness=-1)

        # --- 畫 GT 標註點（綠色） ---
        gt_points = points_pad[0].cpu().numpy()
        gt_mask = mask[0].cpu().numpy()
        gt_points = gt_points[gt_mask]  # 只取有效點
        gt_xs = gt_points[:, 0].astype(int)
        gt_ys = gt_points[:, 1].astype(int)

        for (x, y) in zip(gt_xs, gt_ys):
            cv2.circle(img_np, (x, y), radius=3, color=(0, 255, 0), thickness=-1)

        # 存圖
        out_name = os.path.basename(metas[0]['image_path'])
        out_path = os.path.join(save_dir, f"vis_{out_name}")
        cv2.imwrite(out_path, img_np)
        print(f"[SAVE] {out_path}")
