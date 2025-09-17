import os
import argparse
import yaml
import torch
import math
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models import build_model
from models.diffusion_utils import pixels_to_m11, CosineAbarSchedule
from dataset.dataset import ImageDataset


def make_ddim_steps(T=1000, steps=50, device='cpu'):
    # e.g., 999 → 0 均勻取樣 50 個步
    ts = torch.linspace(T-1, 0, steps, device=device, dtype=torch.long)
    return ts

@torch.no_grad()
def ddim_reverse_step(p_t, eps_pred, abar_t, abar_prev):
    """
    p_t, eps_pred: [B,N,2]
    abar_t, abar_prev: 標量 float tensor（對應當前/前一個時間步的 ᾱ）
    DDIM(eta=0):
      x0_pred = (p_t - sqrt(1-abar_t)*eps_pred) / sqrt(abar_t)
      p_{t-1} = sqrt(abar_prev)*x0_pred + sqrt(1-abar_prev)*eps_pred
    """
    eps = 1e-8
    sqrt_abar_t    = (abar_t + eps).sqrt()
    sqrt_one_mt    = (1.0 - abar_t).clamp_min(0).sqrt()
    x0_pred        = (p_t - sqrt_one_mt * eps_pred) / sqrt_abar_t

    sqrt_abar_prev = abar_prev.clamp(0, 1).sqrt()
    sqrt_one_mp    = (1.0 - abar_prev).clamp_min(0).sqrt()
    p_prev         = sqrt_abar_prev * x0_pred + sqrt_one_mp * eps_pred
    return p_prev
# ---------- collate：把 [N,2] points pad 成固定長度 ----------
def collate_points_padded(batch, max_n=900):
    import torch
    imgs, pts, metas = zip(*batch)   # (img_tile, label_out[N,2], meta)
    imgs = torch.stack(imgs, 0)      # [B,C,H,W]

    B = len(pts)
    padded = torch.full((B, max_n, 2), fill_value=-10.0, dtype=torch.float32)
    mask   = torch.zeros((B, max_n), dtype=torch.bool)

    for i, p in enumerate(pts):
        # p: Tensor [N,2]
        n = int(p.size(0))
        m = min(n, max_n)
        if m > 0:
            padded[i, :m] = p[:m]
            mask[i, :m]   = True

    return imgs, padded, mask, list(metas)

# ---------- 載入 config ----------
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

# ---------- 更穩健的 checkpoint 載入 ----------
def load_checkpoint_into_model(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict):
        if "model_state" in state:
            model.load_state_dict(state["model_state"])
        elif "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            # 可能已經是 state_dict
            model.load_state_dict(state)
    else:
        model.load_state_dict(state)
    return model

if __name__ == "__main__":
    # import torch.multiprocessing as mp
    # mp.set_start_method("spawn", force=True)

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) checkpoint 路徑（改成你的）
    ckpt_path = r"C:\Users\qwert\PycharmProjects\pythonProject\pointdiff_with_logging\output\best_epoch0006_val4.39.pth"

    # 2) 建模（與訓練時一致；若你訓練是灰階，ensure in_ch=1）
    model = build_model(args, training=False)
    model = load_checkpoint_into_model(model, ckpt_path, device)
    model.to(device).eval()
    print(f"[INFO] Loaded checkpoint: {ckpt_path}")

    # 3) DataLoader（測試資料來源：args.test_root）
    #   - 若你在 Windows 曾遇到多進程錯誤，把 num_workers 改成 0 測試
    max_n = 900
    dataset = ImageDataset(
        root=args.test_root,
        mode='points',
        tile_size=(256, 256),
        stride=(256, 256),
        gray=False,                    # 如果你訓練是灰階，改 True
        pad_if_needed=True,
        image_exts=('.jpg', '.png'),
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,                 # 若報 multiprocessing 錯誤，先改 0
        pin_memory=True,
        persistent_workers=True,       # num_workers=0 時請設 False
        prefetch_factor=4,
        collate_fn=collate_points_padded,
    )
    total_abs_err = 0.0
    total_sq_err = 0.0
    n_samples = 0
    # 4) 推論
    # 4. 推論 loop + 指標統計
    import math

    total_abs_err = 0.0
    total_sq_err = 0.0
    n_samples = 0

    with torch.no_grad():
        # 準備 cosine ᾱ 時間表 & DDIM 步
        T = 1000  # 與訓練一致
        steps = 50  # 反向步數；可調 20~100，看速度/精度取捨
        sched = CosineAbarSchedule(T=T)  # 你專案已有的排程
        abar = sched.abar.to(device=device)  # 期望 shape [T]
        t_seq = make_ddim_steps(T=T, steps=steps, device=device)

        clamp_eps = 1e-3  # 避免落到邊界外，grid_sample 取樣會是 0

        for images, points_pad, mask, metas in loader:
            # print(f'images.shape: {images.shape}')  # (B, C, H, W)
            # print(f'points.shape: {points_pad.shape}')  # (B, max_len, 2)
            # print(f'mask.shape: {mask.shape}')  # (B, max_len)
            # print(metas)
            images = images.to(device)  # [B,C,H,W]
            mask = mask.to(device)  # [B,N] 這裡只用來算 GT 人數
            B, C, H, W = images.shape

            # 影像特徵只算一次
            feats = model.encode(images)

            # ---- 從「雜訊」初始化點集（完全不碰 GT points）----
            N = points_pad.shape[1]  # 仍用 collate 的 Nmax，例如 900
            p_t = torch.empty((B, N, 2), device=device).uniform_(
                -1.0 + clamp_eps, 1.0 - clamp_eps
            )

            last_exist_logit = None

            # ---- DDIM 反向去噪：t=T-1 → 0 ----
            for i, t_int in enumerate(t_seq.tolist()):
                t_tensor = torch.full((B, 1), t_int, device=device, dtype=torch.long)  # [B,1]
                # 模型預測 ε 與存在分數
                eps_pred, exist_logit = model.denoise(feats, p_t, t_tensor)
                last_exist_logit = exist_logit  # 記下最後一步的存在分數

                # 取當前/前一 ᾱ
                abar_t = abar[t_int]
                abar_prev = abar[t_seq[i + 1]] if i + 1 < len(t_seq) else torch.tensor(1.0, device=device)

                # 一步反向
                p_t = ddim_reverse_step(p_t, eps_pred, abar_t, abar_prev)

                # 安全：避免出界
                p_t = p_t.clamp(min=-1.0 + clamp_eps, max=1.0 - clamp_eps)

            # ---- 最終人數（只用最後一步的 exist 分數）----
            BIAS_SHIFT = -0 # 可以放到 config；先用這個起手
            exist_prob = torch.sigmoid(last_exist_logit + BIAS_SHIFT)
            pred_count = exist_prob.sum(dim=1)
            gt_count = mask.sum(dim=1)  # [B] 只用來計分

            # 因為 batch=1，取 item
            pred = float(pred_count.item())
            gt = float(gt_count.item())

            abs_err = abs(pred - gt)
            sq_err = (pred - gt) ** 2
            total_abs_err += abs_err
            total_sq_err += sq_err
            n_samples += 1

            name = metas[0].get('image_path', str(metas[0])) if isinstance(metas[0], dict) else str(metas[0])
            print(f"{os.path.basename(name)} | pred={pred:.2f}, gt={gt:.0f}, err={abs_err:.2f}")

    # === Summary ===
    if n_samples == 0:
        print("[WARN] No samples in loader.")
    else:
        MAE = total_abs_err / n_samples
        MSE = total_sq_err / n_samples
        RMSE = math.sqrt(MSE)
        print("\n=== Test Summary (DDIM, noise→denoise) ===")
        print(f"Samples: {n_samples}")
        print(f"MAE  : {MAE:.4f}")
        print(f"MSE  : {MSE:.4f}")
        print(f"RMSE : {RMSE:.4f}")
