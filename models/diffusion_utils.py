# diffusion_utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi

# --- Cosine schedule (Nichol & Dhariwal), 生成 ᾱ[t] 由 1 -> 近 0，t 越大越「更嘈雜」 ---
class CosineAbarSchedule:
    def __init__(self, T:int, s:float=0.008, device="cuda"):
        self.T = T
        t = torch.linspace(0, 1, T, device=device)
        f = torch.cos(((t + s) / (1 + s)) * pi / 2) ** 2
        self.abar = (f / f[0]).clamp(1e-6, 1.0)    # [T], 單調遞減

    def get(self, t_idx: torch.Tensor):
        # t_idx: [B,1] 或 [B,N] (long)，回傳 ᾱ[t] 並自動 broadcast 維度
        return self.abar[t_idx]

# --- 像素座標 -> [-1,1] 正規化（align_corners=False 對應公式: x~ = 2x/W - 1） ---
def pixels_to_m11(points_xy: torch.Tensor, H: int, W: int, scale: float=1.0):
    # points_xy: [B,N,2], 回傳 [B,N,2] in [-1,1]*scale
    x = (2.0 * points_xy[..., 0] / W) - 1.0
    y = (2.0 * points_xy[..., 1] / H) - 1.0
    return torch.stack([x, y], dim=-1) * scale

# --- 前向擴散：p_t = sqrt(ᾱ_t)*p0 + sqrt(1-ᾱ_t)*ε ；回傳 p_t, ε, ᾱ_t ---
def forward_noisy(p0_m11: torch.Tensor, t_idx: torch.Tensor, sched: CosineAbarSchedule):
    # p0_m11: [B,N,2] in [-1,1]; t_idx: [B,1] or [B,N] (long)
    abar_t = sched.get(t_idx).unsqueeze(-1)          # [B,1,1] 或 [B,N,1]
    eps = torch.randn_like(p0_m11)                   # [B,N,2]
    p_t = torch.sqrt(abar_t) * p0_m11 + torch.sqrt(1.0 - abar_t) * eps
    return p_t, eps, abar_t                          

# --- 由 ε̂ 反推 p0（sampling / 輔助監督可用）---
def estimate_p0(p_t: torch.Tensor, eps_pred: torch.Tensor, abar_t: torch.Tensor):
    # p0_hat = (p_t - sqrt(1-ᾱ_t)*ε̂) / sqrt(ᾱ_t)
    return (p_t - torch.sqrt(1.0 - abar_t) * eps_pred) / (torch.sqrt(abar_t) + 1e-6)

# --- ε 損失（遮住 padding，並可用 w(t) 權重，預設 w(t)=1-ᾱ_t -> 噪越大越重） ---
def eps_loss(eps_pred: torch.Tensor, eps_true: torch.Tensor, mask: torch.Tensor, abar_t: torch.Tensor=None):
    # eps_*: [B,N,2], mask: [B,N] bool, abar_t: [B,1,1] 或 [B,N,1]
    l2 = (eps_pred - eps_true).pow(2).sum(-1)  # [B,N]

    # Step 2: 加權 (例如 SNR 權重)
    if abar_t is not None:
        w = (1.0 - abar_t.squeeze(-1)).clamp_min(1e-8)  # [B,N]
        l2 = l2 * w

    # Step 3: 只保留正樣本
    l2 = l2 * mask.float()

    # Step 4: 每張圖的「平均誤差」
    num_pos = mask.sum(dim=1).clamp(min=1)  # [B]，避免除零
    per_img = l2.sum(dim=1) / num_pos  # [B]

    # Step 5: batch 平均
    return per_img.mean()
    # l2 = (eps_pred - eps_true) ** 2                  # [B,N,2]
    # l2 = l2.sum(dim=-1)                              # [B,N]
    # if abar_t is not None:
    #     w = (1.0 - abar_t.squeeze(-1)).clamp(1e-6, 1.0)   # [B,N]
    #     l2 = l2 * w
    # if mask is not None:
    #     l2 = l2[mask]
    # return l2.mean() if l2.numel() > 0 else torch.tensor(0.0, device=eps_pred.device)



def count_primary_loss(
    eps_pred: torch.Tensor,       # [B,N,2]
    eps_true: torch.Tensor,       # [B,N,2]
    exist_logit: torch.Tensor,    # [B,N]  (raw logits，這裡不要先 sigmoid)
    mask: torch.Tensor,           # [B,N]  (bool，True=GT有人)
    abar_t: torch.Tensor = None,  # [B,1,1] or [B,N,1] (optional)
    lambda_count: float = 1.0,    # 主：影像級人數 loss 權重
    lambda_exist: float = 0.5,    # 次：逐點存在 BCE 權重
    lambda_eps: float = 0.2,      # 輔：ε（位置）權重
    exist_pos_boost: float = 1.0, # 正樣本增益（類別不平衡時 >1）
    gate_eps_by_exist: bool = True,  # 用存在機率對 ε loss 加權
    use_smoothl1_count: bool = True, # 人數 loss 用 SmoothL1 或 MSE
    normalize_count: bool = True
):
    B, N, D = eps_pred.shape
    assert D == 2 and eps_true.shape == eps_pred.shape
    assert exist_logit.shape == (B, N)
    assert mask.shape == (B, N) and mask.dtype == torch.bool

    # ---------- 1) 影像級人數（主監督） ----------
    # 預測人數 = sum(sigmoid(logit))
    exist_prob = torch.sigmoid(exist_logit)      # [B,N]
    pred_count = exist_prob.sum(dim=1)           # [B]
    gt_count   = mask.sum(dim=1).to(pred_count.dtype)  # [B]
    #print(pred_count,gt_count)
    if normalize_count:
        denom = torch.tensor(float(N), device=pred_count.device, dtype=pred_count.dtype)
        pc = pred_count / denom
        gc = gt_count   / denom
    else:
        pc, gc = pred_count, gt_count

    if use_smoothl1_count:
        L_count = F.smooth_l1_loss(pred_count, gt_count)
    else:
        L_count = F.mse_loss(pred_count, gt_count)

    # ---------- 2) 逐點存在 BCE（次監督，穩定機率校準） ----------
    # 動態 pos_weight：每張圖的 neg/pos（避免嚴重不平衡）
    pos = mask.sum(dim=1).clamp(min=1)                  # [B]
    neg = (~mask).sum(dim=1).clamp(min=1)               # [B]
    dyn_pos_weight = (neg / pos).detach() * exist_pos_boost  # [B]
    pos_w = dyn_pos_weight.unsqueeze(1).expand_as(exist_logit)  # [B,N]

    bce = F.binary_cross_entropy_with_logits(
        exist_logit, mask.float(), pos_weight=pos_w, reduction='none'
    )  # [B,N]
    L_exist = bce.mean()

    # ---------- 3) ε（位置）輔助 ----------
    L2 = (eps_pred - eps_true).pow(2).sum(dim=-1)       # [B,N]
    if abar_t is not None:
        # 展成 [B,N,1] -> [B,N]
        if abar_t.dim() == 3 and abar_t.size(1) == 1:
            abar_t = abar_t.expand(B, N, 1)
        abar = abar_t.squeeze(-1).clamp(1e-8, 1-1e-8)   # [B,N]
        # 簡單：用 (1-ᾱ) 權重；也可改 snr = ᾱ/(1-ᾱ)
        L2 = L2 * (1.0 - abar)

    if gate_eps_by_exist:
        # 以存在機率進一步 gate（避免對不存在點學位置）
        L2 = L2 * exist_prob

    L2 = L2 * mask.float()
    num_pos = mask.sum(dim=1).clamp(min=1)              # [B]
    L_eps = (L2.sum(dim=1) / num_pos).mean()            # scalar

    # ---------- 4) 總損失（以 Count 為主） ----------
    loss = lambda_count * L_count + lambda_exist * L_exist + lambda_eps * L_eps

    stats = {
        "loss": loss.detach(),
        "L_count": L_count.detach(),
        "L_exist": L_exist.detach(),
        "L_eps": L_eps.detach(),
        "pred_count_mean": pred_count.mean().detach(),
        "gt_count_mean": gt_count.mean().detach(),
    }
    return loss, stats
import torch
import torch.nn.functional as F

@torch.no_grad()
def _make_neg_keep_mask(mask: torch.Tensor, neg_pos_ratio: float) -> torch.Tensor:
    """
    針對 BCE 部分做負樣本下採樣的二元遮罩：
    - 讓每張圖的 neg:pos ≈ neg_pos_ratio:1
    - 只影響 BCE；L2 本來就只在正樣本上
    """
    B, N = mask.shape
    pos_cnt = mask.sum(dim=1)                      # [B]
    neg_cnt = (~mask).sum(dim=1)                   # [B]
    # 目標保留的負樣本數
    tgt_neg = (pos_cnt.to(torch.float32) * neg_pos_ratio).clamp(max=neg_cnt.to(torch.float32))
    # 保留機率（逐圖）
    p_keep = (tgt_neg / neg_cnt.clamp_min(1)).unsqueeze(1)   # [B,1]
    # 對每個負點做伯努利採樣
    keep_prob = p_keep.expand(B, N)
    rand = torch.rand_like(keep_prob)
    neg_keep = (rand < keep_prob) & (~mask)        # [B,N] 只保留負點的一部分
    # 正樣本一律保留
    pos_keep = mask
    keep = pos_keep | neg_keep                     # [B,N]
    # 避免整張圖全被濾掉（極端情況）
    empty_row = keep.sum(dim=1) == 0
    if empty_row.any():
        # 將第一個位置強制保留（不太會影響分布，只是避雷）
        keep[empty_row, 0] = True
    return keep

def loss_exist_eps_balanced(
    eps_pred: torch.Tensor,       # [B,N,2]
    eps_true: torch.Tensor,       # [B,N,2]
    exist_logit: torch.Tensor,    # [B,N] (raw logits)
    mask: torch.Tensor,           # [B,N] (bool)
    abar_t: torch.Tensor = None,  # [B,1,1] or [B,N,1]
    # ---- BCE / focal ----
    use_focal: bool = True,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.75,        # 常數 alpha；若想用動態 alpha 設為 None
    use_pos_weight: bool = True,      # 若 focal_alpha=None，會把 pos_weight -> alpha
    pos_weight_cap: float = 10.0,
    # ---- 負樣本下採樣 ----
    use_neg_subsample: bool = True,
    neg_pos_ratio: float = 3.0,
    # ---- L2（位置） ----
    lambda_exist: float = 1.0,
    lambda_eps: float = 0.2,
    gate_eps_by_exist: bool = True,
    gate_detach: bool = True,
    gate_floor: float = 0.1,
    use_snr_weight: bool = False,
):
    B, N, D = eps_pred.shape
    assert D == 2 and eps_true.shape == eps_pred.shape
    assert exist_logit.shape == (B, N)
    assert mask.shape == (B, N) and mask.dtype == torch.bool

    device = exist_logit.device
    dtype  = exist_logit.dtype

    # 下採樣遮罩（只影響 BCE）
    if use_neg_subsample:
        keep = _make_neg_keep_mask(mask, neg_pos_ratio).to(device)
    else:
        keep = torch.ones_like(mask, dtype=torch.bool, device=device)

    # 動態 pos_weight（逐圖），上限避免爆炸
    if use_pos_weight:
        pos_cnt = mask.sum(dim=1).clamp(min=1)
        neg_cnt = (~mask).sum(dim=1).clamp(min=1)
        pos_w_per_img = (neg_cnt / pos_cnt).to(device=device, dtype=dtype)
        pos_w_per_img = pos_w_per_img.clamp(max=pos_weight_cap).detach()
        pos_w = pos_w_per_img.unsqueeze(1).expand(B, N)  # [B,N]
    else:
        pos_w = torch.ones((B, N), device=device, dtype=dtype)

    # ----- BCE / Focal BCE -----
    if not use_focal:
        bce = F.binary_cross_entropy_with_logits(
            exist_logit, mask.float(), pos_weight=pos_w, reduction='none'
        )  # [B,N]
        bce = bce * keep.float()
        denom = keep.float().sum(dim=1).clamp_min(1.0)        # [B]
        L_exist = (bce.sum(dim=1) / denom).mean()
    else:
        p  = torch.sigmoid(exist_logit)                       # [B,N]
        ce = F.binary_cross_entropy_with_logits(
            exist_logit, mask.float(), reduction='none'
        )  # [B,N] 基礎 CE（穩定）
        pt = torch.where(mask, p, 1 - p)                      # [B,N]
        if focal_alpha is None and use_pos_weight:
            alpha = (pos_w / (pos_w + 1.0)).clamp(0.05, 0.95)
        else:
            alpha = torch.full_like(p, float(focal_alpha))
        foc = alpha * (1.0 - pt).pow(float(focal_gamma)) * ce # [B,N]
        foc = foc * keep.float()
        denom = keep.float().sum(dim=1).clamp_min(1.0)
        L_exist = (foc.sum(dim=1) / denom).mean()

    # ----- 位置 L2 -----
    L2 = (eps_pred - eps_true).pow(2).sum(dim=-1)             # [B,N]
    if abar_t is not None:
        abar = abar_t
        if abar.dim() == 3 and abar.size(1) == 1:             # [B,1,1] -> [B,N,1]
            abar = abar.expand(B, N, 1)
        abar = abar.to(device=device, dtype=dtype).squeeze(-1).clamp(1e-8, 1-1e-8)
        if use_snr_weight:
            snr = (abar / (1.0 - abar).clamp_min(1e-8))
            L2 = L2 * snr
        else:
            L2 = L2 * (1.0 - abar)
    if gate_eps_by_exist:
        gate = torch.sigmoid(exist_logit)
        if gate_detach:
            gate = gate.detach()
        gate = gate.clamp_min(gate_floor)
        L2 = L2 * gate
    L2 = L2 * mask.float()
    num_pos = mask.sum(dim=1).clamp(min=1)
    L_eps = (L2.sum(dim=1) / num_pos).mean()

    loss = lambda_exist * L_exist + lambda_eps * L_eps

    with torch.no_grad():
        pred_count_soft = torch.sigmoid(exist_logit).sum(dim=1).mean()
        gt_count_mean   = mask.sum(dim=1).float().mean()
    stats = {
        "loss": loss.detach(),
        "L_exist": L_exist.detach(),
        "L_eps": L_eps.detach(),
        "pred_count_mean": pred_count_soft.detach(),
        "gt_count_mean": gt_count_mean.detach(),
    }
    return loss, stats

# --- 原本的 loss_exist_eps_balanced 保留 ---

def loss_exist_x0_count(
    p_t, p0, mask, abar_t, eps_pred, exist_logit,
    lambda_exist=1.0, lambda_x0=1.0, lambda_cnt=0.1
):
    """
    改良版 Loss:
      - L_exist: BCE 在存在 mask 上
      - L_x0   : x0_hat vs GT points
      - L_cnt  : soft count vs GT count
    """
    sqrt_ab = (abar_t + 1e-6).sqrt()
    sqrt_om = (1.0 - abar_t).clamp_min(0).sqrt()

    # 反推出 x0_hat
    x0_hat = (p_t - sqrt_om * eps_pred) / sqrt_ab
    x0_hat = x0_hat.clamp(-1+1e-3, 1-1e-3)

    # 1) 座標 L1 (只在有效點上算)
    L_x0 = F.smooth_l1_loss(x0_hat[mask], p0[mask])

    # 2) 存在度
    L_exist = F.binary_cross_entropy_with_logits(exist_logit, mask.float())

    # 3) 數量 loss
    pred_cnt = torch.sigmoid(exist_logit).sum(dim=1)
    gt_cnt   = mask.sum(dim=1).float()
    L_cnt    = F.l1_loss(pred_cnt, gt_cnt)

    loss = lambda_exist*L_exist + lambda_x0*L_x0 + lambda_cnt*L_cnt
    return loss, L_exist, L_x0, L_cnt, pred_cnt.mean().item(), gt_cnt.mean().item()
