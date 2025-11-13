# diffusion_utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi
from scipy.optimize import linear_sum_assignment
import numpy as np
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




def m11_to_pixels(p, H, W):
    x = (p[...,0] + 1) * 0.5 * (W - 1)
    y = (p[...,1] + 1) * 0.5 * (H - 1)
    return torch.stack([x, y], dim=-1)

class hungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1.0, cost_coord: float = 1.0,
                 alpha: float = 0.25, gamma: float = 2.0, large_cost: float = 1e6):
        super().__init__()
        self.cost_class = float(cost_class)
        self.cost_coord = float(cost_coord)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.large = float(large_cost)

    @torch.no_grad()
    def forward(self, pred_logits, pred_points, gt_points, gt_mask):
        """
        pred_logits: [B, N]   (未經 sigmoid)
        pred_points: [B, N, 2]
        gt_points:   [B, N, 2]  # 與 mask 對齊，僅 mask=True 的位置有效
        gt_mask:     [B, N]     # True 表示該槽位有 GT
        return: list[ (pred_idx, gt_idx) ]，每張圖一個 tuple；索引皆相對於 N（全域槽位）
        """
        B, N, _ = pred_points.shape
        device = pred_logits.device
        indices = []

        for b in range(B):
            # 取第 b 張圖的資料
            x = pred_logits[b]            # [N]
            out_pts = pred_points[b]      # [N,2]
            mask_b = gt_mask[b]           # [N]
            tgt_idx_full = mask_b.nonzero(as_tuple=False).squeeze(1)  # [Mb]
            Mb = tgt_idx_full.numel()

            if Mb == 0 or N == 0:
                indices.append((
                    torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, dtype=torch.long, device=device)
                ))
                continue

            tgt_pts = gt_points[b, tgt_idx_full]  # [Mb,2]

            # ---------- 分類成本（只用正類 focal，logits 版，數值穩定） ----------
            # BCE_pos(x) = softplus(-x),  p = sigmoid(x)
            p = x.sigmoid()  # [N]
            pos_cost = self.alpha * ((1.0 - p).pow(self.gamma)) * F.softplus(-x)  # [N]
            cost_class = pos_cost[:, None]  # [N,1]，對每個 GT 一樣 -> broadcast 成 [N,Mb]

            # ---------- 幾何成本（L1 距離；可改成 L2 或 Huber） ----------
            # L1
            cost_coord = torch.cdist(out_pts, tgt_pts, p=1)  # [N,Mb]
            # 若想用 L2：torch.cdist(..., p=2)
            # 或 Huber：
            # beta = 0.2
            # dist = torch.cdist(out_pts, tgt_pts, p=2)
            # cost_coord = torch.where(dist < beta, 0.5*(dist**2)/beta, dist - 0.5*beta)

            # ---------- 總成本 ----------
            C = self.cost_class * cost_class + self.cost_coord * cost_coord  # [N,Mb]

            # ---------- 匹配（兜底處理，避免 NaN/Inf） ----------
            C_np = C.detach().float().cpu().numpy()
            C_np = np.nan_to_num(C_np, nan=self.large, posinf=self.large, neginf=self.large)

            if C_np.size == 0:
                indices.append((
                    torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, dtype=torch.long, device=device)
                ))
                continue

            row, col = linear_sum_assignment(C_np)  # row: N side index, col: Mb side index

            # 轉回全域 N 級索引（pred 直接用 row，gt 需要映射回原始槽位索引）
            pred_idx = torch.as_tensor(row, dtype=torch.long, device=device)
            gt_idx   = tgt_idx_full[torch.as_tensor(col, dtype=torch.long, device=device)]

            indices.append((pred_idx, gt_idx))

        return indices

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


class setCriterion(nn.Module):
    def __init__(self, matcher, lambda_exist=1.0, lambda_x0=1.0, lambda_cnt=0.1, gamma=2.0, alpha=0.9):
        super().__init__()
        self.matcher = matcher
        self.lambda_exist = lambda_exist
        self.lambda_x0 = lambda_x0
        self.lambda_cnt = lambda_cnt
        self.gamma = gamma
        self.alpha = alpha
    def focal_loss_with_logits(self, logits, targets, reduction="mean"):
        x = logits
        y = targets
        ce = torch.clamp(x, min=0) - x * y + torch.log1p(torch.exp(-x.abs()))

        # pt = p  (y=1)；pt = 1-p (y=0)
        p  = torch.sigmoid(x)
        pt = torch.where(y == 1, p, 1 - p).clamp_(1e-6, 1 - 1e-6)

        # alpha_t
        alpha_t = torch.where(y == 1, x.new_tensor(self.alpha), x.new_tensor(1 - self.alpha))

        # Focal
        loss = alpha_t * (1 - pt).pow(self.gamma) * ce

        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:
            return loss

# --- 原本的 loss_exist_eps_balanced 保留 ---

    def forward(
        self, p_t, p0, mask, abar_t, eps_pred, exist_logit
    ):
        """
        改良版 Loss:
          - L_exist: BCE 在存在 mask 上
          - L_x0   : x0_hat vs GT points
          - L_cnt  : soft count vs GT count
        """
        sqrt_ab = (abar_t + 1e-6).sqrt()
        sqrt_om = (1.0 - abar_t).clamp_min(0).sqrt()
        while sqrt_ab.ndim < p_t.ndim:
            sqrt_ab = sqrt_ab.unsqueeze(-1)
            sqrt_om = sqrt_om.unsqueeze(-1)
        # 反推出 x0_hat
        x0_hat = (p_t - sqrt_om * eps_pred) / sqrt_ab
        x0_hat = x0_hat.clamp(-1+1e-3, 1-1e-3)
        # --- 2. 進行匹配 ---
        # 注意：匹配時使用 detach() 的 x0_hat，避免 L_exist 的梯度影響 L_x0
        indices = self.matcher(exist_logit, x0_hat.detach(), p0, mask)
        idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices,mask)
        matched_pred_pts = x0_hat[idx]
        matched_gt_pts = p0[tgt_idx]

        matched_pred_pts_pix = m11_to_pixels(matched_pred_pts, 256, 256)
        matched_gt_pts_pix = m11_to_pixels(matched_gt_pts, 256, 256)
        if matched_pred_pts.shape[0] > 0:
            L_x0 = F.smooth_l1_loss(matched_pred_pts_pix, matched_gt_pts_pix)
        else:
            L_x0 = torch.tensor(0.0, device=x0_hat.device)

        # --- 4. 計算存在度損失 L_exist (對所有 N 個點計算) ---
        target_classes = torch.zeros_like(exist_logit)  # [B, N], 默認都是背景 (0)
        target_classes[idx] = 1.0  # 將匹配上的預測點的目標設為前景 (1)
        L_exist = self.focal_loss_with_logits(exist_logit, target_classes)

        # --- 5. (可選) 計算數量損失 L_cnt ---
        pred_cnt = torch.sigmoid(exist_logit).sum(dim=1)
        gt_cnt = mask.sum(dim=1).float()
        L_cnt = F.l1_loss(pred_cnt, gt_cnt)

        # --- 總損失 ---
        loss = self.lambda_x0 * L_x0 + self.lambda_exist * L_exist + self.lambda_cnt * L_cnt

        # 為了 log，返回各分項損失
        return loss, L_exist, L_x0, L_cnt

    def _get_src_permutation_idx(self, indices):
        # 獲取所有 batch 中被匹配上的 prediction 的索引
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices, mask):
        # 獲取所有 batch 中被匹配上的 ground truth 的索引
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        # 注意：gt 的索引是相對於 mask 之後的，要轉換回相對於 N
        # 但由於我們的 p0 也是 [B, N, 2]，可以直接用 mask 找到原始索引
        # 這裡為了簡化，假設 p0 已經處理好，tgt_idx 直接對應 p0
        # 如果 p0 是打包的，需要更複雜的索引方式
        # 假設 gt_mask 和 p0 的順序是一致的，我們可以這樣還原
        original_tgt_idx = []
        for i, (_, tgt) in enumerate(indices):
            if len(tgt) > 0:
                true_indices = mask[i].nonzero().squeeze(1)
                # ✅ 增加保護，防止 true_indices 為空但 tgt 不為空
                if len(true_indices) > 0:
                    original_tgt_idx.append(true_indices[tgt])
            # 找到第 i 個 batch 的有效 gt 點的原始索引
        if not original_tgt_idx:
            return batch_idx, torch.tensor([], dtype=torch.long, device=batch_idx.device)

        final_tgt_idx = torch.cat(original_tgt_idx)
        return batch_idx, final_tgt_idx
