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
def pixels_to_m11(points_xy, H, W):
    x = 2.0 * points_xy[..., 0] / max(W - 1, 1) - 1.0
    y = 2.0 * points_xy[..., 1] / max(H - 1, 1) - 1.0
    return torch.stack([x, y], dim=-1)

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

    if abar_t is not None:
        w_t = (1.0 - abar_t.squeeze(-1)).clamp_min(1e-8)  # [B,N]
        l2 = l2 * w_t

    pos = mask.float()
    neg = 1.0 - pos
    w_pos = 1.0  # 正樣本
    w_neg = 0.1  # 背景給一點點力道，不是 0

    w = w_pos * pos + w_neg * neg  # [B,N]
    l2 = l2 * w

    per_img = l2.sum(dim=1) / (w.sum(dim=1) + 1e-6)
    L_eps = per_img.mean()

    # Step 5: batch 平均
    return L_eps
    # l2 = (eps_pred - eps_true) ** 2                  # [B,N,2]
    # l2 = l2.sum(dim=-1)                              # [B,N]
    # if abar_t is not None:
    #     w = (1.0 - abar_t.squeeze(-1)).clamp(1e-6, 1.0)   # [B,N]
    #     l2 = l2 * w
    # if mask is not None:
    #     l2 = l2[mask]
    # return l2.mean() if l2.numel() > 0 else torch.tensor(0.0, device=eps_pred.device)


class setCriterion(nn.Module):
    def __init__(
        self,
        matcher,
        lambda_exist: float = 1.0,
        lambda_x0: float = 1.0,
        lambda_cnt: float = 0.1,
        lambda_bg: float = 2.0,
        gamma: float = 2.0,
        alpha: float = 0.6,
    ):
        """
        matcher      : 匈牙利匹配模組
        lambda_exist : L_exist 權重
        lambda_x0    : L_x0    權重
        lambda_cnt   : L_cnt   權重
        lambda_bg    : L_bg    權重
        gamma, alpha : focal loss 超參數
        """
        super().__init__()
        self.matcher = matcher
        self.lambda_exist = lambda_exist
        self.lambda_x0 = lambda_x0
        self.lambda_cnt = lambda_cnt
        self.lambda_bg = lambda_bg
        self.gamma = gamma
        self.alpha = alpha

    # ---------------- Focal loss (跟你原本一樣) ----------------
    def focal_loss_with_logits(self, logits, targets, reduction="mean"):
        """
        logits : [B,N]
        targets: [B,N] in {0,1}
        """
        x = logits
        y = targets

        # binary cross entropy with logits：ce = - [y log p + (1-y) log(1-p)]
        ce = torch.clamp(x, min=0) - x * y + torch.log1p(torch.exp(-x.abs()))

        # p = sigmoid(x)
        p = torch.sigmoid(x)
        pt = torch.where(y == 1, p, 1 - p).clamp_(1e-6, 1 - 1e-6)

        # α_t
        alpha_t = torch.where(y == 1, x.new_tensor(self.alpha), x.new_tensor(1 - self.alpha))

        # focal loss
        loss = alpha_t * (1 - pt).pow(self.gamma) * ce

        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:
            return loss

    # ---------------- Hybrid loss 主體 ----------------
    def forward(
        self,
        p_t: torch.Tensor,        # [B,N,2] 当前 noisy points
        p0: torch.Tensor,         # [B,N,2] GT points (在 [-1,1] 座標)
        mask: torch.Tensor,       # [B,N]   True/1 = 前景 GT
        abar_t: torch.Tensor,     # [B,1,1] or [B,1] \bar{α}_t
        eps_pred: torch.Tensor,   # [B,N,2] 模型預測的 ε_t
        exist_logit: torch.Tensor,# [B,N]   存在度 logits
        lambda_t: torch.Tensor = None,  # scalar 或 [B,1,1]，論文 SNR-based 權重
            aux_weight: float = 1.0,
    ):
        """
        回傳:
          loss  : 總 loss（包含 λ_t * L_eps）
          L_exist, L_x0, L_cnt, L_bg  : 各項方便 log
        """
        eps = 1e-6

        # ---- 1. 先把 abar_t 變成和 p_t 同維度 ----
        # abar_t: [B,1] or [B,1,1] → [B,1,1] → [B,1,1,1...] 再往後 unsqueeze
        if abar_t.dim() == 2:  # [B,1] 的情況
            abar_t = abar_t.unsqueeze(-1)  # [B,1,1]
        sqrt_ab = (abar_t + eps).sqrt()              # [B,1,1]
        sqrt_om = (1.0 - abar_t).clamp_min(0).sqrt() # [B,1,1]
        while sqrt_ab.ndim < p_t.ndim:
            sqrt_ab = sqrt_ab.unsqueeze(-1)  # → [B,1,1,1] ... 對齊 [B,N,2]
            sqrt_om = sqrt_om.unsqueeze(-1)

        # ---- 2. ground-truth noise ε_true + L_eps ----
        eps_true = (p_t - sqrt_ab * p0) / (sqrt_om + eps)  # [B,N,2]
        maskb = mask.bool()

        if maskb.any():
            # per-point mse (sum over xy)
            per_pt = (eps_pred - eps_true).pow(2).sum(dim=-1)  # [B,N]

            # only GT slots
            per_pt = per_pt * maskb.float()  # [B,N]

            # per-image average (divide by #GT in that image)
            den = maskb.float().sum(dim=1).clamp_min(1.0)  # [B]
            per_img = per_pt.sum(dim=1) / den  # [B]

            # batch average
            L_eps = per_img.mean()
        else:
            L_eps = eps_pred.new_tensor(0.0)

        # ---- early return if aux_weight==0 ----
        if (aux_weight is not None) and (float(aux_weight) == 0.0):
            if lambda_t is None:
                loss = L_eps
            else:
                lambda_t_scalar = lambda_t.mean() if isinstance(lambda_t, torch.Tensor) else float(lambda_t)
                loss = lambda_t_scalar * L_eps

            zero = eps_pred.new_tensor(0.0)
            return loss, zero, zero, zero, zero, L_eps
        # =========================================================
        # ---- 3. 反推出 x0_hat（給 x0 loss / 匹配用）----
        x0_hat = (p_t - sqrt_om * eps_pred) / (sqrt_ab + eps)
        x0_hat = x0_hat.clamp(-1 + 1e-3, 1 - 1e-3)


        # ---- 4. 匹配 (Hungarian) ----
        # 注意：使用 detach() 的 x0_hat 去做匹配，避免存在度梯度影響 L_x0
        indices = self.matcher(exist_logit, x0_hat.detach(), p0, mask)
        idx = self._get_src_permutation_idx(indices)           # (batch_idx, src_idx)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        matched_pred_pts = x0_hat[idx]  # [M,2]
        matched_gt_pts   = p0[tgt_idx]  # [M,2]

        matched_pred_pts_pix = m11_to_pixels(matched_pred_pts, 256, 256)
        matched_gt_pts_pix   = m11_to_pixels(matched_gt_pts, 256, 256)

        # ---- 5.L_x0：依「每張圖人數」加權的定位 loss ----
        # ----不做配對-----
        # eps = 1e-6
        #
        # # 先把所有 slot 的座標都轉成像素座標
        # pred_pts_pix = m11_to_pixels(x0_hat, 256, 256)  # [B,N,2]
        # gt_pts_pix   = m11_to_pixels(p0,     256, 256)  # [B,N,2]
        #
        # gt_counts = mask.sum(dim=1).float()  # [B]
        #
        # # 計算每張圖的人數權重（跟你原本一樣）
        # valid = gt_counts > 0
        # if valid.any():
        #     median_cnt = gt_counts[valid].median()
        # else:
        #     median_cnt = gt_counts.new_tensor(1.0)
        #
        # w_raw = (gt_counts + 1.0) / (median_cnt + 1.0)
        # w_img = torch.sqrt(w_raw).clamp(0.4, 4.0)  # 人多的圖權重較大
        #
        # per_img_losses = []
        # per_img_weights = []
        # B = mask.size(0)
        #
        # for b in range(B):
        #     nb = int(gt_counts[b].item())
        #     if nb > 0:
        #         # 這裡假設 p0 / x0_hat 裡「mask=True 的那些 slot」就是該圖的 GT slot
        #         sel = mask[b].bool()          # [N]
        #         pred_b = pred_pts_pix[b, sel] # [nb,2]
        #         gt_b   = gt_pts_pix[b, sel]   # [nb,2]
        #
        #         Lx0_b = F.smooth_l1_loss(pred_b, gt_b, reduction='mean')
        #         per_img_losses.append(Lx0_b)
        #         per_img_weights.append(w_img[b])
        #
        # if per_img_losses:
        #     per_img_losses  = torch.stack(per_img_losses)   # [B_eff]
        #     per_img_weights = torch.stack(per_img_weights)  # [B_eff]
        #     L_x0 = (per_img_losses * per_img_weights).sum() / (per_img_weights.sum() + eps)
        # else:
        #     L_x0 = torch.tensor(0.0, device=x0_hat.device)
        # ----做配對-----
        if matched_pred_pts_pix.shape[0] > 0:
            batch_idx_pred, _ = idx
            batch_idx_tgt, _  = tgt_idx
            # 一般情況下兩者應該相同
            # assert torch.equal(batch_idx_pred, batch_idx_tgt)

            gt_counts = mask.sum(dim=1).float()  # [B]

            valid = gt_counts > 0
            if valid.any():
                median_cnt = gt_counts[valid].median()
            else:
                median_cnt = gt_counts.new_tensor(1.0)

            eps = 1e-6
            w_raw = (gt_counts + 1.0) / (median_cnt + 1.0)  # normalize by median
            w_img = torch.sqrt(w_raw).clamp(0.4,4)  # 人多圖 → 權重大

            per_img_losses = []
            per_img_weights = []
            B = mask.size(0)
            for b in range(B):
                sel = (batch_idx_pred == b)
                if sel.any():
                    pred_b = matched_pred_pts_pix[sel]   # [Mb,2]
                    gt_b   = matched_gt_pts_pix[sel]     # [Mb,2]
                    Lx0_b  = F.smooth_l1_loss(pred_b, gt_b, reduction='mean')
                    per_img_losses.append(Lx0_b)
                    per_img_weights.append(w_img[b])

            if per_img_losses:
                per_img_losses  = torch.stack(per_img_losses)   # [B_eff]
                per_img_weights = torch.stack(per_img_weights)  # [B_eff]
                L_x0 = (per_img_losses * per_img_weights).sum() / (per_img_weights.sum() + eps)
            else:
                L_x0 = torch.tensor(0.0, device=x0_hat.device)
        else:
            L_x0 = torch.tensor(0.0, device=x0_hat.device)

        # ---- 5. L_exist (focal loss on all N) ----
        target_classes = torch.zeros_like(exist_logit)  # [B,N], 默認背景 0
        target_classes[idx] = 1.0                      # 被匹配到 GT 的預測 → 正類 1

        if not torch.isfinite(exist_logit).all():
            print("[WARN] logits not finite before focal_loss")

        L_exist = self.focal_loss_with_logits(exist_logit, target_classes)

        # ---- 6. L_cnt (soft count) ----
        prob_v = torch.sigmoid(exist_logit)    # [B,N]
        pos_mask_pred = (target_classes > 0.5).float()  # [B,N] 1= matched prediction
        pred_cnt = (prob_v * pos_mask_pred).sum(dim=1)  # [B]
        gt_cnt   = mask.sum(dim=1).float()              # [B]
        L_cnt = F.l1_loss(pred_cnt, gt_cnt)

        # ---- 7. 背景損失 L_bg ----
        pos_mask_bool = (target_classes > 0.5)  # [B,N] True = matched preds
        bgmask = (~pos_mask_bool).float()       # [B,N] True = 背景預測點

        bg_ratio = bgmask.sum(1) / (bgmask.size(1) + eps)        # [B]
        bg_mean  = (prob_v * bgmask).sum(1) / (bgmask.sum(1) + eps)  # [B]
        bg_loss_per_img = bg_mean * bg_ratio
        L_bg = bg_loss_per_img.mean()

        # ---- 8. Hybrid Loss：λ_t * L_eps + 其他項 ----
        if lambda_t is None:
            # 沒有給 λ_t 就退化成原本多項 loss
            loss = (
                self.lambda_x0 * L_x0
                + self.lambda_exist * L_exist
                + self.lambda_cnt * L_cnt
                + self.lambda_bg * L_bg
            )
            if not torch.isfinite(loss):
                print("[ERROR] Non-finite loss detected:",
                      "Lex=", float(L_exist),
                      "Lx0=", float(L_x0),
                      "Lcnt=", float(L_cnt),
                      "Lbg=", float(L_bg),
                      )
                raise RuntimeError("Loss became NaN/inf, stop and inspect.")
            return loss, L_exist, L_x0, L_cnt, L_bg
        else:
            # lambda_t 可能是 scalar，也可能是 [B,1,1] → 取平均
            if isinstance(lambda_t, torch.Tensor):
                lambda_t_scalar = lambda_t.mean()
            else:
                lambda_t_scalar = float(lambda_t)

            if lambda_t is None:
                # 這是舊的 fallback，也可以乘上 aux_weight
                loss = aux_weight * (
                        self.lambda_x0 * L_x0
                        + self.lambda_exist * L_exist
                        + self.lambda_cnt * L_cnt
                        + self.lambda_bg * L_bg
                )
            else:
                # lambda_t 處理 L_eps (已經算好了)
                if isinstance(lambda_t, torch.Tensor):
                    lambda_t_scalar = lambda_t.mean()
                else:
                    lambda_t_scalar = float(lambda_t)

                # [MOD] 這裡加上 aux_weight
                loss = (
                        lambda_t_scalar * L_eps
                        + aux_weight * (  # <--- 讓所有輔助 Loss 隨 t 衰減
                                self.lambda_x0 * L_x0
                                + self.lambda_exist * L_exist
                                + self.lambda_cnt * L_cnt
                                + self.lambda_bg * L_bg
                        )
                )
            return loss, L_exist, L_x0, L_cnt, L_bg, L_eps






    # ----------------- 匹配索引工具 -----------------
    def _get_src_permutation_idx(self, indices):
        # 獲取所有 batch 中被匹配上的 prediction 的索引
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

