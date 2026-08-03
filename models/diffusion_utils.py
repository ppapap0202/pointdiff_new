# diffusion_utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi
from scipy.optimize import linear_sum_assignment
import numpy as np
# --- Cosine schedule (Nichol & Dhariwal), 生成 ᾱ[t] 由 1 -> 近 0，t 越大越「更嘈雜」 ---
import torch
import torch.nn.functional as F


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
        pred_logits: [B, N] or [B, N, 2]   (未經 sigmoid/softmax)
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
            # ---------- 分類成本 ----------
            # 支援兩種模式：
            #   (A) BCE/Focal：pred_logits = [N]（舊版）
            #   (B) CE/Softmax：pred_logits = [N,2]，class0=no-object, class1=object（新版）
            if x.dim() == 2 and x.size(-1) == 2:
                p_obj = x.softmax(-1)[:, 1]  # [N]
                cost_class = (-p_obj.clamp_min(1e-8).log())[:, None].expand(-1, Mb)  # [N,Mb]
            else:
                p = x.sigmoid()  # [N]
                pos_cost = self.alpha * ((1.0 - p).pow(self.gamma)) * F.softplus(-x)  # [N]
                cost_class = pos_cost[:, None].expand(-1, Mb)  # [N,Mb]

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
            row_t = torch.as_tensor(row, dtype=torch.long, device=device)
            col_t = torch.as_tensor(col, dtype=torch.long, device=device)
            matched_cost_class = cost_class[row_t, col_t]
            matched_cost_coord = cost_coord[row_t, col_t]
            matched_C = C[row_t, col_t]

            class_term = self.cost_class * matched_cost_class.mean().item()
            coord_term = self.cost_coord * matched_cost_coord.mean().item()
            total_term = matched_C.mean().item()

            # print(
            #     f"[Matcher][img {b}] "
            #     f"pairs={len(row)} | "
            #     f"class_mean={matched_cost_class.mean().item():.6f} "
            #     f"coord_mean={matched_cost_coord.mean().item():.6f} | "
            #     f"class_term={class_term:.6f} "
            #     f"coord_term={coord_term:.6f} "
            #     f"total_mean={total_term:.6f}"
            # )
            # 轉回全域 N 級索引（pred 直接用 row，gt 需要映射回原始槽位索引）
            pred_idx = row_t
            gt_idx = tgt_idx_full[col_t]   # map local Mb-index -> global N-slot index
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

def coverage_loss_bag(
    pred_points_m11: torch.Tensor,   # [B,N,2]
    prob_obj: torch.Tensor,          # [B,N]
    gt_points_m11: torch.Tensor,     # [B,N,2]
    gt_mask: torch.Tensor,           # [B,N]
    pred_valid_mask: torch.Tensor = None,  # [B,N] or None
    H: int = 256,
    W: int = 256,
    topk: int = 5,
    sigma: float = 4.0,
    eps: float = 1e-6,
):
    """
    對每個 GT，看最近 top-k 個 prediction；
    只要其中至少一個是「高分且夠近」，這個 GT 的 coverage 就高。
    """
    B, N, _ = pred_points_m11.shape
    pred_pix = m11_to_pixels(pred_points_m11, H, W)   # [B,N,2]
    gt_pix   = m11_to_pixels(gt_points_m11, H, W)     # [B,N,2]

    losses = []

    for b in range(B):
        gmask = gt_mask[b].bool()
        if not gmask.any():
            continue

        G = gt_pix[b, gmask]   # [Ng,2]

        if pred_valid_mask is not None:
            pmask = pred_valid_mask[b].bool()
            P = pred_pix[b, pmask]     # [Np,2]
            q = prob_obj[b, pmask]      # [Np]
        else:
            P = pred_pix[b]            # [N,2]
            q = prob_obj[b]             # [N]

        if P.numel() == 0:
            # 沒 prediction 可覆蓋，直接給大 loss
            losses.append(G.new_tensor(10.0))
            continue

        # [Ng, Np]
        dist = torch.cdist(G, P, p=2)

        k = min(int(topk), P.size(0))
        dist_topk, topk_idx = torch.topk(dist, k=k, dim=1, largest=False)   # [Ng,k]
        q_topk = q[topk_idx].clamp(min=0.0, max=1.0 - 1e-6)                 # [Ng,k]

        # A GT is covered only by a proposal that is both close and confident.
        geo = torch.exp(-(dist_topk ** 2) / (2.0 * sigma * sigma))          # [Ng,k]
        a = q_topk * geo

        # Clamp closeness into a valid bag-probability term
        a = a.clamp(min=0.0, max=1.0 - 1e-6)

        # bag probability: 至少一個成功覆蓋
        p_bag = 1.0 - torch.prod(1.0 - a, dim=1)                            # [Ng]

        # GT-wise coverage loss
        Lb = -(p_bag + eps).log().mean()
        losses.append(Lb)

    if len(losses) == 0:
        return pred_points_m11.new_tensor(0.0)

    return torch.stack(losses).mean()


def coverage_radius_hinge_loss(
    pred_points_m11: torch.Tensor,
    gt_points_m11: torch.Tensor,
    gt_mask: torch.Tensor,
    pred_valid_mask: torch.Tensor = None,
    H: int = 256,
    W: int = 256,
    cover_radius: float = 6.0,
    hard_weight: float = 0.0,
    hard_cap: float = 1.0,
    dense_weight: float = 0.0,
    dense_radius: float = 16.0,
    dense_norm: float = 4.0,
    weight_cap: float = 6.0,
):
    """
    Directly optimizes the proposal-cover metric: a GT pays no penalty once its
    nearest proposal is inside cover_radius pixels.

    Optional hard/dense weighting focuses the gradient on GTs that are still
    uncovered, especially in dense regions where missed coverage causes severe
    under-counting.
    """
    pred_pix = m11_to_pixels(pred_points_m11, H, W)
    gt_pix = m11_to_pixels(gt_points_m11, H, W)
    radius = max(float(cover_radius), 1e-6)
    hard_weight = max(float(hard_weight), 0.0)
    hard_cap = max(float(hard_cap), 0.0)
    dense_weight = max(float(dense_weight), 0.0)
    dense_radius = max(float(dense_radius), 1e-6)
    dense_norm = max(float(dense_norm), 1e-6)
    weight_cap = max(float(weight_cap), 1.0)
    losses = []

    for b in range(pred_pix.size(0)):
        gmask = gt_mask[b].bool()
        if not gmask.any():
            continue

        G = gt_pix[b, gmask]
        if pred_valid_mask is not None:
            pmask = pred_valid_mask[b].bool()
            P = pred_pix[b, pmask]
        else:
            P = pred_pix[b]

        if P.numel() == 0:
            losses.append(G.new_tensor(10.0))
            continue

        dmin = torch.cdist(G, P, p=2).min(dim=1).values
        per_gt_loss = F.relu(dmin - radius).div(radius)
        weight = torch.ones_like(per_gt_loss)

        if hard_weight > 0:
            # Upweight GTs whose nearest proposal is still outside the cover
            # radius. Detach the factor so the loss gradient stays simple.
            hard_factor = per_gt_loss.detach().clamp(max=hard_cap)
            weight = weight * (1.0 + hard_weight * hard_factor)

        if dense_weight > 0 and G.size(0) > 1:
            gt_dist = torch.cdist(G, G, p=2)
            eye = torch.eye(G.size(0), dtype=torch.bool, device=G.device)
            dense_count = ((gt_dist <= dense_radius) & (~eye)).float().sum(dim=1)
            dense_factor = dense_count.div(dense_norm).clamp(max=1.0)
            weight = weight * (1.0 + dense_weight * dense_factor)

        weight = weight.clamp(max=weight_cap)
        losses.append((per_gt_loss * weight).sum() / weight.sum().clamp_min(1e-6))

    if len(losses) == 0:
        return pred_points_m11.new_tensor(0.0)

    return torch.stack(losses).mean()

def region_duplicate_loss(
    pred_points_m11: torch.Tensor,
    prob_obj: torch.Tensor,
    gt_points_m11: torch.Tensor,
    gt_mask: torch.Tensor,
    pred_valid_mask: torch.Tensor = None,
    H: int = 256,
    W: int = 256,
    region_radius: float = 5.0,
    region_topk: int = 5,
    dense_aware: bool = False,
    neighbor_radius: float = None,
    allow_extra: int = 0,
):
    pred_pix = m11_to_pixels(pred_points_m11, H, W)
    gt_pix = m11_to_pixels(gt_points_m11, H, W)
    losses = []

    for b in range(pred_pix.size(0)):
        gmask = gt_mask[b].bool()
        if not gmask.any():
            continue

        if pred_valid_mask is not None:
            pmask = pred_valid_mask[b].bool()
            P = pred_pix[b, pmask]
            q = prob_obj[b, pmask]
        else:
            P = pred_pix[b]
            q = prob_obj[b]

        if P.numel() == 0:
            continue

        G = gt_pix[b, gmask]
        dist = torch.cdist(G, P, p=2)
        keep_counts = None
        base_keep = max(1, 1 + int(allow_extra))

        if dense_aware:
            dense_r = float(region_radius if neighbor_radius is None else neighbor_radius)
            if G.size(0) > 1:
                gt_dist = torch.cdist(G, G, p=2)
                eye = torch.eye(G.size(0), dtype=torch.bool, device=G.device)
                neighbor_count = ((gt_dist <= dense_r) & (~eye)).long().sum(dim=1)
                keep_counts = (neighbor_count + base_keep).clamp_min(1)
            else:
                keep_counts = torch.ones((G.size(0),), dtype=torch.long, device=G.device) * base_keep

        for gi in range(G.size(0)):
            near_idx = (dist[gi] <= float(region_radius)).nonzero(as_tuple=False).squeeze(1)
            if near_idx.numel() <= 1:
                continue

            if region_topk is not None and int(region_topk) > 0 and near_idx.numel() > int(region_topk):
                _, order = torch.topk(dist[gi, near_idx], k=int(region_topk), largest=False)
                near_idx = near_idx[order]

            q_sorted = torch.sort(q[near_idx], descending=True).values
            keep_n = base_keep
            if keep_counts is not None:
                keep_n = int(keep_counts[gi].item())
            keep_n = max(1, min(keep_n, q_sorted.numel()))

            if q_sorted.numel() > keep_n:
                losses.append((q_sorted[keep_n:] ** 2).mean())

    if len(losses) == 0:
        return prob_obj.new_tensor(0.0)

    return torch.stack(losses).mean()


def duplicate_collapse_loss(
    pred_points_m11: torch.Tensor,
    gt_points_m11: torch.Tensor,
    gt_mask: torch.Tensor,
    matched_indices,
    H: int = 256,
    W: int = 256,
    inner_radius: float = 2.0,
    outer_radius: float = 4.0,
    far_weight: float = 4.0,
):
    """
    Pull near-GT unmatched proposals into the Hungarian matched representative.

    This intentionally encourages duplicates near the same GT to collapse into a
    tight mode, making small-radius NMS/DBSCAN easier. Only unmatched non-GT
    slots near a GT are affected, so far background proposals are left alone.
    """
    pred_pix = m11_to_pixels(pred_points_m11, H, W)
    gt_pix = m11_to_pixels(gt_points_m11, H, W)
    inner = max(float(inner_radius), 1e-6)
    outer = max(float(outer_radius), inner)
    far_weight = max(float(far_weight), 0.0)
    losses = []

    for b, (src_idx, tgt_idx) in enumerate(matched_indices):
        gmask = gt_mask[b].bool()
        if src_idx.numel() == 0 or not gmask.any():
            continue

        matched_mask = torch.zeros(
            (pred_points_m11.size(1),),
            dtype=torch.bool,
            device=pred_points_m11.device,
        )
        matched_mask[src_idx] = True

        # Only regular proposal slots are collapsed. GT teacher slots are kept
        # available as representatives but are not themselves pulled.
        cand_mask = (~matched_mask) & (~gmask)
        cand_idx = cand_mask.nonzero(as_tuple=False).squeeze(1)
        if cand_idx.numel() == 0:
            continue

        reps = pred_pix[b, src_idx]          # [M,2]
        gt_for_reps = gt_pix[b, tgt_idx]     # [M,2]
        cand = pred_pix[b, cand_idx]         # [C,2]

        dist_to_gt = torch.cdist(cand, gt_for_reps, p=2)
        nearest_dist, nearest_rep = dist_to_gt.min(dim=1)
        in_region = nearest_dist <= outer
        if not in_region.any():
            continue

        cand_region = cand[in_region]
        target = reps[nearest_rep[in_region]].detach()
        dist_to_rep = torch.norm(cand_region - target, dim=1)
        dist_to_gt_region = nearest_dist[in_region]

        # Inside the tight duplicate radius, directly attract to the matched
        # representative. Scaling by inner keeps the loss magnitude stable.
        near = dist_to_gt_region <= inner
        parts = []
        if near.any():
            parts.append(F.smooth_l1_loss(
                cand_region[near] / inner,
                target[near] / inner,
                reduction="mean",
            ))

        # Anything still farther than inner from its representative, but within
        # the duplicate neighborhood, receives a stronger squared hinge penalty.
        spread = F.relu(dist_to_rep - inner).div(inner)
        if spread.numel() > 0:
            parts.append(far_weight * spread.pow(2).mean())

        if parts:
            losses.append(torch.stack(parts).sum())

    if len(losses) == 0:
        return pred_points_m11.new_tensor(0.0)

    return torch.stack(losses).mean()


def build_region_representative_targets(
    pred_points_m11: torch.Tensor,
    gt_points_m11: torch.Tensor,
    gt_mask: torch.Tensor,
    pred_valid_mask: torch.Tensor = None,
    H: int = 256,
    W: int = 256,
    region_radius: float = 5.0,
    return_roles: bool = False,
):
    """
    Nearest-GT group target building:
    1. each prediction is assigned to its nearest GT
    2. predictions within region_radius join that GT group
    3. the closest prediction in each GT group is the binary positive
    4. other same-group predictions are duplicate negatives
    5. predictions outside region_radius are background negatives

    When return_roles=True, roles are 0=background, 1=positive, 2=duplicate.
    """
    B, N, _ = pred_points_m11.shape
    pred_pix = m11_to_pixels(pred_points_m11, H, W)
    gt_pix = m11_to_pixels(gt_points_m11, H, W)
    device = pred_points_m11.device

    target_classes = torch.zeros((B, N), dtype=torch.long, device=device)
    target_roles = torch.zeros((B, N), dtype=torch.long, device=device)
    assigned_gt = torch.full((B, N), -1, dtype=torch.long, device=device)
    nearest_dist = torch.full(
        (B, N),
        float("inf"),
        dtype=pred_points_m11.dtype,
        device=device,
    )
    matched_pred = []
    matched_gt = []

    for b in range(B):
        gmask = gt_mask[b].bool()
        if not gmask.any():
            continue

        if pred_valid_mask is not None:
            pmask = pred_valid_mask[b].bool()
        else:
            pmask = torch.ones((N,), dtype=torch.bool, device=device)

        valid_idx = pmask.nonzero(as_tuple=False).squeeze(1)
        if valid_idx.numel() == 0:
            continue

        P = pred_pix[b, valid_idx]  # [Np,2]
        gt_idx_full = gmask.nonzero(as_tuple=False).squeeze(1)
        G = gt_pix[b, gt_idx_full]  # [Ng,2]

        dist = torch.cdist(P, G, p=2)  # [Np,Ng]
        min_dist, nearest_gt_local = dist.min(dim=1)
        nearest_dist[b, valid_idx] = min_dist
        in_region = min_dist <= float(region_radius)
        if not in_region.any():
            continue

        assigned_pred_local = in_region.nonzero(as_tuple=False).squeeze(1)
        assigned_gt_local = nearest_gt_local[assigned_pred_local]
        assigned_pred_global = valid_idx[assigned_pred_local]
        assigned_gt_global = gt_idx_full[assigned_gt_local]
        target_roles[b, assigned_pred_global] = 2
        assigned_gt[b, assigned_pred_global] = assigned_gt_global

        for gi in range(G.size(0)):
            members = assigned_pred_local[assigned_gt_local == gi]
            if members.numel() == 0:
                continue

            member_d = dist[members, gi]
            best_member = members[member_d.argmin()]
            rep_global = valid_idx[best_member]
            gt_global = gt_idx_full[gi]

            target_classes[b, rep_global] = 1
            target_roles[b, rep_global] = 1
            matched_pred.append(torch.tensor([b, rep_global.item()], device=device, dtype=torch.long))
            matched_gt.append(torch.tensor([b, gt_global.item()], device=device, dtype=torch.long))

    if len(matched_pred) > 0:
        matched_pred = torch.stack(matched_pred, dim=0)
        matched_gt = torch.stack(matched_gt, dim=0)
        idx = (matched_pred[:, 0], matched_pred[:, 1])
        tgt_idx = (matched_gt[:, 0], matched_gt[:, 1])
    else:
        empty = torch.empty((0,), dtype=torch.long, device=device)
        idx = (empty, empty)
        tgt_idx = (empty, empty)

    if return_roles:
        valid_mask = (
            pred_valid_mask.bool()
            if pred_valid_mask is not None
            else torch.ones((B, N), dtype=torch.bool, device=device)
        )
        target_info = {
            "roles": target_roles,
            "role_labels_available": True,
            "positive_mask": (target_roles == 1) & valid_mask,
            "duplicate_mask": (target_roles == 2) & valid_mask,
            "background_mask": (target_roles == 0) & valid_mask,
            "assigned_gt": assigned_gt,
            "nearest_dist_px": nearest_dist,
            "pos_radius": float(region_radius),
        }
        return target_classes, idx, tgt_idx, target_info

    return target_classes, idx, tgt_idx

class setCriterion(nn.Module):
    def __init__(
        self,
        matcher,
        lambda_exist: float = 1.0,
        lambda_x0: float = 1.0,
        lambda_cnt: float = 0.1,
        lambda_eps: float = 0.1,  # 新增
        lambda_bg: float = 2.0,
        lambda_cov: float = 0.2,
        lambda_cov_hinge: float = 0.0,
        lambda_dup: float = 0.2,
        lambda_dup_collapse: float = 0.0,
        cov_topk: int = 3,
        cov_sigma: float = 4.0,
        cov_radius: float = 6.0,
        cov_hard_weight: float = 0.0,
        cov_hard_cap: float = 1.0,
        cov_dense_weight: float = 0.0,
        cov_dense_radius: float = 16.0,
        cov_dense_norm: float = 4.0,
        cov_weight_cap: float = 6.0,
        region_radius: float = 5.0,
        region_topk: int = 5,
        exist_label_mode: str = "hungarian",
        exist_pos_radius: float = 6.0,
        exist_duplicate_weight: float = 1.0,
        dup_dense_aware: bool = False,
        dup_neighbor_radius: float = 6.0,
        dup_allow_extra: int = 0,
        dup_collapse_inner_radius: float = 2.0,
        dup_collapse_outer_radius: float = 4.0,
        dup_collapse_far_weight: float = 4.0,
        gamma: float = 2.0,
        alpha: float = 0.6,
        eos_coef: float = 0.3,
    ):
        super().__init__()
        self.matcher = matcher
        self.lambda_exist = lambda_exist
        self.lambda_x0 = lambda_x0
        self.lambda_cnt = lambda_cnt
        self.lambda_bg = lambda_bg
        self.lambda_eps = lambda_eps
        self.lambda_cov = lambda_cov
        self.lambda_cov_hinge = lambda_cov_hinge
        self.lambda_dup = lambda_dup
        self.lambda_dup_collapse = lambda_dup_collapse
        self.cov_topk = cov_topk
        self.cov_sigma = cov_sigma
        self.cov_radius = cov_radius
        self.cov_hard_weight = cov_hard_weight
        self.cov_hard_cap = cov_hard_cap
        self.cov_dense_weight = cov_dense_weight
        self.cov_dense_radius = cov_dense_radius
        self.cov_dense_norm = cov_dense_norm
        self.cov_weight_cap = cov_weight_cap
        self.region_radius = region_radius
        self.region_topk = region_topk
        self.exist_label_mode = str(exist_label_mode).strip().lower().replace("-", "_")
        self.exist_pos_radius = float(exist_pos_radius)
        # Down-weight role=2 slots (same GT group, not the closest one) in the
        # existence loss. They currently count as ordinary negatives, identical to
        # background tens of pixels away, even though evaluation would happily
        # match any of them. 1.0 keeps the previous behaviour.
        self.exist_duplicate_weight = float(exist_duplicate_weight)
        self.dup_dense_aware = bool(dup_dense_aware)
        self.dup_neighbor_radius = dup_neighbor_radius
        self.dup_allow_extra = int(dup_allow_extra)
        self.dup_collapse_inner_radius = dup_collapse_inner_radius
        self.dup_collapse_outer_radius = dup_collapse_outer_radius
        self.dup_collapse_far_weight = dup_collapse_far_weight
        self.gamma = gamma
        self.alpha = alpha


        # CE/softmax 版：兩類 (0=no-object, 1=object)，降低 no-object 權重
        empty_weight = torch.ones(2)
        empty_weight[0] = float(eos_coef)
        self.register_buffer('empty_weight', empty_weight)

    def _match_p2p(self, pred_logits, pred_points, gt_points, gt_mask, pred_valid_mask=None):
        if pred_valid_mask is None:
            return self.matcher(pred_logits, pred_points, gt_points, gt_mask)

        indices = []
        B = pred_points.size(0)
        device = pred_points.device

        for b in range(B):
            valid_idx = pred_valid_mask[b].bool().nonzero(as_tuple=False).squeeze(1)
            if valid_idx.numel() == 0 or not gt_mask[b].bool().any():
                empty = torch.empty(0, dtype=torch.long, device=device)
                indices.append((empty, empty))
                continue

            logits_b = pred_logits[b, valid_idx].unsqueeze(0)
            points_b = pred_points[b, valid_idx].unsqueeze(0)
            src_local, tgt_idx = self.matcher(
                logits_b,
                points_b,
                gt_points[b:b + 1],
                gt_mask[b:b + 1],
            )[0]

            indices.append((valid_idx[src_local], tgt_idx))

        return indices

    def _build_exist_targets(
        self,
        pred_logits,
        pred_points,
        gt_points,
        gt_mask,
        pred_valid_mask=None,
        H: int = 256,
        W: int = 256,
        return_info: bool = False,
    ):
        group_modes = {
            "nearest_gt_group",
            "nearest_gt",
            "gt_group",
            "region_representative",
            "region",
        }
        if self.exist_label_mode in group_modes:
            target_classes, idx, tgt_idx, target_info = build_region_representative_targets(
                pred_points,
                gt_points,
                gt_mask,
                pred_valid_mask=pred_valid_mask,
                H=H,
                W=W,
                region_radius=self.exist_pos_radius,
                return_roles=True,
            )
            indices = []
            for b in range(pred_points.size(0)):
                sel = idx[0] == b
                indices.append((idx[1][sel], tgt_idx[1][sel]))
            if return_info:
                return target_classes, indices, target_info
            return target_classes, indices

        cls_indices = self._match_p2p(
            pred_logits,
            pred_points,
            gt_points,
            gt_mask,
            pred_valid_mask=pred_valid_mask,
        )
        cls_idx = self._get_src_permutation_idx(cls_indices)
        target_classes = torch.zeros(
            (pred_logits.size(0), pred_logits.size(1)),
            dtype=torch.long,
            device=pred_logits.device,
        )
        target_classes[cls_idx] = 1
        if return_info:
            valid_mask = (
                pred_valid_mask.bool()
                if pred_valid_mask is not None
                else torch.ones_like(target_classes, dtype=torch.bool)
            )
            target_info = {
                "roles": target_classes,
                "role_labels_available": False,
                "positive_mask": (target_classes == 1) & valid_mask,
                "duplicate_mask": torch.zeros_like(target_classes, dtype=torch.bool),
                "background_mask": (target_classes == 0) & valid_mask,
                "assigned_gt": torch.full_like(target_classes, -1),
                "nearest_dist_px": torch.full(
                    target_classes.shape,
                    float("inf"),
                    dtype=pred_points.dtype,
                    device=pred_points.device,
                ),
                "pos_radius": None,
            }
            return target_classes, cls_indices, target_info
        return target_classes, cls_indices

    def _reduce_exist_ce(self, per_tok_ce, target_info):
        """Mean over slots, with role=2 (same-group non-closest) down-weighted.

        Falls back to a plain mean when the weight is 1.0 or role labels are not
        available, so the hungarian label mode is unaffected.
        """
        w_dup = float(self.exist_duplicate_weight)
        if abs(w_dup - 1.0) <= 1e-9 or not isinstance(target_info, dict):
            return per_tok_ce.mean()
        if not bool(target_info.get("role_labels_available", False)):
            return per_tok_ce.mean()
        roles = target_info.get("roles")
        if roles is None or roles.shape != per_tok_ce.shape:
            return per_tok_ce.mean()

        weights = torch.ones_like(per_tok_ce)
        weights = torch.where(
            roles == 2,
            weights.new_tensor(w_dup),
            weights,
        )
        return (per_tok_ce * weights).sum() / weights.sum().clamp_min(1e-6)

    # ---------------- Focal loss (跟你原本一樣) ----------------
    def focal_loss_with_logits(self, logits, targets):
        x = logits
        y = targets

        ce = torch.clamp(x, min=0) - x * y + torch.log1p(torch.exp(-x.abs()))
        p = torch.sigmoid(x)
        pt = torch.where(y == 1, p, 1 - p).clamp(1e-6, 1 - 1e-6)

        alpha = x.new_tensor(self.alpha)
        alpha_t = torch.where(y == 1, alpha, 1 - alpha)
        fl = alpha_t * (1 - pt).pow(self.gamma) * ce  # [B,N]

        pos = (y == 1)
        neg = (y == 0)

        fl_pos = fl[pos].mean() if pos.any() else fl.sum() * 0.0
        fl_neg = fl[neg].mean() if neg.any() else fl.sum() * 0.0

        neg_w = 1  # 可調 0.25~1.0
        return fl_pos + neg_w * fl_neg

    # ---------------- Hybrid loss 主體 ----------------
    def p2p_exist_loss(
        self,
        exist_logit: torch.Tensor,
        pred_points_for_cls: torch.Tensor,
        gt_points: torch.Tensor,
        gt_mask: torch.Tensor,
        pred_valid_mask: torch.Tensor = None,
        return_target_info: bool = False,
    ):
        """One-to-one objectness loss used by Lex and random-start Lex."""
        cls_logits = exist_logit
        if cls_logits.dim() == 3 and cls_logits.size(-1) == 1:
            cls_logits = cls_logits.squeeze(-1)

        need_info = bool(return_target_info) or abs(self.exist_duplicate_weight - 1.0) > 1e-9
        target_out = self._build_exist_targets(
            cls_logits,
            pred_points_for_cls.detach(),
            gt_points,
            gt_mask,
            pred_valid_mask=pred_valid_mask,
            return_info=need_info,
        )
        if need_info:
            target_classes, _, target_info = target_out
        else:
            target_classes, _ = target_out
            target_info = None

        if cls_logits.dim() == 3 and cls_logits.size(-1) == 2:
            logits_ce = cls_logits.transpose(1, 2)
            per_tok_ce = F.cross_entropy(
                logits_ce,
                target_classes,
                weight=self.empty_weight,
                reduction='none',
            )
            loss = self._reduce_exist_ce(per_tok_ce, target_info)
        else:
            loss = self.focal_loss_with_logits(
                cls_logits,
                target_classes.to(dtype=cls_logits.dtype),
            )

        if return_target_info:
            return loss, target_classes, target_info
        return loss, target_classes

    def forward(
        self,
        p_t: torch.Tensor,        # [B,N,2] 当前 noisy points
        p0: torch.Tensor,         # [B,N,2] GT points (在 [-1,1] 座標)
        mask: torch.Tensor,       # [B,N]   True/1 = 前景 GT
        pro: torch.Tensor,
        abar_t: torch.Tensor,     # [B,1,1] or [B,1] \bar{α}_t
        eps_pred: torch.Tensor,   # [B,N,2] 模型預測的 ε_t
        exist_logit: torch.Tensor,# [B,N]   存在度 logits
        pred_points_for_cls=None,
        pred_valid_mask=None,
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


        # =========================================================
        # ---- 3. 反推出 x0_hat（給 x0 loss / 匹配用）----
        x0_hat = (p_t - sqrt_om * eps_pred) / (sqrt_ab + eps)
        x0_hat = x0_hat.clamp(-1 + 1e-3, 1 - 1e-3)


        # ---- 4. 匹配 (Hungarian) ----
        # 注意：使用 detach() 的 x0_hat 去做匹配，避免存在度梯度影響 L_x0
        # P2P targets for the merged confidence tokens.
        if pred_points_for_cls is None:
            pred_points_for_cls = x0_hat.detach()

        need_roles = abs(self.exist_duplicate_weight - 1.0) > 1e-9
        if need_roles:
            target_classes, _, exist_target_info = self._build_exist_targets(
                exist_logit,
                pred_points_for_cls.detach(),
                p0,
                mask,
                pred_valid_mask=pred_valid_mask,
                return_info=True,
            )
        else:
            target_classes, _ = self._build_exist_targets(
                exist_logit,
                pred_points_for_cls.detach(),
                p0,
                mask,
                pred_valid_mask=pred_valid_mask,
            )
            exist_target_info = None

        # Coordinate loss stays on original x0_hat, so use coordinate-only P2P matching.
        coord_match_logits = exist_logit.new_zeros((x0_hat.size(0), x0_hat.size(1), 2))
        indices = self.matcher(coord_match_logits, x0_hat.detach(), p0, mask)
        idx = self._get_src_permutation_idx(indices)
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

        
        # ---- 5. L_exist ----
        # target_classes: 0=no-object, 1=Hungarian P2P matched point
        if exist_logit.dim() == 3 and exist_logit.size(-1) == 2:
            if not torch.isfinite(exist_logit).all():
                print("[WARN] cls logits not finite before CE")

            # CrossEntropy expects [B,C,N]
            logits_ce = exist_logit.transpose(1, 2)  # [B,2,N]
            # 先算每個 token 的 CE（不做 reduction）
            per_tok_ce = F.cross_entropy(
                logits_ce,  # [B,2,N]
                target_classes,  # [B,N]
                weight=self.empty_weight,  # [2]
                reduction='none'  # -> [B,N]
            )  # [B,N]

            # 正/負 mask（注意：這裡的 pos 是 matched predictions，不是 GT slot）
            pos_mask = (target_classes == 1)  # [B,N]
            neg_mask = ~pos_mask  # [B,N]

            # 計數
            N_pos = pos_mask.sum().item()
            N_neg = neg_mask.sum().item()

            # 平均 loss（避免除 0）
            pos_loss_mean = (per_tok_ce[pos_mask].mean().item() if N_pos > 0 else 0.0)
            neg_loss_mean = (per_tok_ce[neg_mask].mean().item() if N_neg > 0 else 0.0)

            # 你要看的「總量對比」
            # 注意：per_tok_ce 已經把 class weight 套進去了（背景 * eos_coef、人頭 * w_pos）
            # 所以這裡其實 total = mean * count 就已經反映權重了。
            pos_total = pos_loss_mean * N_pos
            neg_total = neg_loss_mean * N_neg

            # 若你堅持要用你指定的公式形式「再乘 eos_coef」，
            # 那要先用未加權的 neg_loss_mean_raw 來算，不然會重複乘權重。
            # 這裡我也一起算給你看（比較清楚）
            per_tok_ce_raw = F.cross_entropy(
                logits_ce,
                target_classes,
                reduction='none'
            )  # [B,N] 未加權版

            pos_loss_mean_raw = (per_tok_ce_raw[pos_mask].mean().item() if N_pos > 0 else 0.0)
            neg_loss_mean_raw = (per_tok_ce_raw[neg_mask].mean().item() if N_neg > 0 else 0.0)

            eos = float(self.empty_weight[0].item())
            wpos = float(self.empty_weight[1].item())

            pos_total_formula = pos_loss_mean_raw * N_pos * wpos
            neg_total_formula = neg_loss_mean_raw * N_neg * eos

            ratio = (neg_total_formula / (pos_total_formula + 1e-12))
            with torch.no_grad():
                probs = exist_logit.softmax(dim=-1)  # [B,N,2]
                p_neg = probs[..., 0]  # [B,N]
                p_pos = probs[..., 1]  # [B,N]

                # 注意避免空集合
                def safe_mean(x):
                    return x.mean().item() if x.numel() else 0.0

                def safe_q(x, q):
                    if x.numel() == 0:
                        return 0.0
                    x = x.reshape(-1).float()
                    return x.quantile(q).item()


                pos_p_pos_mean = safe_mean(p_pos[pos_mask])  # 正樣本上，P(pos) 平均
                pos_p_pos_p05 = safe_q(p_pos[pos_mask], 0.05)
                pos_p_pos_p50 = safe_q(p_pos[pos_mask], 0.50)

                neg_p_pos_mean = safe_mean(p_pos[neg_mask])  # 負樣本上，P(pos) 平均（越低越好）
                neg_p_pos_p95 = safe_q(p_pos[neg_mask], 0.95)

                neg_p_neg_mean = safe_mean(p_neg[neg_mask])  # 負樣本上，P(neg) 平均（越高越好）

                logit_neg = exist_logit[..., 0]  # [B,N]
                logit_pos = exist_logit[..., 1]  # [B,N]
                margin = logit_pos - logit_neg  # [B,N]

                pos_margin_mean = safe_mean(margin[pos_mask])
                pos_margin_p05 = safe_q(margin[pos_mask], 0.05)
                pos_margin_p50 = safe_q(margin[pos_mask], 0.50)

                neg_margin_mean = safe_mean(margin[neg_mask])
                neg_margin_p95 = safe_q(margin[neg_mask], 0.95)


            # ======= 控制印出頻率（建議每 50 steps 印一次）=======
            # 你可以在 criterion 外面傳 step 進來，或用全域計數器。
            # 這裡提供一個簡單方案：在 module 裡放 self._dbg_step 計數
            if not hasattr(self, "_dbg_step"):
                self._dbg_step = 0
            self._dbg_step += 1

            if (self._dbg_step % 100) == 0:
                print(
                    f"[CE dbg] step={self._dbg_step} "
                    f"N_pos={N_pos} N_neg={N_neg} "
                    f"pos_mean={pos_loss_mean:.4f} neg_mean={neg_loss_mean:.4f} "
                    f"pos_total={pos_total:.2f} neg_total={neg_total:.2f} | "
                    f"(raw) pos_mean={pos_loss_mean_raw:.4f} neg_mean={neg_loss_mean_raw:.4f} "
                    f"w=[bg:{eos:.3g}, pos:{wpos:.3g}] "
                    f"pos_totalF={pos_total_formula:.2f} neg_totalF={neg_total_formula:.2f} "
                    f"ratio(neg/pos)={ratio:.2f}"
                )
                # print(
                #     f"[P dbg] POS tokens: P(pos) mean={pos_p_pos_mean:.4f} p05={pos_p_pos_p05:.4f} p50={pos_p_pos_p50:.4f} | "
                #     f"NEG tokens: P(pos) mean={neg_p_pos_mean:.4f} p95={neg_p_pos_p95:.4f}  P(neg) mean={neg_p_neg_mean:.4f}"
                # )
                print(
                    f"[M dbg] POS margin(pos-neg): mean={pos_margin_mean:.4f} p05={pos_margin_p05:.4f} p50={pos_margin_p50:.4f} | "
                    f"NEG margin(pos-neg): mean={neg_margin_mean:.4f} p95={neg_margin_p95:.4f}"
                )
            # 最後再用原本的方式算 L_exist（或直接 per_tok_ce.mean）
            L_exist = self._reduce_exist_ce(per_tok_ce, exist_target_info)
        else:
            # fallback: 舊版 focal（二分類 sigmoid）
            print("not cross entropy")
            target_classes = target_classes.to(dtype=exist_logit.dtype)
            if not torch.isfinite(exist_logit).all():
                print("[WARN] logits not finite before focal_loss")
            L_exist = self.focal_loss_with_logits(exist_logit, target_classes)

        # ---- 6. L_cnt (soft count) ----
        if exist_logit.dim() == 3 and exist_logit.size(-1) == 2:
            prob_v = exist_logit.softmax(-1)[..., 1]  # [B,N] object prob
            pos_mask_pred = (target_classes == 1).float()
        else:
            prob_v = torch.sigmoid(exist_logit)       # [B,N]
            pos_mask_pred = (target_classes > 0.5).float()

        pred_cnt = prob_v.sum(dim=1)
        gt_cnt = mask.sum(dim=1).float()
        L_cnt = (pred_cnt - gt_cnt).abs() / (gt_cnt + 1.0)
        L_cnt = L_cnt.mean()

        # ---- 6.5. Proposal coverage losses ----
        # Optimize coverage with the random/padded proposal slots. If GT slots
        # are allowed here, the loss can be satisfied by the teacher-like slots
        # and fail to train the inference-time random proposals.
        coverage_valid_mask = ~mask.bool()
        has_coverage_slots = coverage_valid_mask.any(dim=1, keepdim=True)
        coverage_valid_mask = torch.where(
            has_coverage_slots,
            coverage_valid_mask,
            torch.ones_like(coverage_valid_mask),
        )

        L_cov = coverage_loss_bag(
            pred_points_m11=x0_hat,
            prob_obj=prob_v,
            gt_points_m11=p0,
            gt_mask=mask,
            pred_valid_mask=coverage_valid_mask,
            H=256,
            W=256,
            topk=self.cov_topk,
            sigma=self.cov_sigma,
        )
        L_cov_hinge = coverage_radius_hinge_loss(
            pred_points_m11=x0_hat,
            gt_points_m11=p0,
            gt_mask=mask,
            pred_valid_mask=coverage_valid_mask,
            H=256,
            W=256,
            cover_radius=self.cov_radius,
            hard_weight=self.cov_hard_weight,
            hard_cap=self.cov_hard_cap,
            dense_weight=self.cov_dense_weight,
            dense_radius=self.cov_dense_radius,
            dense_norm=self.cov_dense_norm,
            weight_cap=self.cov_weight_cap,
        )

        if pred_points_for_cls is None:
            pred_points_for_cls = x0_hat

        L_dup = region_duplicate_loss(
            pred_points_m11=pred_points_for_cls,
            prob_obj=prob_v,
            gt_points_m11=p0,
            gt_mask=mask,
            pred_valid_mask=pred_valid_mask,
            H=256,
            W=256,
            region_radius=self.region_radius,
            region_topk=self.region_topk,
            dense_aware=self.dup_dense_aware,
            neighbor_radius=self.dup_neighbor_radius,
            allow_extra=self.dup_allow_extra,
        )
        L_dup_collapse = duplicate_collapse_loss(
            pred_points_m11=x0_hat,
            gt_points_m11=p0,
            gt_mask=mask,
            matched_indices=indices,
            H=256,
            W=256,
            inner_radius=self.dup_collapse_inner_radius,
            outer_radius=self.dup_collapse_outer_radius,
            far_weight=self.dup_collapse_far_weight,
        )

        # ---- 7. 背景損失 L_bg ----
        pos_mask_bool = (target_classes == 1) if (target_classes.dtype == torch.long) else (target_classes > 0.5)

        # 取每張圖有效 GT
        B, N, _ = x0_hat.shape
        bg_losses = []
        r_pix = 6.0  # 先從 4~8 像素試
        for b in range(B):
            tgt_idx = mask[b].nonzero(as_tuple=False).squeeze(1)
            if tgt_idx.numel() == 0:
                # 沒 GT：全部非 matched 都是背景
                bgmask_b = (~pos_mask_bool[b])
            else:
                pred_pix = m11_to_pixels(x0_hat[b], 256, 256)  # [N,2]
                # dup_dist = torch.cdist(pred_pix, pred_pix, p=2)  # [N,N]
                # eye = torch.eye(dup_dist.size(0), device=dup_dist.device, dtype=torch.bool)
                # dup_dist.masked_fill_(eye, 1e9)
                #
                # dup_thr = 2.0  # 可改 1.0 / 2.0 / 3.0 pixels 試試
                # dup_mask = dup_dist < dup_thr
                #
                # dup_pairs = torch.triu(dup_mask, diagonal=1).sum().item()
                # dup_points = dup_mask.any(dim=1).sum().item()
                # min_nn_dist = dup_dist.min(dim=1).values.min().item()
                # mean_nn_dist = dup_dist.min(dim=1).values.mean().item()
                gt_pix = m11_to_pixels(p0[b, tgt_idx], 256, 256)  # [Mb,2]
                dist = torch.cdist(pred_pix, gt_pix, p=2)  # [N,Mb]
                dmin = dist.min(dim=1).values  # [N]
                ignore = (dmin < r_pix)  # [N]  # GT附近不當背景
                bgmask_b = (~pos_mask_bool[b]) & (~ignore)  # [N]
                # near_gt_unmatched = (~pos_mask_bool[b]) & ignore
                # if (self._dbg_step % 100) == 0:
                #     st = analyze_pro_similarity(
                #         pred_pix=pred_pix,
                #         pro_feat=pro[b].detach(),  # [N,C]
                #         pos_mask=pos_mask_bool[b],
                #         ignore_mask=ignore,
                #         dup_thr=2.0,
                #         far_thr=8.0,
                #         max_pairs=2000,
                #     )
                #
                #     print(f"[ProSim][img {b}] N={st['N']} near_gt_unmatched={st['near_gt_unmatched_count']}")
                #
                #     if st["near_dup"] is not None:
                #         x = st["near_dup"]
                #         print(
                #             f"  [near_dup] pairs={x['num_pairs']} "
                #             f"pix_mean={x['pix_dist_mean']:.3f} "
                #             f"cos_mean={x['cos_mean']:.4f} cos_p50={x['cos_p50']:.4f} "
                #             f"l2_mean={x['l2_mean']:.4f} l2_p50={x['l2_p50']:.4f}"
                #         )
                #
                #     if st["far_pair"] is not None:
                #         x = st["far_pair"]
                #         print(
                #             f"  [far_pair] pairs={x['num_pairs']} "
                #             f"pix_mean={x['pix_dist_mean']:.3f} "
                #             f"cos_mean={x['cos_mean']:.4f} cos_p50={x['cos_p50']:.4f} "
                #             f"l2_mean={x['l2_mean']:.4f} l2_p50={x['l2_p50']:.4f}"
                #         )
                #
                #     if st["near_gt_unmatched_dup"] is not None:
                #         x = st["near_gt_unmatched_dup"]
                #         print(
                #             f"  [nearGT_unmatched_dup] pairs={x['num_pairs']} "
                #             f"pix_mean={x['pix_dist_mean']:.3f} "
                #             f"cos_mean={x['cos_mean']:.4f} cos_p50={x['cos_p50']:.4f} "
                #             f"l2_mean={x['l2_mean']:.4f} l2_p50={x['l2_p50']:.4f}"
                #         )
                #     print(
                #         f"[DupCheck][img {b}] "
                #         f"dup_pairs={dup_pairs} "
                #         f"dup_points={dup_points}/{pred_pix.size(0)} "
                #         f"min_nn_dist={min_nn_dist:.4f} "
                #         f"mean_nn_dist={mean_nn_dist:.4f}"
                #     )
                #
                #     print(
                #         f"[NearGT-Unmatched][img {b}] "
                #         f"count={near_gt_unmatched.sum().item()} / {pred_pix.size(0)}"
                #     )
            if bgmask_b.any():
                bg_losses.append(prob_v[b][bgmask_b].mean())
            else:
                bg_losses.append(prob_v[b].new_tensor(0.0))

        L_bg = torch.stack(bg_losses).mean()

        # ---- 8. Hybrid Loss：x0-style 單步訓練版 ----
        if isinstance(lambda_t, torch.Tensor):
            lambda_t_scalar = lambda_t.mean()
        elif lambda_t is None:
            lambda_t_scalar = L_eps.new_tensor(1.0)
        else:
            lambda_t_scalar = L_eps.new_tensor(float(lambda_t))

        if aux_weight is None:
            aux_weight_scalar = L_eps.new_tensor(1.0)
        else:
            aux_weight_scalar = L_eps.new_tensor(float(aux_weight))

        eps_term = self.lambda_eps * lambda_t_scalar * L_eps
        aux_term = aux_weight_scalar * (
                self.lambda_x0 * L_x0
                + self.lambda_exist * L_exist
                + self.lambda_cnt * L_cnt
                + self.lambda_bg * L_bg
                + self.lambda_cov * L_cov
                + self.lambda_cov_hinge * L_cov_hinge
                + self.lambda_dup * L_dup
                + self.lambda_dup_collapse * L_dup_collapse
        )

        loss = eps_term + aux_term

        # if hasattr(self, "_dbg_step") and (self._dbg_step % 100) == 0:
        #     print(
        #         f"[LossBalance] "
        #         f"lambda_t={float(lambda_t_scalar):.6f} "
        #         f"aux_weight={float(aux_weight_scalar):.6f} | "
        #         f"Leps={L_eps.item():.6f} "
        #         f"Lx0={L_x0.item():.6f} "
        #         f"Lex={L_exist.item():.6f} | "
        #         f"eps_term={eps_term.item():.6f} "
        #         f"aux_term={aux_term.item():.6f} "
        #         f"ratio_aux/eps={(aux_term.item() / (eps_term.item() + 1e-12)):.3f}"
        #     )

        if not torch.isfinite(loss):
            print(
                "[ERROR] Non-finite loss detected:",
                "Leps=", float(L_eps),
                "Lex=", float(L_exist),
                "Lx0=", float(L_x0),
                "Lcnt=", float(L_cnt),
                "Lbg=", float(L_bg),
                "Lcov_hinge=", float(L_cov_hinge),
                "Ldup=", float(L_dup),
                "LdupCollapse=", float(L_dup_collapse),
            )
            raise RuntimeError("Loss became NaN/inf, stop and inspect.")

        return loss, L_exist, L_x0, L_cnt, L_bg, L_eps, L_cov, L_cov_hinge, L_dup, L_dup_collapse









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
