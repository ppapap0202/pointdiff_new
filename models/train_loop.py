# train_loop.py
import logging
import random
import torch
from models.diffusion_utils import (
    pixels_to_m11,
    forward_noisy,
    coverage_radius_hinge_loss,
    region_duplicate_loss,
)
from models.pointdiff import sample_point_tokens, pool_local_tokens
from models.proposal_prior import (
    build_mixed_x0_prior,
    build_proposal_prior_targets,
    proposal_prior_loss,
    select_guided_prior_points,
)
import numpy as np
from collections import defaultdict


def preserve_rng_state(fn):
    def wrapped(*args, **kwargs):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        cuda_states = None
        if torch.cuda.is_available():
            cuda_states = torch.cuda.get_rng_state_all()
        try:
            return fn(*args, **kwargs)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            if cuda_states is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(cuda_states)

    return wrapped


def full_image_matching_stats(gt_xy, prop_xy, radius: float = 6.0):
    """Return one-to-one GT matches and proposal multiplicity inside radius."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
    from scipy.spatial import cKDTree

    gt_xy = np.asarray(gt_xy, dtype=np.float32).reshape(-1, 2)
    prop_xy = np.asarray(prop_xy, dtype=np.float32).reshape(-1, 2)
    n_gt = int(gt_xy.shape[0])
    n_prop = int(prop_xy.shape[0])

    if n_gt == 0 or n_prop == 0:
        return {
            "gt_count": n_gt,
            "point_count": n_prop,
            "matched_gt": 0,
            "nearby_proposals": 0,
        }

    tree = cKDTree(prop_xy)
    indptr = [0]
    indices = []
    nearby_proposals = 0
    for gt in gt_xy:
        cols = tree.query_ball_point(gt, float(radius))
        nearby_proposals += len(cols)
        indices.extend(cols)
        indptr.append(len(indices))

    if not indices:
        matched_gt = 0
    else:
        graph = csr_matrix(
            (
                np.ones(len(indices), dtype=np.bool_),
                np.asarray(indices, dtype=np.int32),
                np.asarray(indptr, dtype=np.int32),
            ),
            shape=(n_gt, n_prop),
        )
        match = maximum_bipartite_matching(graph, perm_type="column")
        matched_gt = int(np.sum(match >= 0))

    return {
        "gt_count": n_gt,
        "point_count": n_prop,
        "matched_gt": matched_gt,
        "nearby_proposals": int(nearby_proposals),
    }


@torch.no_grad()
def batch_proposal_region_stats(pred_points_m11: torch.Tensor,
                                gt_points_m11: torch.Tensor,
                                gt_mask: torch.Tensor,
                                pred_valid_mask: torch.Tensor = None,
                                radius: float = 6.0,
                                H: int = 256,
                                W: int = 256):
    """
    Lightweight batch-level diagnostics:
    - prop_cov@r: fraction of GT that has at least one nearby proposal
    - multi@r: fraction of GT that has more than one nearby proposal
    """
    pred_pix = m11_to_pixels_batch(pred_points_m11, H, W)
    gt_pix = m11_to_pixels_batch(gt_points_m11, H, W)

    cover_rows = []
    multi_rows = []
    r2 = float(radius) * float(radius)

    for b in range(pred_pix.size(0)):
        gmask = gt_mask[b].bool()
        if not gmask.any():
            continue
        G = gt_pix[b, gmask]  # [Ng,2]
        if pred_valid_mask is not None:
            pmask = pred_valid_mask[b].bool()
            P = pred_pix[b, pmask]
        else:
            P = pred_pix[b]       # [N,2]
        if P.numel() == 0:
            continue
        dist2 = ((G[:, None, :] - P[None, :, :]) ** 2).sum(dim=-1)  # [Ng,N]
        hit_counts = (dist2 <= r2).sum(dim=1).float()               # [Ng]
        cover_rows.append((hit_counts > 0).float().mean())
        multi_rows.append((hit_counts > 1).float().mean())

    if len(cover_rows) == 0:
        zero = pred_points_m11.new_tensor(0.0)
        return zero, zero

    return torch.stack(cover_rows).mean(), torch.stack(multi_rows).mean()
def m11_to_pixels_batch(p_m11: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    p_m11: [B,N,2] in [-1,1]
    return: [B,N,2] in pixel coords
    """
    x = (p_m11[..., 0] + 1) * 0.5 * (W - 1)
    y = (p_m11[..., 1] + 1) * 0.5 * (H - 1)
    return torch.stack([x, y], dim=-1)


def topk_count_margin_loss(
        prob_obj: torch.Tensor,
        gt_mask: torch.Tensor,
        pred_valid_mask: torch.Tensor = None,
        pos_margin: float = 0.35,
        neg_margin: float = 0.10,
        neg_topk: int = 64,
        slack: int = 0,
) -> torch.Tensor:
    """
    Hard-count proxy for threshold/NMS counting.

    Unlike sum(prob)=count, this only shapes the score boundary: the K-th
    proposal should be confidently above threshold, while the strongest
    negatives after K should stay below threshold. It avoids rewarding a flat
    probability field across all slots.
    """
    losses = []
    pos_margin = float(pos_margin)
    neg_margin = float(neg_margin)
    neg_topk = max(1, int(neg_topk))
    slack = max(0, int(slack))

    for b in range(prob_obj.size(0)):
        if pred_valid_mask is not None:
            scores = prob_obj[b][pred_valid_mask[b].bool()]
        else:
            scores = prob_obj[b]

        if scores.numel() == 0:
            losses.append(prob_obj.new_tensor(0.0))
            continue

        scores = torch.sort(scores, descending=True).values
        gt_count = int(gt_mask[b].bool().sum().item())

        if gt_count <= 0:
            losses.append(torch.relu(scores[0] - neg_margin).pow(2))
            continue

        k = min(gt_count, scores.numel())
        pos_boundary = scores[k - 1]
        pos_loss = torch.relu(pos_margin - pos_boundary).pow(2)

        neg_loss = scores.new_tensor(0.0)
        neg_start = min(k + slack, scores.numel())
        if neg_start < scores.numel():
            hard_neg = scores[neg_start:neg_start + neg_topk]
            neg_loss = torch.relu(hard_neg - neg_margin).pow(2).mean()

        losses.append(pos_loss + neg_loss)

    if len(losses) == 0:
        return prob_obj.new_tensor(0.0)

    return torch.stack(losses).mean()


def proposal_hungarian_cover_loss(
        pred_points_m11: torch.Tensor,
        gt_points_m11: torch.Tensor,
        gt_mask: torch.Tensor,
        matcher,
        pred_valid_mask: torch.Tensor = None,
        H: int = 256,
        W: int = 256,
        radius: float = 6.0,
) -> torch.Tensor:
    """
    One-to-one coverage loss for inference-style random proposals.
    Each GT is matched to a unique proposal when possible, so dense regions
    cannot all be explained by the same nearest proposal.
    """
    import torch.nn.functional as F

    if not gt_mask.bool().any():
        return pred_points_m11.new_tensor(0.0)

    match_logits = pred_points_m11.new_zeros((pred_points_m11.size(0), pred_points_m11.size(1), 2))

    if pred_valid_mask is None:
        indices = matcher(match_logits, pred_points_m11.detach(), gt_points_m11, gt_mask)
    else:
        indices = []
        for b in range(pred_points_m11.size(0)):
            valid_idx = pred_valid_mask[b].bool().nonzero(as_tuple=False).squeeze(1)
            if valid_idx.numel() == 0 or not gt_mask[b].bool().any():
                empty = torch.empty(0, dtype=torch.long, device=pred_points_m11.device)
                indices.append((empty, empty))
                continue

            src_local, tgt_idx = matcher(
                match_logits[b, valid_idx].unsqueeze(0),
                pred_points_m11[b, valid_idx].detach().unsqueeze(0),
                gt_points_m11[b:b + 1],
                gt_mask[b:b + 1],
            )[0]
            indices.append((valid_idx[src_local], tgt_idx))

    pred_pix = m11_to_pixels_batch(pred_points_m11, H, W)
    gt_pix = m11_to_pixels_batch(gt_points_m11, H, W)
    scale = max(float(radius), 1e-6)
    losses = []

    for b, (src_idx, tgt_idx) in enumerate(indices):
        if src_idx.numel() == 0:
            continue
        pred_b = pred_pix[b, src_idx] / scale
        gt_b = gt_pix[b, tgt_idx] / scale
        losses.append(F.smooth_l1_loss(pred_b, gt_b, reduction="mean"))

    if len(losses) == 0:
        return pred_points_m11.new_tensor(0.0)

    return torch.stack(losses).mean()


def object_score_from_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.dim() == 3 and logits.size(-1) == 2:
        return logits[..., 1] - logits[..., 0]
    if logits.dim() == 3 and logits.size(-1) == 1:
        return logits.squeeze(-1)
    return logits


def selector_hard_negative_ranking_loss(
        selector_logits: torch.Tensor,
        pred_points_m11: torch.Tensor,
        target_classes: torch.Tensor,
        gt_points_m11: torch.Tensor,
        gt_mask: torch.Tensor,
        pred_valid_mask: torch.Tensor = None,
        target_roles: torch.Tensor = None,
        H: int = 256,
        W: int = 256,
        near_radius: float = 6.0,
        far_radius: float = 12.0,
        margin: float = 1.0,
        neg_topk: int = 96,
        near_weight: float = 1.0,
        far_weight: float = 0.25,
) -> torch.Tensor:
    scores = object_score_from_logits(selector_logits)
    pred_pix = m11_to_pixels_batch(pred_points_m11, H, W)
    gt_pix = m11_to_pixels_batch(gt_points_m11, H, W)
    valid_mask = (
        pred_valid_mask.bool()
        if pred_valid_mask is not None
        else torch.ones_like(target_classes, dtype=torch.bool)
    )
    if target_roles is not None:
        pos_mask = (target_roles == 1) & valid_mask
    else:
        pos_mask = (target_classes == 1) & valid_mask
    losses = []
    neg_topk = max(1, int(neg_topk))
    margin = float(margin)

    for b in range(scores.size(0)):
        pos_scores = scores[b][pos_mask[b]]
        if pos_scores.numel() == 0:
            continue

        gt_idx = gt_mask[b].bool().nonzero(as_tuple=False).squeeze(1)
        neg_mask = valid_mask[b] & (~pos_mask[b])
        if gt_idx.numel() == 0 or not neg_mask.any():
            continue

        if target_roles is not None:
            near_neg = valid_mask[b] & (target_roles[b] == 2)
            far_neg = valid_mask[b] & (target_roles[b] == 0)
        else:
            dmin = torch.cdist(pred_pix[b], gt_pix[b, gt_idx], p=2).min(dim=1).values
            near_neg = neg_mask & (dmin <= float(near_radius))
            far_neg = neg_mask & (dmin >= float(far_radius))

        def margin_loss(neg_scores, weight):
            if neg_scores.numel() == 0 or weight <= 0:
                return None
            k = min(neg_topk, neg_scores.numel())
            hard_neg = neg_scores.topk(k=k, largest=True).values
            pair = torch.relu(margin + hard_neg.unsqueeze(0) - pos_scores.unsqueeze(1))
            return float(weight) * pair.mean()

        near_loss = margin_loss(scores[b][near_neg], near_weight)
        far_loss = margin_loss(scores[b][far_neg], far_weight)
        if near_loss is not None:
            losses.append(near_loss)
        if far_loss is not None:
            losses.append(far_loss)

    if len(losses) == 0:
        return scores.new_tensor(0.0)
    return torch.stack(losses).mean()


def region_one_to_one_confidence_loss(
        selector_logits: torch.Tensor,
        pred_points_m11: torch.Tensor,
        gt_points_m11: torch.Tensor,
        gt_mask: torch.Tensor,
        matcher,
        pred_valid_mask: torch.Tensor = None,
        H: int = 256,
        W: int = 256,
        region_radius: float = 6.0,
        bg_radius: float = 12.0,
        margin: float = 1.0,
        neg_topk: int = 128,
        count_pos_logit: float = 2.0,
        count_neg_logit: float = -2.0,
        count_slack: int = 0,
        pos_logit: float = 0.0,
        duplicate_neg_logit: float = 0.0,
        bg_neg_logit: float = 0.0,
        pos_weight: float = 1.0,
        duplicate_weight: float = 3.0,
        bg_weight: float = 0.5,
        rank_weight: float = 3.0,
        count_weight: float = 1.0,
) -> torch.Tensor:
    """
    Region-aware one-to-one selector loss.

    A coordinate-only Hungarian assignment selects at most one winner per GT.
    Other valid candidates inside that GT region become duplicate negatives.
    This trains confidence ranking without letting confidence choose its own
    positive labels.
    """
    import torch.nn.functional as F

    scores = object_score_from_logits(selector_logits)
    pred_pix = m11_to_pixels_batch(pred_points_m11.detach(), H, W)
    gt_pix = m11_to_pixels_batch(gt_points_m11.detach(), H, W)
    valid_mask = (
        pred_valid_mask.bool()
        if pred_valid_mask is not None
        else torch.ones_like(scores, dtype=torch.bool)
    )

    region_radius = max(float(region_radius), 1e-6)
    bg_radius = max(float(bg_radius), region_radius)
    margin = float(margin)
    neg_topk = max(1, int(neg_topk))
    count_slack = max(0, int(count_slack))

    pos_losses = []
    duplicate_losses = []
    bg_losses = []
    rank_losses = []
    count_losses = []

    for b in range(scores.size(0)):
        valid_idx = valid_mask[b].nonzero(as_tuple=False).squeeze(1)
        if valid_idx.numel() == 0:
            continue

        gt_idx = gt_mask[b].bool().nonzero(as_tuple=False).squeeze(1)
        score_b = scores[b]

        if gt_idx.numel() == 0:
            hard_bg = score_b[valid_idx]
            k_bg = min(neg_topk, hard_bg.numel())
            bg_losses.append(
                F.softplus(
                    hard_bg.topk(k=k_bg, largest=True).values
                    - score_b.new_tensor(float(bg_neg_logit))
                ).mean()
            )
            continue

        match_logits = score_b.new_zeros((1, valid_idx.numel(), 2))
        src_local, tgt_global = matcher(
            match_logits,
            pred_points_m11[b, valid_idx].detach().unsqueeze(0),
            gt_points_m11[b:b + 1],
            gt_mask[b:b + 1],
        )[0]

        d_to_gt = torch.cdist(pred_pix[b, valid_idx], gt_pix[b, gt_idx], p=2)
        dmin, nearest_local_gt = d_to_gt.min(dim=1)

        pos_global = []
        pos_tgt_global = []
        if src_local.numel() > 0:
            matched_global = valid_idx[src_local]
            matched_gt_pix = gt_pix[b, tgt_global]
            matched_pred_pix = pred_pix[b, matched_global]
            matched_dist = torch.norm(matched_pred_pix - matched_gt_pix, dim=1)
            keep = matched_dist <= region_radius
            if keep.any():
                pos_global = matched_global[keep]
                pos_tgt_global = tgt_global[keep]

        pos_mask_b = torch.zeros_like(score_b, dtype=torch.bool)
        if isinstance(pos_global, torch.Tensor) and pos_global.numel() > 0:
            pos_mask_b[pos_global] = True
            pos_scores = score_b[pos_global]
            pos_losses.append(
                F.softplus(
                    score_b.new_tensor(float(pos_logit)) - pos_scores
                ).mean()
            )

        duplicate_valid_local = (dmin <= region_radius) & (~pos_mask_b[valid_idx])
        if duplicate_valid_local.any():
            dup_scores = score_b[valid_idx[duplicate_valid_local]]
            k_dup = min(neg_topk, dup_scores.numel())
            hard_dup = dup_scores.topk(k=k_dup, largest=True).values
            duplicate_losses.append(
                F.softplus(
                    hard_dup - score_b.new_tensor(float(duplicate_neg_logit))
                ).mean()
            )

        bg_valid_local = dmin >= bg_radius
        if bg_valid_local.any():
            bg_scores = score_b[valid_idx[bg_valid_local]]
            k_bg = min(neg_topk, bg_scores.numel())
            hard_bg = bg_scores.topk(k=k_bg, largest=True).values
            bg_losses.append(
                F.softplus(
                    hard_bg - score_b.new_tensor(float(bg_neg_logit))
                ).mean()
            )

        if isinstance(pos_global, torch.Tensor) and pos_global.numel() > 0:
            for pred_i, tgt_i in zip(pos_global, pos_tgt_global):
                tgt_local = (gt_idx == tgt_i).nonzero(as_tuple=False).squeeze(1)
                if tgt_local.numel() == 0:
                    continue
                same_region = (
                    (d_to_gt[:, tgt_local[0]] <= region_radius)
                    & (valid_idx != pred_i)
                )
                if not same_region.any():
                    continue
                neg_scores = score_b[valid_idx[same_region]]
                k_rank = min(neg_topk, neg_scores.numel())
                hard_neg = neg_scores.topk(k=k_rank, largest=True).values
                rank_losses.append(
                    torch.relu(margin + hard_neg - score_b[pred_i]).mean()
                )

        sorted_scores = score_b[valid_idx].sort(descending=True).values
        gt_count = int(gt_idx.numel())
        if sorted_scores.numel() > 0 and gt_count > 0:
            k_pos = min(gt_count, sorted_scores.numel())
            pos_boundary = sorted_scores[k_pos - 1]
            count_loss = torch.relu(
                score_b.new_tensor(float(count_pos_logit)) - pos_boundary
            ).pow(2)
            neg_start = min(k_pos + count_slack, sorted_scores.numel())
            if neg_start < sorted_scores.numel():
                neg_boundary = sorted_scores[neg_start:neg_start + neg_topk]
                count_loss = count_loss + torch.relu(
                    neg_boundary - score_b.new_tensor(float(count_neg_logit))
                ).pow(2).mean()
            count_losses.append(count_loss)

    total = scores.new_tensor(0.0)

    def add_component(losses, weight):
        if not losses or float(weight) <= 0.0:
            return scores.new_tensor(0.0)
        return float(weight) * torch.stack(losses).mean()

    total = total + add_component(pos_losses, pos_weight)
    total = total + add_component(duplicate_losses, duplicate_weight)
    total = total + add_component(bg_losses, bg_weight)
    total = total + add_component(rank_losses, rank_weight)
    total = total + add_component(count_losses, count_weight)
    return total


def group_level_positive_confidence_loss(
        selector_logits: torch.Tensor,
        pred_points_m11: torch.Tensor,
        gt_points_m11: torch.Tensor,
        gt_mask: torch.Tensor,
        pred_valid_mask: torch.Tensor = None,
        H: int = 256,
        W: int = 256,
        group_radius: float = 6.0,
        soft_sigma: float = 4.0,
        temperature: float = 0.7,
        pos_logit: float = 2.0,
        nearest_gt_only: bool = True,
) -> torch.Tensor:
    """
    Encourage each GT-centered candidate group to contain at least one confident
    proposal, without deciding which local candidate must be the winner.
    """
    import torch.nn.functional as F

    scores = object_score_from_logits(selector_logits)
    pred_pix = m11_to_pixels_batch(pred_points_m11.detach(), H, W)
    gt_pix = m11_to_pixels_batch(gt_points_m11.detach(), H, W)
    valid_mask = (
        pred_valid_mask.bool()
        if pred_valid_mask is not None
        else torch.ones_like(scores, dtype=torch.bool)
    )

    group_radius = max(float(group_radius), 1e-6)
    soft_sigma = max(float(soft_sigma), 1e-6)
    temperature = max(float(temperature), 1e-3)
    losses = []

    for b in range(scores.size(0)):
        valid_idx = valid_mask[b].nonzero(as_tuple=False).squeeze(1)
        gt_idx = gt_mask[b].bool().nonzero(as_tuple=False).squeeze(1)
        if valid_idx.numel() == 0 or gt_idx.numel() == 0:
            continue

        d_to_gt = torch.cdist(pred_pix[b, valid_idx], gt_pix[b, gt_idx], p=2)
        _, nearest_local_gt = d_to_gt.min(dim=1)
        score_b = scores[b, valid_idx]

        for gi in range(gt_idx.numel()):
            local_mask = d_to_gt[:, gi] <= group_radius
            if bool(nearest_gt_only):
                local_mask = local_mask & (nearest_local_gt == gi)
            if not local_mask.any():
                continue

            local_scores = score_b[local_mask]
            local_dist = d_to_gt[local_mask, gi]
            weights = torch.exp(-0.5 * (local_dist / soft_sigma).pow(2)).clamp_min(1e-6)
            weights = weights / weights.sum().clamp_min(1e-6)
            group_score = temperature * torch.logsumexp(
                local_scores / temperature + weights.log(),
                dim=0,
            )
            losses.append(F.softplus(scores.new_tensor(float(pos_logit)) - group_score))

    if len(losses) == 0:
        return scores.new_tensor(0.0)
    return torch.stack(losses).mean()


def soft_local_competition_confidence_loss(
        selector_logits: torch.Tensor,
        pred_points_m11: torch.Tensor,
        gt_points_m11: torch.Tensor,
        gt_mask: torch.Tensor,
        matcher,
        pred_valid_mask: torch.Tensor = None,
        H: int = 256,
        W: int = 256,
        winner_radius: float = 6.0,
        compete_radius: float = 10.0,
        soft_sigma: float = 4.0,
        bg_radius: float = 12.0,
        margin: float = 1.0,
        neg_topk: int = 64,
        temperature: float = 1.0,
        nearest_gt_only: bool = True,
        pos_logit: float = 2.0,
        duplicate_neg_logit: float = 0.0,
        bg_neg_logit: float = -0.5,
        pos_weight: float = 0.5,
        rank_weight: float = 2.0,
        duplicate_weight: float = 0.5,
        softmax_weight: float = 1.0,
        bg_weight: float = 0.25,
) -> torch.Tensor:
    """
    Soft GT-centered local competition for selector confidence.

    A coordinate-only Hungarian match picks one winner per GT when a candidate is
    close enough. Nearby non-winner candidates become soft duplicate negatives
    with distance-decayed weights, so no hard local grouping is required.
    """
    import torch.nn.functional as F

    scores = object_score_from_logits(selector_logits)
    pred_pix = m11_to_pixels_batch(pred_points_m11.detach(), H, W)
    gt_pix = m11_to_pixels_batch(gt_points_m11.detach(), H, W)
    valid_mask = (
        pred_valid_mask.bool()
        if pred_valid_mask is not None
        else torch.ones_like(scores, dtype=torch.bool)
    )

    winner_radius = max(float(winner_radius), 1e-6)
    compete_radius = max(float(compete_radius), winner_radius)
    soft_sigma = max(float(soft_sigma), 1e-6)
    bg_radius = max(float(bg_radius), compete_radius)
    margin = float(margin)
    neg_topk = max(1, int(neg_topk))
    temperature = max(float(temperature), 1e-3)

    pos_losses = []
    rank_losses = []
    duplicate_losses = []
    softmax_losses = []
    bg_losses = []

    for b in range(scores.size(0)):
        score_b = scores[b]
        valid_idx = valid_mask[b].nonzero(as_tuple=False).squeeze(1)
        if valid_idx.numel() == 0:
            continue

        gt_idx = gt_mask[b].bool().nonzero(as_tuple=False).squeeze(1)
        if gt_idx.numel() == 0:
            bg_scores = score_b[valid_idx]
            k_bg = min(neg_topk, bg_scores.numel())
            bg_losses.append(
                F.softplus(
                    bg_scores.topk(k=k_bg, largest=True).values
                    - score_b.new_tensor(float(bg_neg_logit))
                ).mean()
            )
            continue

        d_to_gt = torch.cdist(pred_pix[b, valid_idx], gt_pix[b, gt_idx], p=2)
        dmin, nearest_local_gt = d_to_gt.min(dim=1)

        match_logits = score_b.new_zeros((1, valid_idx.numel(), 2))
        src_local, tgt_global = matcher(
            match_logits,
            pred_points_m11[b, valid_idx].detach().unsqueeze(0),
            gt_points_m11[b:b + 1],
            gt_mask[b:b + 1],
        )[0]

        winners = []
        if src_local.numel() > 0:
            for src_i, tgt_i in zip(src_local, tgt_global):
                tgt_local = (gt_idx == tgt_i).nonzero(as_tuple=False).squeeze(1)
                if tgt_local.numel() == 0:
                    continue
                tgt_local_i = int(tgt_local[0].item())
                src_i = int(src_i.item())
                if d_to_gt[src_i, tgt_local_i] <= winner_radius:
                    winners.append((valid_idx[src_i], tgt_local_i))

        winner_mask_b = torch.zeros_like(score_b, dtype=torch.bool)
        for pred_i, _ in winners:
            winner_mask_b[pred_i] = True

        for pred_i, tgt_local_i in winners:
            pos_score = score_b[pred_i]
            pos_losses.append(F.softplus(score_b.new_tensor(float(pos_logit)) - pos_score))

            local_dist = d_to_gt[:, tgt_local_i]
            local_mask = (
                (local_dist <= compete_radius)
                & (valid_idx != pred_i)
                & (~winner_mask_b[valid_idx])
            )
            if bool(nearest_gt_only):
                local_mask = local_mask & (nearest_local_gt == tgt_local_i)
            if not local_mask.any():
                continue

            neg_scores = score_b[valid_idx[local_mask]]
            weights = torch.exp(-0.5 * (local_dist[local_mask] / soft_sigma).pow(2))
            weights = weights.clamp_min(1e-4)

            k_neg = min(neg_topk, neg_scores.numel())
            hard_order = neg_scores.topk(k=k_neg, largest=True).indices
            neg_scores = neg_scores[hard_order]
            weights = weights[hard_order]
            weight_norm = weights.sum().clamp_min(1e-6)

            rank_losses.append(
                (torch.relu(margin + neg_scores - pos_score) * weights).sum() / weight_norm
            )
            duplicate_losses.append(
                (F.softplus(neg_scores - score_b.new_tensor(float(duplicate_neg_logit))) * weights).sum()
                / weight_norm
            )
            local_scores = torch.cat([pos_score.unsqueeze(0), neg_scores], dim=0) / temperature
            target = torch.zeros(1, dtype=torch.long, device=score_b.device)
            softmax_losses.append(F.cross_entropy(local_scores.unsqueeze(0), target))

        bg_mask = valid_mask[b] & (~winner_mask_b)
        if dmin.numel() > 0:
            bg_mask[valid_idx] = bg_mask[valid_idx] & (dmin >= bg_radius)
        if bg_mask.any():
            bg_scores = score_b[bg_mask]
            k_bg = min(neg_topk, bg_scores.numel())
            bg_losses.append(
                F.softplus(
                    bg_scores.topk(k=k_bg, largest=True).values
                    - score_b.new_tensor(float(bg_neg_logit))
                ).mean()
            )

    total = scores.new_tensor(0.0)

    def add_component(losses, weight):
        if not losses or float(weight) <= 0.0:
            return scores.new_tensor(0.0)
        return float(weight) * torch.stack(losses).mean()

    total = total + add_component(pos_losses, pos_weight)
    total = total + add_component(rank_losses, rank_weight)
    total = total + add_component(duplicate_losses, duplicate_weight)
    total = total + add_component(softmax_losses, softmax_weight)
    total = total + add_component(bg_losses, bg_weight)
    return total


def group_quota_confidence_loss(
        selector_logits: torch.Tensor,
        pred_points_m11: torch.Tensor,
        gt_points_m11: torch.Tensor,
        gt_mask: torch.Tensor,
        pred_valid_mask: torch.Tensor = None,
        H: int = 256,
        W: int = 256,
        group_radius: float = 6.0,
        bg_radius: float = 14.0,
        quota_target: float = 1.0,
        margin: float = 1.0,
        neg_topk: int = 64,
        nearest_gt_only: bool = True,
        pos_logit: float = 2.0,
        duplicate_neg_logit: float = 0.0,
        bg_neg_logit: float = -0.5,
        pos_weight: float = 0.5,
        quota_weight: float = 3.0,
        over_weight: float = 1.0,
        under_weight: float = 0.25,
        gap_weight: float = 2.0,
        duplicate_weight: float = 1.5,
        bg_weight: float = 0.25,
) -> torch.Tensor:
    """
    GT-centered quota loss for no-NMS selector training.

    For each GT-local group, the strongest candidate is treated as the current
    winner and the group is penalized when the total object probability exceeds
    roughly one slot. This complements group_pos: group_pos keeps recall alive,
    quota/gap/duplicate terms make duplicates in the same local group expensive.
    """
    import torch.nn.functional as F

    scores = object_score_from_logits(selector_logits)
    pred_pix = m11_to_pixels_batch(pred_points_m11.detach(), H, W)
    gt_pix = m11_to_pixels_batch(gt_points_m11.detach(), H, W)
    valid_mask = (
        pred_valid_mask.bool()
        if pred_valid_mask is not None
        else torch.ones_like(scores, dtype=torch.bool)
    )

    group_radius = max(float(group_radius), 1e-6)
    bg_radius = max(float(bg_radius), group_radius)
    quota_target = max(float(quota_target), 1e-6)
    margin = float(margin)
    neg_topk = max(1, int(neg_topk))

    pos_losses = []
    quota_losses = []
    gap_losses = []
    duplicate_losses = []
    bg_losses = []

    for b in range(scores.size(0)):
        score_b_all = scores[b]
        valid_idx = valid_mask[b].nonzero(as_tuple=False).squeeze(1)
        if valid_idx.numel() == 0:
            continue

        gt_idx = gt_mask[b].bool().nonzero(as_tuple=False).squeeze(1)
        if gt_idx.numel() == 0:
            bg_scores = score_b_all[valid_idx]
            k_bg = min(neg_topk, bg_scores.numel())
            bg_losses.append(
                F.softplus(
                    bg_scores.topk(k=k_bg, largest=True).values
                    - score_b_all.new_tensor(float(bg_neg_logit))
                ).mean()
            )
            continue

        d_to_gt = torch.cdist(pred_pix[b, valid_idx], gt_pix[b, gt_idx], p=2)
        dmin, nearest_local_gt = d_to_gt.min(dim=1)
        score_b = score_b_all[valid_idx]
        prob_b = torch.sigmoid(score_b)

        for gi in range(gt_idx.numel()):
            group_mask = d_to_gt[:, gi] <= group_radius
            if bool(nearest_gt_only):
                group_mask = group_mask & (nearest_local_gt == gi)
            if not group_mask.any():
                continue

            group_scores = score_b[group_mask]
            group_probs = prob_b[group_mask]
            top_vals, top_indices = group_scores.topk(
                k=min(2, group_scores.numel()),
                largest=True,
            )
            winner_score = top_vals[0]
            pos_losses.append(
                F.softplus(score_b_all.new_tensor(float(pos_logit)) - winner_score)
            )

            mass = group_probs.sum()
            over = torch.relu(mass - score_b_all.new_tensor(quota_target))
            under = torch.relu(score_b_all.new_tensor(quota_target) - mass)
            quota_losses.append(
                float(over_weight) * over.pow(2)
                + float(under_weight) * under.pow(2)
            )

            if top_vals.numel() > 1:
                gap_losses.append(
                    torch.relu(score_b_all.new_tensor(margin) + top_vals[1] - top_vals[0])
                )

            if group_scores.numel() > 1:
                dup_mask = torch.ones(
                    group_scores.shape[0],
                    dtype=torch.bool,
                    device=group_scores.device,
                )
                dup_mask[top_indices[0]] = False
                dup_scores = group_scores[dup_mask]
                k_dup = min(neg_topk, dup_scores.numel())
                hard_dup = dup_scores.topk(k=k_dup, largest=True).values
                duplicate_losses.append(
                    F.softplus(
                        hard_dup - score_b_all.new_tensor(float(duplicate_neg_logit))
                    ).mean()
                )

        bg_mask_local = dmin >= bg_radius
        if bg_mask_local.any():
            bg_scores = score_b[bg_mask_local]
            k_bg = min(neg_topk, bg_scores.numel())
            bg_losses.append(
                F.softplus(
                    bg_scores.topk(k=k_bg, largest=True).values
                    - score_b_all.new_tensor(float(bg_neg_logit))
                ).mean()
            )

    total = scores.new_tensor(0.0)

    def add_component(losses, weight):
        if not losses or float(weight) <= 0.0:
            return scores.new_tensor(0.0)
        return float(weight) * torch.stack(losses).mean()

    total = total + add_component(pos_losses, pos_weight)
    total = total + add_component(quota_losses, quota_weight)
    total = total + add_component(gap_losses, gap_weight)
    total = total + add_component(duplicate_losses, duplicate_weight)
    total = total + add_component(bg_losses, bg_weight)
    return total


@torch.no_grad()
def point_nms_count(xy_pix: torch.Tensor, score: torch.Tensor, r: float) -> int:
    """
    xy_pix: [M,2] (pixel)
    score : [M]
    r     : radius in pixels
    return: kept count after greedy NMS
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
        # 跟已保留點算距離（平方距離比較快）
        prev = xy[keep]  # [K,2]
        d2 = ((prev - xy[i]).pow(2).sum(dim=1))  # [K]
        if (d2 >= r2).all():
            keep.append(i)
    return len(keep)

@torch.no_grad()
@preserve_rng_state
def validate_one_epoch(
        model,
        data_loader,
        device,
        sched,
        criterion,
        T: int = 1000,
        seed: int = 7113064165,
        hard_thresh: float = 0.0,
        ddim_steps: int = 50,
        nms_radius: float = 3.0,
        ddim_eta: float = 0.0,
        test_gate_mode: str = "prob_only",
        threshold_sweep=None,
        val_ddim_cover_radius: float = 6.0,
        val_ddim_candidate_low_score_thresh: float = 0.01,
        val_num_realizations: int = 1,
        use_proposal_prior: bool = False,
        proposal_prior_start_t: int = 700,
        proposal_prior_sigma: float = 1.25,
        proposal_prior_mode: str = "occupancy",
        proposal_prior_cell_capacity: int = 2,
        lambda_prior_occupancy: float = 0.0,
        lambda_prior_density: float = 0.0,
        lambda_prior_count: float = 0.0,
):
    model.eval()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if int(val_num_realizations) != 1:
        raise ValueError("validate_one_epoch currently supports val_num_realizations=1 only")

    # ---- supervised loss 統計 ----
    total_loss = 0.0
    n_steps = 0
    run_Lcnt = run_Lexist = run_Lx0 = run_Lbg = run_Leps = run_Lcov = run_Lcover = run_Ldup = run_Lcollapse = 0.0
    run_Lprior_occ = run_Lprior_density = run_Lprior_count = 0.0
    prior_count_abs_sum = 0.0
    prior_count_samples = 0
    guided_slot_sum = 0
    guided_slot_samples = 0

    # ---- 全圖聚合容器（overlap stride 專用）----
    pred_xy_full = defaultdict(list)   # img_idx -> [tensor(M,2), ...]  pixel coords in full image
    pred_sc_full = defaultdict(list)   # img_idx -> [tensor(M,), ...]
    raw_xy_full = defaultdict(list)
    candidate_xy_full = defaultdict(list)
    prior_guided_xy_full = defaultdict(list)
    prior_uniform_xy_full = defaultdict(list)
    gt_xy_full   = defaultdict(list)   # img_idx -> [tensor(K,2), ...]  pixel coords in full image

    # 固定 sampler 設定
    steps = int(ddim_steps)
    thr = float(hard_thresh)
    nms_r = float(nms_radius)
    eta = float(ddim_eta)
    gate_mode = str(test_gate_mode)
    clamp_eps = 1e-3

    def _parse_threshold_sweep(values):
        if values is None or values is False:
            return [thr], False
        if isinstance(values, str):
            raw = values.strip()
            if raw == "" or raw.lower() in {"none", "false", "off"}:
                return [thr], False
            parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
        else:
            parts = list(values)
        parsed = sorted({float(v) for v in parts})
        return (parsed if parsed else [thr]), bool(parsed)

    threshold_values, do_threshold_sweep = _parse_threshold_sweep(threshold_sweep)

    abar_all = sched.abar.to(device=device)

    # ---------------- helper: dedup GT ----------------
    def dedup_points_xy(xy_np, decimals=3):
        if xy_np.shape[0] == 0:
            return xy_np
        key = np.round(xy_np, decimals=decimals)
        _, idx = np.unique(key, axis=0, return_index=True)
        return xy_np[np.sort(idx)]

    # ---------------- main loop ----------------
    with torch.no_grad():
        for images, points_pad, mask, metas in data_loader:
            images     = images.to(device, non_blocking=True)
            points_pad = points_pad.to(device, non_blocking=True)
            mask       = mask.to(device, non_blocking=True)

            B, C, H, W = images.shape
            N = points_pad.size(1)

            # encode once
            feats = model.encode(images)
            cond_cache = model.cond.precompute(*feats)
            prior_occupancy_logits = None
            prior_density = None
            if bool(use_proposal_prior):
                prior_occupancy_logits, prior_density = model.predict_proposal_prior(feats)
                prior_guided_batch = []
                prior_uniform_batch = []
                for b in range(B):
                    k_guided = int(
                        torch.round(prior_density[b].sum())
                        .clamp(0, int(N))
                        .item()
                    )
                    guided_xy = select_guided_prior_points(
                        prior_occupancy_logits[b],
                        prior_density[b],
                        k_guided,
                        mode=str(proposal_prior_mode),
                        density_cell_capacity=int(proposal_prior_cell_capacity),
                    )
                    k_guided = int(guided_xy.size(0))
                    uniform_xy = prior_occupancy_logits.new_empty(
                        (k_guided, 2)
                    ).uniform_(-1.0 + clamp_eps, 1.0 - clamp_eps)
                    prior_guided_batch.append(guided_xy)
                    prior_uniform_batch.append(uniform_xy)

            # pixel -> [-1,1]
            p0 = pixels_to_m11(points_pad, H, W)

            # ---------- (1) 單步 supervised：對齊你原本的 Lex/Lx0/Lcnt/Leps ----------
            t_fixed = 500  # 你想固定的t，0~T-1
            t_int = torch.full((B, 1), t_fixed, device=device, dtype=torch.long)
            p_t, _, abar_t_ = forward_noisy(p0, t_int, sched)

            if abar_t_.dim() == 1:
                abar_t = abar_t_.view(B, 1, 1)
            elif abar_t_.dim() == 2:
                abar_t = abar_t_.view(B, 1, 1)
            else:
                abar_t = abar_t_
            abar_t = abar_t.to(device=device)

            eps_pred, exist_logit, pro, pred_points_for_cls, pred_valid_mask = model.denoise(
                feats, p_t, t_int,
                abar_t=abar_t,
                clamp_eps=1e-6,
                cond_cache=cond_cache,
                need_exist=True,
                selector_prior_maps=(
                    (prior_occupancy_logits, prior_density)
                    if prior_density is not None
                    else None
                ),
            )

            if exist_logit is None:
                raise RuntimeError("validate supervised branch got None exist_logit")

            if exist_logit.dim() == 3 and exist_logit.size(-1) == 1:
                exist_logit = exist_logit.squeeze(-1)

            loss, L_exist, L_x0, L_cnt, L_bg, L_eps, L_cov, L_cover, L_dup, L_collapse = criterion(
                p_t=p_t,
                p0=p0,
                mask=mask,
                pro=pro,
                abar_t=abar_t,
                eps_pred=eps_pred,
                exist_logit=exist_logit,
                pred_points_for_cls=pred_points_for_cls,
                pred_valid_mask=pred_valid_mask,
                lambda_t=None,
                aux_weight=1.0,
            )

            if bool(use_proposal_prior):
                occ_target, density_target, gt_count_prior = build_proposal_prior_targets(
                    points_pad,
                    mask,
                    H,
                    W,
                    prior_density.size(-2),
                    prior_density.size(-1),
                    sigma=float(proposal_prior_sigma),
                )
                Lprior_occ, Lprior_density, Lprior_count, pred_count_prior = proposal_prior_loss(
                    prior_occupancy_logits,
                    prior_density,
                    occ_target,
                    density_target,
                    gt_count_prior,
                )
                loss = (
                    loss
                    + float(lambda_prior_occupancy) * Lprior_occ
                    + float(lambda_prior_density) * Lprior_density
                    + float(lambda_prior_count) * Lprior_count
                )
                run_Lprior_occ += float(Lprior_occ)
                run_Lprior_density += float(Lprior_density)
                run_Lprior_count += float(Lprior_count)
                prior_count_abs_sum += float((pred_count_prior - gt_count_prior).abs().sum())
                prior_count_samples += int(B)

            total_loss += float(loss)
            n_steps += 1
            run_Lexist += float(L_exist)
            run_Lx0    += float(L_x0)
            run_Lcnt   += float(L_cnt)
            run_Leps   += float(L_eps)
            run_Lbg += float(L_bg)
            run_Lcov += float(L_cov)
            run_Lcover += float(L_cover)
            run_Ldup += float(L_dup)
            run_Lcollapse += float(L_collapse)
            # ---------- (2) 多步 DDIM sampling：收集全圖候選點 ----------
            start_t = (
                max(0, min(int(proposal_prior_start_t), int(T) - 1))
                if bool(use_proposal_prior)
                else int(T) - 1
            )
            t_seq = torch.linspace(start_t, 0, steps, device=device).round().long()
            t_seq = torch.unique_consecutive(t_seq)

            if bool(use_proposal_prior):
                x0_prior, guided_counts = build_mixed_x0_prior(
                    prior_occupancy_logits,
                    prior_density,
                    num_slots=N,
                    clamp_eps=clamp_eps,
                    mode=str(proposal_prior_mode),
                    density_cell_capacity=int(proposal_prior_cell_capacity),
                )
                t_init = torch.full(
                    (B, 1),
                    int(t_seq[0].item()),
                    device=device,
                    dtype=torch.long,
                )
                p_t_gen, _, _ = forward_noisy(x0_prior, t_init, sched)
                guided_slot_sum += int(guided_counts.sum().item())
                guided_slot_samples += int(B)
            else:
                p_t_gen = torch.empty((B, N, 2), device=device).uniform_(
                    -1.0 + clamp_eps, 1.0 - clamp_eps
                )

            exist_logit_x0 = None
            pred_points_x0 = None
            pred_valid_x0 = None
            raw_points_x0 = None
            eps = 1e-12

            for si, ti in enumerate(t_seq.tolist()):
                ti = int(ti)
                t_tensor = torch.full((B, 1), ti, device=device, dtype=torch.long)
                abar_ti = abar_all[ti].view(1, 1, 1).expand(B, 1, 1)

                need_exist = (si == len(t_seq) - 1)
                eps_hat, exist_logit_t, pro, pred_points_for_cls, pred_valid_mask  = model.denoise(
                    feats, p_t_gen, t_tensor,
                    abar_t=abar_ti, clamp_eps=1e-6,
                    cond_cache=cond_cache,
                    need_exist=need_exist,
                    selector_prior_maps=(
                        (prior_occupancy_logits, prior_density)
                        if prior_density is not None
                        else None
                    ),
                )
                if need_exist:
                    exist_logit_x0 = exist_logit_t
                    pred_points_x0 = pred_points_for_cls
                    pred_valid_x0 = pred_valid_mask
                    merge_stats_x0 = getattr(model, "last_merge_stats", None)
                    if merge_stats_x0 is not None:
                        raw_points_x0 = merge_stats_x0.get("raw_x0_hat")
                if si + 1 < len(t_seq):
                    ti_prev = int(t_seq[si + 1].item())
                    abar_prev = abar_all[ti_prev].view(1, 1, 1).expand(B, 1, 1)
                    eta_step = eta
                else:
                    abar_prev = torch.ones((B, 1, 1), device=device)
                    eta_step = 0.0

                sqrt_ab_t = abar_ti.clamp(0.0, 1.0).add(eps).sqrt()
                sqrt_om_t = (1.0 - abar_ti).clamp_min(0.0).sqrt()
                x0_hat = (p_t_gen - sqrt_om_t * eps_hat) / sqrt_ab_t

                alpha_t = (abar_ti / (abar_prev + eps)).clamp_min(eps)
                sigma2 = (eta_step ** 2) * (
                    ((1.0 - abar_prev).clamp_min(0.0) / (1.0 - abar_ti).clamp_min(eps))
                    * (1.0 - alpha_t).clamp_min(0.0)
                )
                sigma = sigma2.clamp_min(0.0).sqrt()
                c_eps = (1.0 - abar_prev - sigma2).clamp_min(0.0).sqrt()

                p_t_gen = (abar_prev + eps).sqrt() * x0_hat + c_eps * eps_hat
                if eta_step > 0.0:
                    p_t_gen = p_t_gen + sigma * torch.randn_like(p_t_gen)
                p_t_gen = p_t_gen.clamp(min=-1.0 + clamp_eps, max=1.0 - clamp_eps)

            if exist_logit_x0 is None:
                raise RuntimeError("DDIM last step did not compute exist_logit_x0. Check need_exist logic.")
            if pred_points_x0 is None:
                raise RuntimeError("DDIM last step did not return merged prediction points.")
            if raw_points_x0 is None:
                raise RuntimeError("DDIM validation needs model.last_merge_stats['raw_x0_hat'] for raw cover.")

            if exist_logit_x0.dim() == 3 and exist_logit_x0.size(-1) == 1:
                exist_logit_x0 = exist_logit_x0.squeeze(-1)

            if exist_logit_x0.dim() == 3 and exist_logit_x0.size(-1) == 2:
                exist_prob_sample = exist_logit_x0.softmax(-1)[..., 1]  # [B,N] object prob
                if gate_mode == "argmax_only":
                    pos_mask_sample = exist_logit_x0.argmax(-1) == 1
                elif gate_mode == "argmax_or_prob":
                    pos_mask_sample = (exist_logit_x0.argmax(-1) == 1) | (exist_prob_sample > thr)
                else:
                    pos_mask_sample = exist_prob_sample > thr
            else:
                exist_prob_sample = torch.sigmoid(exist_logit_x0)  # [B,N]
                pos_mask_sample = exist_prob_sample > thr
                print("not cross entropy")
            if pred_valid_x0 is not None:
                pos_mask_sample = pos_mask_sample & pred_valid_x0.bool()

            x0_hat = pred_points_x0.detach()                   # [B,N,2], aligned with exist_prob_sample
            x0_pix = m11_to_pixels_batch(x0_hat, H, W)         # tile pixel coords
            raw_x0_pix = m11_to_pixels_batch(raw_points_x0.detach().to(device), H, W)
            candidate_mask_sample = pos_mask_sample & (
                exist_prob_sample > float(val_ddim_candidate_low_score_thresh)
            )

            # ---- 收集到全圖容器 ----
            for b in range(B):
                meta_b = metas[b] if isinstance(metas, (list, tuple)) else metas
                img_idx = int(meta_b["img_index"])
                top  = float(meta_b["tile_top"])
                left = float(meta_b["tile_left"])
                h_full, w_full = meta_b["orig_size"]

                # GT: tile pixel -> full pixel
                gt_xy_tile = points_pad[b, mask[b]]  # [G,2] tile pixel
                if gt_xy_tile.numel() > 0:
                    gt_xy = gt_xy_tile.clone()
                    gt_xy[:, 0] += left
                    gt_xy[:, 1] += top
                    gt_xy_full[img_idx].append(gt_xy.detach().cpu())

                if bool(use_proposal_prior):
                    guided_xy = m11_to_pixels_batch(
                        prior_guided_batch[b].unsqueeze(0), H, W
                    ).squeeze(0)
                    uniform_xy = m11_to_pixels_batch(
                        prior_uniform_batch[b].unsqueeze(0), H, W
                    ).squeeze(0)
                    if guided_xy.numel() > 0:
                        guided_xy[:, 0] = (
                            guided_xy[:, 0] + left
                        ).clamp(0, float(w_full) - 1)
                        guided_xy[:, 1] = (
                            guided_xy[:, 1] + top
                        ).clamp(0, float(h_full) - 1)
                        prior_guided_xy_full[img_idx].append(
                            guided_xy.detach().cpu()
                        )
                    if uniform_xy.numel() > 0:
                        uniform_xy[:, 0] = (
                            uniform_xy[:, 0] + left
                        ).clamp(0, float(w_full) - 1)
                        uniform_xy[:, 1] = (
                            uniform_xy[:, 1] + top
                        ).clamp(0, float(h_full) - 1)
                        prior_uniform_xy_full[img_idx].append(
                            uniform_xy.detach().cpu()
                        )

                raw_xy = raw_x0_pix[b].clone()
                raw_xy[:, 0] = (raw_xy[:, 0] + left).clamp(0, float(w_full) - 1)
                raw_xy[:, 1] = (raw_xy[:, 1] + top).clamp(0, float(h_full) - 1)
                raw_xy_full[img_idx].append(raw_xy.detach().cpu())

                candidate_mask_b = candidate_mask_sample[b]
                if candidate_mask_b.any():
                    candidate_xy = x0_pix[b, candidate_mask_b].clone()
                    candidate_xy[:, 0] = (candidate_xy[:, 0] + left).clamp(0, float(w_full) - 1)
                    candidate_xy[:, 1] = (candidate_xy[:, 1] + top).clamp(0, float(h_full) - 1)
                    candidate_xy_full[img_idx].append(candidate_xy.detach().cpu())

                # Pred candidates: tile pixel -> full pixel
                prob_b = exist_prob_sample[b]
                if do_threshold_sweep:
                    cand = pred_valid_x0[b].bool() if pred_valid_x0 is not None else torch.ones_like(prob_b, dtype=torch.bool)
                else:
                    cand = pos_mask_sample[b]
                if cand.any():
                    xy_tile = x0_pix[b, cand]      # [M,2]
                    sc      = prob_b[cand]         # [M]
                    xy = xy_tile.clone()
                    xy[:, 0] += left
                    xy[:, 1] += top
                    pred_xy_full[img_idx].append(xy.detach().cpu())
                    pred_sc_full[img_idx].append(sc.detach().cpu())

    # ---------------- after loop: per-image NMS & metrics ----------------
    img_ids = sorted(set(
        list(gt_xy_full.keys())
        + list(pred_xy_full.keys())
        + list(raw_xy_full.keys())
        + list(candidate_xy_full.keys())
        + list(prior_guided_xy_full.keys())
        + list(prior_uniform_xy_full.keys())
    ))
    errs_by_thresh = {float(t): [] for t in threshold_values}
    no_nms_errs_by_thresh = {float(t): [] for t in threshold_values}
    no_nms_totals_by_thresh = {
        float(t): {
            "gt": 0,
            "selected": 0,
            "matched": 0,
            "nearby": 0,
        }
        for t in threshold_values
    }
    total_gt_ddim = 0
    total_raw_matched = 0
    total_candidate_matched = 0
    total_candidate_nearby = 0
    total_candidates = 0
    total_prior_guided_matched = 0
    total_prior_uniform_matched = 0
    total_prior_guided_nearby = 0
    total_prior_uniform_nearby = 0
    total_prior_gt = 0
    total_prior_points = 0

    for img_idx in img_ids:
        # GT count (dedup)
        if len(gt_xy_full[img_idx]) > 0:
            gt_xy = torch.cat(gt_xy_full[img_idx], dim=0).numpy()
            gt_xy = dedup_points_xy(gt_xy, decimals=3)
            gt_cnt = float(gt_xy.shape[0])
        else:
            gt_xy = np.zeros((0, 2), dtype=np.float32)
            gt_cnt = 0.0

        raw_xy = (
            torch.cat(raw_xy_full[img_idx], dim=0).numpy()
            if len(raw_xy_full[img_idx]) > 0
            else np.zeros((0, 2), dtype=np.float32)
        )
        candidate_xy = (
            torch.cat(candidate_xy_full[img_idx], dim=0).numpy()
            if len(candidate_xy_full[img_idx]) > 0
            else np.zeros((0, 2), dtype=np.float32)
        )
        raw_stats = full_image_matching_stats(gt_xy, raw_xy, val_ddim_cover_radius)
        candidate_stats = full_image_matching_stats(gt_xy, candidate_xy, val_ddim_cover_radius)
        total_gt_ddim += int(raw_stats["gt_count"])
        total_raw_matched += int(raw_stats["matched_gt"])
        total_candidate_matched += int(candidate_stats["matched_gt"])
        total_candidate_nearby += int(candidate_stats["nearby_proposals"])
        total_candidates += int(candidate_stats["point_count"])

        if bool(use_proposal_prior):
            guided_prior_xy = (
                torch.cat(prior_guided_xy_full[img_idx], dim=0).numpy()
                if len(prior_guided_xy_full[img_idx]) > 0
                else np.zeros((0, 2), dtype=np.float32)
            )
            uniform_prior_xy = (
                torch.cat(prior_uniform_xy_full[img_idx], dim=0).numpy()
                if len(prior_uniform_xy_full[img_idx]) > 0
                else np.zeros((0, 2), dtype=np.float32)
            )
            guided_prior_stats = full_image_matching_stats(
                gt_xy, guided_prior_xy, val_ddim_cover_radius
            )
            uniform_prior_stats = full_image_matching_stats(
                gt_xy, uniform_prior_xy, val_ddim_cover_radius
            )
            total_prior_gt += int(guided_prior_stats["gt_count"])
            total_prior_guided_matched += int(
                guided_prior_stats["matched_gt"]
            )
            total_prior_uniform_matched += int(
                uniform_prior_stats["matched_gt"]
            )
            total_prior_guided_nearby += int(
                guided_prior_stats["nearby_proposals"]
            )
            total_prior_uniform_nearby += int(
                uniform_prior_stats["nearby_proposals"]
            )
            total_prior_points += int(guided_prior_stats["point_count"])

        # Pred hard count (full-image NMS)
        if len(pred_xy_full[img_idx]) > 0:
            xy = torch.cat(pred_xy_full[img_idx], dim=0)   # (M,2)
            sc = torch.cat(pred_sc_full[img_idx], dim=0)   # (M,)
            for thr_i in threshold_values:
                if do_threshold_sweep:
                    keep = sc > float(thr_i)
                    xy_selected = xy[keep] if keep.any() else xy.new_zeros((0, 2))
                    sc_selected = sc[keep] if keep.any() else sc.new_zeros((0,))
                    pred_cnt_hard = (
                        float(point_nms_count(xy_selected, sc_selected, r=nms_r))
                        if keep.any()
                        else 0.0
                    )
                else:
                    xy_selected = xy
                    pred_cnt_hard = float(point_nms_count(xy, sc, r=nms_r))
                errs_by_thresh[float(thr_i)].append(pred_cnt_hard - gt_cnt)

                pred_cnt_no_nms = float(xy_selected.size(0))
                no_nms_errs_by_thresh[float(thr_i)].append(pred_cnt_no_nms - gt_cnt)
                xy_selected_np = (
                    xy_selected.numpy()
                    if xy_selected.numel() > 0
                    else np.zeros((0, 2), dtype=np.float32)
                )
                no_nms_stats = full_image_matching_stats(
                    gt_xy,
                    xy_selected_np,
                    val_ddim_cover_radius,
                )
                no_nms_totals = no_nms_totals_by_thresh[float(thr_i)]
                no_nms_totals["gt"] += int(no_nms_stats["gt_count"])
                no_nms_totals["selected"] += int(no_nms_stats["point_count"])
                no_nms_totals["matched"] += int(no_nms_stats["matched_gt"])
                no_nms_totals["nearby"] += int(no_nms_stats["nearby_proposals"])
        else:
            for thr_i in threshold_values:
                errs_by_thresh[float(thr_i)].append(-gt_cnt)
                no_nms_errs_by_thresh[float(thr_i)].append(-gt_cnt)
                no_nms_totals = no_nms_totals_by_thresh[float(thr_i)]
                no_nms_totals["gt"] += int(gt_xy.shape[0])

    thresh_metrics = []
    for thr_i in threshold_values:
        errs_hard = np.array(errs_by_thresh[float(thr_i)], dtype=np.float32)
        mae_i = float(np.mean(np.abs(errs_hard))) if errs_hard.size else 0.0
        rmse_i = float(np.sqrt(np.mean(errs_hard**2))) if errs_hard.size else 0.0
        thresh_metrics.append((mae_i, rmse_i, float(thr_i)))

    mae_hard_img, rmse_hard_img, best_thresh = min(thresh_metrics, key=lambda x: x[0])

    no_nms_thresh_metrics = []
    for thr_i in threshold_values:
        thr_key = float(thr_i)
        errs_no_nms = np.array(no_nms_errs_by_thresh[thr_key], dtype=np.float32)
        no_nms_mae_i = (
            float(np.mean(np.abs(errs_no_nms))) if errs_no_nms.size else 0.0
        )
        no_nms_rmse_i = (
            float(np.sqrt(np.mean(errs_no_nms**2))) if errs_no_nms.size else 0.0
        )
        no_nms_totals = no_nms_totals_by_thresh[thr_key]
        no_nms_gt = int(no_nms_totals["gt"])
        no_nms_selected = int(no_nms_totals["selected"])
        no_nms_matched = int(no_nms_totals["matched"])
        no_nms_nearby = int(no_nms_totals["nearby"])
        no_nms_thresh_metrics.append((
            no_nms_mae_i,
            no_nms_rmse_i,
            thr_key,
            float(no_nms_matched / max(no_nms_gt, 1)),
            float(no_nms_matched / max(no_nms_selected, 1)),
            float(no_nms_nearby / max(no_nms_gt, 1)),
            float(no_nms_selected / max(no_nms_gt, 1)),
            no_nms_gt,
            no_nms_selected,
            no_nms_matched,
        ))

    (
        no_nms_mae_img,
        no_nms_rmse_img,
        no_nms_best_thresh,
        no_nms_recall,
        no_nms_precision,
        no_nms_dup,
        no_nms_selected_per_gt,
        no_nms_total_gt,
        no_nms_total_selected,
        no_nms_total_matched,
    ) = min(no_nms_thresh_metrics, key=lambda x: x[0])

    val_ddim_metrics = {
        "val_ddim_raw_cover@6": float(total_raw_matched / max(total_gt_ddim, 1)),
        "val_ddim_candidate_cover@6": float(total_candidate_matched / max(total_gt_ddim, 1)),
        "val_ddim_dup@6": float(total_candidate_nearby / max(total_gt_ddim, 1)),
        "val_ddim_candidates_per_gt": float(total_candidates / max(total_gt_ddim, 1)),
        "val_ddim_total_gt": int(total_gt_ddim),
        "val_ddim_total_candidates": int(total_candidates),
        "val_conf_no_nms_mae": float(no_nms_mae_img),
        "val_conf_no_nms_rmse": float(no_nms_rmse_img),
        "val_conf_no_nms_thr": float(no_nms_best_thresh),
        "val_conf_no_nms_recall@6": float(no_nms_recall),
        "val_conf_no_nms_precision@6": float(no_nms_precision),
        "val_conf_no_nms_dup@6": float(no_nms_dup),
        "val_conf_no_nms_selected_per_gt": float(no_nms_selected_per_gt),
        "val_conf_no_nms_total_gt": int(no_nms_total_gt),
        "val_conf_no_nms_total_selected": int(no_nms_total_selected),
        "val_conf_no_nms_total_matched": int(no_nms_total_matched),
        "val_prior_count_mae": float(
            prior_count_abs_sum / max(prior_count_samples, 1)
        ),
        "val_prior_guided_slots": float(
            guided_slot_sum / max(guided_slot_samples, 1)
        ),
        "val_prior_guided_cover@6": float(
            total_prior_guided_matched / max(total_prior_gt, 1)
        ),
        "val_prior_uniform_cover@6": float(
            total_prior_uniform_matched / max(total_prior_gt, 1)
        ),
        "val_prior_cover_gain@6": float(
            (total_prior_guided_matched - total_prior_uniform_matched)
            / max(total_prior_gt, 1)
        ),
        "val_prior_guided_dup@6": float(
            total_prior_guided_nearby / max(total_prior_gt, 1)
        ),
        "val_prior_uniform_dup@6": float(
            total_prior_uniform_nearby / max(total_prior_gt, 1)
        ),
        "val_prior_points_per_gt": float(
            total_prior_points / max(total_prior_gt, 1)
        ),
    }

    # supervised averages
    if n_steps > 0:
        avg_loss   = total_loss / n_steps
        avg_Lexist = run_Lexist / n_steps
        avg_Lx0    = run_Lx0 / n_steps
        avg_Lcnt   = run_Lcnt / n_steps
        avg_Leps   = run_Leps / n_steps
        avg_Lbg = run_Lbg / n_steps
        avg_Lcov = run_Lcov / n_steps
        avg_Lcover = run_Lcover / n_steps
        avg_Ldup = run_Ldup / n_steps
        avg_Lcollapse = run_Lcollapse / n_steps
        avg_Lprior_occ = run_Lprior_occ / n_steps
        avg_Lprior_density = run_Lprior_density / n_steps
        avg_Lprior_count = run_Lprior_count / n_steps
    else:
        avg_loss = avg_Lexist = avg_Lx0 = avg_Lcnt = avg_Leps = avg_Lbg = avg_Lcov = avg_Lcover = avg_Ldup = avg_Lcollapse = 0.0
        avg_Lprior_occ = avg_Lprior_density = avg_Lprior_count = 0.0

    logging.info(
        f"[val] loss={avg_loss:.4f} Lex={avg_Lexist:.4f} Lx0={avg_Lx0:.4f} "
        f"Lcnt={avg_Lcnt:.4f} Leps={avg_Leps:.4f} Lbg={avg_Lbg:.4f} "
        f"Lcov={avg_Lcov:.4f} Lcover={avg_Lcover:.4f} Ldup={avg_Ldup:.4f} "
        f"Lcollapse={avg_Lcollapse:.4f} | "
        f"Lprior_occ={avg_Lprior_occ:.4f} "
        f"Lprior_density={avg_Lprior_density:.4f} "
        f"Lprior_count={avg_Lprior_count:.4f} | "
        f"FULL-IMG hard: MAE={mae_hard_img:.2f} RMSE={rmse_hard_img:.2f} "
        f"thr={best_thresh:.3f} sweep={int(do_threshold_sweep)} (N={len(img_ids)}) | "
        f"DDIM steps={steps} realizations=1 "
        f"val_ddim_raw_cover@6={val_ddim_metrics['val_ddim_raw_cover@6']:.4f} "
        f"val_ddim_candidate_cover@6={val_ddim_metrics['val_ddim_candidate_cover@6']:.4f} "
        f"val_ddim_dup@6={val_ddim_metrics['val_ddim_dup@6']:.4f} "
        f"val_ddim_candidates_per_gt={val_ddim_metrics['val_ddim_candidates_per_gt']:.4f} "
        f"val_conf_no_nms_mae={val_ddim_metrics['val_conf_no_nms_mae']:.2f} "
        f"val_conf_no_nms_rmse={val_ddim_metrics['val_conf_no_nms_rmse']:.2f} "
        f"val_conf_no_nms_thr={val_ddim_metrics['val_conf_no_nms_thr']:.3f} "
        f"val_conf_no_nms_recall@6={val_ddim_metrics['val_conf_no_nms_recall@6']:.4f} "
        f"val_conf_no_nms_precision@6={val_ddim_metrics['val_conf_no_nms_precision@6']:.4f} "
        f"val_conf_no_nms_dup@6={val_ddim_metrics['val_conf_no_nms_dup@6']:.4f} "
        f"val_conf_no_nms_selected_per_gt={val_ddim_metrics['val_conf_no_nms_selected_per_gt']:.4f} "
        f"val_prior_count_mae={val_ddim_metrics['val_prior_count_mae']:.4f} "
        f"val_prior_guided_slots={val_ddim_metrics['val_prior_guided_slots']:.2f} "
        f"val_prior_guided_cover@6={val_ddim_metrics['val_prior_guided_cover@6']:.4f} "
        f"val_prior_uniform_cover@6={val_ddim_metrics['val_prior_uniform_cover@6']:.4f} "
        f"val_prior_cover_gain@6={val_ddim_metrics['val_prior_cover_gain@6']:.4f} "
        f"val_prior_guided_dup@6={val_ddim_metrics['val_prior_guided_dup@6']:.4f} "
        f"val_prior_uniform_dup@6={val_ddim_metrics['val_prior_uniform_dup@6']:.4f} "
        f"val_prior_points_per_gt={val_ddim_metrics['val_prior_points_per_gt']:.4f}"
    )

    return avg_loss, mae_hard_img, val_ddim_metrics






def train_one_epoch(
        model,
        data_loader,
        device,
        optimizer,
        criterion,
        scaler,
        sched,
        T: int = 1000,
        K: int = 10,                 # 保留介面相容，這版不使用
        log_every: int = 10,
        max_norm: float = 1.0,
        lambda_gt_loss: float = 1.0,
        lambda_cnt_val: float = 0.00,
        lambda_rand_cover: float = 0.0,
        lambda_rand_match: float = 0.0,
        lambda_rand_exist: float = 0.0,
        lambda_rand_count: float = 0.0,
        lambda_rand_count_margin: float = 0.0,
        lambda_rand_bg: float = 0.0,
        lambda_rand_dup: float = 0.0,
        lambda_rand_rank: float = 0.0,
        lambda_rand_one_to_one_conf: float = 0.0,
        lambda_rand_group_pos: float = 0.0,
        lambda_rand_soft_compete: float = 0.0,
        lambda_rand_group_quota: float = 0.0,
        rand_cover_t_min: int = 700,
        rand_cover_t_max: int = 999,
        rand_cover_radius: float = 6.0,
        rand_bg_ignore_radius: float = 6.0,
        rand_dup_radius: float = 6.0,
        rand_dup_topk: int = 12,
        rand_dup_dense_aware: bool = False,
        rand_dup_neighbor_radius: float = 6.0,
        rand_dup_allow_extra: int = 0,
        rand_rank_near_radius: float = 6.0,
        rand_rank_far_radius: float = 12.0,
        rand_rank_margin: float = 1.0,
        rand_rank_neg_topk: int = 96,
        rand_rank_far_weight: float = 0.25,
        rand_one_to_one_radius: float = 6.0,
        rand_one_to_one_bg_radius: float = 12.0,
        rand_one_to_one_margin: float = 1.0,
        rand_one_to_one_neg_topk: int = 128,
        rand_one_to_one_count_pos_logit: float = 2.0,
        rand_one_to_one_count_neg_logit: float = -2.0,
        rand_one_to_one_count_slack: int = 0,
        rand_one_to_one_pos_logit: float = 0.0,
        rand_one_to_one_dup_neg_logit: float = 0.0,
        rand_one_to_one_bg_neg_logit: float = 0.0,
        rand_one_to_one_pos_weight: float = 1.0,
        rand_one_to_one_dup_weight: float = 3.0,
        rand_one_to_one_bg_weight: float = 0.5,
        rand_one_to_one_rank_weight: float = 3.0,
        rand_one_to_one_count_weight: float = 1.0,
        rand_group_pos_radius: float = 6.0,
        rand_group_pos_sigma: float = 4.0,
        rand_group_pos_temperature: float = 0.7,
        rand_group_pos_logit: float = 2.0,
        rand_group_pos_nearest_gt_only: bool = True,
        rand_soft_compete_winner_radius: float = 6.0,
        rand_soft_compete_radius: float = 10.0,
        rand_soft_compete_sigma: float = 4.0,
        rand_soft_compete_bg_radius: float = 12.0,
        rand_soft_compete_margin: float = 1.0,
        rand_soft_compete_neg_topk: int = 64,
        rand_soft_compete_temperature: float = 1.0,
        rand_soft_compete_nearest_gt_only: bool = True,
        rand_soft_compete_pos_logit: float = 2.0,
        rand_soft_compete_dup_neg_logit: float = 0.0,
        rand_soft_compete_bg_neg_logit: float = -0.5,
        rand_soft_compete_pos_weight: float = 0.5,
        rand_soft_compete_rank_weight: float = 2.0,
        rand_soft_compete_dup_weight: float = 0.5,
        rand_soft_compete_softmax_weight: float = 1.0,
        rand_soft_compete_bg_weight: float = 0.25,
        rand_group_quota_radius: float = 6.0,
        rand_group_quota_bg_radius: float = 14.0,
        rand_group_quota_target: float = 1.0,
        rand_group_quota_margin: float = 1.0,
        rand_group_quota_neg_topk: int = 64,
        rand_group_quota_nearest_gt_only: bool = True,
        rand_group_quota_pos_logit: float = 2.0,
        rand_group_quota_dup_neg_logit: float = 0.0,
        rand_group_quota_bg_neg_logit: float = -0.5,
        rand_group_quota_pos_weight: float = 0.5,
        rand_group_quota_weight: float = 3.0,
        rand_group_quota_over_weight: float = 1.0,
        rand_group_quota_under_weight: float = 0.25,
        rand_group_quota_gap_weight: float = 2.0,
        rand_group_quota_dup_weight: float = 1.5,
        rand_group_quota_bg_weight: float = 0.25,
        rand_count_pos_margin: float = 0.35,
        rand_count_neg_margin: float = 0.10,
        rand_count_neg_topk: int = 64,
        rand_count_slack: int = 0,
        use_proposal_prior: bool = False,
        proposal_prior_sigma: float = 1.25,
        proposal_prior_mode: str = "occupancy",
        proposal_prior_cell_capacity: int = 2,
        lambda_prior_occupancy: float = 0.0,
        lambda_prior_density: float = 0.0,
        lambda_prior_count: float = 0.0,
):
    """
    單步 x0 訓練：
    - 每張圖隨機抽一個 t ∈ [0, T-1]
    - 從 p0 前向加噪得到 p_t
    - 模型只做一次 denoise
    - 以 Lx0 / Lex 為主，Leps 為輔助
    """
    import torch
    import torch.nn.functional as F
    from torch.cuda.amp import autocast

    model.train()
    if bool(getattr(model, "freeze_base_for_prior", False)):
        model.eval()
        model.proposal_prior_head.train()
    elif bool(getattr(model, "freeze_selector_only", False)):
        model.eval()
        for module_name in getattr(model, "selector_finetune_module_names", ()):
            module = getattr(model, module_name, None)
            if module is not None:
                module.train()

    epoch_loss_sum = torch.zeros((), device=device)
    epoch_step_cnt = 0

    bucket_loss     = torch.zeros((), device=device)
    bucket_Lex      = torch.zeros((), device=device)
    bucket_Lx0      = torch.zeros((), device=device)
    bucket_Lcnt     = torch.zeros((), device=device)
    bucket_Lcnt_val = torch.zeros((), device=device)
    bucket_Lbg      = torch.zeros((), device=device)
    bucket_Leps     = torch.zeros((), device=device)
    bucket_Lcov = torch.zeros((), device=device)
    bucket_Lcover = torch.zeros((), device=device)
    bucket_Lrandcover = torch.zeros((), device=device)
    bucket_Lrandmatch = torch.zeros((), device=device)
    bucket_Lrandexist = torch.zeros((), device=device)
    bucket_Lrandcount = torch.zeros((), device=device)
    bucket_Lrandcountmargin = torch.zeros((), device=device)
    bucket_Lrandbg = torch.zeros((), device=device)
    bucket_Lranddup = torch.zeros((), device=device)
    bucket_Lrandrank = torch.zeros((), device=device)
    bucket_Lrand1to1 = torch.zeros((), device=device)
    bucket_Lrandgrp = torch.zeros((), device=device)
    bucket_Lrandsoft = torch.zeros((), device=device)
    bucket_Lrandquota = torch.zeros((), device=device)
    bucket_Ldup = torch.zeros((), device=device)
    bucket_Lcollapse = torch.zeros((), device=device)
    bucket_Lprior_occ = torch.zeros((), device=device)
    bucket_Lprior_density = torch.zeros((), device=device)
    bucket_Lprior_count = torch.zeros((), device=device)
    bucket_k = 0

    T_int = int(T)
    if T_int <= 1:
        raise ValueError(f"T must be > 1, got T={T_int}")

    for step, (images, points_pad, mask, metas) in enumerate(data_loader, start=1):
        images     = images.to(device, non_blocking=True)      # [B,C,H,W]
        points_pad = points_pad.to(device, non_blocking=True)  # [B,N,2]
        mask       = mask.to(device, non_blocking=True)        # [B,N]

        B, C, H, W = images.shape
        feats = model.encode(images)

        if bool(getattr(model, "freeze_base_for_prior", False)):
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                prior_occupancy_logits, prior_density = model.predict_proposal_prior(feats)
                occ_target, density_target, gt_count_prior = build_proposal_prior_targets(
                    points_pad,
                    mask,
                    H,
                    W,
                    prior_density.size(-2),
                    prior_density.size(-1),
                    sigma=float(proposal_prior_sigma),
                )
                L_prior_occ, L_prior_density, L_prior_count, pred_count_prior = proposal_prior_loss(
                    prior_occupancy_logits,
                    prior_density,
                    occ_target,
                    density_target,
                    gt_count_prior,
                )
                loss = (
                    float(lambda_prior_occupancy) * L_prior_occ
                    + float(lambda_prior_density) * L_prior_density
                    + float(lambda_prior_count) * L_prior_count
                )

            scaler.scale(loss).backward()
            if max_norm is not None and max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [
                        p for p in model.proposal_prior_head.parameters()
                        if p.requires_grad
                    ],
                    max_norm,
                )
            scaler.step(optimizer)
            scaler.update()

            epoch_loss_sum += loss.detach()
            epoch_step_cnt += 1
            bucket_loss += loss.detach()
            bucket_Lprior_occ += L_prior_occ.detach()
            bucket_Lprior_density += L_prior_density.detach()
            bucket_Lprior_count += L_prior_count.detach()
            bucket_k += 1

            if step % log_every == 0:
                inv_k = 1.0 / max(1, bucket_k)
                msg = (
                    f"[train-prior] it={step:05d} "
                    f"loss={(bucket_loss * inv_k).item():.4f} "
                    f"Lprior_occ={(bucket_Lprior_occ * inv_k).item():.4f} "
                    f"Lprior_density={(bucket_Lprior_density * inv_k).item():.4f} "
                    f"Lprior_count={(bucket_Lprior_count * inv_k).item():.4f} "
                    f"pred_count={pred_count_prior.mean().item():.2f} "
                    f"gt_count={gt_count_prior.mean().item():.2f}"
                )
                print(msg)
                logging.info(msg)
                bucket_loss.zero_()
                bucket_Lprior_occ.zero_()
                bucket_Lprior_density.zero_()
                bucket_Lprior_count.zero_()
                bucket_k = 0
            continue

        cond_cache = model.cond.precompute(*feats)

        # GT points in [-1, 1]
        p0 = pixels_to_m11(points_pad, H, W)

        # ---- 每張圖隨機抽一個 t ∈ [0, T-1] ----
        t_cur = torch.randint(
            low=0, high=T_int, size=(B, 1),
            device=device, dtype=torch.long
        )

        # 前向加噪：p0 -> p_t
        p_t, _, _ = forward_noisy(p0, t_cur, sched)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            L_rand_cover = torch.zeros((), device=device)
            L_rand_match = torch.zeros((), device=device)
            L_rand_exist = torch.zeros((), device=device)
            L_rand_count = torch.zeros((), device=device)
            L_rand_count_margin = torch.zeros((), device=device)
            L_rand_bg = torch.zeros((), device=device)
            L_rand_dup = torch.zeros((), device=device)
            L_rand_rank = torch.zeros((), device=device)
            L_rand_one_to_one = torch.zeros((), device=device)
            L_rand_group_pos = torch.zeros((), device=device)
            L_rand_soft_compete = torch.zeros((), device=device)
            L_rand_group_quota = torch.zeros((), device=device)
            rand_points_for_stats = None
            rand_valid_for_stats = None
            L_prior_occ = torch.zeros((), device=device)
            L_prior_density = torch.zeros((), device=device)
            L_prior_count = torch.zeros((), device=device)
            prior_occupancy_logits = None
            prior_density = None
            abar_cur = sched.get(t_cur).unsqueeze(-1)   # [B,1,1]

            if bool(use_proposal_prior):
                prior_occupancy_logits, prior_density = model.predict_proposal_prior(feats)
                occ_target, density_target, gt_count_prior = build_proposal_prior_targets(
                    points_pad,
                    mask,
                    H,
                    W,
                    prior_density.size(-2),
                    prior_density.size(-1),
                    sigma=float(proposal_prior_sigma),
                )
                L_prior_occ, L_prior_density, L_prior_count, _ = proposal_prior_loss(
                    prior_occupancy_logits,
                    prior_density,
                    occ_target,
                    density_target,
                    gt_count_prior,
                )

            eps_pred, exist_logit, pro, pred_points_for_cls, pred_valid_mask = model.denoise(
                feats, p_t, t_cur,
                abar_t=abar_cur,
                clamp_eps=1e-6,
                cond_cache=cond_cache,
                need_exist=True,
                selector_prior_maps=(
                    (prior_occupancy_logits, prior_density)
                    if prior_density is not None
                    else None
                ),
            )

            if exist_logit is None:
                raise RuntimeError("denoise returned None exist_logit")

            exist_logit = torch.clamp(exist_logit, -30.0, 30.0)



            loss, L_exist, L_x0, L_cnt, L_bg, Leps, Lcov, Lcover, Ldup, Lcollapse = criterion(
                p_t=p_t,
                p0=p0,
                mask=mask,
                pro=pro,
                abar_t=abar_cur,
                eps_pred=eps_pred,
                exist_logit=exist_logit,
                pred_points_for_cls=pred_points_for_cls,
                pred_valid_mask=pred_valid_mask,
                lambda_t=None,
                aux_weight=1,
            )

            # ---- 額外 count validation-style loss ----
            loss = float(lambda_gt_loss) * loss
            loss = (
                loss
                + float(lambda_prior_occupancy) * L_prior_occ
                + float(lambda_prior_density) * L_prior_density
                + float(lambda_prior_count) * L_prior_count
            )

            cls_logits = exist_logit
            if cls_logits.dim() == 3 and cls_logits.size(-1) == 1:
                cls_logits = cls_logits.squeeze(-1)

            if cls_logits.dim() == 3 and cls_logits.size(-1) == 2:
                prob_obj = cls_logits.softmax(-1)[..., 1]
            else:
                prob_obj = torch.sigmoid(cls_logits)

            pred_cnt_v = prob_obj.sum(dim=1)
            gt_cnt = mask.sum(dim=1).float()
            L_cnt_val = F.smooth_l1_loss(pred_cnt_v, gt_cnt)

            loss = loss + lambda_cnt_val * L_cnt_val

            if (
                lambda_rand_cover > 0
                or lambda_rand_match > 0
                or lambda_rand_exist > 0
                or lambda_rand_count > 0
                or lambda_rand_count_margin > 0
                or lambda_rand_bg > 0
                or lambda_rand_dup > 0
                or lambda_rand_rank > 0
                or lambda_rand_one_to_one_conf > 0
                or lambda_rand_group_pos > 0
                or lambda_rand_soft_compete > 0
                or lambda_rand_group_quota > 0
            ):
                t_low = max(0, min(int(rand_cover_t_min), T_int - 1))
                t_high = max(t_low, min(int(rand_cover_t_max), T_int - 1))
                t_rand = torch.randint(
                    low=t_low,
                    high=t_high + 1,
                    size=(B, 1),
                    device=device,
                    dtype=torch.long,
                )
                if bool(use_proposal_prior):
                    x0_prior, _ = build_mixed_x0_prior(
                        prior_occupancy_logits,
                        prior_density,
                        num_slots=p0.size(1),
                        clamp_eps=1e-3,
                        mode=str(proposal_prior_mode),
                        density_cell_capacity=int(proposal_prior_cell_capacity),
                    )
                    p_rand, _, _ = forward_noisy(x0_prior, t_rand, sched)
                else:
                    p_rand = torch.empty_like(p0).uniform_(
                        -1.0 + 1e-3, 1.0 - 1e-3
                    )
                abar_rand = sched.get(t_rand).unsqueeze(-1)
                need_rand_exist = (
                    float(lambda_rand_exist) > 0.0
                    or float(lambda_rand_count) > 0.0
                    or float(lambda_rand_count_margin) > 0.0
                    or float(lambda_rand_bg) > 0.0
                    or float(lambda_rand_dup) > 0.0
                    or float(lambda_rand_rank) > 0.0
                    or float(lambda_rand_one_to_one_conf) > 0.0
                    or float(lambda_rand_group_pos) > 0.0
                    or float(lambda_rand_soft_compete) > 0.0
                    or float(lambda_rand_group_quota) > 0.0
                )

                _, rand_exist_logit, _, rand_x0_hat, rand_valid_mask = model.denoise(
                    feats,
                    p_rand,
                    t_rand,
                    abar_t=abar_rand,
                    clamp_eps=1e-6,
                    cond_cache=cond_cache,
                    need_exist=need_rand_exist,
                    selector_prior_maps=(
                        (prior_occupancy_logits, prior_density)
                        if prior_density is not None
                        else None
                    ),
                )
                if rand_x0_hat is None:
                    raise RuntimeError("random-start auxiliary branch did not return x0_hat")

                if need_rand_exist:
                    if rand_exist_logit is None:
                        raise RuntimeError("random-start auxiliary branch did not return exist logits")
                    rand_exist_logit = torch.clamp(rand_exist_logit, -30.0, 30.0)
                    L_rand_exist, rand_target_classes, rand_target_info = criterion.p2p_exist_loss(
                        exist_logit=rand_exist_logit,
                        pred_points_for_cls=rand_x0_hat,
                        gt_points=p0,
                        gt_mask=mask,
                        pred_valid_mask=rand_valid_mask,
                        return_target_info=True,
                    )
                    rand_target_roles = (
                        rand_target_info.get("roles")
                        if (
                            isinstance(rand_target_info, dict)
                            and bool(rand_target_info.get("role_labels_available", False))
                        )
                        else None
                    )
                    loss = loss + float(lambda_rand_exist) * L_rand_exist

                    if rand_exist_logit.dim() == 3 and rand_exist_logit.size(-1) == 2:
                        rand_prob_obj = rand_exist_logit.softmax(-1)[..., 1]
                    else:
                        rand_prob_obj = torch.sigmoid(rand_exist_logit)

                    if rand_valid_mask is not None:
                        rand_valid_float = rand_valid_mask.to(dtype=rand_prob_obj.dtype)
                        rand_valid_bool = rand_valid_mask.bool()
                    else:
                        rand_valid_float = torch.ones_like(rand_prob_obj)
                        rand_valid_bool = torch.ones_like(rand_prob_obj, dtype=torch.bool)

                    if lambda_rand_count > 0:
                        rand_pred_cnt = (rand_prob_obj * rand_valid_float).sum(dim=1)
                        gt_cnt_norm = mask.sum(dim=1).float()
                        L_rand_count = ((rand_pred_cnt - gt_cnt_norm).abs() / (gt_cnt_norm + 1.0)).mean()
                        loss = loss + float(lambda_rand_count) * L_rand_count

                    if lambda_rand_count_margin > 0:
                        L_rand_count_margin = topk_count_margin_loss(
                            prob_obj=rand_prob_obj,
                            gt_mask=mask,
                            pred_valid_mask=rand_valid_mask,
                            pos_margin=float(rand_count_pos_margin),
                            neg_margin=float(rand_count_neg_margin),
                            neg_topk=int(rand_count_neg_topk),
                            slack=int(rand_count_slack),
                        )
                        loss = loss + float(lambda_rand_count_margin) * L_rand_count_margin

                    if lambda_rand_bg > 0:
                        rand_bg_losses = []
                        rand_pos_mask = rand_target_classes == 1
                        rand_pred_pix = m11_to_pixels_batch(rand_x0_hat, H, W)
                        gt_pix = m11_to_pixels_batch(p0, H, W)
                        ignore_radius = max(float(rand_bg_ignore_radius), 0.0)
                        for b in range(B):
                            if rand_target_roles is not None:
                                bg_mask = rand_valid_bool[b] & (rand_target_roles[b] == 0)
                            else:
                                bg_mask = rand_valid_bool[b] & (~rand_pos_mask[b])
                                gt_idx = mask[b].nonzero(as_tuple=False).squeeze(1)
                                if gt_idx.numel() > 0 and ignore_radius > 0:
                                    dmin = torch.cdist(rand_pred_pix[b], gt_pix[b, gt_idx], p=2).min(dim=1).values
                                    bg_mask = bg_mask & (dmin >= ignore_radius)
                            if bg_mask.any():
                                rand_bg_losses.append(rand_prob_obj[b][bg_mask].mean())
                            else:
                                rand_bg_losses.append(rand_prob_obj[b].new_tensor(0.0))
                        L_rand_bg = torch.stack(rand_bg_losses).mean()
                        loss = loss + float(lambda_rand_bg) * L_rand_bg

                    if lambda_rand_dup > 0:
                        L_rand_dup = region_duplicate_loss(
                            pred_points_m11=rand_x0_hat,
                            prob_obj=rand_prob_obj,
                            gt_points_m11=p0,
                            gt_mask=mask,
                            pred_valid_mask=rand_valid_mask,
                            H=H,
                            W=W,
                            region_radius=float(rand_dup_radius),
                            region_topk=int(rand_dup_topk),
                            dense_aware=bool(rand_dup_dense_aware),
                            neighbor_radius=float(rand_dup_neighbor_radius),
                            allow_extra=int(rand_dup_allow_extra),
                        )
                        loss = loss + float(lambda_rand_dup) * L_rand_dup

                    if lambda_rand_rank > 0:
                        L_rand_rank = selector_hard_negative_ranking_loss(
                            selector_logits=rand_exist_logit,
                            pred_points_m11=rand_x0_hat,
                            target_classes=rand_target_classes,
                            gt_points_m11=p0,
                            gt_mask=mask,
                            pred_valid_mask=rand_valid_mask,
                            target_roles=rand_target_roles,
                            H=H,
                            W=W,
                            near_radius=float(rand_rank_near_radius),
                            far_radius=float(rand_rank_far_radius),
                            margin=float(rand_rank_margin),
                            neg_topk=int(rand_rank_neg_topk),
                            near_weight=1.0,
                            far_weight=float(rand_rank_far_weight),
                        )
                        loss = loss + float(lambda_rand_rank) * L_rand_rank

                    if lambda_rand_one_to_one_conf > 0:
                        L_rand_one_to_one = region_one_to_one_confidence_loss(
                            selector_logits=rand_exist_logit,
                            pred_points_m11=rand_x0_hat,
                            gt_points_m11=p0,
                            gt_mask=mask,
                            matcher=criterion.matcher,
                            pred_valid_mask=rand_valid_mask,
                            H=H,
                            W=W,
                            region_radius=float(rand_one_to_one_radius),
                            bg_radius=float(rand_one_to_one_bg_radius),
                            margin=float(rand_one_to_one_margin),
                            neg_topk=int(rand_one_to_one_neg_topk),
                            count_pos_logit=float(rand_one_to_one_count_pos_logit),
                            count_neg_logit=float(rand_one_to_one_count_neg_logit),
                            count_slack=int(rand_one_to_one_count_slack),
                            pos_logit=float(rand_one_to_one_pos_logit),
                            duplicate_neg_logit=float(rand_one_to_one_dup_neg_logit),
                            bg_neg_logit=float(rand_one_to_one_bg_neg_logit),
                            pos_weight=float(rand_one_to_one_pos_weight),
                            duplicate_weight=float(rand_one_to_one_dup_weight),
                            bg_weight=float(rand_one_to_one_bg_weight),
                            rank_weight=float(rand_one_to_one_rank_weight),
                            count_weight=float(rand_one_to_one_count_weight),
                        )
                        loss = loss + float(lambda_rand_one_to_one_conf) * L_rand_one_to_one

                    if lambda_rand_group_pos > 0:
                        L_rand_group_pos = group_level_positive_confidence_loss(
                            selector_logits=rand_exist_logit,
                            pred_points_m11=rand_x0_hat,
                            gt_points_m11=p0,
                            gt_mask=mask,
                            pred_valid_mask=rand_valid_mask,
                            H=H,
                            W=W,
                            group_radius=float(rand_group_pos_radius),
                            soft_sigma=float(rand_group_pos_sigma),
                            temperature=float(rand_group_pos_temperature),
                            pos_logit=float(rand_group_pos_logit),
                            nearest_gt_only=bool(rand_group_pos_nearest_gt_only),
                        )
                        loss = loss + float(lambda_rand_group_pos) * L_rand_group_pos

                    if lambda_rand_soft_compete > 0:
                        L_rand_soft_compete = soft_local_competition_confidence_loss(
                            selector_logits=rand_exist_logit,
                            pred_points_m11=rand_x0_hat,
                            gt_points_m11=p0,
                            gt_mask=mask,
                            matcher=criterion.matcher,
                            pred_valid_mask=rand_valid_mask,
                            H=H,
                            W=W,
                            winner_radius=float(rand_soft_compete_winner_radius),
                            compete_radius=float(rand_soft_compete_radius),
                            soft_sigma=float(rand_soft_compete_sigma),
                            bg_radius=float(rand_soft_compete_bg_radius),
                            margin=float(rand_soft_compete_margin),
                            neg_topk=int(rand_soft_compete_neg_topk),
                            temperature=float(rand_soft_compete_temperature),
                            nearest_gt_only=bool(rand_soft_compete_nearest_gt_only),
                            pos_logit=float(rand_soft_compete_pos_logit),
                            duplicate_neg_logit=float(rand_soft_compete_dup_neg_logit),
                            bg_neg_logit=float(rand_soft_compete_bg_neg_logit),
                            pos_weight=float(rand_soft_compete_pos_weight),
                            rank_weight=float(rand_soft_compete_rank_weight),
                            duplicate_weight=float(rand_soft_compete_dup_weight),
                            softmax_weight=float(rand_soft_compete_softmax_weight),
                            bg_weight=float(rand_soft_compete_bg_weight),
                        )
                        loss = loss + float(lambda_rand_soft_compete) * L_rand_soft_compete

                    if lambda_rand_group_quota > 0:
                        L_rand_group_quota = group_quota_confidence_loss(
                            selector_logits=rand_exist_logit,
                            pred_points_m11=rand_x0_hat,
                            gt_points_m11=p0,
                            gt_mask=mask,
                            pred_valid_mask=rand_valid_mask,
                            H=H,
                            W=W,
                            group_radius=float(rand_group_quota_radius),
                            bg_radius=float(rand_group_quota_bg_radius),
                            quota_target=float(rand_group_quota_target),
                            margin=float(rand_group_quota_margin),
                            neg_topk=int(rand_group_quota_neg_topk),
                            nearest_gt_only=bool(rand_group_quota_nearest_gt_only),
                            pos_logit=float(rand_group_quota_pos_logit),
                            duplicate_neg_logit=float(rand_group_quota_dup_neg_logit),
                            bg_neg_logit=float(rand_group_quota_bg_neg_logit),
                            pos_weight=float(rand_group_quota_pos_weight),
                            quota_weight=float(rand_group_quota_weight),
                            over_weight=float(rand_group_quota_over_weight),
                            under_weight=float(rand_group_quota_under_weight),
                            gap_weight=float(rand_group_quota_gap_weight),
                            duplicate_weight=float(rand_group_quota_dup_weight),
                            bg_weight=float(rand_group_quota_bg_weight),
                        )
                        loss = loss + float(lambda_rand_group_quota) * L_rand_group_quota

                L_rand_cover = coverage_radius_hinge_loss(
                    pred_points_m11=rand_x0_hat,
                    gt_points_m11=p0,
                    gt_mask=mask,
                    pred_valid_mask=rand_valid_mask,
                    H=H,
                    W=W,
                    cover_radius=float(rand_cover_radius),
                    hard_weight=float(getattr(criterion, "cov_hard_weight", 0.0)),
                    hard_cap=float(getattr(criterion, "cov_hard_cap", 1.0)),
                    dense_weight=float(getattr(criterion, "cov_dense_weight", 0.0)),
                    dense_radius=float(getattr(criterion, "cov_dense_radius", 16.0)),
                    dense_norm=float(getattr(criterion, "cov_dense_norm", 4.0)),
                    weight_cap=float(getattr(criterion, "cov_weight_cap", 6.0)),
                )
                loss = loss + float(lambda_rand_cover) * L_rand_cover
                rand_points_for_stats = rand_x0_hat
                rand_valid_for_stats = rand_valid_mask

                if lambda_rand_match > 0:
                    L_rand_match = proposal_hungarian_cover_loss(
                        pred_points_m11=rand_x0_hat,
                        gt_points_m11=p0,
                        gt_mask=mask,
                        matcher=criterion.matcher,
                        pred_valid_mask=rand_valid_mask,
                        H=H,
                        W=W,
                        radius=float(rand_cover_radius),
                    )
                    loss = loss + float(lambda_rand_match) * L_rand_match

        scaler.scale(loss).backward()
        if max_norm is not None and max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss_sum += loss.detach()
        epoch_step_cnt += 1

        bucket_loss     += loss.detach()
        bucket_Lex      += L_exist.detach()
        bucket_Lx0      += L_x0.detach()
        bucket_Lcnt     += L_cnt.detach()
        bucket_Lcnt_val += L_cnt_val.detach()
        bucket_Lbg      += L_bg.detach()
        bucket_Leps     += Leps.detach()
        bucket_Lcov     += Lcov.detach()
        bucket_Lcover   += Lcover.detach()
        bucket_Lrandcover += L_rand_cover.detach()
        bucket_Lrandmatch += L_rand_match.detach()
        bucket_Lrandexist += L_rand_exist.detach()
        bucket_Lrandcount += L_rand_count.detach()
        bucket_Lrandcountmargin += L_rand_count_margin.detach()
        bucket_Lrandbg += L_rand_bg.detach()
        bucket_Lranddup += L_rand_dup.detach()
        bucket_Lrandrank += L_rand_rank.detach()
        bucket_Lrand1to1 += L_rand_one_to_one.detach()
        bucket_Lrandgrp += L_rand_group_pos.detach()
        bucket_Lrandsoft += L_rand_soft_compete.detach()
        bucket_Lrandquota += L_rand_group_quota.detach()
        bucket_Ldup     += Ldup.detach()
        bucket_Lcollapse += Lcollapse.detach()
        bucket_Lprior_occ += L_prior_occ.detach()
        bucket_Lprior_density += L_prior_density.detach()
        bucket_Lprior_count += L_prior_count.detach()
        bucket_k += 1

        if step % log_every == 0:
            prop_cov, multi_ratio = batch_proposal_region_stats(
                pred_points_m11=pred_points_for_cls.detach() if pred_points_for_cls is not None else p_t.detach(),
                gt_points_m11=p0.detach(),
                gt_mask=mask.detach(),
                radius=6.0,
                H=H,
                W=W,
            )
            if rand_points_for_stats is not None:
                rand_prop_cov, rand_multi_ratio = batch_proposal_region_stats(
                    pred_points_m11=rand_points_for_stats.detach(),
                    gt_points_m11=p0.detach(),
                    gt_mask=mask.detach(),
                    pred_valid_mask=rand_valid_for_stats.detach() if rand_valid_for_stats is not None else None,
                    radius=float(rand_cover_radius),
                    H=H,
                    W=W,
                )
            else:
                rand_prop_cov = torch.zeros((), device=device)
                rand_multi_ratio = torch.zeros((), device=device)
            inv_k = 1.0 / max(1, bucket_k)
            msg = (
                f"[train-x0] it={step:05d} "
                f"loss={(bucket_loss * inv_k).item():.4f} "
                f"Lex={(bucket_Lex * inv_k).item():.4f} "
                f"Lx0={(bucket_Lx0 * inv_k).item():.4f} "
                f"Lcnt={(bucket_Lcnt * inv_k).item():.4f} "
                f"Lcnt_val={(bucket_Lcnt_val * inv_k).item():.4f} "
                f"Lbg={(bucket_Lbg * inv_k).item():.4f} "
                f"Leps={(bucket_Leps * inv_k).item():.4f} "
                f"Lcov={(bucket_Lcov * inv_k).item():.4f} "
                f"Lcover={(bucket_Lcover * inv_k).item():.4f} "
                f"Lrandcover={(bucket_Lrandcover * inv_k).item():.4f} "
                f"Lrandmatch={(bucket_Lrandmatch * inv_k).item():.4f} "
                f"Lrandexist={(bucket_Lrandexist * inv_k).item():.4f} "
                f"Lrandcount={(bucket_Lrandcount * inv_k).item():.4f} "
                f"Lrandcm={(bucket_Lrandcountmargin * inv_k).item():.4f} "
                f"Lrandbg={(bucket_Lrandbg * inv_k).item():.4f} "
                f"Lranddup={(bucket_Lranddup * inv_k).item():.4f} "
                f"Lrandrank={(bucket_Lrandrank * inv_k).item():.4f} "
                f"Lrand1to1={(bucket_Lrand1to1 * inv_k).item():.4f} "
                f"Lrandgrp={(bucket_Lrandgrp * inv_k).item():.4f} "
                f"Lrandsoft={(bucket_Lrandsoft * inv_k).item():.4f} "
                f"Lrandquota={(bucket_Lrandquota * inv_k).item():.4f} "
                f"Ldup={(bucket_Ldup * inv_k).item():.4f} "
                f"Lcollapse={(bucket_Lcollapse * inv_k).item():.4f} "
                f"Lprior_occ={(bucket_Lprior_occ * inv_k).item():.4f} "
                f"Lprior_density={(bucket_Lprior_density * inv_k).item():.4f} "
                f"Lprior_count={(bucket_Lprior_count * inv_k).item():.4f} "
                f"prop_cov@6={prop_cov.item():.4f} "
                f"multi@6={multi_ratio.item():.4f} "
                f"rand_prop_cov@6={rand_prop_cov.item():.4f} "
                f"rand_multi@6={rand_multi_ratio.item():.4f} "
            )
            print(msg)
            logging.info(msg)

            bucket_loss.zero_()
            bucket_Lex.zero_()
            bucket_Lx0.zero_()
            bucket_Lcnt.zero_()
            bucket_Lcnt_val.zero_()
            bucket_Lbg.zero_()
            bucket_Leps.zero_()
            bucket_Lcov.zero_()
            bucket_Lcover.zero_()
            bucket_Lrandcover.zero_()
            bucket_Lrandmatch.zero_()
            bucket_Lrandexist.zero_()
            bucket_Lrandcount.zero_()
            bucket_Lrandcountmargin.zero_()
            bucket_Lrandbg.zero_()
            bucket_Lranddup.zero_()
            bucket_Lrandrank.zero_()
            bucket_Lrand1to1.zero_()
            bucket_Lrandgrp.zero_()
            bucket_Lrandsoft.zero_()
            bucket_Lrandquota.zero_()
            bucket_Ldup.zero_()
            bucket_Lcollapse.zero_()
            bucket_Lprior_occ.zero_()
            bucket_Lprior_density.zero_()
            bucket_Lprior_count.zero_()
            bucket_k = 0

    if epoch_step_cnt == 0:
        return 0.0
    return (epoch_loss_sum / epoch_step_cnt).item()


