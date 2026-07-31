import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProposalPriorHead(nn.Module):
    """Predict occupancy logits and a count-preserving density map from P4."""

    def __init__(self, in_ch: int = 128, hidden: int = 64):
        super().__init__()
        groups = 8 if hidden % 8 == 0 else 1
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(),
        )
        self.occupancy = nn.Conv2d(hidden, 1, 1)
        self.density = nn.Conv2d(hidden, 1, 1)

        # Start with a low count so a fresh head does not immediately consume
        # all 900 slots with guided proposals.
        nn.init.constant_(self.occupancy.bias, -4.0)
        nn.init.constant_(self.density.bias, -6.0)

    def forward(self, p4: torch.Tensor):
        feat = self.body(p4)
        occupancy_logits = self.occupancy(feat)
        density = F.softplus(self.density(feat))
        return occupancy_logits, density


def _gaussian_kernel(sigma: float, device, dtype, normalize: bool):
    sigma = max(float(sigma), 1e-3)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma * sigma))
    if normalize:
        kernel = kernel / kernel.sum().clamp_min(1e-12)
    else:
        kernel = kernel / kernel.max().clamp_min(1e-12)
    return kernel.view(1, 1, kernel.size(0), kernel.size(1)), radius


@torch.no_grad()
def build_proposal_prior_targets(
        points_xy: torch.Tensor,
        point_mask: torch.Tensor,
        image_h: int,
        image_w: int,
        map_h: int,
        map_w: int,
        sigma: float = 1.25,
):
    """Rasterize point labels into occupancy and count-preserving density maps."""
    batch_size = points_xy.size(0)
    impulses = points_xy.new_zeros((batch_size, 1, map_h, map_w))

    x_cell = torch.floor(points_xy[..., 0] * float(map_w) / max(float(image_w), 1.0))
    y_cell = torch.floor(points_xy[..., 1] * float(map_h) / max(float(image_h), 1.0))
    x_cell = x_cell.long().clamp(0, map_w - 1)
    y_cell = y_cell.long().clamp(0, map_h - 1)

    for b in range(batch_size):
        valid = point_mask[b].bool()
        if not valid.any():
            continue
        linear = y_cell[b, valid] * map_w + x_cell[b, valid]
        impulses[b, 0].view(-1).scatter_add_(
            0,
            linear,
            torch.ones_like(linear, dtype=impulses.dtype),
        )

    density_kernel, radius = _gaussian_kernel(
        sigma, impulses.device, impulses.dtype, normalize=True
    )
    occupancy_kernel, _ = _gaussian_kernel(
        sigma, impulses.device, impulses.dtype, normalize=False
    )
    density_target = F.conv2d(impulses, density_kernel, padding=radius)
    occupancy_target = F.conv2d(impulses, occupancy_kernel, padding=radius).clamp_(0.0, 1.0)

    # Convolution loses mass at image borders. Renormalize each non-empty map
    # so density.sum() remains the number of annotated points.
    gt_count = point_mask.sum(dim=1).to(dtype=density_target.dtype)
    map_sum = density_target.sum(dim=(1, 2, 3))
    scale = torch.where(
        gt_count > 0,
        gt_count / map_sum.clamp_min(1e-12),
        torch.zeros_like(gt_count),
    )
    density_target = density_target * scale.view(-1, 1, 1, 1)
    return occupancy_target, density_target, gt_count


def proposal_prior_loss(
        occupancy_logits: torch.Tensor,
        density: torch.Tensor,
        occupancy_target: torch.Tensor,
        density_target: torch.Tensor,
        gt_count: torch.Tensor,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
):
    """Return occupancy focal loss, density-map loss, and normalized count loss."""
    ce = F.binary_cross_entropy_with_logits(
        occupancy_logits, occupancy_target, reduction="none"
    )
    prob = occupancy_logits.sigmoid()
    p_t = prob * occupancy_target + (1.0 - prob) * (1.0 - occupancy_target)
    alpha_t = (
        float(focal_alpha) * occupancy_target
        + (1.0 - float(focal_alpha)) * (1.0 - occupancy_target)
    )
    occupancy_loss = (alpha_t * (1.0 - p_t).pow(float(focal_gamma)) * ce).mean()

    normalizer = (gt_count + 1.0).clamp_min(1.0)
    density_error = F.smooth_l1_loss(density, density_target, reduction="none")
    density_loss = (
        density_error.sum(dim=(1, 2, 3)) / normalizer
    ).mean()

    pred_count = density.sum(dim=(1, 2, 3))
    count_loss = F.smooth_l1_loss(
        pred_count / normalizer,
        gt_count / normalizer,
    )
    return occupancy_loss, density_loss, count_loss, pred_count


@torch.no_grad()
def select_occupancy_points(occupancy_prob: torch.Tensor, k: int):
    """Select exactly k P4 cell centers, preferring 3x3 local maxima."""
    _, map_h, map_w = occupancy_prob.shape
    k = int(k)
    if k <= 0:
        return occupancy_prob.new_empty((0, 2))

    pooled = F.max_pool2d(
        occupancy_prob.unsqueeze(0),
        kernel_size=3,
        stride=1,
        padding=1,
    ).squeeze(0)
    flat_scores = occupancy_prob.flatten()
    peak_mask = (occupancy_prob >= pooled).flatten()
    peak_indices = peak_mask.nonzero(as_tuple=False).squeeze(1)

    if peak_indices.numel() > 0:
        peak_order = flat_scores[peak_indices].argsort(descending=True)
        selected = peak_indices[peak_order[:k]]
    else:
        selected = peak_indices

    if selected.numel() < k:
        used = torch.zeros_like(flat_scores, dtype=torch.bool)
        if selected.numel() > 0:
            used[selected] = True
        remaining_scores = flat_scores.masked_fill(used, float("-inf"))
        fill_count = min(k - selected.numel(), remaining_scores.numel() - int(used.sum()))
        if fill_count > 0:
            fill = remaining_scores.topk(fill_count).indices
            selected = torch.cat([selected, fill], dim=0)

    y = torch.div(selected, map_w, rounding_mode="floor")
    x = selected % map_w
    x_m11 = ((x.to(dtype=occupancy_prob.dtype) + 0.5) / float(map_w)) * 2.0 - 1.0
    y_m11 = ((y.to(dtype=occupancy_prob.dtype) + 0.5) / float(map_h)) * 2.0 - 1.0
    return torch.stack([x_m11, y_m11], dim=-1)


def _subcell_offsets(count: int, device, dtype):
    if count <= 1:
        return torch.zeros((max(count, 0), 2), device=device, dtype=dtype)

    base = [
        (-0.25, -0.25),
        (0.25, 0.25),
        (-0.25, 0.25),
        (0.25, -0.25),
        (0.0, 0.0),
        (-0.35, 0.0),
        (0.35, 0.0),
        (0.0, -0.35),
        (0.0, 0.35),
    ]
    if count <= len(base):
        return torch.tensor(base[:count], device=device, dtype=dtype)

    side = int(math.ceil(math.sqrt(count)))
    coords = torch.linspace(-0.35, 0.35, side, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    return torch.stack([xx.flatten(), yy.flatten()], dim=-1)[:count]


@torch.no_grad()
def select_density_points(
        density: torch.Tensor,
        k: int,
        cell_capacity: int = 2,
):
    """Allocate guided points from density mass with a per-cell capacity cap."""
    _, map_h, map_w = density.shape
    k = int(k)
    if k <= 0:
        return density.new_empty((0, 2))

    capacity = max(1, int(cell_capacity))
    max_points = int(map_h) * int(map_w) * capacity
    k = min(k, max_points)

    flat = density.flatten().clamp_min(0)
    if flat.numel() == 0:
        return density.new_empty((0, 2))

    base = torch.floor(flat).long().clamp_(0, capacity)
    total_base = int(base.sum().item())

    if total_base > k:
        scores = flat + torch.linspace(
            0,
            -1e-6,
            flat.numel(),
            device=flat.device,
            dtype=flat.dtype,
        )
        order = scores.argsort(descending=True)
        quota = torch.zeros_like(base)
        remaining = k
        for idx in order.tolist():
            if remaining <= 0:
                break
            take = min(int(base[idx].item()), remaining)
            if take > 0:
                quota[idx] = take
                remaining -= take
    else:
        quota = base.clone()
        remaining = k - total_base
        if remaining > 0:
            frac = flat - torch.floor(flat)
            tie_break = torch.linspace(
                0,
                -1e-6,
                flat.numel(),
                device=flat.device,
                dtype=flat.dtype,
            )
            scores = frac + tie_break
            scores = scores.masked_fill(quota >= capacity, float("-inf"))
            while remaining > 0 and torch.isfinite(scores).any():
                idx = int(scores.argmax().item())
                quota[idx] += 1
                remaining -= 1
                if quota[idx] >= capacity:
                    scores[idx] = float("-inf")

    selected = quota.nonzero(as_tuple=False).squeeze(1)
    if selected.numel() == 0:
        return density.new_empty((0, 2))

    points = []
    for idx in selected.tolist():
        q = int(quota[idx].item())
        if q <= 0:
            continue
        y = idx // int(map_w)
        x = idx % int(map_w)
        offsets = _subcell_offsets(q, density.device, density.dtype)
        xy = density.new_empty((q, 2))
        xy[:, 0] = (float(x) + 0.5 + offsets[:, 0]) / float(map_w)
        xy[:, 1] = (float(y) + 0.5 + offsets[:, 1]) / float(map_h)
        xy = xy * 2.0 - 1.0
        points.append(xy.clamp(-1.0 + 1e-6, 1.0 - 1e-6))

    return torch.cat(points, dim=0) if points else density.new_empty((0, 2))


@torch.no_grad()
def select_guided_prior_points(
        occupancy_logits: torch.Tensor,
        density: torch.Tensor,
        k: int,
        mode: str = "occupancy",
        density_cell_capacity: int = 2,
):
    mode = str(mode or "occupancy").strip().lower()
    if mode in {"density", "density_only", "density-only"}:
        return select_density_points(
            density,
            k,
            cell_capacity=int(density_cell_capacity),
        )
    if mode in {"occupancy", "occupancy_topk", "occupancy_top-k"}:
        return select_occupancy_points(occupancy_logits.sigmoid(), k)
    raise ValueError(f"Unsupported proposal_prior_mode={mode!r}")


@torch.no_grad()
def build_mixed_x0_prior(
        occupancy_logits: torch.Tensor,
        density: torch.Tensor,
        num_slots: int = 900,
        clamp_eps: float = 1e-3,
        mode: str = "occupancy",
        density_cell_capacity: int = 2,
):
    """
    Build exactly num_slots x0 proposals per image.

    K_guided = clamp(round(density.sum()), 0, num_slots)
    K_random = num_slots - K_guided
    """
    batch_size = occupancy_logits.size(0)
    priors = []
    guided_counts = []

    for b in range(batch_size):
        k_guided = int(
            torch.round(density[b].sum()).clamp(0, int(num_slots)).item()
        )
        guided = select_guided_prior_points(
            occupancy_logits[b],
            density[b],
            k_guided,
            mode=mode,
            density_cell_capacity=int(density_cell_capacity),
        )
        k_guided = int(guided.size(0))
        k_random = int(num_slots) - k_guided
        random_xy = occupancy_logits.new_empty((k_random, 2)).uniform_(
            -1.0 + float(clamp_eps),
            1.0 - float(clamp_eps),
        )
        prior = torch.cat([guided, random_xy], dim=0)
        if prior.size(0) != int(num_slots):
            raise RuntimeError(
                f"proposal prior produced {prior.size(0)} slots, expected {num_slots}"
            )
        prior = prior[torch.randperm(prior.size(0), device=prior.device)]
        priors.append(prior)
        guided_counts.append(k_guided)

    return torch.stack(priors, dim=0), torch.tensor(
        guided_counts,
        device=occupancy_logits.device,
        dtype=torch.long,
    )
