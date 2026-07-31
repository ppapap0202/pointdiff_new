"""Diagnose whether selector local competition actually discriminates.

Two probes, sharing one DDIM forward pass:

1. Group winner margin
   For every GT, collect valid merged candidates within `val_ddim_cover_radius`
   and report top1-top2 score gap. A gap near zero means same-group candidates
   are indistinguishable to the confidence head, so any competition mechanism is
   only breaking ties by chance.

2. LocalCompetitionSelector score_shift
   Split the per-slot logit shift into "has neighbor" and "no neighbor" groups.
   The module is supposed to only act on neighbors, so a large shift on the
   no-neighbor group means score_delta is mostly an unconditional bias rather
   than a competition term.

Read-only: no checkpoint, config or training state is modified.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import models.pointdiff as pointdiff_mod
from dataset import build_dataset
from main import collate_points_padded, configure_trainable_params, load_model_state
from models import Diffusion_schedule, build_model
from models.diffusion_utils import forward_noisy, pixels_to_m11
from models.pointdiff import (
    sample_selector_prior_features,
    selector_object_logit,
    selector_relative_geometry_features,
)
from models.proposal_prior import build_mixed_x0_prior
from models.train_loop import m11_to_pixels_batch


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ddim_step(p_t, eps_pred, abar_t, abar_prev, eta: float):
    eps = 1e-12
    sqrt_ab_t = abar_t.clamp(0.0, 1.0).add(eps).sqrt()
    sqrt_om_t = (1.0 - abar_t).clamp_min(0.0).sqrt()
    x0_hat = (p_t - sqrt_om_t * eps_pred) / sqrt_ab_t

    alpha_t = (abar_t / (abar_prev + eps)).clamp_min(eps)
    sigma2 = (eta ** 2) * (
        ((1.0 - abar_prev).clamp_min(0.0) / (1.0 - abar_t).clamp_min(eps))
        * (1.0 - alpha_t).clamp_min(0.0)
    )
    sigma = sigma2.clamp_min(0.0).sqrt()
    c_eps = (1.0 - abar_prev - sigma2).clamp_min(0.0).sqrt()

    out = (abar_prev + eps).sqrt() * x0_hat + c_eps * eps_pred
    if eta > 0.0:
        out = out + sigma * torch.randn_like(out)
    return out


def describe(values, percentiles=(1, 5, 25, 50, 75, 95, 99)):
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        return {"count": 0}
    out = {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "abs_mean": float(np.abs(arr).mean()),
    }
    for p in percentiles:
        out[f"p{p}"] = float(np.percentile(arr, p))
    return out


class CompetitionProbe:
    """Capture pre/post competition scores by wrapping the module's forward."""

    def __init__(self, module):
        self.module = module
        self._orig_forward = None
        self.enabled = False
        self.last = None

    def __enter__(self):
        if self.module is None:
            return self
        self._orig_forward = self.module.forward

        def patched(logits, feat, points_m11, valid_mask):
            out = self._orig_forward(logits, feat, points_m11, valid_mask)
            if self.enabled:
                self.last = self._capture(logits, out, points_m11, valid_mask)
            return out

        self.module.forward = patched
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.module is not None and self._orig_forward is not None:
            # Remove the instance attribute so the class method takes over again.
            try:
                del self.module.forward
            except AttributeError:
                self.module.forward = self._orig_forward
        return False

    @torch.no_grad()
    def _capture(self, logits, out, points_m11, valid_mask):
        comp = self.module
        pre = selector_object_logit(logits.float())
        post = selector_object_logit(out.float())
        shift = post - pre

        valid = valid_mask.to(device=pre.device, dtype=torch.bool)
        pix = comp._pixel_points(points_m11.to(pre.device))
        dist = torch.cdist(pix, pix, p=2)
        allowed = (dist <= float(comp.radius)) & valid[:, :, None] & valid[:, None, :]
        if comp.exclude_self:
            eye = torch.eye(allowed.size(1), dtype=torch.bool, device=allowed.device)
            allowed = allowed & ~eye.unsqueeze(0)
        neighbor_count = allowed.sum(dim=-1)

        return {
            "pre": pre.detach(),
            "post": post.detach(),
            "shift": shift.detach(),
            "valid": valid.detach(),
            "neighbor_count": neighbor_count.detach(),
        }


def _rank(x):
    """Ordinal ranks 0..n-1. Ties are broken arbitrarily; scores/distances are
    continuous so exact ties are rare enough not to bias the correlation."""
    order = torch.argsort(x)
    ranks = torch.empty_like(x)
    ranks[order] = torch.arange(x.numel(), dtype=x.dtype, device=x.device)
    return ranks


def _spearman(a, b):
    if a.numel() < 2:
        return None
    ra = _rank(a)
    rb = _rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float((ra.norm() * rb.norm()).item())
    if denom < 1e-12:
        return None
    return float((ra @ rb).item() / denom)


def _pairwise_cos(feat):
    """Mean pairwise cosine similarity of rows in feat [n,C], n>=2."""
    f = torch.nn.functional.normalize(feat.float(), dim=-1, eps=1e-8)
    sim = f @ f.t()
    n = f.size(0)
    iu = torch.triu_indices(n, n, offset=1, device=f.device)
    return sim[iu[0], iu[1]]


class MergeProbe:
    """Capture pro features on both sides of merge_slots_same_pixel.

    The merge averages `pro` across every slot inside merge_radius_px. This probe
    measures how much within-cluster variation that averaging destroys, against
    two references: same-GT-group pairs that survive the merge, and random pairs.
    """

    def __init__(self):
        self._orig = None
        self.enabled = False
        self.last = None

    def __enter__(self):
        self._orig = pointdiff_mod.merge_slots_same_pixel

        def patched(x0_hat, pro, H, W, max_slots=None, weight=None,
                    weight_min=0.0, merge_radius_px=0.0):
            out = self._orig(
                x0_hat, pro, H, W, max_slots=max_slots, weight=weight,
                weight_min=weight_min, merge_radius_px=merge_radius_px,
            )
            if self.enabled:
                self.last = self._capture(
                    x0_hat, pro, out, H, W, weight, weight_min, merge_radius_px
                )
            return out

        pointdiff_mod.merge_slots_same_pixel = patched
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._orig is not None:
            pointdiff_mod.merge_slots_same_pixel = self._orig
        return False

    @staticmethod
    def _cluster_assignment(pts, radius_px, weight_b, weight_min):
        """Replicate the greedy clustering inside merge_slots_same_pixel."""
        n = pts.size(0)
        device = pts.device
        if radius_px > 0.0:
            if weight_b is not None:
                ow = weight_b.to(device=device, dtype=pts.dtype).clamp_min(float(weight_min))
            else:
                ow = torch.ones((n,), device=device, dtype=pts.dtype)
            order = torch.argsort(ow, descending=True)
            assigned = torch.zeros((n,), device=device, dtype=torch.bool)
            inverse = torch.empty((n,), device=device, dtype=torch.long)
            count = 0
            for idx in order:
                i = int(idx.item())
                if bool(assigned[i]):
                    continue
                dist = torch.norm(pts - pts[i], dim=1)
                members = (~assigned) & (dist < radius_px)
                inverse[members] = count
                assigned[members] = True
                count += 1
            return inverse, count
        pts_int = torch.round(pts).long()
        pts_int[:, 0].clamp_(0, 10 ** 6)
        _, inverse = torch.unique(pts_int, dim=0, return_inverse=True)
        count = int(inverse.max().item()) + 1 if inverse.numel() > 0 else 0
        return inverse, count

    @torch.no_grad()
    def _capture(self, x0_hat, pro, out, H, W, weight, weight_min, merge_radius_px):
        merged_xy, merged_pro, merged_mask = out
        x = (x0_hat[..., 0] + 1.0) * 0.5 * (W - 1)
        y = (x0_hat[..., 1] + 1.0) * 0.5 * (H - 1)
        pts = torch.stack([x, y], dim=-1)

        inverses = []
        for b in range(pts.size(0)):
            w_b = weight[b] if weight is not None else None
            inv, _ = self._cluster_assignment(
                pts[b], float(merge_radius_px), w_b, float(weight_min)
            )
            inverses.append(inv)
        return {
            "raw_pro": pro.detach(),
            "inverse": torch.stack(inverses, dim=0),
            "merged_pro": merged_pro.detach(),
            "merged_xy": merged_xy.detach(),
            "merged_mask": merged_mask.detach(),
        }


@torch.no_grad()
def group_margin_stats(gt_pix, cand_pix, score, radius):
    """Per-GT group margins. gt_pix [G,2], cand_pix [M,2], score [M]."""
    if gt_pix.numel() == 0 or cand_pix.numel() == 0:
        return None

    dist = torch.cdist(gt_pix, cand_pix, p=2)  # [G,M]
    within = dist <= float(radius)
    sizes = within.sum(dim=1)

    rows = {
        "group_size": [],
        "margin": [],
        "top1": [],
        "top2": [],
        "winner_is_nearest": [],
        "winner_dist": [],
        "nearest_dist": [],
        # Does a higher score mean closer to GT? Negative Spearman = yes.
        "spearman_score_vs_dist": [],
        # Where the winner sits in the distance ordering, 0=nearest, 1=farthest.
        # Expectation under random selection is 0.5.
        "winner_dist_rank_norm": [],
    }
    rows["_z_score"] = []
    rows["_z_dist"] = []
    multi = (sizes >= 2).nonzero(as_tuple=False).squeeze(1)
    singles = int((sizes == 1).sum().item())
    empty = int((sizes == 0).sum().item())

    for gi in multi.tolist():
        member = within[gi].nonzero(as_tuple=False).squeeze(1)
        s = score[member]
        d = dist[gi, member]
        order = torch.argsort(s, descending=True)
        top1 = float(s[order[0]].item())
        top2 = float(s[order[1]].item())
        nearest_local = int(torch.argmin(d).item())

        rows["group_size"].append(int(member.numel()))
        rows["margin"].append(top1 - top2)
        rows["top1"].append(top1)
        rows["top2"].append(top2)
        rows["winner_is_nearest"].append(float(int(order[0].item()) == nearest_local))
        rows["winner_dist"].append(float(d[order[0]].item()))
        rows["nearest_dist"].append(float(d[nearest_local].item()))

        rho = _spearman(s, d)
        if rho is not None:
            rows["spearman_score_vs_dist"].append(rho)

        d_rank = _rank(d)
        rows["winner_dist_rank_norm"].append(
            float(d_rank[order[0]].item()) / float(member.numel() - 1)
        )

        # Within-group z-scores, pooled later into one robust correlation that
        # is not dominated by tiny groups the way per-group Spearman is.
        s_std = float(s.std(unbiased=False).item())
        d_std = float(d.std(unbiased=False).item())
        if s_std > 1e-9 and d_std > 1e-9:
            rows["_z_score"].extend(((s - s.mean()) / s_std).tolist())
            rows["_z_dist"].extend(((d - d.mean()) / d_std).tolist())

    rows["_singleton_groups"] = singles
    rows["_empty_groups"] = empty
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size. Use a small value if a training run holds the GPU.")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Stop early for a quick probe.")
    parser.add_argument("--group_radius", type=float, default=None,
                        help="GT group radius in pixels. Defaults to val_ddim_cover_radius.")
    parser.add_argument("--output_json", default=None)
    args_cli = parser.parse_args()

    cfg = load_config(args_cli.config)
    cfg["ckpt_path"] = args_cli.ckpt_path
    cfg["resume_training"] = False
    if args_cli.batch_size is not None:
        cfg["batch_size"] = int(args_cli.batch_size)
    if args_cli.num_workers is not None:
        cfg["num_workers"] = int(args_cli.num_workers)
        cfg["val_num_workers"] = int(args_cli.num_workers)
    args = SimpleNamespace(**cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    _, val_data = build_dataset(args)
    val_num_workers = int(getattr(args, "val_num_workers", getattr(args, "num_workers", 0)))
    loader_kwargs = {
        "batch_size": int(args.batch_size),
        "shuffle": False,
        "num_workers": val_num_workers,
        "pin_memory": bool(getattr(args, "val_pin_memory", True)),
        "collate_fn": collate_points_padded,
    }
    if val_num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(getattr(args, "val_persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(getattr(args, "val_prefetch_factor", 2))
    val_loader = DataLoader(val_data, **loader_kwargs)

    model = build_model(args, training=True).to(device)
    configure_trainable_params(model, args)
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    state_dict = (
        checkpoint["model_state"]
        if isinstance(checkpoint, dict) and "model_state" in checkpoint
        else checkpoint
    )
    incompatible = load_model_state(
        model,
        state_dict,
        shape_compatible_only=bool(getattr(args, "load_shape_compatible_only", False)),
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            "[WARN] checkpoint loaded with non-strict keys | "
            f"missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}"
        )
    model.eval()

    comp = getattr(model, "selector_local_competition", None)
    comp_config = None
    if comp is not None:
        strength = float(comp._effective_strength(torch.float32, torch.device("cpu")).item())
        comp_config = {
            "radius_px": float(comp.radius),
            "temperature": float(comp.temperature),
            "max_strength": float(comp.max_strength),
            "residual_scale": float(comp.residual_scale),
            "exclude_self": bool(comp.exclude_self),
            "learned_effective_strength": strength,
            "init_strength_from_config": float(
                getattr(args, "selector_local_competition_init_strength", float("nan"))
            ),
        }

    sched, _ = Diffusion_schedule(args.diffusion_T, device=device, signal_scale=args.signal_scale)
    abar_all = sched.abar.to(device=device)
    T = int(args.diffusion_T)
    steps = int(args.ddim_steps)
    eta = float(getattr(args, "ddim_eta", 0.0))
    start_t = (
        max(0, min(int(getattr(args, "proposal_prior_start_t", T - 1)), T - 1))
        if bool(getattr(args, "use_proposal_prior", False))
        else T - 1
    )
    t_seq = torch.linspace(start_t, 0, steps, device=device).round().long()
    t_seq = torch.unique_consecutive(t_seq)
    clamp_eps = 1e-3
    group_radius = float(
        args_cli.group_radius
        if args_cli.group_radius is not None
        else getattr(args, "val_ddim_cover_radius", 6.0)
    )

    margin_rows = {
        "group_size": [], "margin": [], "top1": [], "top2": [],
        "winner_is_nearest": [], "winner_dist": [], "nearest_dist": [],
        "spearman_score_vs_dist": [], "winner_dist_rank_norm": [],
    }
    margin_rows_pre = {k: [] for k in margin_rows}
    z_score_pool = []
    z_dist_pool = []
    singleton_groups = 0
    empty_groups = 0

    shift_with_nb = []
    shift_no_nb = []
    neighbor_counts = []
    # Per-group: does the pre-competition leader get suppressed less than the rest?
    winner_shift = []
    loser_shift = []

    # probe5: if a head learned to rank purely by one feature, how often would the
    # group winner be the candidate nearest to GT? Upper bound per feature.
    oracle_hits = defaultdict(int)
    oracle_hits_rev = defaultdict(int)
    oracle_total = 0

    cos_within_cluster = []      # merged-away pairs, measured before merging
    cos_within_gt_group = []      # surviving same-GT-group pairs, after merging
    cos_random_pairs = []         # unrelated valid pairs, after merging
    cluster_sizes = []
    slots_dropped = []

    with torch.no_grad(), CompetitionProbe(comp) as probe, MergeProbe() as mprobe:
        iterator = enumerate(tqdm(val_loader, desc="selector competition probe"))
        for bi, (images, points_pad, mask, metas) in iterator:
            if args_cli.max_batches is not None and bi >= int(args_cli.max_batches):
                break

            images = images.to(device, non_blocking=True)
            points_pad = points_pad.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            B, _, H, W = images.shape
            N = points_pad.size(1)

            feats = model.encode(images)
            cond_cache = model.cond.precompute(*feats)
            selector_prior_maps = None
            if bool(getattr(args, "use_proposal_prior", False)):
                prior_occupancy_logits, prior_density = model.predict_proposal_prior(feats)
                selector_prior_maps = (prior_occupancy_logits, prior_density)

            # Match validate_one_epoch RNG/order: a supervised branch runs before
            # DDIM. Keep the probe off so it does not pollute the statistics.
            probe.enabled = False
            mprobe.enabled = False
            p0 = pixels_to_m11(points_pad, H, W)
            t_fixed = torch.full((B, 1), 500, device=device, dtype=torch.long)
            p_t_supervised, _, abar_supervised = forward_noisy(p0, t_fixed, sched)
            if abar_supervised.dim() in (1, 2):
                abar_supervised = abar_supervised.view(B, 1, 1)
            model.denoise(
                feats, p_t_supervised, t_fixed,
                abar_t=abar_supervised.to(device=device),
                clamp_eps=1e-6, cond_cache=cond_cache, need_exist=True,
                selector_prior_maps=selector_prior_maps,
            )

            if bool(getattr(args, "use_proposal_prior", False)):
                x0_prior, _ = build_mixed_x0_prior(
                    prior_occupancy_logits, prior_density,
                    num_slots=N, clamp_eps=clamp_eps,
                    mode=str(getattr(args, "proposal_prior_mode", "occupancy")),
                    density_cell_capacity=int(getattr(args, "proposal_prior_cell_capacity", 2)),
                )
                t_init = torch.full((B, 1), int(t_seq[0].item()), device=device, dtype=torch.long)
                p_t_gen, _, _ = forward_noisy(x0_prior, t_init, sched)
            else:
                p_t_gen = torch.empty((B, N, 2), device=device).uniform_(
                    -1.0 + clamp_eps, 1.0 - clamp_eps
                )

            pred_points_x0 = None
            pred_valid_x0 = None
            for si, ti in enumerate(t_seq.tolist()):
                ti = int(ti)
                t_tensor = torch.full((B, 1), ti, device=device, dtype=torch.long)
                abar_t = abar_all[ti].view(1, 1, 1).expand(B, 1, 1)
                need_exist = si == len(t_seq) - 1
                probe.enabled = bool(need_exist)
                mprobe.enabled = bool(need_exist)
                eps_hat, _, _, pred_points_for_cls, pred_valid_mask = model.denoise(
                    feats, p_t_gen, t_tensor,
                    abar_t=abar_t, clamp_eps=1e-6, cond_cache=cond_cache,
                    need_exist=need_exist, selector_prior_maps=selector_prior_maps,
                )
                probe.enabled = False
                mprobe.enabled = False
                if need_exist:
                    pred_points_x0 = pred_points_for_cls
                    pred_valid_x0 = pred_valid_mask

                if si + 1 < len(t_seq):
                    ti_prev = int(t_seq[si + 1].item())
                    abar_prev = abar_all[ti_prev].view(1, 1, 1).expand(B, 1, 1)
                    eta_step = eta
                else:
                    abar_prev = torch.ones((B, 1, 1), device=device)
                    eta_step = 0.0
                p_t_gen = ddim_step(p_t_gen, eps_hat, abar_t, abar_prev, eta_step).clamp(
                    min=-1.0 + clamp_eps, max=1.0 - clamp_eps
                )

            if probe.last is None:
                raise RuntimeError(
                    "LocalCompetitionSelector was never invoked. "
                    "Check selector_local_competition in the config."
                )
            cap = probe.last
            pre_score = cap["pre"]
            post_score = cap["post"]
            shift = cap["shift"]
            valid = cap["valid"]
            nb_count = cap["neighbor_count"]

            # ---- probe 3: score_shift split by neighbor presence ----
            has_nb = (nb_count > 0) & valid
            no_nb = (nb_count == 0) & valid
            if has_nb.any():
                shift_with_nb.append(shift[has_nb].float().cpu().numpy())
                neighbor_counts.append(nb_count[has_nb].float().cpu().numpy())
            if no_nb.any():
                shift_no_nb.append(shift[no_nb].float().cpu().numpy())

            # ---- probe 2: per-GT group margins, in tile pixel space ----
            cand_pix_all = m11_to_pixels_batch(pred_points_x0.detach(), H, W)

            relgeom_all = selector_relative_geometry_features(
                pred_points_x0.detach(),
                pred_valid_x0 if pred_valid_x0 is not None else valid,
                k=int(getattr(args, "selector_relgeom_k", 8)),
                H=H, W=W,
            )
            prior_feat_all = sample_selector_prior_features(
                selector_prior_maps,
                pred_points_x0.detach(),
                pred_valid_x0 if pred_valid_x0 is not None else valid,
            )
            for b in range(B):
                gmask = mask[b].bool()
                if not gmask.any():
                    continue
                vmask = valid[b]
                if pred_valid_x0 is not None:
                    vmask = vmask & pred_valid_x0[b].bool()
                if not vmask.any():
                    continue

                gt_pix_b = points_pad[b, gmask].float()
                cand_pix_b = cand_pix_all[b, vmask].float()
                post_b = post_score[b, vmask].float()
                pre_b = pre_score[b, vmask].float()
                shift_b = shift[b, vmask].float()

                rows = group_margin_stats(gt_pix_b, cand_pix_b, post_b, group_radius)
                if rows is not None:
                    for key in margin_rows:
                        margin_rows[key].extend(rows[key])
                    z_score_pool.extend(rows["_z_score"])
                    z_dist_pool.extend(rows["_z_dist"])
                    singleton_groups += rows["_singleton_groups"]
                    empty_groups += rows["_empty_groups"]

                if mprobe.last is not None:
                    mcap = mprobe.last
                    inv_b = mcap["inverse"][b]
                    raw_pro_b = mcap["raw_pro"][b]
                    mm_b = mcap["merged_mask"][b].bool()
                    merged_pro_b = mcap["merged_pro"][b]
                    merged_xy_b = mcap["merged_xy"][b]

                    n_clusters = int(inv_b.max().item()) + 1 if inv_b.numel() else 0
                    slots_dropped.append(int(inv_b.numel() - n_clusters))
                    counts = torch.bincount(inv_b, minlength=max(n_clusters, 1))
                    cluster_sizes.extend(counts[:n_clusters].tolist())

                    # Pairs the merge collapses into one slot: their feature
                    # differences are destroyed outright.
                    for ci in (counts >= 2).nonzero(as_tuple=False).squeeze(1).tolist():
                        mem = (inv_b == ci).nonzero(as_tuple=False).squeeze(1)
                        cos_within_cluster.append(
                            _pairwise_cos(raw_pro_b[mem]).cpu().numpy()
                        )

                    mvalid_idx = mm_b.nonzero(as_tuple=False).squeeze(1)
                    if mvalid_idx.numel() >= 2:
                        mxy_pix = m11_to_pixels_batch(
                            merged_xy_b[mvalid_idx].unsqueeze(0), H, W
                        ).squeeze(0)
                        dg = torch.cdist(gt_pix_b, mxy_pix, p=2)
                        wg = dg <= group_radius
                        for gi in range(wg.size(0)):
                            mem = wg[gi].nonzero(as_tuple=False).squeeze(1)
                            if mem.numel() >= 2:
                                cos_within_gt_group.append(
                                    _pairwise_cos(merged_pro_b[mvalid_idx[mem]]).cpu().numpy()
                                )
                        # Strided rather than random, so the probe does not consume
                        # RNG and shift the prior draws of later batches.
                        step = max(1, mvalid_idx.numel() // 32)
                        ref = mvalid_idx[torch.arange(
                            0, mvalid_idx.numel(), step, device=mvalid_idx.device
                        )][:32]
                        if ref.numel() >= 2:
                            cos_random_pairs.append(
                                _pairwise_cos(merged_pro_b[ref]).cpu().numpy()
                            )

                # ---- probe 5: per-feature oracle ceiling ----
                rg_b = relgeom_all[b, vmask].float()
                criteria = {
                    "selector_score_current": post_b,
                    "centroid_dist": rg_b[:, 2],
                    "centroid_rank": rg_b[:, 16],
                    "local_scale": rg_b[:, 15],
                    "nn1_dist": rg_b[:, 3 + 2],
                }
                if prior_feat_all is not None:
                    pf_b = prior_feat_all[b, vmask].float()
                    criteria["prior_occupancy"] = pf_b[:, 0]
                    criteria["prior_density"] = pf_b[:, 1]

                d_or = torch.cdist(gt_pix_b, cand_pix_b, p=2)
                w_or = d_or <= group_radius
                for gi in range(w_or.size(0)):
                    mem = w_or[gi].nonzero(as_tuple=False).squeeze(1)
                    if mem.numel() < 2:
                        continue
                    nearest_local = int(torch.argmin(d_or[gi, mem]).item())
                    oracle_total += 1
                    for cname, cval in criteria.items():
                        v = cval[mem]
                        if int(torch.argmax(v).item()) == nearest_local:
                            oracle_hits[cname] += 1
                        if int(torch.argmin(v).item()) == nearest_local:
                            oracle_hits_rev[cname] += 1

                rows_pre = group_margin_stats(gt_pix_b, cand_pix_b, pre_b, group_radius)
                if rows_pre is not None:
                    for key in margin_rows_pre:
                        margin_rows_pre[key].extend(rows_pre[key])

                # Winner-vs-loser shift, grouped by the pre-competition leader.
                d = torch.cdist(gt_pix_b, cand_pix_b, p=2)
                within = d <= group_radius
                for gi in range(within.size(0)):
                    member = within[gi].nonzero(as_tuple=False).squeeze(1)
                    if member.numel() < 2:
                        continue
                    lead = int(torch.argmax(pre_b[member]).item())
                    winner_shift.append(float(shift_b[member[lead]].item()))
                    others = torch.ones(member.numel(), dtype=torch.bool)
                    others[lead] = False
                    loser_shift.append(float(shift_b[member[others]].mean().item()))

            probe.last = None

    # ---------------- aggregate ----------------
    margins = np.asarray(margin_rows["margin"], dtype=np.float64)
    margins_pre = np.asarray(margin_rows_pre["margin"], dtype=np.float64)
    top1 = np.asarray(margin_rows["top1"], dtype=np.float64)
    top2 = np.asarray(margin_rows["top2"], dtype=np.float64)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))

    prob_margin = sigmoid(top1) - sigmoid(top2) if top1.size else np.zeros(0)

    shift_nb = np.concatenate(shift_with_nb) if shift_with_nb else np.zeros(0)
    shift_alone = np.concatenate(shift_no_nb) if shift_no_nb else np.zeros(0)
    nb_hist = np.concatenate(neighbor_counts) if neighbor_counts else np.zeros(0)

    # ---- random baselines, computed from the observed group-size distribution ----
    sizes_arr = np.asarray(margin_rows["group_size"], dtype=np.int64)
    size_hist = {}
    random_nearest_rate = None
    if sizes_arr.size:
        uniq, cnt = np.unique(sizes_arr, return_counts=True)
        size_hist = {int(u): int(c) for u, c in zip(uniq, cnt)}
        # Picking a winner uniformly at random hits the nearest member with
        # probability 1/size, so the baseline is E[1/size] over the same groups.
        random_nearest_rate = float(np.mean(1.0 / sizes_arr))

    zs = np.asarray(z_score_pool, dtype=np.float64)
    zd = np.asarray(z_dist_pool, dtype=np.float64)
    pooled_corr = None
    if zs.size > 1 and zs.std() > 1e-9 and zd.std() > 1e-9:
        pooled_corr = float(np.corrcoef(zs, zd)[0, 1])

    observed_rate = (
        float(np.mean(margin_rows["winner_is_nearest"]))
        if margin_rows["winner_is_nearest"]
        else None
    )

    probe2 = {
        "group_radius_px": group_radius,
        "groups_with_2plus_candidates": int(margins.size),
        "group_size_histogram": size_hist,
        "random_baseline_nearest_rate": random_nearest_rate,
        "winner_is_nearest_lift_over_random": (
            float(observed_rate - random_nearest_rate)
            if (observed_rate is not None and random_nearest_rate is not None)
            else None
        ),
        "spearman_score_vs_dist_per_group": describe(margin_rows["spearman_score_vs_dist"]),
        "pooled_within_group_corr_score_vs_dist": pooled_corr,
        "winner_dist_rank_norm": describe(margin_rows["winner_dist_rank_norm"]),
        "groups_with_1_candidate": int(singleton_groups),
        "groups_with_0_candidates": int(empty_groups),
        "group_size": describe(margin_rows["group_size"]),
        "margin_logit_post_competition": describe(margins),
        "margin_logit_pre_competition": describe(margins_pre),
        "margin_prob_post_competition": describe(prob_margin),
        "top1_logit": describe(top1),
        "top2_logit": describe(top2),
        "winner_is_nearest_to_gt_rate": (
            float(np.mean(margin_rows["winner_is_nearest"])) if margin_rows["winner_is_nearest"] else None
        ),
        "winner_dist_px": describe(margin_rows["winner_dist"]),
        "nearest_dist_px": describe(margin_rows["nearest_dist"]),
    }

    probe3 = {
        "module_config": comp_config,
        "shift_with_neighbor": describe(shift_nb),
        "shift_without_neighbor": describe(shift_alone),
        "neighbor_count_when_present": describe(nb_hist),
        "group_leader_shift": describe(winner_shift),
        "group_nonleader_mean_shift": describe(loser_shift),
    }
    if shift_nb.size and shift_alone.size:
        denom = max(abs(float(shift_nb.mean())), 1e-9)
        probe3["bias_ratio_no_neighbor_over_with_neighbor"] = float(
            abs(float(shift_alone.mean())) / denom
        )
    if winner_shift and loser_shift:
        probe3["leader_minus_nonleader_shift"] = float(
            np.mean(winner_shift) - np.mean(loser_shift)
        )

    # ---------------- verdicts (heuristic thresholds, stated explicitly) ----------------
    verdicts = {}
    if margins.size:
        med = float(np.median(margins))
        if med < 0.1:
            level = "INDISTINGUISHABLE (median logit margin < 0.1)"
        elif med < 0.5:
            level = "WEAK (median logit margin < 0.5)"
        else:
            level = "SEPARABLE (median logit margin >= 0.5)"
        verdicts["probe2_4e_feature_indistinguishable"] = {
            "median_margin_logit": med,
            "median_margin_prob": float(np.median(prob_margin)),
            "level": level,
            "note": "Thresholds are heuristic, referenced against pos_logit targets of 2.0-2.5 in the config.",
        }
    if pooled_corr is not None:
        # Score should fall as distance to GT grows, i.e. a negative correlation.
        if pooled_corr > -0.05:
            level = "NO POSITION SIGNAL (|corr| < 0.05) -> ranking axis is unrelated to GT distance"
        elif pooled_corr > -0.2:
            level = "WEAK POSITION SIGNAL"
        else:
            level = "CLEAR POSITION SIGNAL (score tracks GT proximity)"
        verdicts["probe2b_does_score_track_gt_proximity"] = {
            "pooled_within_group_corr": pooled_corr,
            "mean_per_group_spearman": (
                float(np.mean(margin_rows["spearman_score_vs_dist"]))
                if margin_rows["spearman_score_vs_dist"]
                else None
            ),
            "winner_is_nearest_rate": observed_rate,
            "random_baseline": random_nearest_rate,
            "lift_over_random": (
                float(observed_rate - random_nearest_rate)
                if (observed_rate is not None and random_nearest_rate is not None)
                else None
            ),
            "mean_winner_dist_rank_norm": (
                float(np.mean(margin_rows["winner_dist_rank_norm"]))
                if margin_rows["winner_dist_rank_norm"]
                else None
            ),
            "level": level,
            "implication": (
                "No/weak signal => the representation lacks usable tie-break "
                "information, so fixing loss conflicts alone will not fix winner "
                "selection. Clear signal => the head can rank correctly and the "
                "errors come from conflicting supervision."
            ),
        }

    if shift_nb.size and shift_alone.size:
        ratio = probe3["bias_ratio_no_neighbor_over_with_neighbor"]
        if ratio > 0.5:
            level = "MOSTLY GLOBAL BIAS (no-neighbor shift > 50% of with-neighbor shift)"
        elif ratio > 0.2:
            level = "PARTIAL BIAS"
        else:
            level = "MOSTLY NEIGHBOR-DRIVEN"
        verdicts["probe3_4b_delta_is_global_bias"] = {
            "mean_shift_with_neighbor": float(shift_nb.mean()),
            "mean_shift_without_neighbor": float(shift_alone.mean()),
            "ratio": ratio,
            "level": level,
        }

    cwc = np.concatenate(cos_within_cluster) if cos_within_cluster else np.zeros(0)
    cwg = np.concatenate(cos_within_gt_group) if cos_within_gt_group else np.zeros(0)
    crp = np.concatenate(cos_random_pairs) if cos_random_pairs else np.zeros(0)

    probe4 = {
        "merge_radius_px": float(getattr(args, "selector_merge_radius_px", 0.0)),
        "conf_weighted_merge": bool(getattr(args, "selector_conf_weighted_merge", False)),
        "cluster_size": describe(cluster_sizes),
        "slots_dropped_per_image": describe(slots_dropped),
        "pro_cos_within_merged_cluster": describe(cwc),
        "pro_cos_within_gt_group_after_merge": describe(cwg),
        "pro_cos_reference_pairs_after_merge": describe(crp),
    }
    if cwc.size and crp.size:
        # 1.0 means merged-away pairs were already identical relative to the
        # spread of unrelated pairs, i.e. the merge destroyed nothing.
        span = 1.0 - float(crp.mean())
        probe4["merge_destroyed_fraction_of_available_spread"] = (
            float((1.0 - float(cwc.mean())) / span) if abs(span) > 1e-9 else None
        )

    verdicts_merge = {}
    if cwc.size and cwg.size and crp.size:
        if float(cwc.mean()) > 0.99:
            level = "MERGE IS HARMLESS (collapsed pairs were near-identical)"
        elif float(cwc.mean()) > float(cwg.mean()):
            level = "MERGE DESTROYS REAL VARIATION (but less than same-group spread)"
        else:
            level = "MERGE DESTROYS MORE VARIATION THAN IT PRESERVES"
        verdicts_merge = {
            "mean_cos_collapsed_pairs": float(cwc.mean()),
            "mean_cos_same_gt_group_survivors": float(cwg.mean()),
            "mean_cos_reference_pairs": float(crp.mean()),
            "level": level,
        }
        verdicts["probe4_merge_feature_averaging"] = verdicts_merge

    probe5 = {
        "groups_scored": int(oracle_total),
        "random_baseline": random_nearest_rate,
        "note": (
            "Ranking by each feature alone, taking the better of ascending and "
            "descending order. This is the ceiling a head could reach if it "
            "learned to use that feature perfectly."
        ),
        "ceilings": {},
    }
    if oracle_total > 0:
        for cname in sorted(set(list(oracle_hits.keys()) + list(oracle_hits_rev.keys()))):
            fwd = oracle_hits[cname] / oracle_total
            rev = oracle_hits_rev[cname] / oracle_total
            best = max(fwd, rev)
            probe5["ceilings"][cname] = {
                "descending": float(fwd),
                "ascending": float(rev),
                "best": float(best),
                "lift_over_random": (
                    float(best - random_nearest_rate)
                    if random_nearest_rate is not None else None
                ),
            }

    result = {
        "config": os.path.abspath(args_cli.config),
        "ckpt_path": args_cli.ckpt_path,
        "ddim_steps": steps,
        "proposal_prior_start_t": start_t,
        "batches_used": (int(args_cli.max_batches) if args_cli.max_batches is not None else "all"),
        "probe2_group_winner_margin": probe2,
        "probe3_local_competition_shift": probe3,
        "probe4_merge_feature_averaging": probe4,
        "probe5_feature_oracle_ceiling": probe5,
        "verdicts": verdicts,
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args_cli.output_json:
        out_dir = os.path.dirname(os.path.abspath(args_cli.output_json))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args_cli.output_json, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
