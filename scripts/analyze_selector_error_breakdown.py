import argparse
import json
import os
import sys
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataset import build_dataset
from main import collate_points_padded, configure_trainable_params, load_model_state
from models import Diffusion_schedule, build_model
from models.diffusion_utils import forward_noisy, pixels_to_m11
from models.proposal_prior import build_mixed_x0_prior
from models.train_loop import m11_to_pixels_batch


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_float_list(text):
    if text is None:
        return None
    if isinstance(text, (list, tuple)):
        return [float(v) for v in text]
    raw = str(text).strip()
    if raw == "":
        return None
    return [float(v.strip()) for v in raw.replace(";", ",").split(",") if v.strip()]


def dedup_points_xy(xy_np: np.ndarray, decimals: int = 3) -> np.ndarray:
    if xy_np.shape[0] == 0:
        return xy_np.reshape(0, 2)
    key = np.round(xy_np.reshape(-1, 2), decimals=decimals)
    _, idx = np.unique(key, axis=0, return_index=True)
    return xy_np.reshape(-1, 2)[np.sort(idx)]


def selected_error_breakdown(gt_xy, pred_xy, radius: float):
    gt_xy = np.asarray(gt_xy, dtype=np.float32).reshape(-1, 2)
    pred_xy = np.asarray(pred_xy, dtype=np.float32).reshape(-1, 2)
    n_gt = int(gt_xy.shape[0])
    n_pred = int(pred_xy.shape[0])
    out = {
        "gt": n_gt,
        "selected": n_pred,
        "matched_tp": 0,
        "near_duplicate_fp": 0,
        "far_background_fp": 0,
        "near_selected": 0,
        "unmatched_fp": n_pred,
    }
    if n_gt == 0:
        out["far_background_fp"] = n_pred
        return out
    if n_pred == 0:
        return out

    pred_tree = cKDTree(pred_xy)
    indptr = [0]
    indices = []
    for gt in gt_xy:
        cols = pred_tree.query_ball_point(gt, float(radius))
        indices.extend(cols)
        indptr.append(len(indices))

    matched_pred = np.zeros(n_pred, dtype=bool)
    if indices:
        graph = csr_matrix(
            (
                np.ones(len(indices), dtype=np.bool_),
                np.asarray(indices, dtype=np.int32),
                np.asarray(indptr, dtype=np.int32),
            ),
            shape=(n_gt, n_pred),
        )
        col_to_row = maximum_bipartite_matching(graph, perm_type="row")
        matched_pred = col_to_row >= 0

    gt_tree = cKDTree(gt_xy)
    nearest_dist, _ = gt_tree.query(pred_xy, k=1)
    near_pred = nearest_dist <= float(radius)

    out["matched_tp"] = int(matched_pred.sum())
    out["near_selected"] = int(near_pred.sum())
    out["near_duplicate_fp"] = int((near_pred & ~matched_pred).sum())
    out["far_background_fp"] = int((~near_pred).sum())
    out["unmatched_fp"] = int(n_pred - matched_pred.sum())
    return out


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--thresholds", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--include_per_image", action="store_true")
    args_cli = parser.parse_args()

    cfg = load_config(args_cli.config)
    cfg["ckpt_path"] = args_cli.ckpt_path
    cfg["resume_training"] = False
    if args_cli.batch_size is not None:
        cfg["batch_size"] = int(args_cli.batch_size)
    if args_cli.num_workers is not None:
        cfg["num_workers"] = int(args_cli.num_workers)
        cfg["val_num_workers"] = int(args_cli.num_workers)
    thresholds = parse_float_list(args_cli.thresholds)
    if thresholds is None:
        thresholds = parse_float_list(cfg.get("threshold_sweep", None))
    if thresholds is None:
        thresholds = [float(cfg.get("hard_thresh", 0.5))]
    thresholds = sorted({float(v) for v in thresholds})
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
    state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
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
    radius = float(getattr(args, "val_ddim_cover_radius", 6.0))
    gate_mode = str(getattr(args, "test_gate_mode", "prob_only"))

    pred_xy_full = defaultdict(list)
    pred_sc_full = defaultdict(list)
    gt_xy_full = defaultdict(list)

    with torch.no_grad():
        for images, points_pad, mask, metas in tqdm(val_loader, desc="DDIM selector breakdown"):
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

            # Match validate_one_epoch RNG/order: a supervised branch is run
            # before DDIM, including one forward_noisy draw and one denoise call.
            p0 = pixels_to_m11(points_pad, H, W)
            t_fixed = torch.full((B, 1), 500, device=device, dtype=torch.long)
            p_t_supervised, _, abar_supervised = forward_noisy(p0, t_fixed, sched)
            if abar_supervised.dim() == 1:
                abar_supervised = abar_supervised.view(B, 1, 1)
            elif abar_supervised.dim() == 2:
                abar_supervised = abar_supervised.view(B, 1, 1)
            model.denoise(
                feats,
                p_t_supervised,
                t_fixed,
                abar_t=abar_supervised.to(device=device),
                clamp_eps=1e-6,
                cond_cache=cond_cache,
                need_exist=True,
                selector_prior_maps=selector_prior_maps,
            )

            if bool(getattr(args, "use_proposal_prior", False)):
                x0_prior, _ = build_mixed_x0_prior(
                    prior_occupancy_logits,
                    prior_density,
                    num_slots=N,
                    clamp_eps=clamp_eps,
                    mode=str(getattr(args, "proposal_prior_mode", "occupancy")),
                    density_cell_capacity=int(getattr(args, "proposal_prior_cell_capacity", 2)),
                )
                t_init = torch.full((B, 1), int(t_seq[0].item()), device=device, dtype=torch.long)
                p_t_gen, _, _ = forward_noisy(
                    x0_prior,
                    t_init,
                    sched,
                )
            else:
                p_t_gen = torch.empty((B, N, 2), device=device).uniform_(
                    -1.0 + clamp_eps,
                    1.0 - clamp_eps,
                )

            exist_logit_x0 = None
            pred_points_x0 = None
            pred_valid_x0 = None
            for si, ti in enumerate(t_seq.tolist()):
                ti = int(ti)
                t_tensor = torch.full((B, 1), ti, device=device, dtype=torch.long)
                abar_t = abar_all[ti].view(1, 1, 1).expand(B, 1, 1)
                need_exist = si == len(t_seq) - 1
                eps_hat, exist_logit_t, _, pred_points_for_cls, pred_valid_mask = model.denoise(
                    feats,
                    p_t_gen,
                    t_tensor,
                    abar_t=abar_t,
                    clamp_eps=1e-6,
                    cond_cache=cond_cache,
                    need_exist=need_exist,
                    selector_prior_maps=selector_prior_maps,
                )
                if need_exist:
                    exist_logit_x0 = exist_logit_t
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
                    min=-1.0 + clamp_eps,
                    max=1.0 - clamp_eps,
                )

            if exist_logit_x0 is None or pred_points_x0 is None:
                raise RuntimeError("DDIM loop did not return selector logits and points")
            if exist_logit_x0.dim() == 3 and exist_logit_x0.size(-1) == 2:
                prob = exist_logit_x0.softmax(-1)[..., 1]
                if gate_mode == "argmax_only":
                    base_cand = exist_logit_x0.argmax(-1) == 1
                elif gate_mode == "argmax_or_prob":
                    base_cand = exist_logit_x0.argmax(-1) == 1
                else:
                    base_cand = torch.ones_like(prob, dtype=torch.bool)
            else:
                prob = torch.sigmoid(exist_logit_x0.squeeze(-1) if exist_logit_x0.dim() == 3 else exist_logit_x0)
                base_cand = torch.ones_like(prob, dtype=torch.bool)
            if pred_valid_x0 is not None:
                base_cand = base_cand & pred_valid_x0.bool()

            x0_pix = m11_to_pixels_batch(pred_points_x0.detach(), H, W)
            for b in range(B):
                meta = metas[b] if isinstance(metas, (list, tuple)) else metas
                img_idx = int(meta["img_index"])
                top = float(meta["tile_top"])
                left = float(meta["tile_left"])
                h_full, w_full = meta["orig_size"]

                gt_tile = points_pad[b, mask[b]]
                if gt_tile.numel() > 0:
                    gt = gt_tile.clone()
                    gt[:, 0] += left
                    gt[:, 1] += top
                    gt_xy_full[img_idx].append(gt.detach().cpu())

                cand = base_cand[b]
                if cand.any():
                    xy = x0_pix[b, cand].clone()
                    xy[:, 0] = (xy[:, 0] + left).clamp(0, float(w_full) - 1)
                    xy[:, 1] = (xy[:, 1] + top).clamp(0, float(h_full) - 1)
                    pred_xy_full[img_idx].append(xy.detach().cpu())
                    pred_sc_full[img_idx].append(prob[b, cand].detach().cpu())

    img_ids = sorted(set(list(gt_xy_full.keys()) + list(pred_xy_full.keys())))
    totals_by_thr = {
        float(t): {
            "gt": 0,
            "selected": 0,
            "matched_tp": 0,
            "near_duplicate_fp": 0,
            "far_background_fp": 0,
            "near_selected": 0,
            "unmatched_fp": 0,
        }
        for t in thresholds
    }
    per_image = []
    for img_idx in img_ids:
        gt_xy = (
            torch.cat(gt_xy_full[img_idx], dim=0).numpy()
            if len(gt_xy_full[img_idx]) > 0
            else np.zeros((0, 2), dtype=np.float32)
        )
        gt_xy = dedup_points_xy(gt_xy, decimals=3)
        xy_all = (
            torch.cat(pred_xy_full[img_idx], dim=0)
            if len(pred_xy_full[img_idx]) > 0
            else torch.zeros((0, 2), dtype=torch.float32)
        )
        sc_all = (
            torch.cat(pred_sc_full[img_idx], dim=0)
            if len(pred_sc_full[img_idx]) > 0
            else torch.zeros((0,), dtype=torch.float32)
        )
        for thr in thresholds:
            keep = sc_all > float(thr)
            xy = xy_all[keep].numpy() if keep.any() else np.zeros((0, 2), dtype=np.float32)
            stats = selected_error_breakdown(gt_xy, xy, radius=radius)
            total = totals_by_thr[float(thr)]
            for key in total:
                total[key] += int(stats[key])
            per_image.append({"img_index": int(img_idx), "threshold": float(thr), **stats})

    summary_by_thr = {}
    for thr, total in totals_by_thr.items():
        gt = max(int(total["gt"]), 1)
        selected = max(int(total["selected"]), 1)
        fp = max(int(total["unmatched_fp"]), 1)
        summary_by_thr[str(thr)] = {
            **total,
            "recall": float(total["matched_tp"] / gt),
            "precision": float(total["matched_tp"] / selected),
            "selected_per_gt": float(total["selected"] / gt),
            "near_duplicate_fp_per_gt": float(total["near_duplicate_fp"] / gt),
            "far_background_fp_per_gt": float(total["far_background_fp"] / gt),
            "near_duplicate_fp_fraction_of_fp": float(total["near_duplicate_fp"] / fp),
            "far_background_fp_fraction_of_fp": float(total["far_background_fp"] / fp),
        }

    result = {
        "config": os.path.abspath(args_cli.config),
        "ckpt_path": args_cli.ckpt_path,
        "radius": radius,
        "ddim_steps": steps,
        "proposal_prior_start_t": start_t,
        "thresholds": thresholds,
        "summary_by_threshold": summary_by_thr,
    }
    if args_cli.include_per_image:
        result["per_image"] = per_image
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
