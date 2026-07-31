"""Check what exist_pos_radius actually labels, against the 6px eval radius.

build_region_representative_targets assigns every candidate within
exist_pos_radius (currently 32px) to its nearest GT: the closest one becomes the
positive, all the rest become role=2 "duplicates". Evaluation, however, calls
anything farther than val_ddim_cover_radius (6px) from every GT a
far-background false positive.

This script reproduces the training-time random-start branch (prior -> noise at
rand_cover_t_min -> one denoise pass, matching models/train_loop.py) and reports:

  * how much of the role=2 population actually sits beyond the eval radius
  * what confidence those mislabelled points receive, i.e. how many become FPs
  * how the role split would change under smaller exist_pos_radius values

Read-only: no checkpoint, config or training state is modified.
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataset import build_dataset
from main import collate_points_padded, configure_trainable_params, load_model_state
from models import Diffusion_schedule, build_model
from models.diffusion_utils import (
    build_region_representative_targets,
    forward_noisy,
    pixels_to_m11,
)
from models.pointdiff import selector_object_prob
from models.proposal_prior import build_mixed_x0_prior


def describe(values, percentiles=(1, 5, 25, 50, 75, 95, 99)):
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    out = {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
    for p in percentiles:
        out[f"p{p}"] = float(np.percentile(arr, p))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--radii", default=None,
                        help="exist_pos_radius values to compare, e.g. '32,8,6'. "
                             "Defaults to the config value plus 8 and 6.")
    parser.add_argument("--eval_radius", type=float, default=None,
                        help="Defaults to val_ddim_cover_radius.")
    parser.add_argument("--score_thresh", type=float, default=None,
                        help="Defaults to hard_thresh used at validation (0.6).")
    parser.add_argument("--output_json", default=None)
    args_cli = parser.parse_args()

    cfg = yaml.safe_load(open(args_cli.config, encoding="utf-8"))
    cfg["ckpt_path"] = args_cli.ckpt_path
    cfg["resume_training"] = False
    if args_cli.batch_size is not None:
        cfg["batch_size"] = int(args_cli.batch_size)
    if args_cli.num_workers is not None:
        cfg["num_workers"] = int(args_cli.num_workers)
        cfg["val_num_workers"] = int(args_cli.num_workers)
    args = SimpleNamespace(**cfg)

    eval_radius = float(
        args_cli.eval_radius
        if args_cli.eval_radius is not None
        else getattr(args, "val_ddim_cover_radius", 6.0)
    )
    cfg_radius = float(getattr(args, "exist_pos_radius", 32.0))
    if args_cli.radii:
        radii = [float(v) for v in str(args_cli.radii).replace(";", ",").split(",") if v.strip()]
    else:
        radii = sorted({cfg_radius, 8.0, 6.0}, reverse=True)
    score_thresh = float(
        args_cli.score_thresh if args_cli.score_thresh is not None else 0.6
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    _, val_data = build_dataset(args)
    nw = int(getattr(args, "val_num_workers", getattr(args, "num_workers", 0)))
    loader_kwargs = {
        "batch_size": int(args.batch_size), "shuffle": False, "num_workers": nw,
        "pin_memory": True, "collate_fn": collate_points_padded,
    }
    if nw > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    val_loader = DataLoader(val_data, **loader_kwargs)

    model = build_model(args, training=True).to(device)
    configure_trainable_params(model, args)
    ck = torch.load(args.ckpt_path, map_location=device)
    sd = ck["model_state"] if isinstance(ck, dict) and "model_state" in ck else ck
    inc = load_model_state(
        model, sd, shape_compatible_only=bool(getattr(args, "load_shape_compatible_only", False))
    )
    if inc.missing_keys or inc.unexpected_keys:
        print(f"[WARN] non-strict load | missing={inc.missing_keys} unexpected={inc.unexpected_keys}")
    model.eval()

    sched, _ = Diffusion_schedule(args.diffusion_T, device=device, signal_scale=args.signal_scale)
    T_int = int(args.diffusion_T)
    t_low = max(0, min(int(getattr(args, "rand_cover_t_min", 50)), T_int - 1))
    t_high = max(t_low, min(int(getattr(args, "rand_cover_t_max", 50)), T_int - 1))

    # per-radius accumulators
    stats = {r: {"pos": 0, "dup": 0, "bg": 0,
                 "dup_beyond_eval": 0, "dup_within_eval": 0,
                 "dup_beyond_eval_scores": [], "dup_within_eval_scores": [],
                 "bg_scores": [], "pos_scores": [],
                 "dup_dists": []} for r in radii}
    valid_total = 0

    with torch.no_grad():
        for bi, (images, points_pad, mask, metas) in enumerate(
                tqdm(val_loader, desc="exist label assignment")):
            if args_cli.max_batches is not None and bi >= int(args_cli.max_batches):
                break
            images = images.to(device, non_blocking=True)
            points_pad = points_pad.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            B, _, H, W = images.shape

            feats = model.encode(images)
            cond_cache = model.cond.precompute(*feats)
            prior_maps = None
            if bool(getattr(args, "use_proposal_prior", False)):
                occ_logits, dens = model.predict_proposal_prior(feats)
                prior_maps = (occ_logits, dens)

            p0 = pixels_to_m11(points_pad, H, W)

            # Mirror the training random-start branch.
            t_rand = torch.randint(low=t_low, high=t_high + 1, size=(B, 1),
                                   device=device, dtype=torch.long)
            if prior_maps is not None:
                x0_prior, _ = build_mixed_x0_prior(
                    occ_logits, dens, num_slots=p0.size(1), clamp_eps=1e-3,
                    mode=str(getattr(args, "proposal_prior_mode", "occupancy")),
                    density_cell_capacity=int(getattr(args, "proposal_prior_cell_capacity", 2)),
                )
                p_rand, _, _ = forward_noisy(x0_prior, t_rand, sched)
            else:
                p_rand = torch.empty_like(p0).uniform_(-1.0 + 1e-3, 1.0 - 1e-3)
            abar_rand = sched.get(t_rand).unsqueeze(-1)

            _, rand_logit, _, rand_x0, rand_valid = model.denoise(
                feats, p_rand, t_rand, abar_t=abar_rand, clamp_eps=1e-6,
                cond_cache=cond_cache, need_exist=True, selector_prior_maps=prior_maps,
            )
            prob = selector_object_prob(torch.clamp(rand_logit, -30.0, 30.0))
            vmask = rand_valid.bool() if rand_valid is not None else torch.ones_like(prob, dtype=torch.bool)
            valid_total += int(vmask.sum().item())

            for r in radii:
                _, _, _, info = build_region_representative_targets(
                    rand_x0, p0, mask, pred_valid_mask=rand_valid,
                    H=H, W=W, region_radius=float(r), return_roles=True,
                )
                pos_m = info["positive_mask"]
                dup_m = info["duplicate_mask"]
                bg_m = info["background_mask"]
                nd = info["nearest_dist_px"]

                s = stats[r]
                s["pos"] += int(pos_m.sum().item())
                s["dup"] += int(dup_m.sum().item())
                s["bg"] += int(bg_m.sum().item())

                beyond = dup_m & (nd > eval_radius)
                within = dup_m & (nd <= eval_radius)
                s["dup_beyond_eval"] += int(beyond.sum().item())
                s["dup_within_eval"] += int(within.sum().item())
                if beyond.any():
                    s["dup_beyond_eval_scores"].append(prob[beyond].float().cpu().numpy())
                if within.any():
                    s["dup_within_eval_scores"].append(prob[within].float().cpu().numpy())
                if bg_m.any():
                    s["bg_scores"].append(prob[bg_m].float().cpu().numpy())
                if pos_m.any():
                    s["pos_scores"].append(prob[pos_m].float().cpu().numpy())
                if dup_m.any():
                    s["dup_dists"].append(nd[dup_m].float().cpu().numpy())

    def cat(x):
        return np.concatenate(x) if x else np.zeros(0)

    summary = {}
    for r in radii:
        s = stats[r]
        dup = max(s["dup"], 1)
        beyond_scores = cat(s["dup_beyond_eval_scores"])
        within_scores = cat(s["dup_within_eval_scores"])
        bg_scores = cat(s["bg_scores"])
        pos_scores = cat(s["pos_scores"])
        summary[str(r)] = {
            "counts": {"positive": s["pos"], "duplicate": s["dup"],
                       "background": s["bg"], "valid_total": valid_total},
            "duplicate_beyond_eval_radius": s["dup_beyond_eval"],
            "duplicate_within_eval_radius": s["dup_within_eval"],
            "fraction_of_duplicates_beyond_eval_radius": float(s["dup_beyond_eval"] / dup),
            "duplicate_nearest_dist_px": describe(cat(s["dup_dists"])),
            "score_duplicate_beyond_eval": describe(beyond_scores),
            "score_duplicate_within_eval": describe(within_scores),
            "score_background": describe(bg_scores),
            "score_positive": describe(pos_scores),
            f"duplicate_beyond_eval_over_thresh_{score_thresh}": (
                float((beyond_scores > score_thresh).mean()) if beyond_scores.size else None
            ),
            f"positive_over_thresh_{score_thresh}": (
                float((pos_scores > score_thresh).mean()) if pos_scores.size else None
            ),
        }

    result = {
        "config": os.path.abspath(args_cli.config),
        "ckpt_path": args_cli.ckpt_path,
        "config_exist_pos_radius": cfg_radius,
        "eval_radius_px": eval_radius,
        "score_thresh": score_thresh,
        "branch": "training random-start (prior -> forward_noisy at rand_cover_t -> one denoise)",
        "by_exist_pos_radius": summary,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args_cli.output_json:
        d = os.path.dirname(os.path.abspath(args_cli.output_json))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(args_cli.output_json, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
