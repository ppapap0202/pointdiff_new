import argparse
import csv
import os
from collections import defaultdict, deque
from itertools import product

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

from dataset.dataset import ImageDataset
from models import build_model
from models.diffusion_utils import CosineAbarSchedule
from models.proposal_prior import build_mixed_x0_prior
from test import (
    collate_points_padded,
    dedup_points_xy,
    ddim_reverse_step_eta,
    get_image_key_from_meta,
    load_checkpoint_into_model,
    make_ddim_steps,
    nearest_gt_distance_stats,
    set_seed,
)


def parse_csv_ints(text):
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_csv_floats(text):
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_csv_strings(text):
    return [x.strip() for x in str(text).split(",") if x.strip()]


STAGE_XY_KEYS = ("raw_xy", "merged_xy")


def make_stage_cache_path(cache_path):
    root, ext = os.path.splitext(cache_path)
    ext = ext or ".pt"
    return f"{root}_stage_xy{ext}"


def strip_stage_xy(cache):
    images = {}
    for img_key, item in cache["images"].items():
        images[img_key] = {
            key: value
            for key, value in item.items()
            if key not in STAGE_XY_KEYS
        }

    meta = dict(cache.get("meta", {}))
    meta["contains_stage_xy"] = False
    return {
        **cache,
        "meta": meta,
        "images": images,
    }


def parse_args():
    def load_config(yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    base = argparse.ArgumentParser()
    base.add_argument("--config", default=r"config/train.yaml", type=str)
    args0, _ = base.parse_known_args()

    cfg = load_config(args0.config)
    parser = argparse.ArgumentParser(parents=[base], add_help=False)
    for key, value in cfg.items():
        parser.add_argument(f"--{key}", type=type(value), default=value)

    parser.add_argument("--scan_save_dir", type=str, default="")
    parser.add_argument("--scan_cache_path", type=str, default="")
    parser.add_argument("--refresh_cache", action="store_true")
    parser.add_argument("--scan_low_score_thresh", type=float, default=0.01)
    parser.add_argument("--scan_save_stage_xy", action="store_true")

    parser.add_argument("--scan_k", type=str, default=",21")
    parser.add_argument("--scan_dense_kdist_thresh", type=str, default="2.0")
    parser.add_argument("--scan_alpha", type=str, default="0.65")
    parser.add_argument("--scan_min_eps", type=str, default="2.0")
    parser.add_argument("--scan_max_eps", type=str, default="8.0")
    parser.add_argument("--scan_dense_min_eps", type=str, default="1.0")
    parser.add_argument("--scan_min_eps_mode", type=str, default="count_gate")
    parser.add_argument("--scan_min_samples", type=int, default=1)
    parser.add_argument("--scan_connect_rule", type=str, default="mutual", choices=["mutual", "either", "mean"])
    parser.add_argument("--scan_score_power", type=float, default=2.0)
    parser.add_argument("--scan_drop_noise", action="store_true")

    parser.add_argument("--scan_count_gate_c4_dup", type=str, default="20,28")
    parser.add_argument("--scan_count_gate_c6_dup", type=str, default="31,42")
    parser.add_argument("--scan_count_gate_growth84_dense", type=str, default="2.4,3.0")
    parser.add_argument("--scan_count_gate_growth124_dense", type=str, default="4.0")
    parser.add_argument("--scan_count_gate_dup_min_eps", type=str, default="2.0,3.0")
    parser.add_argument("--scan_count_gate_dense_alpha", type=str, default="0.25,0.35,0.5")
    parser.add_argument("--scan_count_gate_dense_max_eps", type=str, default="1.0,1.5,2.0")

    parser.add_argument("--watch_images", type=str, default="IMG_165.jpg,IMG_36.jpg,IMG_104.jpg,IMG_50.jpg")
    parser.add_argument("--save_all_per_image", action="store_true")

    return parser.parse_args()


def make_param_grid(args):
    ks = parse_csv_ints(args.scan_k)
    dense_threshes = parse_csv_floats(args.scan_dense_kdist_thresh)
    alphas = parse_csv_floats(args.scan_alpha)
    min_eps_values = parse_csv_floats(args.scan_min_eps)
    max_eps_values = parse_csv_floats(args.scan_max_eps)
    dense_min_eps_values = parse_csv_floats(args.scan_dense_min_eps)
    min_eps_modes = parse_csv_strings(args.scan_min_eps_mode)
    count_gate_c4_dups = parse_csv_floats(args.scan_count_gate_c4_dup)
    count_gate_c6_dups = parse_csv_floats(args.scan_count_gate_c6_dup)
    count_gate_growth84_denses = parse_csv_floats(args.scan_count_gate_growth84_dense)
    count_gate_growth124_denses = parse_csv_floats(args.scan_count_gate_growth124_dense)
    count_gate_dup_min_eps_values = parse_csv_floats(args.scan_count_gate_dup_min_eps)
    count_gate_dense_alphas = parse_csv_floats(args.scan_count_gate_dense_alpha)
    count_gate_dense_max_eps_values = parse_csv_floats(args.scan_count_gate_dense_max_eps)

    valid_modes = {"fixed", "adaptive", "count_gate"}
    bad_modes = sorted(set(min_eps_modes) - valid_modes)
    if bad_modes:
        raise ValueError(f"--scan_min_eps_mode supports {sorted(valid_modes)}, got {bad_modes}")

    grid = []
    for min_eps_mode in min_eps_modes:
        if min_eps_mode == "fixed":
            dense_thresh_iter = [0.0]
            dense_min_eps_iter = [dense_min_eps_values[0]]
            count_gate_iter = [(0.0, 0.0, 0.0, 0.0, min_eps_values[0], alphas[0], max_eps_values[0])]
        elif min_eps_mode == "adaptive":
            dense_thresh_iter = dense_threshes
            dense_min_eps_iter = dense_min_eps_values
            count_gate_iter = [(0.0, 0.0, 0.0, 0.0, min_eps_values[0], alphas[0], max_eps_values[0])]
        else:
            dense_thresh_iter = [0.0]
            dense_min_eps_iter = dense_min_eps_values
            count_gate_iter = product(
                count_gate_c4_dups,
                count_gate_c6_dups,
                count_gate_growth84_denses,
                count_gate_growth124_denses,
                count_gate_dup_min_eps_values,
                count_gate_dense_alphas,
                count_gate_dense_max_eps_values,
            )

        for k, dense_thresh, alpha, min_eps, max_eps, dense_min_eps, count_gate_values in product(
                ks,
                dense_thresh_iter,
                alphas,
                min_eps_values,
                max_eps_values,
                dense_min_eps_iter,
                count_gate_iter,
        ):
            (
                c4_dup,
                c6_dup,
                growth84_dense,
                growth124_dense,
                dup_min_eps,
                dense_alpha,
                dense_max_eps,
            ) = count_gate_values
            grid.append({
                "k": int(k),
                "dense_kdist_thresh": float(dense_thresh),
                "alpha": float(alpha),
                "min_eps": float(min_eps),
                "max_eps": float(max_eps),
                "dense_min_eps": float(dense_min_eps),
                "min_eps_mode": str(min_eps_mode),
                "min_samples": int(args.scan_min_samples),
                "connect_rule": str(args.scan_connect_rule),
                "score_power": float(args.scan_score_power),
                "keep_noise": not bool(args.scan_drop_noise),
                "count_gate_c4_dup": float(c4_dup),
                "count_gate_c6_dup": float(c6_dup),
                "count_gate_growth84_dense": float(growth84_dense),
                "count_gate_growth124_dense": float(growth124_dense),
                "count_gate_dup_min_eps": float(dup_min_eps),
                "count_gate_dense_alpha": float(dense_alpha),
                "count_gate_dense_max_eps": float(dense_max_eps),
            })
    return grid


def param_id(params):
    base = (
        f"{params['min_eps_mode']}_k{params['k']}_a{params['alpha']:.3g}"
        f"_min{params['min_eps']:.3g}_dmin{params['dense_min_eps']:.3g}_max{params['max_eps']:.3g}"
    )
    if params["min_eps_mode"] == "adaptive":
        return f"{base}_thr{params['dense_kdist_thresh']:.3g}"
    if params["min_eps_mode"] == "count_gate":
        return (
            f"{base}_c4{params['count_gate_c4_dup']:.3g}"
            f"_c6{params['count_gate_c6_dup']:.3g}"
            f"_g84{params['count_gate_growth84_dense']:.3g}"
            f"_g124{params['count_gate_growth124_dense']:.3g}"
            f"_dupmin{params['count_gate_dup_min_eps']:.3g}"
            f"_da{params['count_gate_dense_alpha']:.3g}"
            f"_dmax{params['count_gate_dense_max_eps']:.3g}"
        )
    return base


def norm_points_to_global_xy(pts_norm, h_tile, w_tile, left, top, h_full, w_full):
    if pts_norm.numel() == 0:
        return torch.zeros((0, 2), dtype=torch.float32)
    pts_norm = pts_norm.detach().float().cpu()
    xs = (pts_norm[:, 0] + 1) * 0.5 * (w_tile - 1)
    ys = (pts_norm[:, 1] + 1) * 0.5 * (h_tile - 1)
    xs_g = (xs + left).clamp(0, w_full - 1)
    ys_g = (ys + top).clamp(0, h_full - 1)
    return torch.stack([xs_g, ys_g], dim=1).float()


def build_candidate_cache(args, cache_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(getattr(args, "seed", 7113064165))
    set_seed(seed)

    assert args.ckpt_path, "Please set --ckpt_path"
    assert args.test_root, "Please set --test_root"

    model = build_model(args, training=False)
    model = load_checkpoint_into_model(model, args.ckpt_path, device)
    model.to(device).eval()

    dataset = ImageDataset(
        root=args.test_root,
        mode="points",
        tile_size=(256, 256),
        stride=(256, 256),
        gray=False,
        pad_if_needed=True,
        image_exts=(".jpg", ".png"),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_points_padded,
    )

    t_total = int(args.diffusion_T)
    steps = int(args.ddim_steps)
    sched = CosineAbarSchedule(T=t_total)
    abar = sched.abar.to(device=device)
    use_proposal_prior = bool(getattr(args, "use_proposal_prior", False))
    prior_start_t = int(getattr(args, "proposal_prior_start_t", 700))
    t_seq = make_ddim_steps(
        T=t_total,
        steps=steps,
        device=device,
        start_t=prior_start_t if use_proposal_prior else None,
    )
    eps = float(args.eps_clip)
    low_score_thresh = float(args.scan_low_score_thresh)
    save_stage_xy = bool(getattr(args, "scan_save_stage_xy", False))

    per_image_points_xy = defaultdict(list)
    per_image_points_prob = defaultdict(list)
    per_image_raw_points_xy = defaultdict(list)
    per_image_merged_points_xy = defaultdict(list)
    per_image_gt_points_xy = defaultdict(list)
    per_image_size = {}
    per_image_proposal_count = defaultdict(int)
    per_image_raw_slot_count = defaultdict(int)
    per_image_merged_valid_count = defaultdict(int)
    per_image_merge_drop_count = defaultdict(int)
    per_image_threshold_count = defaultdict(int)

    print(f"[INFO] Building cache from {len(dataset)} tiles")
    print(f"[INFO] Loaded checkpoint: {args.ckpt_path}")

    with torch.no_grad():
        for images, points_pad, mask, metas in tqdm(
                loader,
                total=len(loader),
                desc="DDIM cache inference",
                dynamic_ncols=True,
        ):
            images = images.to(device)
            mask = mask.to(device)
            batch_size, _, h_tile, w_tile = images.shape
            num_points = points_pad.shape[1]

            feats = model.encode(images)
            prior_occupancy_logits = None
            prior_density = None
            if use_proposal_prior:
                prior_occupancy_logits, prior_density = model.predict_proposal_prior(feats)
            p0_list = []
            prob_list = []
            posmask_list = []
            validmask_list = []
            merge_stats_list = []
            runs = int(getattr(args, "num_realizations", 1))

            for _ in range(runs):
                if use_proposal_prior:
                    x0_prior, _ = build_mixed_x0_prior(
                        prior_occupancy_logits,
                        prior_density,
                        num_slots=num_points,
                        clamp_eps=eps,
                        mode=str(getattr(args, "proposal_prior_mode", "occupancy")),
                        density_cell_capacity=int(getattr(args, "proposal_prior_cell_capacity", 2)),
                    )
                    abar_t0 = abar[int(t_seq[0].item())].view(1, 1, 1)
                    p_t = (
                        abar_t0.sqrt() * x0_prior
                        + (1.0 - abar_t0).clamp_min(0.0).sqrt()
                        * torch.randn_like(x0_prior)
                    )
                else:
                    p_t = torch.empty(
                        (batch_size, num_points, 2), device=device
                    ).uniform_(-1.0 + eps, 1.0 - eps)
                exist_prob_last = None
                pos_mask_last = None
                pred_points_last = None
                pred_valid_last = None
                merge_stats_last = None

                for i, t_int in enumerate(t_seq.tolist()):
                    t_int = int(t_int)
                    t_tensor = torch.full((batch_size, 1), t_int, device=device, dtype=torch.long)
                    abar_t = abar[t_int].view(1, 1, 1).expand(batch_size, 1, 1)
                    need_exist = (i == len(t_seq) - 1)

                    eps_pred, exist_logit, _, pred_points_for_cls, pred_valid_mask = model.denoise(
                        feats,
                        p_t,
                        t_tensor,
                        abar_t=abar_t,
                        clamp_eps=1e-6,
                        need_exist=need_exist,
                        selector_prior_maps=(
                            (prior_occupancy_logits, prior_density)
                            if prior_density is not None
                            else None
                        ),
                    )

                    if need_exist:
                        if exist_logit is None:
                            raise RuntimeError("need_exist=True but denoise returned None exist_logit")
                        if pred_points_for_cls is None:
                            raise RuntimeError("need_exist=True but denoise returned no prediction points")
                        pred_points_last = pred_points_for_cls
                        pred_valid_last = pred_valid_mask
                        model_merge_stats = getattr(model, "last_merge_stats", None)
                        if model_merge_stats is not None:
                            merge_stats_last = {
                                key: value.detach().cpu()
                                for key, value in model_merge_stats.items()
                            }

                        if exist_logit.dim() == 3 and exist_logit.size(-1) == 1:
                            exist_logit = exist_logit.squeeze(-1)

                        if exist_logit.dim() == 3 and exist_logit.size(-1) == 2:
                            prob_pos = torch.softmax(exist_logit, dim=-1)[..., 1]
                            exist_prob_last = prob_pos
                            gate_mode = getattr(args, "test_gate_mode", "argmax_or_prob")
                            if gate_mode == "prob_only":
                                pos_mask_last = prob_pos > float(args.hard_thresh)
                            elif gate_mode == "argmax_only":
                                pos_mask_last = exist_logit.argmax(-1) == 1
                            else:
                                pos_mask_last = (exist_logit.argmax(-1) == 1) | (prob_pos > float(args.hard_thresh))
                        elif exist_logit.dim() == 2:
                            exist_prob_last = torch.sigmoid(exist_logit)
                            pos_mask_last = exist_prob_last > float(args.hard_thresh)
                        else:
                            raise RuntimeError(f"Unexpected exist_logit shape: {tuple(exist_logit.shape)}")

                        if pred_valid_last is not None:
                            pos_mask_last = pos_mask_last & pred_valid_last.bool()

                    if i + 1 < len(t_seq):
                        abar_prev = abar[int(t_seq[i + 1].item())].view(1, 1, 1).expand(batch_size, 1, 1)
                        eta_step = float(getattr(args, "ddim_eta", 0.3))
                    else:
                        abar_prev = torch.ones((batch_size, 1, 1), device=device)
                        eta_step = 0.0

                    p_t = ddim_reverse_step_eta(p_t, eps_pred, abar_t, abar_prev, eta=eta_step)
                    p_t = p_t.clamp(-1.0 + eps, 1.0 - eps)

                if exist_prob_last is None or pos_mask_last is None or pred_points_last is None:
                    raise RuntimeError("DDIM loop finished without final probabilities or points")

                p0_list.append(pred_points_last.detach())
                prob_list.append(exist_prob_last.detach())
                posmask_list.append(pos_mask_last.detach())
                if pred_valid_last is None:
                    validmask_list.append(torch.ones_like(pos_mask_last, dtype=torch.bool).detach())
                else:
                    validmask_list.append(pred_valid_last.bool().detach())
                merge_stats_list.append(merge_stats_last)

            for b in range(batch_size):
                meta = metas[b]
                img_key = get_image_key_from_meta(meta)
                h_full, w_full = meta["orig_size"]
                left = int(meta["tile_left"])
                top = int(meta["tile_top"])

                if img_key not in per_image_size:
                    per_image_size[img_key] = (int(h_full), int(w_full))

                points_b = points_pad[b]
                mask_b = mask[b].detach().cpu()
                gt_b = points_b[mask_b]
                if gt_b.numel() > 0:
                    gt_x = (gt_b[:, 0] + left).round().long().clamp(0, w_full - 1)
                    gt_y = (gt_b[:, 1] + top).round().long().clamp(0, h_full - 1)
                    per_image_gt_points_xy[img_key].append(torch.stack([gt_x, gt_y], dim=1))

                for run_idx in range(runs):
                    pts_norm_all = p0_list[run_idx][b]
                    sc_all = prob_list[run_idx][b]
                    pm = posmask_list[run_idx][b]
                    vm = validmask_list[run_idx][b]
                    merge_stats = merge_stats_list[run_idx]

                    raw_count = int(num_points)
                    merged_count = int(vm.sum().item())
                    if merge_stats is not None:
                        raw_count = int(merge_stats["raw_slot_count"][b].item())
                        merged_count = int(merge_stats["merged_valid_count"][b].item())

                    per_image_raw_slot_count[img_key] += raw_count
                    per_image_merged_valid_count[img_key] += merged_count
                    per_image_merge_drop_count[img_key] += max(0, raw_count - merged_count)
                    per_image_proposal_count[img_key] += merged_count
                    per_image_threshold_count[img_key] += int(pm.sum().item())

                    if save_stage_xy:
                        if merge_stats is not None and "raw_x0_hat" in merge_stats:
                            raw_norm = merge_stats["raw_x0_hat"][b]
                            raw_xy = norm_points_to_global_xy(raw_norm, h_tile, w_tile, left, top, h_full, w_full)
                            per_image_raw_points_xy[img_key].append(raw_xy)
                        if merge_stats is not None and "merged_x0_hat" in merge_stats:
                            merged_norm_all = merge_stats["merged_x0_hat"][b]
                            if "merged_valid_mask" in merge_stats:
                                merged_vm = merge_stats["merged_valid_mask"][b].bool()
                            else:
                                merged_vm = vm.detach().cpu().bool()
                            merged_norm = merged_norm_all[merged_vm]
                        else:
                            merged_norm = pts_norm_all[vm]
                        merged_xy = norm_points_to_global_xy(merged_norm, h_tile, w_tile, left, top, h_full, w_full)
                        per_image_merged_points_xy[img_key].append(merged_xy)

                    pts_norm = pts_norm_all[pm]
                    sc = sc_all[pm]
                    if sc.numel() == 0:
                        continue

                    keep_low = sc > low_score_thresh
                    pts_norm = pts_norm[keep_low]
                    sc = sc[keep_low]
                    if sc.numel() == 0:
                        continue

                    xs = (pts_norm[:, 0] + 1) * 0.5 * (w_tile - 1)
                    ys = (pts_norm[:, 1] + 1) * 0.5 * (h_tile - 1)
                    xs_g = (xs + left).clamp(0, w_full - 1)
                    ys_g = (ys + top).clamp(0, h_full - 1)

                    per_image_points_xy[img_key].append(torch.stack([xs_g, ys_g], dim=1).detach().cpu())
                    per_image_points_prob[img_key].append(sc.detach().cpu())

    images_out = {}
    for img_key in sorted(per_image_size.keys()):
        if len(per_image_gt_points_xy[img_key]) > 0:
            gt_xy = torch.cat(per_image_gt_points_xy[img_key], dim=0).float()
            gt_np = dedup_points_xy(gt_xy.numpy().astype(np.float32), decimals=3)
            gt_xy = torch.from_numpy(gt_np).float()
        else:
            gt_xy = torch.zeros((0, 2), dtype=torch.float32)

        if len(per_image_points_xy[img_key]) > 0:
            xy = torch.cat(per_image_points_xy[img_key], dim=0).float()
            score = torch.cat(per_image_points_prob[img_key], dim=0).float()
        else:
            xy = torch.zeros((0, 2), dtype=torch.float32)
            score = torch.zeros((0,), dtype=torch.float32)

        if save_stage_xy and len(per_image_raw_points_xy[img_key]) > 0:
            raw_xy = torch.cat(per_image_raw_points_xy[img_key], dim=0).float()
        else:
            raw_xy = None

        if save_stage_xy and len(per_image_merged_points_xy[img_key]) > 0:
            merged_xy = torch.cat(per_image_merged_points_xy[img_key], dim=0).float()
        else:
            merged_xy = None

        images_out[img_key] = {
            "xy": xy,
            "score": score,
            "gt_xy": gt_xy,
            "gt_count": float(gt_xy.size(0)),
            "raw_slot_count": int(per_image_raw_slot_count[img_key]),
            "merged_valid_count": int(per_image_merged_valid_count[img_key]),
            "merge_drop_count": int(per_image_merge_drop_count[img_key]),
            "threshold_count": int(per_image_threshold_count[img_key]),
            "proposal_count": int(per_image_proposal_count[img_key]),
            "candidate_count": int(xy.size(0)),
            "size": tuple(per_image_size[img_key]),
        }
        if raw_xy is not None:
            images_out[img_key]["raw_xy"] = raw_xy
        if merged_xy is not None:
            images_out[img_key]["merged_xy"] = merged_xy

    cache = {
        "meta": {
            "ckpt_path": str(args.ckpt_path),
            "test_root": str(args.test_root),
            "hard_thresh": float(args.hard_thresh),
            "ddim_steps": int(args.ddim_steps),
            "num_realizations": int(getattr(args, "num_realizations", 1)),
            "seed": int(getattr(args, "seed", 7113064165)),
            "low_score_thresh": low_score_thresh,
            "contains_stage_xy": bool(save_stage_xy),
        },
        "images": images_out,
    }
    light_cache = strip_stage_xy(cache)

    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    torch.save(light_cache, cache_path)
    print(f"[SAVE-CACHE] {cache_path}")
    if save_stage_xy:
        stage_cache_path = make_stage_cache_path(cache_path)
        torch.save(cache, stage_cache_path)
        print(f"[SAVE-CACHE-STAGE-XY] {stage_cache_path}")
    return light_cache


def compute_local_count_features(dist):
    n = int(dist.size(0))
    if n == 0:
        empty = torch.zeros((0,), dtype=torch.float32)
        return {
            "count_r4": empty,
            "count_r6": empty,
            "count_r8": empty,
            "count_r12": empty,
            "growth84": empty,
            "growth124": empty,
        }

    count_r4 = (dist <= 4.0).sum(dim=1).float()
    count_r6 = (dist <= 6.0).sum(dim=1).float()
    count_r8 = (dist <= 8.0).sum(dim=1).float()
    count_r12 = (dist <= 12.0).sum(dim=1).float()
    denom = count_r4.clamp_min(1.0)
    return {
        "count_r4": count_r4,
        "count_r6": count_r6,
        "count_r8": count_r8,
        "count_r12": count_r12,
        "growth84": count_r8 / denom,
        "growth124": count_r12 / denom,
    }


def empty_gate_info(n):
    return {
        "dense_mask": torch.zeros((n,), dtype=torch.bool),
        "duplicate_mask": torch.zeros((n,), dtype=torch.bool),
        "broad_dense_mask": torch.zeros((n,), dtype=torch.bool),
        "min_eps_used": torch.zeros((n,), dtype=torch.float32),
        "alpha_used": torch.zeros((n,), dtype=torch.float32),
        "max_eps_used": torch.zeros((n,), dtype=torch.float32),
    }


def cluster_from_precomputed(pts, scores, dist, kth_dist, params, local_counts=None):
    pts = pts.float().cpu()
    scores = scores.float().cpu().clamp_min(0.0)
    n = int(pts.size(0))

    if n == 0:
        empty_xy = torch.zeros((0, 2), dtype=torch.float32)
        empty_1d = torch.zeros((0,), dtype=torch.float32)
        return empty_xy, empty_1d, empty_1d, empty_gate_info(0), torch.zeros((0,), dtype=torch.long), 0

    if n == 1:
        centers = pts.clone()
        radii = torch.full((1,), float(params["min_eps"]), dtype=torch.float32)
        gate_info = empty_gate_info(1)
        gate_info["min_eps_used"] = radii.clone()
        gate_info["alpha_used"] = torch.full((1,), float(params["alpha"]), dtype=torch.float32)
        gate_info["max_eps_used"] = torch.full((1,), float(params["max_eps"]), dtype=torch.float32)
        sizes = torch.ones((1,), dtype=torch.long)
        return centers, scores.clone(), radii, gate_info, sizes, 0

    alpha_used = torch.full_like(kth_dist, float(params["alpha"]))
    max_eps_used = torch.full_like(kth_dist, float(params["max_eps"]))

    if params["min_eps_mode"] == "count_gate":
        if local_counts is None:
            local_counts = compute_local_count_features(dist)

        compact_duplicate = (
            (local_counts["count_r4"] >= float(params["count_gate_c4_dup"]))
            | (local_counts["count_r6"] >= float(params["count_gate_c6_dup"]))
        )
        broad_dense_mask = (
            (local_counts["growth84"] >= float(params["count_gate_growth84_dense"]))
            | (local_counts["growth124"] >= float(params["count_gate_growth124_dense"]))
        )
        dense_mask = broad_dense_mask & (
            local_counts["count_r8"] >= float(params["count_gate_c4_dup"])
        )
        duplicate_mask = compact_duplicate & ~dense_mask
        min_eps_used = torch.full_like(kth_dist, float(params["min_eps"]))
        min_eps_used = torch.where(
            duplicate_mask,
            torch.full_like(kth_dist, float(params["count_gate_dup_min_eps"])),
            min_eps_used,
        )
        min_eps_used = torch.where(
            dense_mask,
            torch.full_like(kth_dist, float(params["dense_min_eps"])),
            min_eps_used,
        )
        alpha_used = torch.where(
            dense_mask,
            torch.full_like(kth_dist, float(params["count_gate_dense_alpha"])),
            alpha_used,
        )
        max_eps_used = torch.where(
            dense_mask,
            torch.full_like(kth_dist, float(params["count_gate_dense_max_eps"])),
            max_eps_used,
        )
    elif params["min_eps_mode"] == "adaptive":
        dense_mask = kth_dist <= float(params["dense_kdist_thresh"])
        duplicate_mask = torch.zeros_like(kth_dist, dtype=torch.bool)
        broad_dense_mask = dense_mask.clone()
        min_eps_used = torch.where(
            dense_mask,
            torch.full_like(kth_dist, float(params["dense_min_eps"])),
            torch.full_like(kth_dist, float(params["min_eps"])),
        )
    else:
        dense_mask = torch.zeros_like(kth_dist, dtype=torch.bool)
        duplicate_mask = torch.zeros_like(kth_dist, dtype=torch.bool)
        broad_dense_mask = torch.zeros_like(kth_dist, dtype=torch.bool)
        min_eps_used = torch.full_like(kth_dist, float(params["min_eps"]))

    radii = torch.minimum(torch.maximum(alpha_used * kth_dist, min_eps_used), max_eps_used)
    gate_info = {
        "dense_mask": dense_mask,
        "duplicate_mask": duplicate_mask,
        "broad_dense_mask": broad_dense_mask,
        "min_eps_used": min_eps_used,
        "alpha_used": alpha_used,
        "max_eps_used": max_eps_used,
    }

    ri = radii[:, None]
    rj = radii[None, :]
    if params["connect_rule"] == "either":
        eps_pair = torch.maximum(ri, rj)
    elif params["connect_rule"] == "mean":
        eps_pair = 0.5 * (ri + rj)
    else:
        eps_pair = torch.minimum(ri, rj)

    adjacency = dist <= eps_pair
    adjacency.fill_diagonal_(True)
    core = adjacency.sum(dim=1) >= max(1, int(params["min_samples"]))

    labels = torch.full((n,), -1, dtype=torch.long)
    visited = torch.zeros((n,), dtype=torch.bool)
    adjacency_np = adjacency.numpy()
    core_np = core.numpy()
    labels_np = labels.numpy()
    visited_np = visited.numpy()

    cluster_id = 0
    for start in range(n):
        if visited_np[start]:
            continue
        visited_np[start] = True
        if not core_np[start]:
            continue
        labels_np[start] = cluster_id
        q = deque(np.flatnonzero(adjacency_np[start]).tolist())
        while q:
            j = q.popleft()
            if not visited_np[j]:
                visited_np[j] = True
                if core_np[j]:
                    q.extend(np.flatnonzero(adjacency_np[j]).tolist())
            if labels_np[j] < 0:
                labels_np[j] = cluster_id
        cluster_id += 1

    labels = torch.from_numpy(labels_np.copy()).long()
    noise_mask = labels < 0
    num_noise = int(noise_mask.sum().item())

    centers = []
    cluster_scores = []
    cluster_sizes = []
    for cid in range(cluster_id):
        idx = torch.nonzero(labels == cid, as_tuple=False).squeeze(1)
        weights = scores[idx].clamp_min(1e-6).pow(float(params["score_power"]))
        centers.append((pts[idx] * weights[:, None]).sum(dim=0) / weights.sum())
        cluster_scores.append(scores[idx].max())
        cluster_sizes.append(torch.tensor(idx.numel(), dtype=torch.long))

    if params["keep_noise"] and num_noise > 0:
        noise_idx = torch.nonzero(noise_mask, as_tuple=False).squeeze(1)
        for idx in noise_idx.tolist():
            centers.append(pts[idx])
            cluster_scores.append(scores[idx])
            cluster_sizes.append(torch.tensor(1, dtype=torch.long))

    if len(centers) == 0:
        empty_xy = torch.zeros((0, 2), dtype=torch.float32)
        empty_1d = torch.zeros((0,), dtype=torch.float32)
        empty_long = torch.zeros((0,), dtype=torch.long)
        return empty_xy, empty_1d, radii, gate_info, empty_long, num_noise

    centers = torch.stack(centers, dim=0).float()
    cluster_scores = torch.stack(cluster_scores, dim=0).float()
    cluster_sizes = torch.stack(cluster_sizes, dim=0).long()
    order = cluster_scores.argsort(descending=True)
    return centers[order], cluster_scores[order], radii, gate_info, cluster_sizes[order], num_noise


def empty_stats():
    return {
        "sum_abs": 0.0,
        "sum_sq": 0.0,
        "sum_signed": 0.0,
        "sum_pred": 0.0,
        "sum_gt": 0.0,
        "sum_dense_ratio": 0.0,
        "sum_duplicate_ratio": 0.0,
        "sum_broad_dense_ratio": 0.0,
        "sum_min_eps_mean": 0.0,
        "sum_alpha_mean": 0.0,
        "sum_max_eps_mean": 0.0,
        "sum_radius_mean": 0.0,
        "sum_radius_p50": 0.0,
        "sum_radius_p90": 0.0,
        "sum_cluster_p50": 0.0,
        "sum_cluster_p90": 0.0,
        "over": 0,
        "under": 0,
        "n": 0,
        "worst_abs_err": -1.0,
        "worst_image": "",
    }


def update_stats(stats, row):
    signed = float(row["pred_adbscan"]) - float(row["gt"])
    abs_err = abs(signed)
    stats["sum_abs"] += abs_err
    stats["sum_sq"] += signed * signed
    stats["sum_signed"] += signed
    stats["sum_pred"] += float(row["pred_adbscan"])
    stats["sum_gt"] += float(row["gt"])
    stats["sum_dense_ratio"] += float(row["dense_point_ratio"])
    stats["sum_duplicate_ratio"] += float(row["duplicate_point_ratio"])
    stats["sum_broad_dense_ratio"] += float(row["broad_dense_point_ratio"])
    stats["sum_min_eps_mean"] += float(row["min_eps_mean"])
    stats["sum_alpha_mean"] += float(row["alpha_mean"])
    stats["sum_max_eps_mean"] += float(row["max_eps_mean"])
    stats["sum_radius_mean"] += float(row["radius_mean"])
    stats["sum_radius_p50"] += float(row["radius_p50"])
    stats["sum_radius_p90"] += float(row["radius_p90"])
    stats["sum_cluster_p50"] += float(row["cluster_p50_gt_dist"])
    stats["sum_cluster_p90"] += float(row["cluster_p90_gt_dist"])
    stats["over"] += int(signed > 0)
    stats["under"] += int(signed < 0)
    stats["n"] += 1
    if abs_err > stats["worst_abs_err"]:
        stats["worst_abs_err"] = abs_err
        stats["worst_image"] = str(row["image_path"])


def finalize_stats(stats):
    n = max(1, int(stats["n"]))
    return {
        "MAE": stats["sum_abs"] / n,
        "RMSE": float(np.sqrt(stats["sum_sq"] / n)),
        "bias": stats["sum_signed"] / n,
        "avg_pred": stats["sum_pred"] / n,
        "avg_gt": stats["sum_gt"] / n,
        "over": int(stats["over"]),
        "under": int(stats["under"]),
        "dense_point_ratio_mean": stats["sum_dense_ratio"] / n,
        "duplicate_point_ratio_mean": stats["sum_duplicate_ratio"] / n,
        "broad_dense_point_ratio_mean": stats["sum_broad_dense_ratio"] / n,
        "min_eps_mean_avg": stats["sum_min_eps_mean"] / n,
        "alpha_mean_avg": stats["sum_alpha_mean"] / n,
        "max_eps_mean_avg": stats["sum_max_eps_mean"] / n,
        "radius_mean_avg": stats["sum_radius_mean"] / n,
        "radius_p50_avg": stats["sum_radius_p50"] / n,
        "radius_p90_avg": stats["sum_radius_p90"] / n,
        "cluster_p50_gt_dist_avg": stats["sum_cluster_p50"] / n,
        "cluster_p90_gt_dist_avg": stats["sum_cluster_p90"] / n,
        "worst_abs_err": stats["worst_abs_err"],
        "worst_image": stats["worst_image"],
    }


def evaluate_grid(cache, grid, watch_images, save_all_per_image=False):
    combo_stats = {param_id(params): empty_stats() for params in grid}
    combo_params = {param_id(params): params for params in grid}
    rows_by_combo = {param_id(params): [] for params in grid}
    watch_basenames = set(watch_images)
    watch_errors = {param_id(params): {} for params in grid}

    unique_ks = sorted({int(params["k"]) for params in grid})
    max_k = max(unique_ks) if unique_ks else 1

    images = cache["images"]
    for img_key, item in tqdm(images.items(), total=len(images), desc="Scanning images", dynamic_ncols=True):
        xy = item["xy"].float().cpu()
        score = item["score"].float().cpu()
        gt_xy = item["gt_xy"].float().cpu()
        gt_np = gt_xy.numpy().astype(np.float32)
        gt_count = float(item["gt_count"])

        if xy.numel() == 0:
            dist = torch.zeros((0, 0), dtype=torch.float32)
            kth_by_k = {k: torch.zeros((0,), dtype=torch.float32) for k in unique_ks}
            local_counts = compute_local_count_features(dist)
        else:
            dist = torch.cdist(xy, xy)
            dist_for_knn = dist.clone()
            dist_for_knn.fill_diagonal_(float("inf"))
            kth_by_k = {}
            n = int(xy.size(0))
            for k in unique_ks:
                kth = max(1, min(int(k), max(1, n - 1)))
                kth_by_k[k] = torch.kthvalue(dist_for_knn, kth, dim=1).values
            local_counts = compute_local_count_features(dist)

        for params in grid:
            pid = param_id(params)
            centers, _, radii, gate_info, _, num_noise = cluster_from_precomputed(
                xy,
                score,
                dist,
                kth_by_k[int(params["k"])],
                params,
                local_counts=local_counts,
            )
            centers_np = centers.numpy().astype(np.float32)
            stats_cluster = nearest_gt_distance_stats(centers_np, gt_np)

            radii_np = radii.numpy().astype(np.float32)
            dense_np = gate_info["dense_mask"].numpy().astype(bool)
            duplicate_np = gate_info["duplicate_mask"].numpy().astype(bool)
            broad_dense_np = gate_info["broad_dense_mask"].numpy().astype(bool)
            min_eps_np = gate_info["min_eps_used"].numpy().astype(np.float32)
            alpha_np = gate_info["alpha_used"].numpy().astype(np.float32)
            max_eps_np = gate_info["max_eps_used"].numpy().astype(np.float32)
            pred = float(centers.size(0))
            row = {
                "param_id": pid,
                "image_path": img_key,
                "pred_adbscan": pred,
                "gt": gt_count,
                "abs_err": abs(pred - gt_count),
                "raw_slot_count": int(item.get("raw_slot_count", item["proposal_count"])),
                "merged_valid_count": int(item.get("merged_valid_count", item["proposal_count"])),
                "merge_drop_count": int(item.get("merge_drop_count", 0)),
                "threshold_count": int(item.get("threshold_count", item["candidate_count"])),
                "proposal_count": int(item["proposal_count"]),
                "candidate_count": int(item["candidate_count"]),
                "cluster_count": int(centers.size(0)),
                "noise_count": int(num_noise),
                "radius_mean": float(np.mean(radii_np)) if radii_np.size else 0.0,
                "radius_p50": float(np.percentile(radii_np, 50)) if radii_np.size else 0.0,
                "radius_p90": float(np.percentile(radii_np, 90)) if radii_np.size else 0.0,
                "min_eps_mean": float(np.mean(min_eps_np)) if min_eps_np.size else 0.0,
                "alpha_mean": float(np.mean(alpha_np)) if alpha_np.size else 0.0,
                "max_eps_mean": float(np.mean(max_eps_np)) if max_eps_np.size else 0.0,
                "dense_point_ratio": float(np.mean(dense_np)) if dense_np.size else 0.0,
                "duplicate_point_ratio": float(np.mean(duplicate_np)) if duplicate_np.size else 0.0,
                "broad_dense_point_ratio": float(np.mean(broad_dense_np)) if broad_dense_np.size else 0.0,
                "cluster_p50_gt_dist": stats_cluster["p50"],
                "cluster_p90_gt_dist": stats_cluster["p90"],
            }
            update_stats(combo_stats[pid], row)
            if save_all_per_image:
                rows_by_combo[pid].append(row)
            elif os.path.basename(str(img_key)) in watch_basenames:
                rows_by_combo[pid].append(row)

            base = os.path.basename(str(img_key))
            if base in watch_basenames:
                watch_errors[pid][f"err_{base}"] = float(row["abs_err"])
                watch_errors[pid][f"pred_{base}"] = float(row["pred_adbscan"])
                watch_errors[pid][f"dense_{base}"] = float(row["dense_point_ratio"])
                watch_errors[pid][f"dup_{base}"] = float(row["duplicate_point_ratio"])

        del dist

    result_rows = []
    for pid, stats in combo_stats.items():
        params = combo_params[pid]
        final = finalize_stats(stats)
        row = {
            "param_id": pid,
            "k": params["k"],
            "dense_kdist_thresh": params["dense_kdist_thresh"],
            "alpha": params["alpha"],
            "min_eps": params["min_eps"],
            "dense_min_eps": params["dense_min_eps"],
            "max_eps": params["max_eps"],
            "min_eps_mode": params["min_eps_mode"],
            "min_samples": params["min_samples"],
            "connect_rule": params["connect_rule"],
            "score_power": params["score_power"],
            "keep_noise": params["keep_noise"],
            "count_gate_c4_dup": params["count_gate_c4_dup"],
            "count_gate_c6_dup": params["count_gate_c6_dup"],
            "count_gate_growth84_dense": params["count_gate_growth84_dense"],
            "count_gate_growth124_dense": params["count_gate_growth124_dense"],
            "count_gate_dup_min_eps": params["count_gate_dup_min_eps"],
            "count_gate_dense_alpha": params["count_gate_dense_alpha"],
            "count_gate_dense_max_eps": params["count_gate_dense_max_eps"],
            **final,
            **watch_errors[pid],
        }
        result_rows.append(row)

    result_rows.sort(key=lambda r: (float(r["MAE"]), abs(float(r["bias"]))))
    best_pid = result_rows[0]["param_id"] if result_rows else ""
    best_rows = rows_by_combo.get(best_pid, [])
    return result_rows, best_pid, best_rows


def write_csv(path, rows, fallback_fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else fallback_fields
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    save_dir = args.scan_save_dir or f"{args.save_dir}_ADBSCAN_SCAN"
    os.makedirs(save_dir, exist_ok=True)
    cache_path = args.scan_cache_path or os.path.join(save_dir, "adaptive_dbscan_candidate_cache.pt")
    stage_cache_path = make_stage_cache_path(cache_path)
    needs_stage_cache = bool(args.scan_save_stage_xy)

    if (
            os.path.exists(cache_path)
            and not bool(args.refresh_cache)
            and (not needs_stage_cache or os.path.exists(stage_cache_path))
    ):
        print(f"[LOAD-CACHE] {cache_path}")
        cache = torch.load(cache_path, map_location="cpu")
    else:
        if os.path.exists(cache_path) and needs_stage_cache and not os.path.exists(stage_cache_path):
            print(f"[INFO] Stage-xy cache missing, rebuilding both cache variants: {stage_cache_path}")
        cache = build_candidate_cache(args, cache_path)

    grid = make_param_grid(args)
    watch_images = parse_csv_strings(args.watch_images)
    print(f"[INFO] Images in cache: {len(cache['images'])}")
    print(f"[INFO] Grid size: {len(grid)}")
    print(f"[INFO] Watch images: {watch_images}")

    result_rows, best_pid, best_rows = evaluate_grid(
        cache,
        grid,
        watch_images=watch_images,
        save_all_per_image=bool(args.save_all_per_image),
    )

    grid_csv = os.path.join(save_dir, "adbscan_grid_results.csv")
    write_csv(grid_csv, result_rows, fallback_fields=["param_id"])
    print(f"[SAVE-CSV] {grid_csv}")

    if result_rows:
        best = result_rows[0]
        print("\n[BEST]")
        for key in [
            "param_id",
            "MAE",
            "RMSE",
            "bias",
            "avg_pred",
            "avg_gt",
            "over",
            "under",
            "dense_point_ratio_mean",
            "duplicate_point_ratio_mean",
            "broad_dense_point_ratio_mean",
            "min_eps_mean_avg",
            "alpha_mean_avg",
            "max_eps_mean_avg",
            "radius_p90_avg",
            "worst_abs_err",
            "worst_image",
        ]:
            print(f"  {key} = {best.get(key)}")

    if best_rows:
        best_csv = os.path.join(save_dir, f"best_per_image_{best_pid}.csv")
        write_csv(best_csv, best_rows, fallback_fields=["param_id", "image_path"])
        print(f"[SAVE-CSV] {best_csv}")


if __name__ == "__main__":
    main()
