import argparse
import json
import logging
import os
import sys
from types import SimpleNamespace

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import yaml
from torch.utils.data import DataLoader

from dataset import build_dataset
from main import collate_points_padded, configure_trainable_params, load_model_state
from models import Diffusion_schedule, HungarianMatcher, SetCriterion, build_model
from models.train_loop import validate_one_epoch


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--thresholds", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
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
    thresholds = parse_float_list(args_cli.thresholds)
    if thresholds is not None:
        cfg["threshold_sweep"] = thresholds

    args = SimpleNamespace(**cfg)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("device=%s", device)
    _, val_data = build_dataset(args)
    val_num_workers = int(getattr(args, "val_num_workers", getattr(args, "num_workers", 0)))
    val_loader_kwargs = {
        "batch_size": int(args.batch_size),
        "shuffle": False,
        "num_workers": val_num_workers,
        "pin_memory": bool(getattr(args, "val_pin_memory", True)),
        "collate_fn": collate_points_padded,
    }
    if val_num_workers > 0:
        val_loader_kwargs["persistent_workers"] = bool(getattr(args, "val_persistent_workers", True))
        val_loader_kwargs["prefetch_factor"] = int(getattr(args, "val_prefetch_factor", 2))
    val_loader = DataLoader(val_data, **val_loader_kwargs)

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
        logging.warning(
            "checkpoint loaded with non-strict keys | missing=%s unexpected=%s",
            incompatible.missing_keys,
            incompatible.unexpected_keys,
        )

    matcher = HungarianMatcher(cost_class=0.1, cost_coord=1.0)
    criterion = SetCriterion(
        matcher=matcher,
        lambda_exist=args.lambda_exist,
        lambda_x0=args.lambda_x0,
        lambda_cnt=args.lambda_cnt,
        lambda_bg=args.lambda_bg,
        lambda_eps=args.lambda_eps,
        lambda_cov=args.lambda_cov,
        lambda_cov_hinge=args.lambda_cov_hinge,
        lambda_dup=args.lambda_dup,
        lambda_dup_collapse=float(getattr(args, "lambda_dup_collapse", 0.0)),
        cov_topk=args.cov_topk,
        cov_sigma=args.cov_sigma,
        cov_radius=args.cov_radius,
        cov_hard_weight=float(getattr(args, "cov_hard_weight", 0.0)),
        cov_hard_cap=float(getattr(args, "cov_hard_cap", 1.0)),
        cov_dense_weight=float(getattr(args, "cov_dense_weight", 0.0)),
        cov_dense_radius=float(getattr(args, "cov_dense_radius", 16.0)),
        cov_dense_norm=float(getattr(args, "cov_dense_norm", 4.0)),
        cov_weight_cap=float(getattr(args, "cov_weight_cap", 6.0)),
        region_radius=args.region_radius,
        region_topk=args.region_topk,
        exist_label_mode=str(getattr(args, "exist_label_mode", "hungarian")),
        exist_pos_radius=float(getattr(args, "exist_pos_radius", 6.0)),
        dup_dense_aware=bool(getattr(args, "dup_dense_aware", False)),
        dup_neighbor_radius=float(getattr(args, "dup_neighbor_radius", args.region_radius)),
        dup_allow_extra=int(getattr(args, "dup_allow_extra", 0)),
        dup_collapse_inner_radius=float(getattr(args, "dup_collapse_inner_radius", 2.0)),
        dup_collapse_outer_radius=float(getattr(args, "dup_collapse_outer_radius", 4.0)),
        dup_collapse_far_weight=float(getattr(args, "dup_collapse_far_weight", 4.0)),
    ).to(device)
    sched, _ = Diffusion_schedule(args.diffusion_T, device=device, signal_scale=args.signal_scale)

    val_loss, val_mae, metrics = validate_one_epoch(
        model,
        val_loader,
        device,
        sched,
        criterion,
        args.diffusion_T,
        args.seed,
        hard_thresh=args.hard_thresh,
        ddim_steps=args.ddim_steps,
        nms_radius=float(getattr(args, "nms_radius", 3.0)),
        ddim_eta=float(getattr(args, "ddim_eta", 0.0)),
        test_gate_mode=str(getattr(args, "test_gate_mode", "prob_only")),
        threshold_sweep=getattr(args, "threshold_sweep", None),
        val_ddim_cover_radius=float(getattr(args, "val_ddim_cover_radius", 6.0)),
        val_ddim_candidate_low_score_thresh=float(
            getattr(args, "val_ddim_candidate_low_score_thresh", 0.01)
        ),
        val_num_realizations=int(getattr(args, "val_num_realizations", 1)),
        use_proposal_prior=bool(getattr(args, "use_proposal_prior", False)),
        proposal_prior_start_t=int(getattr(args, "proposal_prior_start_t", 700)),
        proposal_prior_sigma=float(getattr(args, "proposal_prior_sigma", 1.25)),
        proposal_prior_mode=str(getattr(args, "proposal_prior_mode", "occupancy")),
        proposal_prior_cell_capacity=int(getattr(args, "proposal_prior_cell_capacity", 2)),
        lambda_prior_occupancy=float(getattr(args, "lambda_prior_occupancy", 0.0)),
        lambda_prior_density=float(getattr(args, "lambda_prior_density", 0.0)),
        lambda_prior_count=float(getattr(args, "lambda_prior_count", 0.0)),
    )

    result = {
        "config": os.path.abspath(args_cli.config),
        "ckpt_path": args.ckpt_path,
        "threshold_sweep": getattr(args, "threshold_sweep", None),
        "val_loss": float(val_loss),
        "val_MAE": float(val_mae),
        **metrics,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args_cli.output_json:
        os.makedirs(os.path.dirname(args_cli.output_json), exist_ok=True)
        with open(args_cli.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
