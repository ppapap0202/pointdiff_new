import logging
from datetime import datetime
import os
if os.environ.get("POINTDIFF_CUDA_LAUNCH_BLOCKING", "").strip().lower() in {"1", "true", "yes", "on"}:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import argparse
import yaml
from dataset import build_dataset, dataset_pos_neg_stats
from torch.utils.data import DataLoader
from models import build_model, build_optimizers, Diffusion_schedule,HungarianMatcher,SetCriterion
from models.train_loop import train_one_epoch,validate_one_epoch
import torch
import time
from visualize import visualization

# --- Logging 初始化 ---
def setup_logging(out_dir=None):
    os.makedirs('logs', exist_ok=True)
    log_file = os.path.join('logs', f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    if out_dir is not None:
        logging.info(f'out_dir={out_dir}')


def parse_args():
    def load_config(yaml_path):
        with open(yaml_path, 'r', encoding="utf-8") as f:
            return yaml.safe_load(f)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=r'config\train.yaml', type=str)
    args, remaining_argv = parser.parse_known_args()
    cfg = load_config(args.config)
    #print(cfg)
    parser = argparse.ArgumentParser(parents=[parser],add_help=False)
    for key, value in cfg.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)
    args = parser.parse_args()
    return args


def data(args):
    train_data, val_data= build_dataset(args)
    return train_data,val_data

def collate_points_padded(batch):
    import torch
    imgs, pts, metas = zip(*batch)
    imgs = torch.stack(imgs, 0)  # (B,C,H,W)
    B, C, H, W = imgs.shape

    # 計算此 batch 內最大點數
    max_n = 900#max(p.size(0) for p in pts)
    B = len(pts)
    padded = torch.empty((B, max_n, 2), dtype=torch.float32)  # 先不填
    mask = torch.zeros((B, max_n), dtype=torch.bool)

    padded[..., 0].uniform_(0, W - 1)
    padded[..., 1].uniform_(0, H - 1)

    for i, p in enumerate(pts):
        n = p.size(0)
        if n > 0:
            n = min(n, max_n)
            padded[i, :n] = p[:n]
            mask[i, :n] = True

    return imgs, padded, mask, list(metas)


def load_model_state(model, state_dict, shape_compatible_only=False):
    if not shape_compatible_only:
        return model.load_state_dict(state_dict, strict=False)

    model_state = model.state_dict()
    compatible = {}
    skipped_shapes = []
    for key, value in state_dict.items():
        target = model_state.get(key)
        if target is None:
            continue
        if target.shape == value.shape:
            compatible[key] = value
        else:
            skipped_shapes.append((key, tuple(value.shape), tuple(target.shape)))

    incompatible = model.load_state_dict(compatible, strict=False)
    loaded_numel = sum(value.numel() for value in compatible.values())
    total_numel = sum(value.numel() for value in model_state.values())
    logging.info(
        "shape-compatible checkpoint load: "
        f"loaded_tensors={len(compatible)}/{len(model_state)} "
        f"loaded_numel={loaded_numel}/{total_numel} "
        f"skipped_shape_tensors={len(skipped_shapes)}"
    )
    for key, source_shape, target_shape in skipped_shapes[:20]:
        logging.info(
            f"shape-skip {key}: checkpoint={source_shape} model={target_shape}"
        )
    if len(skipped_shapes) > 20:
        logging.info(f"shape-skip: {len(skipped_shapes) - 20} more tensors omitted")
    return incompatible


def configure_trainable_params(model, args):
    if bool(getattr(args, "freeze_base_for_prior", False)) and bool(getattr(args, "freeze_selector_only", False)):
        raise ValueError("freeze_base_for_prior and freeze_selector_only are mutually exclusive")

    if bool(getattr(args, "freeze_selector_only", False)):
        trainable_prefixes = [
            "conf_from_pro.",
            "conf_pos_proj.",
            "selector_local_proj.",
        ]
        module_names = ["conf_from_pro", "conf_pos_proj", "selector_local_proj"]
        selector_dim = int(getattr(args, "selector_dim", getattr(args, "cond_c", 64)))
        selector_fusion = str(getattr(args, "selector_feature_fusion", "add")).strip().lower().replace("-", "_")
        if selector_fusion != "add" or selector_dim != int(getattr(args, "cond_c", selector_dim)):
            trainable_prefixes.append("selector_fuse_proj.")
            module_names.append("selector_fuse_proj")
        if bool(getattr(args, "selector_relative_geometry", False)):
            trainable_prefixes.append("selector_relgeom_proj.")
            module_names.append("selector_relgeom_proj")
        if bool(getattr(args, "train_selector_prior_proj", True)):
            trainable_prefixes.append("selector_prior_proj.")
            module_names.append("selector_prior_proj")
        if bool(getattr(args, "selector_interaction", False)):
            trainable_prefixes.append("selector_interaction.")
            module_names.append("selector_interaction")
        if bool(getattr(args, "selector_local_competition", False)):
            trainable_prefixes.append("selector_local_competition.")
            module_names.append("selector_local_competition")

        for name, param in model.named_parameters():
            param.requires_grad = any(name.startswith(prefix) for prefix in trainable_prefixes)

        model.freeze_selector_only = True
        model.selector_finetune_module_names = tuple(module_names)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(
            "freeze_selector_only=True "
            f"trainable_params={trainable} modules={','.join(module_names)}"
        )
        return

    if bool(getattr(args, "freeze_base_for_prior", False)):
        if not bool(getattr(args, "use_proposal_prior", False)):
            raise ValueError("freeze_base_for_prior=True requires use_proposal_prior=True")
        prior_mode = str(getattr(args, "proposal_prior_mode", "occupancy")).strip().lower()
        for name, param in model.named_parameters():
            train_prior_param = name.startswith("proposal_prior_head.")
            if prior_mode in {"density", "density_only", "density-only"}:
                train_prior_param = train_prior_param and ".occupancy." not in name
            param.requires_grad = train_prior_param
        model.freeze_base_for_prior = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"freeze_base_for_prior=True trainable_params={trainable}")


def main(args):
    #讀取參數
    device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
    import logging
    logging.info(f'device={device}')
    #訓練資料處理
    train_data,val_data=data(args)

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,  # 加速 CPU→GPU 拷貝
        persistent_workers=True,  # 避免每個 epoch 重啟 worker
        prefetch_factor=4,  # 每個 worker 預取 4 個 batch
        collate_fn=collate_points_padded
    )

    val_num_workers = int(getattr(args, "val_num_workers", args.num_workers))
    val_loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": val_num_workers,
        "pin_memory": bool(getattr(args, "val_pin_memory", True)),
        "collate_fn": collate_points_padded,
    }
    if val_num_workers > 0:
        val_loader_kwargs["persistent_workers"] = bool(
            getattr(args, "val_persistent_workers", True)
        )
        val_loader_kwargs["prefetch_factor"] = int(
            getattr(args, "val_prefetch_factor", 2)
        )
    val_loader = DataLoader(val_data, **val_loader_kwargs)
    # for a,b,c in train_data:
    #     visualization(a,b,c)
    # _ = dataset_pos_neg_stats(train_loader)
    # _ = dataset_pos_neg_stats(val_loader)
    for imgs, pts, mask, metas in train_loader:
        logging.info(f'images.shape: {imgs.shape}')  # (B, C, H, W)
        logging.info(f'points.shape: {pts.shape}')  # (B, max_len, 2)
        logging.info(f'mask.shape: {mask.shape}')  # (B, max_len)
        logging.info(metas)
        break
    model = build_model(args, training=True).to(device)
    configure_trainable_params(model, args)
    #print(model)
    optim = build_optimizers(model, lr=args.lr, lr_backbone=args.lr_backbone, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    matcher = HungarianMatcher(cost_class=0.1, cost_coord=1.0)  # 權重需要調參
    criterion = SetCriterion(matcher=matcher,
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
                             exist_duplicate_weight=float(getattr(args, "exist_duplicate_weight", 1.0)),
                             dup_dense_aware=bool(getattr(args, "dup_dense_aware", False)),
                             dup_neighbor_radius=float(getattr(args, "dup_neighbor_radius", args.region_radius)),
                             dup_allow_extra=int(getattr(args, "dup_allow_extra", 0)),
                             dup_collapse_inner_radius=float(getattr(args, "dup_collapse_inner_radius", 2.0)),
                             dup_collapse_outer_radius=float(getattr(args, "dup_collapse_outer_radius", 4.0)),
                             dup_collapse_far_weight=float(getattr(args, "dup_collapse_far_weight", 4.0))).to(device)
    T=args.diffusion_T
    sched, signal_scale = Diffusion_schedule(T, device=device, signal_scale=args.signal_scale)

    best_val = 1e9
    best_mae = 1e9
    best_cover = -1.0
    start_epoch = 1
    last_val_loss = best_val
    last_val_MAE = best_mae
    last_val_ddim_metrics = {
        "val_ddim_raw_cover@6": 0.0,
        "val_ddim_candidate_cover@6": 0.0,
        "val_ddim_dup@6": 0.0,
        "val_ddim_candidates_per_gt": 0.0,
    }
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt_path = str(getattr(args, "ckpt_path", "") or "")
    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location=device)
    #
    # # 載入模型與優化器參數
        state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
        incompatible = load_model_state(
            model,
            state_dict,
            shape_compatible_only=bool(
                getattr(args, "load_shape_compatible_only", False)
            ),
        )
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print(
                "[WARN] checkpoint loaded with non-strict keys | "
                f"missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}"
            )
        resume_training = bool(getattr(args, "resume_training", False))
        reset_optimizer = bool(getattr(args, "reset_optimizer", False))
        if resume_training and not reset_optimizer and isinstance(checkpoint, dict) and "optim_state" in checkpoint:
            optim.load_state_dict(checkpoint["optim_state"])
        if not reset_optimizer and isinstance(checkpoint, dict) and "scaler_state" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state"])
        if resume_training and isinstance(checkpoint, dict):
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            best_val = float(checkpoint.get("best_val", best_val))
            best_mae = float(checkpoint.get("best_mae", best_mae))
            checkpoint_metrics = checkpoint.get("val_ddim_metrics", {})
            best_cover = float(checkpoint.get(
                "best_cover",
                checkpoint_metrics.get("val_ddim_candidate_cover@6", best_cover),
            ))
            last_val_loss = best_val
            last_val_MAE = best_mae
            if isinstance(checkpoint_metrics, dict):
                last_val_ddim_metrics.update(checkpoint_metrics)
            logging.info(
                f"resume_training=True start_epoch={start_epoch} "
                f"best_val={best_val:.4f} best_mae={best_mae:.4f} "
                f"best_cover={best_cover:.4f} reset_optimizer={int(reset_optimizer)}"
            )
        print("loaded checkpoint", ckpt_path)

    print('start training')
    validate_every = max(
        1,
        int(getattr(args, "validate_every", getattr(args, "val_every", 1))),
    )
    validate_final = bool(getattr(args, "validate_final", True))

    for epoch in range(start_epoch, args.epochs+1):

        t0 = time.time()
        tr_loss = train_one_epoch(
            model,
            train_loader,
            device,
            optim,
            criterion,
            scaler,
            sched,
            args.diffusion_T,
            args.K,
            args.log_every,
            args.max_norm,
            lambda_gt_loss=float(getattr(args, "lambda_gt_loss", 1.0)),
            lambda_cnt_val=float(getattr(args, "lambda_cnt_val", 0.0)),
            lambda_rand_cover=float(getattr(args, "lambda_rand_cover", 0.0)),
            lambda_rand_match=float(getattr(args, "lambda_rand_match", 0.0)),
            lambda_rand_exist=float(getattr(args, "lambda_rand_exist", 0.0)),
            lambda_rand_count=float(getattr(args, "lambda_rand_count", 0.0)),
            lambda_rand_count_margin=float(getattr(args, "lambda_rand_count_margin", 0.0)),
            lambda_rand_bg=float(getattr(args, "lambda_rand_bg", 0.0)),
            lambda_rand_dup=float(getattr(args, "lambda_rand_dup", 0.0)),
            lambda_rand_rank=float(getattr(args, "lambda_rand_rank", 0.0)),
            lambda_rand_one_to_one_conf=float(getattr(args, "lambda_rand_one_to_one_conf", 0.0)),
            lambda_rand_group_pos=float(getattr(args, "lambda_rand_group_pos", 0.0)),
            lambda_rand_soft_compete=float(getattr(args, "lambda_rand_soft_compete", 0.0)),
            lambda_rand_group_quota=float(getattr(args, "lambda_rand_group_quota", 0.0)),
            rand_cover_t_min=int(getattr(args, "rand_cover_t_min", 700)),
            rand_cover_t_max=int(getattr(args, "rand_cover_t_max", args.diffusion_T - 1)),
            rand_cover_radius=float(getattr(args, "rand_cover_radius", 6.0)),
            rand_bg_ignore_radius=float(getattr(args, "rand_bg_ignore_radius", 6.0)),
            rand_bg_topk=int(getattr(args, "rand_bg_topk", 0)),
            rand_dup_radius=float(getattr(args, "rand_dup_radius", 6.0)),
            rand_dup_topk=int(getattr(args, "rand_dup_topk", 12)),
            rand_dup_dense_aware=bool(getattr(args, "rand_dup_dense_aware", False)),
            rand_dup_neighbor_radius=float(getattr(args, "rand_dup_neighbor_radius", 6.0)),
            rand_dup_allow_extra=int(getattr(args, "rand_dup_allow_extra", 0)),
            rand_rank_near_radius=float(getattr(args, "rand_rank_near_radius", 6.0)),
            rand_rank_far_radius=float(getattr(args, "rand_rank_far_radius", 12.0)),
            rand_rank_margin=float(getattr(args, "rand_rank_margin", 1.0)),
            rand_rank_neg_topk=int(getattr(args, "rand_rank_neg_topk", 96)),
            rand_rank_far_weight=float(getattr(args, "rand_rank_far_weight", 0.25)),
            rand_one_to_one_radius=float(getattr(args, "rand_one_to_one_radius", 6.0)),
            rand_one_to_one_bg_radius=float(getattr(args, "rand_one_to_one_bg_radius", 12.0)),
            rand_one_to_one_margin=float(getattr(args, "rand_one_to_one_margin", 1.0)),
            rand_one_to_one_neg_topk=int(getattr(args, "rand_one_to_one_neg_topk", 128)),
            rand_one_to_one_count_pos_logit=float(getattr(args, "rand_one_to_one_count_pos_logit", 2.0)),
            rand_one_to_one_count_neg_logit=float(getattr(args, "rand_one_to_one_count_neg_logit", -2.0)),
            rand_one_to_one_count_slack=int(getattr(args, "rand_one_to_one_count_slack", 0)),
            rand_one_to_one_pos_logit=float(getattr(args, "rand_one_to_one_pos_logit", 0.0)),
            rand_one_to_one_dup_neg_logit=float(getattr(args, "rand_one_to_one_dup_neg_logit", 0.0)),
            rand_one_to_one_bg_neg_logit=float(getattr(args, "rand_one_to_one_bg_neg_logit", 0.0)),
            rand_one_to_one_pos_weight=float(getattr(args, "rand_one_to_one_pos_weight", 1.0)),
            rand_one_to_one_dup_weight=float(getattr(args, "rand_one_to_one_dup_weight", 3.0)),
            rand_one_to_one_bg_weight=float(getattr(args, "rand_one_to_one_bg_weight", 0.5)),
            rand_one_to_one_rank_weight=float(getattr(args, "rand_one_to_one_rank_weight", 3.0)),
            rand_one_to_one_count_weight=float(getattr(args, "rand_one_to_one_count_weight", 1.0)),
            rand_group_pos_radius=float(getattr(args, "rand_group_pos_radius", 6.0)),
            rand_group_pos_sigma=float(getattr(args, "rand_group_pos_sigma", 4.0)),
            rand_group_pos_temperature=float(getattr(args, "rand_group_pos_temperature", 0.7)),
            rand_group_pos_logit=float(getattr(args, "rand_group_pos_logit", 2.0)),
            rand_group_pos_nearest_gt_only=bool(getattr(args, "rand_group_pos_nearest_gt_only", True)),
            rand_soft_compete_winner_radius=float(getattr(args, "rand_soft_compete_winner_radius", 6.0)),
            rand_soft_compete_radius=float(getattr(args, "rand_soft_compete_radius", 10.0)),
            rand_soft_compete_sigma=float(getattr(args, "rand_soft_compete_sigma", 4.0)),
            rand_soft_compete_bg_radius=float(getattr(args, "rand_soft_compete_bg_radius", 12.0)),
            rand_soft_compete_margin=float(getattr(args, "rand_soft_compete_margin", 1.0)),
            rand_soft_compete_neg_topk=int(getattr(args, "rand_soft_compete_neg_topk", 64)),
            rand_soft_compete_temperature=float(getattr(args, "rand_soft_compete_temperature", 1.0)),
            rand_soft_compete_nearest_gt_only=bool(getattr(args, "rand_soft_compete_nearest_gt_only", True)),
            rand_soft_compete_pos_logit=float(getattr(args, "rand_soft_compete_pos_logit", 2.0)),
            rand_soft_compete_dup_neg_logit=float(getattr(args, "rand_soft_compete_dup_neg_logit", 0.0)),
            rand_soft_compete_bg_neg_logit=float(getattr(args, "rand_soft_compete_bg_neg_logit", -0.5)),
            rand_soft_compete_pos_weight=float(getattr(args, "rand_soft_compete_pos_weight", 0.5)),
            rand_soft_compete_rank_weight=float(getattr(args, "rand_soft_compete_rank_weight", 2.0)),
            rand_soft_compete_dup_weight=float(getattr(args, "rand_soft_compete_dup_weight", 0.5)),
            rand_soft_compete_softmax_weight=float(getattr(args, "rand_soft_compete_softmax_weight", 1.0)),
            rand_soft_compete_bg_weight=float(getattr(args, "rand_soft_compete_bg_weight", 0.25)),
            rand_group_quota_radius=float(getattr(args, "rand_group_quota_radius", 6.0)),
            rand_group_quota_bg_radius=float(getattr(args, "rand_group_quota_bg_radius", 14.0)),
            rand_group_quota_target=float(getattr(args, "rand_group_quota_target", 1.0)),
            rand_group_quota_margin=float(getattr(args, "rand_group_quota_margin", 1.0)),
            rand_group_quota_neg_topk=int(getattr(args, "rand_group_quota_neg_topk", 64)),
            rand_group_quota_nearest_gt_only=bool(getattr(args, "rand_group_quota_nearest_gt_only", True)),
            rand_group_quota_pos_logit=float(getattr(args, "rand_group_quota_pos_logit", 2.0)),
            rand_group_quota_dup_neg_logit=float(getattr(args, "rand_group_quota_dup_neg_logit", 0.0)),
            rand_group_quota_bg_neg_logit=float(getattr(args, "rand_group_quota_bg_neg_logit", -0.5)),
            rand_group_quota_pos_weight=float(getattr(args, "rand_group_quota_pos_weight", 0.5)),
            rand_group_quota_weight=float(getattr(args, "rand_group_quota_weight", 3.0)),
            rand_group_quota_over_weight=float(getattr(args, "rand_group_quota_over_weight", 1.0)),
            rand_group_quota_under_weight=float(getattr(args, "rand_group_quota_under_weight", 0.25)),
            rand_group_quota_gap_weight=float(getattr(args, "rand_group_quota_gap_weight", 2.0)),
            rand_group_quota_dup_weight=float(getattr(args, "rand_group_quota_dup_weight", 1.5)),
            rand_group_quota_bg_weight=float(getattr(args, "rand_group_quota_bg_weight", 0.25)),
            rand_count_pos_margin=float(getattr(args, "rand_count_pos_margin", 0.35)),
            rand_count_neg_margin=float(getattr(args, "rand_count_neg_margin", 0.10)),
            rand_count_neg_topk=int(getattr(args, "rand_count_neg_topk", 64)),
            rand_count_slack=int(getattr(args, "rand_count_slack", 0)),
            use_proposal_prior=bool(getattr(args, "use_proposal_prior", False)),
            proposal_prior_sigma=float(getattr(args, "proposal_prior_sigma", 1.25)),
            proposal_prior_mode=str(getattr(args, "proposal_prior_mode", "occupancy")),
            proposal_prior_cell_capacity=int(getattr(args, "proposal_prior_cell_capacity", 2)),
            lambda_prior_occupancy=float(getattr(args, "lambda_prior_occupancy", 0.0)),
            lambda_prior_density=float(getattr(args, "lambda_prior_density", 0.0)),
            lambda_prior_count=float(getattr(args, "lambda_prior_count", 0.0)),
        )
        t1 = time.time()
        should_validate = (
            validate_every <= 1
            or epoch == start_epoch
            or epoch % validate_every == 0
            or (validate_final and epoch == args.epochs)
        )
        if should_validate:
            val_loss, val_MAE, val_ddim_metrics = validate_one_epoch(
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
            last_val_loss = val_loss
            last_val_MAE = val_MAE
            last_val_ddim_metrics = val_ddim_metrics
        else:
            val_loss = last_val_loss
            val_MAE = last_val_MAE
            val_ddim_metrics = last_val_ddim_metrics
        t2 = time.time()
        if should_validate:
            logging.info(
                f"[Epoch {epoch:04d}] train={tr_loss:.4f}  val={val_loss:.4f} val_MAE={val_MAE:.4f} "
                f"val_ddim_raw_cover@6={val_ddim_metrics['val_ddim_raw_cover@6']:.4f} "
                f"val_ddim_candidate_cover@6={val_ddim_metrics['val_ddim_candidate_cover@6']:.4f} "
                f"val_ddim_dup@6={val_ddim_metrics['val_ddim_dup@6']:.4f} "
                f"val_ddim_candidates_per_gt={val_ddim_metrics['val_ddim_candidates_per_gt']:.4f} "
                f"val_conf_no_nms_mae={val_ddim_metrics.get('val_conf_no_nms_mae', 0.0):.2f} "
                f"val_conf_no_nms_rmse={val_ddim_metrics.get('val_conf_no_nms_rmse', 0.0):.2f} "
                f"val_conf_no_nms_thr={val_ddim_metrics.get('val_conf_no_nms_thr', 0.0):.3f} "
                f"val_conf_no_nms_recall@6={val_ddim_metrics.get('val_conf_no_nms_recall@6', 0.0):.4f} "
                f"val_conf_no_nms_precision@6={val_ddim_metrics.get('val_conf_no_nms_precision@6', 0.0):.4f} "
                f"val_conf_no_nms_dup@6={val_ddim_metrics.get('val_conf_no_nms_dup@6', 0.0):.4f} "
                f"val_conf_no_nms_selected_per_gt={val_ddim_metrics.get('val_conf_no_nms_selected_per_gt', 0.0):.4f} "
                f"val_prior_count_mae={val_ddim_metrics.get('val_prior_count_mae', 0.0):.4f} "
                f"val_prior_guided_slots={val_ddim_metrics.get('val_prior_guided_slots', 0.0):.2f} "
                f"val_prior_guided_cover@6={val_ddim_metrics.get('val_prior_guided_cover@6', 0.0):.4f} "
                f"val_prior_uniform_cover@6={val_ddim_metrics.get('val_prior_uniform_cover@6', 0.0):.4f} "
                f"val_prior_cover_gain@6={val_ddim_metrics.get('val_prior_cover_gain@6', 0.0):.4f} "
                f"val_prior_guided_dup@6={val_ddim_metrics.get('val_prior_guided_dup@6', 0.0):.4f} "
                f"val_prior_uniform_dup@6={val_ddim_metrics.get('val_prior_uniform_dup@6', 0.0):.4f} "
                f"val_prior_points_per_gt={val_ddim_metrics.get('val_prior_points_per_gt', 0.0):.4f}"
            )
        else:
            logging.info(
                f"[Epoch {epoch:04d}] train={tr_loss:.4f} "
                f"val=SKIPPED validate_every={validate_every} "
                f"last_val={val_loss:.4f} last_val_MAE={val_MAE:.4f}"
            )
        candidate_cover = float(val_ddim_metrics["val_ddim_candidate_cover@6"])
        is_best_cover = should_validate and candidate_cover > best_cover
        if is_best_cover:
            best_cover = candidate_cover
        last_path = os.path.join(args.out_dir, f"last_epoch{epoch:04d}.pth")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optim.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_val": best_val,
            "best_mae": best_mae,
            "best_cover": best_cover,
            "val_ddim_metrics": val_ddim_metrics,
        }, last_path)
        if should_validate and val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(args.out_dir, f"best_epoch{epoch:04d}_val{val_loss:.2f}.pth")
            print('save model',best_path)
            torch.save(model.state_dict(), best_path)
        if should_validate and val_MAE < best_mae:
            best_mae = val_MAE
            best_mae_path = os.path.join(args.out_dir, f"best_mae_epoch{epoch:04d}_mae{val_MAE:.2f}.pth")
            print('save best mae model', best_mae_path)
            torch.save(model.state_dict(), best_mae_path)
        if is_best_cover:
            best_cover_path = os.path.join(
                args.out_dir,
                f"best_cover_epoch{epoch:04d}_cover{candidate_cover:.4f}.pth",
            )
            print('save best cover model', best_cover_path)
            torch.save(model.state_dict(), best_cover_path)
        t4 = time.time()
        print("train_sec", t1 - t0, "val_sec", t2 - t1, "one epoch", (t4 - t0) )



if __name__ == '__main__':
    args = parse_args()
    setup_logging(args.out_dir)
    main(args)
