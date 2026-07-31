from .pointdiff import ModelBuilder
import torch
from .diffusion_utils import CosineAbarSchedule,setCriterion,hungarianMatcher

# create the main model
def build_model(cfg, training: bool):
   model = ModelBuilder(
      in_ch=cfg.in_ch,
      fpn_c=cfg.fpn_c,
      cond_c=cfg.cond_c,
      t_dim=cfg.t_dim,
      with_score=cfg.with_score,
      conf_pos_bands=int(getattr(cfg, "conf_pos_bands", 8)),
      selector_local_features=bool(getattr(cfg, "selector_local_features", False)),
      selector_prior_features=bool(getattr(cfg, "selector_prior_features", False)),
      selector_relative_geometry=bool(getattr(cfg, "selector_relative_geometry", False)),
      selector_relgeom_k=int(getattr(cfg, "selector_relgeom_k", 8)),
      num_refine=int(getattr(cfg, "num_refine", 3)),
      use_proposal_prior=bool(getattr(cfg, "use_proposal_prior", False)),
      proposal_prior_hidden=int(getattr(cfg, "proposal_prior_hidden", 64)),
      selector_refine_gate=bool(getattr(cfg, "selector_refine_gate", False)),
      selector_refine_gate_strength=float(getattr(cfg, "selector_refine_gate_strength", 0.0)),
      selector_refine_gate_detach=bool(getattr(cfg, "selector_refine_gate_detach", True)),
      selector_conf_weighted_merge=bool(getattr(cfg, "selector_conf_weighted_merge", False)),
      selector_merge_weight_min=float(getattr(cfg, "selector_merge_weight_min", 0.05)),
      selector_merge_radius_px=float(getattr(cfg, "selector_merge_radius_px", 0.0)),
      selector_merge_weight_detach=bool(getattr(cfg, "selector_merge_weight_detach", True)),
      selector_refine_point_gate=bool(getattr(cfg, "selector_refine_point_gate", False)),
      selector_refine_point_gate_strength=float(getattr(cfg, "selector_refine_point_gate_strength", 0.0)),
      selector_refine_point_gate_min=float(getattr(cfg, "selector_refine_point_gate_min", 0.25)),
      selector_refine_point_gate_detach=bool(getattr(cfg, "selector_refine_point_gate_detach", True)),
      selector_interaction=bool(getattr(cfg, "selector_interaction", False)),
      selector_interaction_layers=int(getattr(cfg, "selector_interaction_layers", 1)),
      selector_interaction_heads=int(getattr(cfg, "selector_interaction_heads", 4)),
      selector_interaction_radius=float(getattr(cfg, "selector_interaction_radius", 12.0)),
      selector_interaction_dropout=float(getattr(cfg, "selector_interaction_dropout", 0.1)),
      selector_local_competition=bool(getattr(cfg, "selector_local_competition", False)),
      selector_local_competition_radius=float(getattr(cfg, "selector_local_competition_radius", 6.0)),
      selector_local_competition_temperature=float(getattr(cfg, "selector_local_competition_temperature", 0.7)),
      selector_local_competition_strength=float(getattr(cfg, "selector_local_competition_strength", 0.35)),
      selector_local_competition_init_strength=float(getattr(cfg, "selector_local_competition_init_strength", 0.05)),
      selector_local_competition_residual_scale=float(getattr(cfg, "selector_local_competition_residual_scale", 0.25)),
      selector_local_competition_exclude_self=bool(getattr(cfg, "selector_local_competition_exclude_self", True)),
      selector_dim=int(getattr(cfg, "selector_dim", getattr(cfg, "cond_c", 64))),
      selector_feature_fusion=str(getattr(cfg, "selector_feature_fusion", "add")),
   )
   model.train(training)
   return model


def build_optimizers(model, lr: float, lr_backbone: float, weight_decay: float = 1e-4):
   # backbone 小 lr，其他(temb/cond/head/FPN) 大 lr
   back_params, other_params = [], []
   for n, p in model.named_parameters():
      if not p.requires_grad:
         continue
      (back_params if n.startswith("backbone") else other_params).append(p)
   lr = float(lr)
   lr_backbone = float(lr_backbone)
   optim = torch.optim.AdamW(
      [{"params": back_params, "lr": lr_backbone},
       {"params": other_params, "lr": lr}],
      weight_decay=weight_decay
   )
   return optim

def Diffusion_schedule(T,device,signal_scale):
   sched = CosineAbarSchedule(T=T, device=device)
   signal_scale = float(signal_scale)
   return sched,signal_scale

def HungarianMatcher(cost_class=2.0, cost_coord=5.0):
    return hungarianMatcher(cost_class=cost_class, cost_coord=cost_coord)

def SetCriterion(
    matcher,
    lambda_exist=2.0,
    lambda_x0=5.0,
    lambda_cnt=1.0,
    lambda_bg=2,
    lambda_eps=0.1,
    lambda_cov=0.2,
    lambda_cov_hinge=0.0,
    lambda_dup=0.2,
    lambda_dup_collapse=0.0,
    cov_topk=3,
    cov_sigma=4.0,
    cov_radius=6.0,
    cov_hard_weight=0.0,
    cov_hard_cap=1.0,
    cov_dense_weight=0.0,
    cov_dense_radius=16.0,
    cov_dense_norm=4.0,
    cov_weight_cap=6.0,
    region_radius=5.0,
    region_topk=5,
    exist_label_mode="hungarian",
    exist_pos_radius=6.0,
    dup_dense_aware=False,
    dup_neighbor_radius=6.0,
    dup_allow_extra=0,
    dup_collapse_inner_radius=2.0,
    dup_collapse_outer_radius=4.0,
    dup_collapse_far_weight=4.0,
):
    return setCriterion(
        matcher,
        lambda_exist=lambda_exist,
        lambda_x0=lambda_x0,
        lambda_cnt=lambda_cnt,
        lambda_bg=lambda_bg,
        lambda_eps=lambda_eps,
        lambda_cov=lambda_cov,
        lambda_cov_hinge=lambda_cov_hinge,
        lambda_dup=lambda_dup,
        lambda_dup_collapse=lambda_dup_collapse,
        cov_topk=cov_topk,
        cov_sigma=cov_sigma,
        cov_radius=cov_radius,
        cov_hard_weight=cov_hard_weight,
        cov_hard_cap=cov_hard_cap,
        cov_dense_weight=cov_dense_weight,
        cov_dense_radius=cov_dense_radius,
        cov_dense_norm=cov_dense_norm,
        cov_weight_cap=cov_weight_cap,
        region_radius=region_radius,
        region_topk=region_topk,
        exist_label_mode=exist_label_mode,
        exist_pos_radius=exist_pos_radius,
        dup_dense_aware=dup_dense_aware,
        dup_neighbor_radius=dup_neighbor_radius,
        dup_allow_extra=dup_allow_extra,
        dup_collapse_inner_radius=dup_collapse_inner_radius,
        dup_collapse_outer_radius=dup_collapse_outer_radius,
        dup_collapse_far_weight=dup_collapse_far_weight,
    )
