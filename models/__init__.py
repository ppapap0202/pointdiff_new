from .pointdiff import ModelBuilder
import torch
from .diffusion_utils import CosineAbarSchedule

# create the main model
def build_model(cfg, training: bool):
   """
   會從 cfg 讀取必要的設定，建立 ModelBuilder 實例。
   你可以依專案風格擴充。
   """
   # 讀設定（給預設，避免缺欄位）
   model = ModelBuilder(in_ch=cfg.in_ch, fpn_c=cfg.fpn_c, cond_c=cfg.cond_c, t_dim=cfg.t_dim, with_score=cfg.with_score)
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
