# train_loop.py
import logging
import torch
from torch.cuda.amp import autocast, GradScaler
from models.diffusion_utils import pixels_to_m11, forward_noisy, count_primary_loss, loss_exist_eps_balanced
import time


@torch.no_grad()
def validate_one_epoch(model, data_loader, device, sched, signal_scale=1.0, T: int = 1000):
    model.eval()
    total, steps = 0.0, 0
    run_Lcount = run_Lexist = run_Leps = 0.0
    run_predC = run_gtC = 0.0

    # ★ 新增：收集每張圖的 (pred, gt, err, meta)
    per_image_records = []  # list of dicts

    for images, points_pad, mask, metas in data_loader:
        images     = images.to(device, non_blocking=True)
        points_pad = points_pad.to(device, non_blocking=True)
        mask       = mask.to(device, non_blocking=True)  # [B,N] bool
        B, _, H, W = images.shape

        feats = model.encode(images)
        p0 = pixels_to_m11(points_pad, H, W, scale=signal_scale)

        t = torch.randint(0, T, (B, 1), device=device, dtype=torch.long)
        p_t, eps_true, abar_t = forward_noisy(p0, t, sched)

        eps_pred, exist_logit = model.denoise(feats, p_t, t, abar_t=abar_t, clamp_eps=1e-6)

        # 原本的 loss（batch 平均）
        loss, stats = loss_exist_eps_balanced(
                eps_pred=eps_pred,
                eps_true=eps_true,
                exist_logit=exist_logit,  # raw logits
                mask=mask,
                abar_t=abar_t,
                use_focal=True,
                focal_gamma=2.0,
                focal_alpha=0.75,  # 常數 alpha；若想用動態 alpha 可設為 None
                use_pos_weight=True,  # 動態 pos_weight（再映射成 alpha）
                pos_weight_cap=10.0,  # 動態 pos_weight 的上限
                # ---- 負樣本下採樣 ----
                use_neg_subsample=True,
                neg_pos_ratio=3.0,  # 目標 neg:pos ≈ 3:1
                # ---- L2（位置） ----
                lambda_exist=1.0,
                lambda_eps=0.2,
                gate_eps_by_exist=True,
                gate_detach=True,
                gate_floor=0.1,
                use_snr_weight=False
            )
        total += float(loss); steps += 1
        run_Lexist += float(stats["L_exist"])
        run_Leps   += float(stats["L_eps"])
        run_predC  += float(stats["pred_count_mean"])
        run_gtC    += float(stats["gt_count_mean"])

        # ★ 新增：每張圖的 pred_count / gt_count
        exist_prob  = torch.sigmoid(exist_logit)            # [B,N]
        pred_count  = exist_prob.sum(dim=1)                 # [B]
        gt_count    = mask.sum(dim=1).to(pred_count.dtype)  # [B]
        abs_err     = (pred_count - gt_count).abs()

        # ★ 新增：把這個 batch 的每張圖記錄下來
        # metas[i] 如果是路徑/檔名最好；若不是，也可用 index
        for i in range(B):
            per_image_records.append({
                "meta": metas[i] if isinstance(metas, (list, tuple)) else str(metas),
                "pred": float(pred_count[i]),
                "gt": float(gt_count[i]),
                "abs_err": float(abs_err[i]),
            })

    # 驗證摘要（保留你原本的 log）
    if steps > 0:
        logging.info(
            f"[val] loss={total/steps:.4f} "
            f"Lcnt={run_Lcount/steps:.4f} Lex={run_Lexist/steps:.4f} Leps={run_Leps/steps:.4f} "
            f"predCnt={run_predC/steps:.2f} gtCnt={run_gtC/steps:.2f}"
        )

    # ★ 新增：印出誤差最大的前 5 張（或想要的數量）
    if per_image_records:
        per_image_records.sort(key=lambda d: d["abs_err"], reverse=True)
        topk = per_image_records[:5]
        msg = " | ".join([f" pred={r['pred']:.1f} gt={r['gt']:.1f} err={r['abs_err']:.1f}" for r in topk])
        logging.info(f"[val-top5] {msg}")

    return total / max(steps, 1)



def train_one_epoch(
    model, data_loader, device, optimizer, scaler, sched,
    signal_scale: float = 1.0, T: int = 1000, max_norm: float = 1.0,
    K: int = 3,                      # 每個 batch 同時抽 K 個 timestep
    clamp_eps: float = 1e-3,         # clamp 座標到 [-1+eps, 1-eps]
    log_every: int = 10,             # 每多少步印一次時間
):
    model.train()
    running = 0.0
    step = 0
    t_data_total = 0.0
    t_gpu_total = 0.0

    run_Lcount = 0.0
    run_Lexist = 0.0
    run_Leps   = 0.0
    run_predC  = 0.0
    run_gtC    = 0.0

    for images, points_pad, mask, metas in data_loader:
        t0 = time.time()

        images     = images.to(device, non_blocking=True)   # [B,C,H,W]
        points_pad = points_pad.to(device, non_blocking=True)  # [B,N,2]
        mask       = mask.to(device, non_blocking=True)     # [B,N]
        B, _, H, W = images.shape

        feats = model.encode(images)
        p0 = pixels_to_m11(points_pad, H, W, scale=signal_scale)

        loss_acc = 0.0
        Lc_b = 0.0
        Le_b = 0.0
        Lp_b = 0.0
        predC_b = 0.0
        gtC_b = 0.0
        t1 = time.time()

        for k in range(K):
            t = torch.randint(0, T, (B, 1), device=device, dtype=torch.long)

            with autocast(enabled=(device.type == "cuda")):
                p_t, eps_true, abar_t = forward_noisy(p0, t, sched)
                p_t = p_t.clamp(min=-1.0 + clamp_eps, max=1.0 - clamp_eps)

                eps_pred, exist_logit = model.denoise(feats, p_t, t, abar_t=abar_t, clamp_eps=1e-6)
                loss, stats = loss_exist_eps_balanced(
                    eps_pred=eps_pred,
                    eps_true=eps_true,
                    exist_logit=exist_logit,  # raw logits
                    mask=mask,
                    abar_t=abar_t,
                    use_focal= True,
                    focal_gamma = 2.0,
                    focal_alpha = 0.75,  # 常數 alpha；若想用動態 alpha 可設為 None
                    use_pos_weight= True,  # 動態 pos_weight（再映射成 alpha）
                    pos_weight_cap = 10.0,  # 動態 pos_weight 的上限
                    # ---- 負樣本下採樣 ----
                    use_neg_subsample = True,
                    neg_pos_ratio = 3.0,  # 目標 neg:pos ≈ 3:1
                    # ---- L2（位置） ----
                    lambda_exist = 1.0,
                    lambda_eps = 0.2,
                    gate_eps_by_exist = True,
                    gate_detach = True,
                    gate_floor = 0.1,
                    use_snr_weight = False
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            if max_norm and max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            loss_acc += float(loss)


            Le_b   += float(stats["L_exist"])
            Lp_b   += float(stats["L_eps"])
            predC_b+= float(stats["pred_count_mean"])
            gtC_b  += float(stats["gt_count_mean"])

        t2 = time.time()
        Le_b   /= K
        Lp_b   /= K
        predC_b/= K
        gtC_b  /= K

        running += loss_acc
        step += 1
        t_data_total += (t1 - t0)
        t_gpu_total  += (t2 - t1)

        if (step % log_every) == 0:
            logging.info(
                f"[train] step={step:05d} "
                f"data={t_data_total/step:.3f}s gpu={t_gpu_total/step:.3f}s "
                f"loss={running/step:.4f}"
            )

    return running / max(step, 1)
