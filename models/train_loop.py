# train_loop.py
import logging
import torch
from torch.cuda.amp import autocast, GradScaler
from models.diffusion_utils import pixels_to_m11, forward_noisy
import torch.nn.functional as F
import time

def unroll_once_with_feats(feats, images, points_pad, mask, t_seq, abar, clamp_eps, W, H):
    B, _, _, _ = images.shape
    N = points_pad.shape[1]

    # init p_t（用同一個隨機種子保證兩次一致）
    g = torch.Generator(device=images.device)
    g.manual_seed(12345)  # 保證重現
    p_t = torch.empty((B, N, 2), device=images.device, generator=g).uniform_(-1.0 + clamp_eps, 1.0 - clamp_eps)

    last_exist_logit = None
    for i, t_int in enumerate(t_seq.tolist()):
        t_tensor = torch.full((B, 1), t_int, device=images.device, dtype=torch.long)
        eps_pred, exist_logit = model.denoise(feats, p_t, t_tensor)
        last_exist_logit = exist_logit

        abar_t   = abar[t_int]
        abar_prev = abar[t_seq[i + 1]] if i + 1 < len(t_seq) else torch.tensor(1.0, device=images.device)
        p_t = ddim_reverse_step(p_t, eps_pred, abar_t, abar_prev)
        p_t = p_t.clamp(min=-1.0 + clamp_eps, max=1.0 - clamp_eps)

    # 座標 loss（歸一化空間）
    mask_bool = mask.bool()
    # 先把 GT 轉成 [-1,1]
    p0 = points_pad.to(dtype=p_t.dtype).clone()
    p0[...,0] = p0[...,0] / (W-1) * 2 - 1
    p0[...,1] = p0[...,1] / (H-1) * 2 - 1
    Lx0 = F.smooth_l1_loss(p_t[mask_bool], p0[mask_bool]) if mask_bool.any() else p_t.new_tensor(0.0)

    # 轉像素誤差（直覺用）
    scale = torch.tensor([(W-1)/2.0, (H-1)/2.0], device=p_t.device, dtype=p_t.dtype)
    diff_px = (p_t - p0).abs() * scale
    l2_px = torch.sqrt(diff_px[...,0]**2 + diff_px[...,1]**2 + 1e-12)
    mae_l2_px = l2_px[mask_bool].mean().item() if mask_bool.any() else float('nan')

    # 計數
    exist_prob = torch.sigmoid(last_exist_logit)
    pred_cnt = exist_prob.sum(dim=1)               # [B]
    gt_cnt   = mask.sum(dim=1).float()             # [B]
    mae_cnt  = (pred_cnt - gt_cnt).abs().mean().item()

    return {
        "p0": p0.detach(),
        "p_end": p_t.detach(),
        "Lx0": float(Lx0.item()),
        "mae_l2_px": float(mae_l2_px),
        "pred_cnt": pred_cnt.detach().cpu().numpy(),
        "gt_cnt": gt_cnt.detach().cpu().numpy(),
        "mae_cnt": float(mae_cnt),
        "exist_prob": exist_prob.detach()
    }
@torch.no_grad()
def validate_one_epoch(model, data_loader, device, sched, criterion, T: int = 1000):
    import logging
    model.eval()

    # ---- 累積器（supervised loss 統計）----
    total_loss = 0.0
    n_steps = 0
    run_Lcnt = run_Lexist = run_Laux = 0.0   # Laux: 這裡代表 Lx0
    run_predC = run_gtC = 0.0

    # ---- 短步 DDIM 統計 ----
    total_mae, total_mse, total_imgs = 0.0, 0.0, 0

    # ---- 每張圖紀錄 ----
    per_image_records = []

    # 預先取出 abar（1D: [T]）
    abar_all = sched.abar.to(device=device)

    for images, points_pad, mask, metas in data_loader:
        images     = images.to(device, non_blocking=True)      # [B,C,H,W]
        points_pad = points_pad.to(device, non_blocking=True)  # [B,N,2] (pixels)
        mask       = mask.to(device, non_blocking=True)        # [B,N] (bool/0-1)

        B, C, H, W = images.shape
        N = points_pad.size(1)

        # 影像只 encode 一次
        feats = model.encode(images)

        # pixel -> [-1,1]（若已是 [-1,1]，可直接 p0 = points_pad）
        p0 = pixels_to_m11(points_pad, H, W)                   # [B,N,2]

        # --- 單步 supervised（x0 + count loss）---
        t_int = torch.randint(0, T, (B, 1), device=device, dtype=torch.long)  # [B,1]
        p_t, eps_true, abar_t_ = forward_noisy(p0, t_int, sched)              # p_t:[B,N,2], abar_t_:[B] or [B,1]
        # 轉成 [B,1,1]
        if abar_t_.dim() == 1:  # [B]
            abar_t = abar_t_.view(B, 1, 1)
        elif abar_t_.dim() == 2:  # [B,1]
            abar_t = abar_t_.view(B, 1, 1)
        else:
            abar_t = abar_t_
        abar_t = abar_t.to(device=device)

        eps_pred, exist_logit = model.denoise(feats, p_t, t_int, abar_t=abar_t, clamp_eps=1e-6)

        # （如果模型輸出 [B,N,1]，壓成 [B,N] ）
        if exist_logit is not None and exist_logit.dim() == 3 and exist_logit.size(-1) == 1:
            exist_logit = exist_logit.squeeze(-1)

        loss, L_exist, L_x0, L_cnt = criterion(
            p_t=p_t, p0=p0, mask=mask, abar_t=abar_t,
            eps_pred=eps_pred, exist_logit=exist_logit,
        )

        # ---- 累計 supervised 統計 ----
        total_loss += float(loss)
        n_steps    += 1
        run_Lcnt   += float(L_cnt)
        run_Lexist += float(L_exist)
        run_Laux   += float(L_x0)
        # run_predC  += float(predC_mean)
        # run_gtC    += float(gtC_mean)

        # ---- 短步 DDIM 模擬（10步）----
        steps = 10
        # 注意：這裡用等距的 long 值（從 T-1 到 0），OK
        t_seq = torch.linspace(T - 1, 0, steps, device=device, dtype=torch.long)

        clamp_eps = 1e-3
        p_t_gen = torch.empty((B, N, 2), device=device).uniform_(-1.0 + clamp_eps, 1.0 - clamp_eps)

        last_exist_logit = None
        for i in range(steps):
            ti = int(t_seq[i].item())
            # [B,1] 的 t tensor
            t_tensor = torch.full((B, 1), ti, device=device, dtype=torch.long)

            # 準備當前與前一步的 abar -> [B,1,1]
            abar_ti = abar_all[ti].view(1, 1, 1).expand(B, 1, 1)                # [B,1,1]
            eps_hat, exist_logit_t = model.denoise(feats, p_t_gen, t_tensor, abar_t=abar_ti, clamp_eps=1e-6)
            last_exist_logit = exist_logit_t

            # sqrt abar_ti / 1-abar_ti
            sqrt_ab_t  = abar_ti.clamp(1e-6, 1.0).sqrt()
            sqrt_om_t  = (1.0 - abar_ti).clamp_min(0).sqrt()
            x0_hat     = (p_t_gen - sqrt_om_t * eps_hat) / (sqrt_ab_t + 1e-12)

            # 下一步 abar
            if i + 1 < steps:
                ti_prev = int(t_seq[i + 1].item())
                abar_prev = abar_all[ti_prev].view(1, 1, 1).expand(B, 1, 1)
            else:
                abar_prev = torch.ones((B, 1, 1), device=device)

            sqrt_ab_prev = abar_prev.clamp(1e-6, 1.0).sqrt()
            sqrt_om_prev = (1.0 - abar_prev).clamp_min(0).sqrt()

            # eta=0 DDIM
            p_t_gen = sqrt_ab_prev * x0_hat + sqrt_om_prev * eps_hat
            p_t_gen = p_t_gen.clamp(min=-1.0 + clamp_eps, max=1.0 - clamp_eps)

        # 用 sampling 的存在機率估人數
        if last_exist_logit is not None:
            if last_exist_logit.dim() == 3 and last_exist_logit.size(-1) == 1:
                last_exist_logit = last_exist_logit.squeeze(-1)  # [B,N]
            exist_prob_sample = torch.sigmoid(last_exist_logit)  # [B,N]
            pred_cnt = exist_prob_sample.sum(dim=1).cpu().numpy()
        else:
            # 極端不會發生，但保底
            pred_cnt = torch.zeros(B, device=device).cpu().numpy()

        gt_cnt = mask.sum(dim=1).float().cpu().numpy()

        total_mae += float(abs(pred_cnt - gt_cnt).sum())
        total_mse += float(((pred_cnt - gt_cnt) ** 2).sum())
        total_imgs += len(gt_cnt)

        # 紀錄每張圖誤差（用 sampling 的結果）
        for i in range(B):
            meta_i = None
            try:
                meta_i = metas[i] if isinstance(metas, (list, tuple)) else metas
            except Exception:
                meta_i = ""
            per_image_records.append({
                "meta": str(meta_i),
                "pred": float(pred_cnt[i]),
                "gt":   float(gt_cnt[i]),
                "abs_err": float(abs(pred_cnt[i] - gt_cnt[i])),
            })

    # ---- 輸出 summary ----
    if n_steps > 0:
        avg_loss   = total_loss / n_steps
        avg_Lexist = run_Lexist / n_steps
        avg_Lcnt   = run_Lcnt   / n_steps
        avg_Lx0    = run_Laux   / n_steps
        # avg_predC  = run_predC  / n_steps
        # avg_gtC    = run_gtC    / n_steps
    else:
        avg_loss = avg_Lexist = avg_Lx0 = avg_Lcnt = 0.0

    if total_imgs > 0:
        avg_mae  = total_mae / total_imgs
        avg_rmse = (total_mse / total_imgs) ** 0.5
    else:
        avg_mae = avg_rmse = 0.0

    logging.info(
        f"[val] loss={avg_loss:.4f} Lex={avg_Lexist:.4f} Lx0={avg_Lx0:.4f} Lcnt={avg_Lcnt:.4f} "
        f" | MAE={avg_mae:.2f} RMSE={avg_rmse:.2f}"
    )

    if per_image_records:
        per_image_records.sort(key=lambda d: d["abs_err"], reverse=True)
        topk = per_image_records[:5]
        msg = " | ".join([f"pred={r['pred']:.1f} gt={r['gt']:.1f} err={r['abs_err']:.1f}" for r in topk])
        logging.info(f"[val-top5] {msg}")

    return avg_loss, avg_mae






def train_one_epoch(
        model,
        data_loader,
        device,
        optimizer,
        criterion,
        scaler: GradScaler,
        sched,
        T: int = 1000,
        K: int = 10,  # unroll 步數（建議 5~20）
        loss_mode: str = "x0_count",  # "eps" 或 "x0_count"（目前僅用於日誌顯示）
        lambda_exist: float = 1.0,
        lambda_eps: float = 1.0,  # 只在 eps 模式用（若你的 loss 會用到）
        lambda_x0: float = 1.0,  # 只在 x0_count 模式用
        lambda_cnt: float = 1.0,  # 只在 x0_count 模式用
        log_every: int = 10,
        max_norm: float = 1.0,):
    """
    多步（短鏈）訓練：隨機取 t_start，從 p_{t_start} 開始 unroll K 步，每步都計 loss，最後平均。
    - model 需提供：
        feats = model.encode(images)
        eps_pred, exist_logit = model.denoise(feats, p_t, t_idx, abar_t=..., clamp_eps=...)
    - data_loader 輸出：(images[B,C,H,W], points_pad[B,N,2](pixels), mask[B,N], metas)
    - sched: CosineAbarSchedule，提供 .abar (tensor 長度 T)
    """

    model.train()

    # ===== 供 epoch 統計 =====
    epoch_loss_sum = 0.0
    epoch_step_cnt = 0

    # bucket（分段顯示）
    bucket_loss = 0.0
    bucket_Lex  = 0.0
    bucket_Laux = 0.0  # Laux = Leps 或 Lx0
    bucket_Lcnt = 0.0
    bucket_predC = 0.0
    bucket_gtC   = 0.0
    bucket_k = 0

    # ---- 小工具：對齊模型輸出 N 到 GT 的 N ----
    def align_pred_N(eps_pred, exist_logit, N_gt):
        B_ = eps_pred.size(0)
        N_pr = eps_pred.size(1)
        if N_pr == N_gt:
            return eps_pred, exist_logit
        if N_pr > N_gt:
            eps_pred   = eps_pred[:, :N_gt]
            if exist_logit is not None:
                exist_logit = exist_logit[:, :N_gt]
            return eps_pred, exist_logit
        # N_pr < N_gt：pad
        pad = N_gt - N_pr
        pad_eps   = torch.zeros(B_, pad, eps_pred.size(-1), device=eps_pred.device, dtype=eps_pred.dtype)
        eps_pred  = torch.cat([eps_pred, pad_eps], dim=1)
        if exist_logit is not None:
            pad_exist = torch.full((B_, pad), -10.0, device=exist_logit.device, dtype=exist_logit.dtype)  # 強負
            exist_logit = torch.cat([exist_logit, pad_exist], dim=1)
        return eps_pred, exist_logit

    # 參數檢查
    T_int = int(T)
    if T_int <= 1:
        raise ValueError(f"T must be > 1, got T={T_int}")
    K_int = int(K)
    K_eff_global = max(1, min(K_int, T_int - 1))

    for step, (images, points_pad, mask, metas) in enumerate(data_loader, start=1):
        #print(f"--- Batch {step} loaded. Starting computation. ---")  # ✅ 2. 檢查點2
        images     = images.to(device, non_blocking=True)   # [B,C,H,W]
        points_pad = points_pad.to(device, non_blocking=True)  # [B,N,2] (像素座標)
        mask       = mask.to(device, non_blocking=True)        # [B,N]   True=前景

        B, C, H, W = images.shape
        N_gt = points_pad.size(1)

        # encode 一次
        feats = model.encode(images)

        # 若 points_pad 已是 [-1,1]，可改成 p0 = points_pad
        p0 = pixels_to_m11(points_pad, H, W)  # [B,N,2]

        # ---- 隨機起點 t_start ∈ [K_eff, T-1] ----
        K_eff = K_eff_global
        low, high = K_eff, T_int  # randint 的 high 為開區間
        if low >= high:
            low = max(1, high - 1)
        t_start = torch.randint(low=low, high=high, size=(B, 1), device=device, dtype=torch.long)  # [B,1]

        # 從真實 p0 前向加噪到 p_{t_start}
        p_t, _, _ = forward_noisy(p0, t_start, sched)  # [B,N,2]

        optimizer.zero_grad(set_to_none=True)

        loss_steps = []
        Lex_steps  = []
        Lx0_steps  = []
        Lcnt_steps = []
        predC_steps = []
        gtC_steps   = []

        with autocast():
            for k in range(K_eff):
                # --- 當前時間步 ---
                t_cur = (t_start - k).clamp(min=0)              # [B,1]
                idx = t_cur.squeeze(-1).long()                  # [B]
                abar_cur = sched.abar.index_select(0, idx)      # [B]
                abar_cur = abar_cur.view(B, 1, 1).to(device)    # [B,1,1]

                # 預測
                eps_pred, exist_logit = model.denoise(
                    feats, p_t, t_cur, abar_t=abar_cur, clamp_eps=1e-6
                )
                # N 對齊（保險）
                eps_pred, exist_logit = align_pred_N(eps_pred, exist_logit, N_gt)
                # 損失（請確保 loss_exist_x0_count 內部對空 mask 做了防呆）
                loss_k, L_exist, L_x0, L_cnt = criterion(
                    p_t=p_t, p0=p0, mask=mask, abar_t=abar_cur,
                    eps_pred=eps_pred, exist_logit=exist_logit,
                )

                loss_steps.append(loss_k)
                Lex_steps.append(L_exist)
                Lx0_steps.append(L_x0)
                Lcnt_steps.append(L_cnt)
                # predC_steps.append(predC)
                # gtC_steps.append(gtC)

                # --- DDIM 反推一步：p_t -> p_{t-1} ---
                # x0_hat = (p_t - sqrt(1-abar_t)*eps) / sqrt(abar_t)
                sqrt_ab_t = abar_cur.clamp_min(1e-12).sqrt()
                sqrt_om_t = (1.0 - abar_cur).clamp_min(0).sqrt()
                x0_hat = (p_t - sqrt_om_t * eps_pred) / (sqrt_ab_t + 1e-12)

                t_prev = (t_cur - 1).clamp(min=0)
                idx_prev = t_prev.squeeze(-1).long()
                abar_prev = sched.abar.index_select(0, idx_prev).view(B, 1, 1).to(device)
                sqrt_ab_p = abar_prev.clamp_min(1e-12).sqrt()
                sqrt_om_p = (1.0 - abar_prev).clamp_min(0).sqrt()

                # eta=0 的 DDIM（deterministic）
                p_t_next = sqrt_ab_p * x0_hat + sqrt_om_p * eps_pred
                # 穩定性：限制在合法範圍
                p_t = p_t_next.clamp(min=-1.0 + 1e-3, max=1.0 - 1e-3).detach()  # 只讓梯度回到本步 eps_pred
            #print(f"--- Batch {step} computation finished. ---")
            # 聚合 K 步（平均較穩）
            loss = torch.stack(loss_steps).mean()
            Lex  = torch.stack(Lex_steps).mean()
            Lx0  = torch.stack(Lx0_steps).mean()
            Lcnt = torch.stack(Lcnt_steps).mean()
            # predC_mean = torch.stack([torch.as_tensor(x, device=device, dtype=loss.dtype) for x in predC_steps]).mean()
            # gtC_mean   = torch.stack([torch.as_tensor(x, device=device, dtype=loss.dtype) for x in gtC_steps]).mean()

        # 反傳 + 更新
        scaler.scale(loss).backward()
        if max_norm is not None and max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()

        # ===== 統計 =====
        epoch_loss_sum += float(loss)
        epoch_step_cnt += 1

        bucket_loss += float(loss)
        bucket_Lex  += float(Lex)
        bucket_Laux += float(Lx0)  # 目前我們以 Lx0 當作 Laux（若你切換 eps 模式，可改成 Leps）
        bucket_Lcnt += float(Lcnt)
        # bucket_predC += float(predC_mean)
        # bucket_gtC   += float(gtC_mean)
        bucket_k += 1

        if step % log_every == 0:
            msg = (f"[train-unroll] it={step:05d} loss={bucket_loss / bucket_k:.4f} "
                   f"Lex={bucket_Lex / bucket_k:.4f} "
                   f"{'Leps' if loss_mode == 'eps' else 'Lx0'}={bucket_Laux / bucket_k:.4f} ")
            if loss_mode != "eps":
                msg += f"Lcnt={bucket_Lcnt / bucket_k:.4f} "
            print(msg)

            # reset bucket
            bucket_loss = bucket_Lex = bucket_Laux = bucket_Lcnt  = 0.0
            bucket_k = 0

    # 避免除以 0
    if epoch_step_cnt == 0:
        return 0.0
    return epoch_loss_sum / epoch_step_cnt

