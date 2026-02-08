# train_loop.py
import logging
import torch
from models.diffusion_utils import pixels_to_m11, forward_noisy
from models.pointdiff import sample_point_tokens, pool_local_tokens
import numpy as np
from collections import defaultdict
def m11_to_pixels_batch(p_m11: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    p_m11: [B,N,2] in [-1,1]
    return: [B,N,2] in pixel coords
    """
    x = (p_m11[..., 0] + 1) * 0.5 * (W - 1)
    y = (p_m11[..., 1] + 1) * 0.5 * (H - 1)
    return torch.stack([x, y], dim=-1)

@torch.no_grad()
def point_nms_count(xy_pix: torch.Tensor, score: torch.Tensor, r: float) -> int:
    """
    xy_pix: [M,2] (pixel)
    score : [M]
    r     : radius in pixels
    return: kept count after greedy NMS
    """
    M = xy_pix.size(0)
    if M == 0:
        return 0
    order = score.argsort(descending=True)
    xy = xy_pix[order]

    keep = []
    r2 = float(r) * float(r)
    for i in range(M):
        if not keep:
            keep.append(i)
            continue
        # 跟已保留點算距離（平方距離比較快）
        prev = xy[keep]  # [K,2]
        d2 = ((prev - xy[i]).pow(2).sum(dim=1))  # [K]
        if (d2 >= r2).all():
            keep.append(i)
    return len(keep)

@torch.no_grad()
def validate_one_epoch(model, data_loader, device, sched, criterion, T: int = 1000, seed: int = 7113064165):
    model.eval()
    torch.manual_seed(seed)

    # ---- supervised loss 統計 ----
    total_loss = 0.0
    n_steps = 0
    run_Lcnt = run_Lexist = run_Lx0 = run_Leps = 0.0

    # ---- 全圖聚合容器（overlap stride 專用）----
    pred_xy_full = defaultdict(list)   # img_idx -> [tensor(M,2), ...]  pixel coords in full image
    pred_sc_full = defaultdict(list)   # img_idx -> [tensor(M,), ...]
    gt_xy_full   = defaultdict(list)   # img_idx -> [tensor(K,2), ...]  pixel coords in full image

    # 固定 sampler 設定
    steps = 50
    thr   = 0.45
    nms_r = 3.0
    clamp_eps = 1e-3

    abar_all = sched.abar.to(device=device)

    # ---------------- helper: dedup GT ----------------
    def dedup_points_xy(xy_np, decimals=3):
        if xy_np.shape[0] == 0:
            return xy_np
        key = np.round(xy_np, decimals=decimals)
        _, idx = np.unique(key, axis=0, return_index=True)
        return xy_np[np.sort(idx)]

    # ---------------- main loop ----------------
    with torch.no_grad():
        for images, points_pad, mask, metas in data_loader:
            images     = images.to(device, non_blocking=True)
            points_pad = points_pad.to(device, non_blocking=True)
            mask       = mask.to(device, non_blocking=True)

            B, C, H, W = images.shape
            N = points_pad.size(1)

            # encode once
            feats = model.encode(images)
            cond_cache = model.cond.precompute(*feats)

            # pixel -> [-1,1]
            p0 = pixels_to_m11(points_pad, H, W)

            # ---------- (1) 單步 supervised：對齊你原本的 Lex/Lx0/Lcnt/Leps ----------
            t_int = torch.randint(0, T, (B, 1), device=device, dtype=torch.long)
            p_t, eps_true, abar_t_ = forward_noisy(p0, t_int, sched)

            if abar_t_.dim() == 1:
                abar_t = abar_t_.view(B, 1, 1)
            elif abar_t_.dim() == 2:
                abar_t = abar_t_.view(B, 1, 1)
            else:
                abar_t = abar_t_
            abar_t = abar_t.to(device=device)

            eps_pred, exist_logit = model.denoise(
                feats, p_t, t_int, abar_t=abar_t, clamp_eps=1e-6, cond_cache=cond_cache
            )
            if exist_logit is not None and exist_logit.dim() == 3 and exist_logit.size(-1) == 1:
                exist_logit = exist_logit.squeeze(-1)

            loss, L_exist, L_x0, L_cnt, L_eps = criterion(
                p_t=p_t, p0=p0, mask=mask, abar_t=abar_t,
                eps_pred=eps_pred, exist_logit=exist_logit,
            )

            total_loss += float(loss)
            n_steps += 1
            run_Lexist += float(L_exist)
            run_Lx0    += float(L_x0)
            run_Lcnt   += float(L_cnt)
            run_Leps   += float(L_eps)

            # ---------- (2) 多步 DDIM sampling：收集全圖候選點 ----------
            t_seq = torch.linspace(T - 1, 0, steps, device=device, dtype=torch.long)
            p_t_gen = torch.empty((B, N, 2), device=device).uniform_(-1.0 + clamp_eps, 1.0 - clamp_eps)

            exist_logit_x0 = None

            for si, ti in enumerate(t_seq.tolist()):
                ti = int(ti)
                t_tensor = torch.full((B, 1), ti, device=device, dtype=torch.long)
                abar_ti = abar_all[ti].view(1, 1, 1).expand(B, 1, 1)

                need_exist = (si == len(t_seq) - 1)
                eps_hat, exist_logit_t = model.denoise(
                    feats, p_t_gen, t_tensor,
                    abar_t=abar_ti,
                    clamp_eps=1e-6,
                    cond_cache=cond_cache,
                    need_exist=need_exist
                )
                if need_exist:
                    exist_logit_x0 = exist_logit_t

                # x0_hat
                sqrt_ab_t = abar_ti.clamp(1e-6, 1.0).sqrt()
                sqrt_om_t = (1.0 - abar_ti).clamp_min(0).sqrt()
                x0_hat = (p_t_gen - sqrt_om_t * eps_hat) / (sqrt_ab_t + 1e-12)

                # next abar
                if si + 1 < len(t_seq):
                    ti_prev = int(t_seq[si + 1].item())
                    abar_prev = abar_all[ti_prev].view(1, 1, 1).expand(B, 1, 1)
                else:
                    abar_prev = torch.ones((B, 1, 1), device=device)

                sqrt_ab_prev = abar_prev.clamp(1e-6, 1.0).sqrt()
                sqrt_om_prev = (1.0 - abar_prev).clamp_min(0).sqrt()
                p_t_gen = sqrt_ab_prev * x0_hat + sqrt_om_prev * eps_hat
                p_t_gen = p_t_gen.clamp(min=-1.0 + clamp_eps, max=1.0 - clamp_eps)

            if exist_logit_x0 is None:
                raise RuntimeError("DDIM last step did not compute exist_logit_x0. Check need_exist logic.")

            if exist_logit_x0.dim() == 3 and exist_logit_x0.size(-1) == 1:
                exist_logit_x0 = exist_logit_x0.squeeze(-1)

            exist_prob_sample = torch.sigmoid(exist_logit_x0)  # [B,N]
            x0_hat = p_t_gen.detach()                          # [B,N,2]
            x0_pix = m11_to_pixels_batch(x0_hat, H, W)         # tile pixel coords

            # ---- 收集到全圖容器 ----
            for b in range(B):
                meta_b = metas[b] if isinstance(metas, (list, tuple)) else metas
                img_idx = int(meta_b["img_index"])
                top  = float(meta_b["tile_top"])
                left = float(meta_b["tile_left"])

                # GT: tile pixel -> full pixel
                gt_xy_tile = points_pad[b, mask[b]]  # [G,2] tile pixel
                if gt_xy_tile.numel() > 0:
                    gt_xy = gt_xy_tile.clone()
                    gt_xy[:, 0] += left
                    gt_xy[:, 1] += top
                    gt_xy_full[img_idx].append(gt_xy.detach().cpu())

                # Pred candidates: tile pixel -> full pixel
                prob_b = exist_prob_sample[b]
                cand = prob_b > thr
                if cand.any():
                    xy_tile = x0_pix[b, cand]      # [M,2]
                    sc      = prob_b[cand]         # [M]
                    xy = xy_tile.clone()
                    xy[:, 0] += left
                    xy[:, 1] += top
                    pred_xy_full[img_idx].append(xy.detach().cpu())
                    pred_sc_full[img_idx].append(sc.detach().cpu())

    # ---------------- after loop: per-image NMS & metrics ----------------
    img_ids = sorted(set(list(gt_xy_full.keys()) + list(pred_xy_full.keys())))
    errs_hard = []

    for img_idx in img_ids:
        # GT count (dedup)
        if len(gt_xy_full[img_idx]) > 0:
            gt_xy = torch.cat(gt_xy_full[img_idx], dim=0).numpy()
            gt_xy = dedup_points_xy(gt_xy, decimals=3)
            gt_cnt = float(gt_xy.shape[0])
        else:
            gt_cnt = 0.0

        # Pred hard count (full-image NMS)
        if len(pred_xy_full[img_idx]) > 0:
            xy = torch.cat(pred_xy_full[img_idx], dim=0)   # (M,2)
            sc = torch.cat(pred_sc_full[img_idx], dim=0)   # (M,)
            pred_cnt_hard = float(point_nms_count(xy, sc, r=nms_r))
        else:
            pred_cnt_hard = 0.0

        errs_hard.append(pred_cnt_hard - gt_cnt)

    errs_hard = np.array(errs_hard, dtype=np.float32)
    mae_hard_img  = float(np.mean(np.abs(errs_hard))) if errs_hard.size else 0.0
    rmse_hard_img = float(np.sqrt(np.mean(errs_hard**2))) if errs_hard.size else 0.0

    # supervised averages
    if n_steps > 0:
        avg_loss   = total_loss / n_steps
        avg_Lexist = run_Lexist / n_steps
        avg_Lx0    = run_Lx0 / n_steps
        avg_Lcnt   = run_Lcnt / n_steps
        avg_Leps   = run_Leps / n_steps
    else:
        avg_loss = avg_Lexist = avg_Lx0 = avg_Lcnt = avg_Leps = 0.0

    logging.info(
        f"[val] loss={avg_loss:.4f} Lex={avg_Lexist:.4f} Lx0={avg_Lx0:.4f} "
        f"Lcnt={avg_Lcnt:.4f} Leps={avg_Leps:.4f} | "
        f"FULL-IMG hard: MAE={mae_hard_img:.2f} RMSE={rmse_hard_img:.2f} (N={len(img_ids)})"
    )

    return avg_loss, mae_hard_img






def train_one_epoch(
        model,
        data_loader,
        device,
        optimizer,
        criterion,
        scaler,   # GradScaler
        sched,
        T: int = 1000,
        K: int = 10,  # unroll 步數（建議 5~20）
        log_every: int = 10,
        max_norm: float = 1.0,
        ### NEW: 兩個新權重（短鏈口徑）
        lambda_cnt_val: float = 0.01,
        lambda_bg: float = 100.,
    ):
    """
    多步（短鏈）訓練：隨機取 t_start，從 p_{t_start} 開始 unroll K 步，每步都計 loss，最後平均。
    - model 需提供：
        feats = model.encode(images)
        eps_pred, exist_logit = model.denoise(feats, p_t, t_idx, abar_t=..., clamp_eps=...)
    - data_loader 輸出：(images[B,C,H,W], points_pad[B,N,2](pixels), mask[B,N], metas)
    - sched: CosineAbarSchedule，提供 .abar (tensor 長度 T)
    """
    import torch
    import torch.nn.functional as F
    from torch.cuda.amp import autocast

    model.train()

    # ============================================================
    # ### CHANGED (1): 統計變數改成「GPU tensor 累積」，避免每 step .item()/float() 同步
    # ============================================================
    epoch_loss_sum = torch.zeros((), device=device)   # scalar tensor on GPU
    epoch_step_cnt = 0

    bucket_loss     = torch.zeros((), device=device)
    bucket_Lex      = torch.zeros((), device=device)
    bucket_Laux     = torch.zeros((), device=device)  # Laux = Lx0
    bucket_Lcnt     = torch.zeros((), device=device)
    bucket_Lcnt_val = torch.zeros((), device=device)
    bucket_Lbg      = torch.zeros((), device=device)
    bucket_Leps     = torch.zeros((), device=device)
    bucket_k = 0
    # ============================================================

    # 參數檢查
    T_int = int(T)
    if T_int <= 1:
        raise ValueError(f"T must be > 1, got T={T_int}")
    K_int = int(K)
    K_eff_global = max(1, min(K_int, T_int - 1))

    for step, (images, points_pad, mask, metas) in enumerate(data_loader, start=1):
        images     = images.to(device, non_blocking=True)      # [B,C,H,W]
        points_pad = points_pad.to(device, non_blocking=True)  # [B,N,2]
        mask       = mask.to(device, non_blocking=True)        # [B,N]

        B, C, H, W = images.shape
        N_gt = points_pad.size(1)  # e.g., 900

        # encode 一次
        feats = model.encode(images)
        cond_cache = model.cond.precompute(*feats)

        # 若 points_pad 已是 [-1,1]，可改成 p0 = points_pad
        p0 = pixels_to_m11(points_pad, H, W)  # [B,N,2]

        # ---- 隨機起點 t_start ∈ [K_eff, T-1] ----
        K_eff = K_eff_global
        low, high = K_eff, T_int
        if low >= high:
            low = max(1, high - 1)
        t_start = torch.randint(low=low, high=high, size=(B, 1), device=device, dtype=torch.long)

        # 從真實 p0 前向加噪到 p_{t_start}
        p_t, _, _ = forward_noisy(p0, t_start, sched)  # [B,N,2]

        optimizer.zero_grad(set_to_none=True)

        loss_steps = []
        Lex_steps  = []
        Lx0_steps  = []
        Lcnt_steps = []
        Lbg_steps  = []
        Leps_steps = []

        with autocast():
            for k in range(K_eff):
                t_cur = (t_start - k).clamp(min=0)          # [B,1]
                abar_cur = sched.get(t_cur).unsqueeze(-1)   # [B,1,1]

                t_prev = (t_cur - 1).clamp(min=0)           # [B,1]
                abar_prev = sched.get(t_prev).unsqueeze(-1) # [B,1,1]

                need_exist = (k >= K_eff - 4)

                eps_pred, exist_logit = model.denoise(
                    feats, p_t, t_cur,
                    abar_t=abar_cur, clamp_eps=1e-6,
                    cond_cache=cond_cache,
                    need_exist=need_exist
                )

                # ===== lambda_t =====
                eps = 1e-8
                alpha_t = abar_cur / (abar_prev + eps)
                beta_t = 1 - alpha_t
                snr_t = abar_cur / (1 - abar_cur + eps)
                lambda_t = ((1 - beta_t) * (1 - abar_cur) / (beta_t + eps)) / ((1.0 + snr_t) ** 1.0)
                lambda_t_scalar = lambda_t.mean()

                # ===== exist gate（只在最後幾步算分類）=====
                if not need_exist:
                    exist_logit = torch.zeros((B, N_gt), device=device, dtype=eps_pred.dtype)
                    lambda_t_scalar = lambda_t_scalar * 0.0
                    aux_weight_scalar = lambda_t_scalar * 0.0
                else:
                    if exist_logit is None:
                        raise RuntimeError("need_exist=True but denoise returned None exist_logit")
                    exist_logit = torch.clamp(exist_logit, -30.0, 30.0)
                    aux_weight_scalar = (abar_cur ** 2).mean()

                # ===== loss =====
                loss_k, L_exist, L_x0, L_cnt, L_bg, Leps = criterion(
                    p_t=p_t, p0=p0, mask=mask, abar_t=abar_cur,
                    eps_pred=eps_pred, exist_logit=exist_logit,
                    lambda_t=lambda_t_scalar,
                    aux_weight=aux_weight_scalar,
                )
                loss_steps.append(loss_k)
                Lex_steps.append(L_exist)
                Lx0_steps.append(L_x0)
                Lcnt_steps.append(L_cnt)
                Lbg_steps.append(L_bg)
                Leps_steps.append(Leps)

                # --- DDIM 反推一步：p_t -> p_{t-1} ---
                sqrt_ab_t = abar_cur.clamp_min(1e-12).sqrt()
                sqrt_om_t = (1.0 - abar_cur).clamp_min(0).sqrt()
                x0_hat = (p_t - sqrt_om_t * eps_pred) / (sqrt_ab_t + 1e-12)

                sqrt_ab_p = abar_prev.clamp_min(1e-12).sqrt()
                sqrt_om_p = (1.0 - abar_prev).clamp_min(0).sqrt()
                p_t_next = sqrt_ab_p * x0_hat + sqrt_om_p * eps_pred

                p_t = p_t_next.detach().clamp(-1.0 + 1e-3, 1.0 - 1e-3)

            # # === 短鏈結束後：用「最終 x0」對齊驗證口徑，計算 L_cnt_val 與 L_bg ===
            # x0_like = x0_hat  # 顯存緊可用 x0_hat.detach()
            #
            # pf_val = model.cond.forward_cached(cond_cache, x0_like)
            # x0_like = x0_hat  # 顯存緊可用 x0_hat.detach()

            # # 1) pf_hat: [B,N,4C]
            # pf_hat = model.cond.forward_cached(cond_cache, x0_like)
            #
            # # 2) local tokens at x0_like: [S, BN, C]
            # tok4 = sample_point_tokens(cond_cache["q4"], x0_like, patch=model.cond.patch)
            # tok8 = sample_point_tokens(cond_cache["q8"], x0_like, patch=model.cond.patch)
            # tok16 = sample_point_tokens(cond_cache["q16"], x0_like, patch=model.cond.patch)
            # local_tokens_exist = torch.cat([tok4, tok8, tok16], dim=0)  # [S, BN, C]
            # # 3) pool -> [B,N,2C]
            # B, N, _ = x0_like.shape
            # pooled = pool_local_tokens(local_tokens_exist, B, N, use_max=True)  # mean+max
            # # 4) concat -> [B,N,6C]
            # conf_in = torch.cat([pf_hat, pooled], dim=-1)
            # logit_v = model.conf_head(conf_in)
            # if logit_v.dim() == 3 and logit_v.size(-1) == 1:
            #     logit_v = logit_v.squeeze(-1)
            prob_v  = torch.sigmoid(exist_logit)

            pred_cnt_v = prob_v.sum(dim=1)
            gt_cnt     = mask.sum(dim=1).float()
            L_cnt_val  = F.smooth_l1_loss(pred_cnt_v, gt_cnt)

            # 聚合 K 步（平均較穩）+ 加上短鏈口徑 loss
            loss = torch.stack(loss_steps).mean()
            Lex  = torch.stack(Lex_steps).mean()
            Lx0  = torch.stack(Lx0_steps).mean()
            Lcnt = torch.stack(Lcnt_steps).mean()
            Lbg  = torch.stack(Lbg_steps).mean()
            Leps = torch.stack(Leps_steps).mean()

            loss = loss + lambda_cnt_val * L_cnt_val

        # 反傳 + 更新
        scaler.scale(loss).backward()
        if max_norm is not None and max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()

        # ============================================================
        # ### CHANGED (2): 這裡不再用 float(loss)/float(Lex)...（那些會強制同步）
        #                 改成用 GPU tensor detach 後累積；只在 log 時才 .item()
        # ============================================================
        epoch_loss_sum += loss.detach()
        epoch_step_cnt += 1

        bucket_loss     += loss.detach()
        bucket_Lex      += Lex.detach()
        bucket_Laux     += Lx0.detach()
        bucket_Lcnt     += Lcnt.detach()
        bucket_Lcnt_val += L_cnt_val.detach()
        bucket_Lbg      += Lbg.detach()
        bucket_Leps     += Leps.detach()
        bucket_k += 1
        # ============================================================

        # ============================================================
        # ### CHANGED (3): log 時才同步一次（.item() 只出現在這裡）
        # ============================================================
        if step % log_every == 0:
            inv_k = 1.0 / max(1, bucket_k)

            msg = (f"[train-unroll] it={step:05d} "
                   f"loss={(bucket_loss * inv_k).item():.4f} "
                   f"Lex={(bucket_Lex * inv_k).item():.4f} "
                   f"Lx0={(bucket_Laux * inv_k).item():.4f} "
                   f"Lcnt={(bucket_Lcnt * inv_k).item():.4f} "
                   f"Lcnt_val={(bucket_Lcnt_val * inv_k).item():.4f} "
                   f"Lbg={(bucket_Lbg * inv_k).item():.4f} "
                   f"Leps={(bucket_Leps * inv_k).item():.4f}")
            print(msg)

            bucket_loss.zero_()
            bucket_Lex.zero_()
            bucket_Laux.zero_()
            bucket_Lcnt.zero_()
            bucket_Lcnt_val.zero_()
            bucket_Lbg.zero_()
            bucket_Leps.zero_()
            bucket_k = 0
        # ============================================================

    # ============================================================
    # ### CHANGED (4): return 時才 .item() 一次
    # ============================================================
    if epoch_step_cnt == 0:
        return 0.0
    return (epoch_loss_sum / epoch_step_cnt).item()
    # ============================================================


