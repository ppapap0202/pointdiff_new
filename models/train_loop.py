# train_loop.py
import logging
import torch
from torch.cuda.amp import autocast, GradScaler
from models.diffusion_utils import pixels_to_m11, forward_noisy
import torch.nn.functional as F
import time


# @torch.no_grad()
# def validate_one_epoch(model, data_loader, device, sched, criterion, T: int = 1000):
#     import logging
#     model.eval()
#
#     # ---- 累積器（supervised loss 統計）----
#     total_loss = 0.0
#     n_steps = 0
#     run_Lcnt = run_Lexist = run_Laux = 0.0   # Laux: 這裡代表 Lx0
#     run_predC = run_gtC = 0.0
#
#     # ---- 短步 DDIM 統計 ----
#     total_mae, total_mse, total_imgs = 0.0, 0.0, 0
#
#     # ---- 每張圖紀錄 ----
#     per_image_records = []
#
#     # 預先取出 abar（1D: [T]）
#     abar_all = sched.abar.to(device=device)
#
#     for images, points_pad, mask, metas in data_loader:
#         images     = images.to(device, non_blocking=True)      # [B,C,H,W]
#         points_pad = points_pad.to(device, non_blocking=True)  # [B,N,2] (pixels)
#         mask       = mask.to(device, non_blocking=True)        # [B,N] (bool/0-1)
#
#         B, C, H, W = images.shape
#         N = points_pad.size(1)
#
#         # 影像只 encode 一次
#         feats = model.encode(images)
#
#         # pixel -> [-1,1]（若已是 [-1,1]，可直接 p0 = points_pad）
#         p0 = pixels_to_m11(points_pad, H, W)                   # [B,N,2]
#
#         # --- 單步 supervised（x0 + count loss）---
#         t_int = torch.randint(0, T, (B, 1), device=device, dtype=torch.long)  # [B,1]
#         p_t, eps_true, abar_t_ = forward_noisy(p0, t_int, sched)              # p_t:[B,N,2], abar_t_:[B] or [B,1]
#         # 轉成 [B,1,1]
#         if abar_t_.dim() == 1:  # [B]
#             abar_t = abar_t_.view(B, 1, 1)
#         elif abar_t_.dim() == 2:  # [B,1]
#             abar_t = abar_t_.view(B, 1, 1)
#         else:
#             abar_t = abar_t_
#         abar_t = abar_t.to(device=device)
#
#         eps_pred, exist_logit = model.denoise(feats, p_t, t_int, abar_t=abar_t, clamp_eps=1e-6)
#
#         # （如果模型輸出 [B,N,1]，壓成 [B,N] ）
#         if exist_logit is not None and exist_logit.dim() == 3 and exist_logit.size(-1) == 1:
#             exist_logit = exist_logit.squeeze(-1)
#
#         loss, L_exist, L_x0, L_cnt = criterion(
#             p_t=p_t, p0=p0, mask=mask, abar_t=abar_t,
#             eps_pred=eps_pred, exist_logit=exist_logit,
#         )
#
#         # ---- 累計 supervised 統計 ----
#         total_loss += float(loss)
#         n_steps    += 1
#         run_Lcnt   += float(L_cnt)
#         run_Lexist += float(L_exist)
#         run_Laux   += float(L_x0)
#         # run_predC  += float(predC_mean)
#         # run_gtC    += float(gtC_mean)
#
#         # ---- 短步 DDIM 模擬（10步）----
#         steps = 20
#         # 注意：這裡用等距的 long 值（從 T-1 到 0），OK
#         t_seq = torch.linspace(T - 1, 0, steps, device=device, dtype=torch.long)
#
#         clamp_eps = 1e-3
#         p_t_gen = torch.empty((B, N, 2), device=device).uniform_(-1.0 + clamp_eps, 1.0 - clamp_eps)
#
#         last_exist_logit = None
#         for i in range(steps):
#             ti = int(t_seq[i].item())
#             # [B,1] 的 t tensor
#             t_tensor = torch.full((B, 1), ti, device=device, dtype=torch.long)
#
#             # 準備當前與前一步的 abar -> [B,1,1]
#             abar_ti = abar_all[ti].view(1, 1, 1).expand(B, 1, 1)                # [B,1,1]
#             eps_hat, exist_logit_t = model.denoise(feats, p_t_gen, t_tensor, abar_t=abar_ti, clamp_eps=1e-6)
#             last_exist_logit = exist_logit_t
#
#             # sqrt abar_ti / 1-abar_ti
#             sqrt_ab_t  = abar_ti.clamp(1e-6, 1.0).sqrt()
#             sqrt_om_t  = (1.0 - abar_ti).clamp_min(0).sqrt()
#             x0_hat     = (p_t_gen - sqrt_om_t * eps_hat) / (sqrt_ab_t + 1e-12)
#
#             # 下一步 abar
#             if i + 1 < steps:
#                 ti_prev = int(t_seq[i + 1].item())
#                 abar_prev = abar_all[ti_prev].view(1, 1, 1).expand(B, 1, 1)
#             else:
#                 abar_prev = torch.ones((B, 1, 1), device=device)
#
#             sqrt_ab_prev = abar_prev.clamp(1e-6, 1.0).sqrt()
#             sqrt_om_prev = (1.0 - abar_prev).clamp_min(0).sqrt()
#
#             # eta=0 DDIM
#             p_t_gen = sqrt_ab_prev * x0_hat + sqrt_om_prev * eps_hat
#             p_t_gen = p_t_gen.clamp(min=-1.0 + clamp_eps, max=1.0 - clamp_eps)
#
#         # 走完 DDIM：p_t_gen ≈ x0_hat
#         x0_hat = p_t_gen.detach()  # [B,N,2]
#
#         # 在最終 x0 位置重新取樣評分（與推論一致）
#         pf_hat = model.cond(*feats, x0_hat)  # [B,N,cond*3]
#         exist_logit_x0 = model.conf_head(pf_hat)  # [B,N]
#         exist_prob_sample = torch.sigmoid(exist_logit_x0)  # [B,N]
#         pred_cnt = exist_prob_sample.sum(dim=1).cpu().numpy()
#
#
#         gt_cnt = mask.sum(dim=1).float().cpu().numpy()
#
#         total_mae += float(abs(pred_cnt - gt_cnt).sum())
#         total_mse += float(((pred_cnt - gt_cnt) ** 2).sum())
#         total_imgs += len(gt_cnt)
#
#         # 紀錄每張圖誤差（用 sampling 的結果）
#         for i in range(B):
#             meta_i = None
#             try:
#                 meta_i = metas[i] if isinstance(metas, (list, tuple)) else metas
#             except Exception:
#                 meta_i = ""
#             per_image_records.append({
#                 "meta": str(meta_i),
#                 "pred": float(pred_cnt[i]),
#                 "gt":   float(gt_cnt[i]),
#                 "abs_err": float(abs(pred_cnt[i] - gt_cnt[i])),
#             })
#
#     # ---- 輸出 summary ----
#     if n_steps > 0:
#         avg_loss   = total_loss / n_steps
#         avg_Lexist = run_Lexist / n_steps
#         avg_Lcnt   = run_Lcnt   / n_steps
#         avg_Lx0    = run_Laux   / n_steps
#         # avg_predC  = run_predC  / n_steps
#         # avg_gtC    = run_gtC    / n_steps
#     else:
#         avg_loss = avg_Lexist = avg_Lx0 = avg_Lcnt = 0.0
#
#     if total_imgs > 0:
#         avg_mae  = total_mae / total_imgs
#         avg_rmse = (total_mse / total_imgs) ** 0.5
#     else:
#         avg_mae = avg_rmse = 0.0
#
#     logging.info(
#         f"[val] loss={avg_loss:.4f} Lex={avg_Lexist:.4f} Lx0={avg_Lx0:.4f} Lcnt={avg_Lcnt:.4f} "
#         f" | MAE={avg_mae:.2f} RMSE={avg_rmse:.2f}"
#     )
#
#     if per_image_records:
#         per_image_records.sort(key=lambda d: d["abs_err"], reverse=True)
#         topk = per_image_records[:5]
#         msg = " | ".join([f"pred={r['pred']:.1f} gt={r['gt']:.1f} err={r['abs_err']:.1f}" for r in topk])
#         logging.info(f"[val-top5] {msg}")
#
#     return avg_loss, avg_mae
@torch.no_grad()
def validate_one_epoch(model, data_loader, device, sched, criterion, T: int = 1000):
    import logging, numpy as np
    model.eval()

    # ---- 累積器（supervised loss 統計）----
    total_loss = 0.0
    n_steps = 0
    run_Lcnt = run_Lexist = run_Laux = 0.0   # Laux: 這裡代表 Lx0

    # ---- 短步 DDIM 統計（與推論一致）----
    total_mae, total_mse, total_imgs = 0.0, 0.0, 0

    # ---- 額外統計：單步口徑 & GT+隨機診斷 ----
    total_mae_soft = 0.0                         # 單步口徑 MAE（對齊 L_cnt 口徑）
    total_mae_gtmix = 0.0                        # GT+隨機候選點 的 MAE
    sum_p_gt = 0.0; sum_n_gt = 0                 # 平均 p(GT)
    sum_p_rand = 0.0; sum_n_rand = 0             # 平均 p(rand)
    pairwise_gt_better = 0.0; pairwise_pairs = 0 # 簡易 AUC-ish: P(p(GT)>p(rand))

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
        if abar_t_.dim() == 1:      # [B]
            abar_t = abar_t_.view(B, 1, 1)
        elif abar_t_.dim() == 2:    # [B,1]
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

        # ---- 單步口徑的軟計數（對齊 L_cnt）→ MAE_soft ----
        exist_prob_supervised = torch.sigmoid(exist_logit)                  # [B,N]
        pred_cnt_supervised   = exist_prob_supervised.sum(dim=1).cpu().numpy()
        gt_cnt_np             = mask.sum(dim=1).float().cpu().numpy()
        total_mae_soft       += float(np.abs(pred_cnt_supervised - gt_cnt_np).sum())

        # ---- 診斷：GT + 隨機候選點（不走 DDIM）----
        clamp_eps = 1e-3
        p_candidates = torch.empty_like(p0)                 # [B,N,2]
        gt_counts = mask.sum(dim=1)                         # [B]
        gt_len_list = gt_counts.tolist()

        for b in range(B):
            nb = int(gt_counts[b].item())
            if nb > 0:
                p_candidates[b, :nb] = p0[b, mask[b]]      # 前 nb 放 GT 座標
            m = N - nb
            if m > 0:
                u = torch.rand((m, 2), device=p0.device)   # [0,1)
                rand_pts = (u * (1.0 - 2*clamp_eps) + clamp_eps) * 2.0 - 1.0  # (-1+eps,1-eps)
                p_candidates[b, nb:] = rand_pts

        pf_cand = model.cond(*feats, p_candidates)         # [B,N,cond*3]
        logit_cand = model.conf_head(pf_cand)              # [B,N] or [B,N,1]
        if logit_cand.dim() == 3 and logit_cand.size(-1) == 1:
            logit_cand = logit_cand.squeeze(-1)
        prob_cand = torch.sigmoid(logit_cand)              # [B,N]

        pred_cnt_gtmix = prob_cand.sum(dim=1)              # [B]
        gt_cnt_img     = mask.sum(dim=1).float()           # [B]
        total_mae_gtmix += float(torch.abs(pred_cnt_gtmix - gt_cnt_img).sum().item())

        # 針對 GT 與 隨機 的平均分數與 pairwise (AUC-ish)
        for b in range(B):
            nb = int(gt_len_list[b]); m = N - nb
            if nb > 0:
                sum_p_gt += float(prob_cand[b, :nb].mean().item())
                sum_n_gt += 1
            if m > 0:
                sum_p_rand += float(prob_cand[b, nb:].mean().item())
                sum_n_rand += 1
            if nb > 0 and m > 0:
                gt_scores = prob_cand[b, :nb].unsqueeze(1)   # [nb,1]
                rd_scores = prob_cand[b, nb:].unsqueeze(0)   # [1,m]
                comp = (gt_scores > rd_scores).float().mean().item()
                pairwise_gt_better += comp
                pairwise_pairs += 1

        # ---- 多步 DDIM 模擬（與推論一致的 steps/序列）----
        steps = 50  # 建議與你的測試腳本一致
        t_seq = torch.linspace(T-1, 0, steps, device=device, dtype=torch.long)

        p_t_gen = torch.empty((B, N, 2), device=device).uniform_(-1.0 + clamp_eps, 1.0 - clamp_eps)

        for i, ti in enumerate(t_seq.tolist()):
            t_tensor = torch.full((B, 1), ti, device=device, dtype=torch.long)

            # [B,1,1]
            abar_ti = abar_all[ti].view(1, 1, 1).expand(B, 1, 1)
            eps_hat, _ = model.denoise(feats, p_t_gen, t_tensor, abar_t=abar_ti, clamp_eps=1e-6)

            sqrt_ab_t  = abar_ti.clamp(1e-6, 1.0).sqrt()
            sqrt_om_t  = (1.0 - abar_ti).clamp_min(0).sqrt()
            x0_hat     = (p_t_gen - sqrt_om_t * eps_hat) / (sqrt_ab_t + 1e-12)

            if i + 1 < len(t_seq):
                abar_prev = abar_all[t_seq[i + 1]].view(1, 1, 1).expand(B, 1, 1)
            else:
                abar_prev = torch.ones((B, 1, 1), device=device)

            sqrt_ab_prev = abar_prev.clamp(1e-6, 1.0).sqrt()
            sqrt_om_prev = (1.0 - abar_prev).clamp_min(0).sqrt()

            # eta=0 DDIM
            p_t_gen = sqrt_ab_prev * x0_hat + sqrt_om_prev * eps_hat
            p_t_gen = p_t_gen.clamp(min=-1.0 + clamp_eps, max=1.0 - clamp_eps)

        # 走完 DDIM：p_t_gen ≈ x0_hat；在最終 x0 位置重算存在度（與推論一致）
        x0_hat = p_t_gen.detach()                   # [B,N,2]
        pf_hat = model.cond(*feats, x0_hat)         # [B,N,cond*3]
        exist_logit_x0 = model.conf_head(pf_hat)    # [B,N] or [B,N,1]
        if exist_logit_x0.dim() == 3 and exist_logit_x0.size(-1) == 1:
            exist_logit_x0 = exist_logit_x0.squeeze(-1)
        exist_prob_sample = torch.sigmoid(exist_logit_x0)  # [B,N]
        pred_cnt = exist_prob_sample.sum(dim=1).cpu().numpy()

        gt_cnt = mask.sum(dim=1).float().cpu().numpy()

        total_mae += float(np.abs(pred_cnt - gt_cnt).sum())
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
    else:
        avg_loss = avg_Lexist = avg_Lx0 = avg_Lcnt = 0.0

    if total_imgs > 0:
        avg_mae  = total_mae / total_imgs
        avg_rmse = (total_mse / total_imgs) ** 0.5
        avg_mae_soft  = total_mae_soft / total_imgs
        avg_mae_gtmix = total_mae_gtmix / total_imgs
    else:
        avg_mae = avg_rmse = avg_mae_soft = avg_mae_gtmix = 0.0

    if sum_n_gt > 0:
        avg_p_gt = sum_p_gt / sum_n_gt
    else:
        avg_p_gt = float('nan')
    if sum_n_rand > 0:
        avg_p_rand = sum_p_rand / sum_n_rand
    else:
        avg_p_rand = float('nan')
    if pairwise_pairs > 0:
        aucish = pairwise_gt_better / pairwise_pairs      # 0.5~1.0 越高越好
    else:
        aucish = float('nan')

    logging.info(
        f"[val] loss={avg_loss:.4f} Lex={avg_Lexist:.4f} Lx0={avg_Lx0:.4f} Lcnt={avg_Lcnt:.4f} "
        f"| MAE={avg_mae:.2f} RMSE={avg_rmse:.2f} "
        f"(MAE_soft={avg_mae_soft:.2f}; GT+Rand: MAE_gtmix={avg_mae_gtmix:.2f}, "
        f"avg_p_GT={avg_p_gt:.3f}, avg_p_rand={avg_p_rand:.3f}, AUC~={aucish:.3f})"
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
        log_every: int = 10,
        max_norm: float = 1.0,
        ### NEW: 兩個新權重（短鏈口徑）
        lambda_cnt_val: float = 0.05,
        lambda_bg: float = 25.,        # 推薦 0.5~1.0
    ):
    """
    多步（短鏈）訓練：隨機取 t_start，從 p_{t_start} 開始 unroll K 步，每步都計 loss，最後平均。
    - model 需提供：
        feats = model.encode(images)
        eps_pred, exist_logit = model.denoise(feats, p_t, t_idx, abar_t=..., clamp_eps=...)
    - data_loader 輸出：(images[B,C,H,W], points_pad[B,N,2](pixels), mask[B,N], metas)
    - sched: CosineAbarSchedule，提供 .abar (tensor 長度 T)
    """
    import torch.nn.functional as F  # ### NEW: 確保有 F

    model.train()

    # ===== 供 epoch 統計 =====
    epoch_loss_sum = 0.0
    epoch_step_cnt = 0

    # bucket（分段顯示）
    bucket_loss = 0.0
    bucket_Lex  = 0.0
    bucket_Laux = 0.0  # Laux = Leps 或 Lx0
    bucket_Lcnt = 0.0
    ### NEW: 新增兩個 bucket
    bucket_Lcnt_val = 0.0
    bucket_Lbg = 0.0

    bucket_k = 0

    # 參數檢查
    T_int = int(T)
    if T_int <= 1:
        raise ValueError(f"T must be > 1, got T={T_int}")
    K_int = int(K)
    K_eff_global = max(1, min(K_int, T_int - 1))

    for step, (images, points_pad, mask, metas) in enumerate(data_loader, start=1):
        images     = images.to(device, non_blocking=True)   # [B,C,H,W]
        points_pad = points_pad.to(device, non_blocking=True)  # [B,N,2] (像素座標)
        mask       = mask.to(device, non_blocking=True)        # [B,N]   True=前景

        B, C, H, W = images.shape
        N_gt = points_pad.size(1)  # e.g., 900

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

        with autocast():
            for k in range(K_eff):
                # --- 當前時間步 ---
                t_cur = (t_start - k).clamp(min=0)              # [B,1]
                abar_cur = sched.get(t_cur).unsqueeze(-1)       # [B,1,1]

                # DDIM 需要的上一個時間步
                t_prev = (t_cur - 1).clamp(min=0)               # [B,1]
                abar_prev = sched.get(t_prev).unsqueeze(-1)     # [B,1,1]

                # 預測
                eps_pred, exist_logit = model.denoise(
                    feats, p_t, t_cur, abar_t=abar_cur, clamp_eps=1e-6
                )

                # 損失（單步口徑）
                loss_k, L_exist, L_x0, L_cnt = criterion(
                    p_t=p_t, p0=p0, mask=mask, abar_t=abar_cur,
                    eps_pred=eps_pred, exist_logit=exist_logit,
                )
                loss_steps.append(loss_k)
                Lex_steps.append(L_exist)
                Lx0_steps.append(L_x0)
                Lcnt_steps.append(L_cnt)

                # --- DDIM 反推一步：p_t -> p_{t-1} ---
                # x0_hat = (p_t - sqrt(1-abar_t)*eps) / sqrt(abar_t)
                sqrt_ab_t = abar_cur.clamp_min(1e-12).sqrt()
                sqrt_om_t = (1.0 - abar_cur).clamp_min(0).sqrt()
                x0_hat = (p_t - sqrt_om_t * eps_pred) / (sqrt_ab_t + 1e-12)

                # eta=0 的 DDIM（deterministic）
                sqrt_ab_p = abar_prev.clamp_min(1e-12).sqrt()
                sqrt_om_p = (1.0 - abar_prev).clamp_min(0).sqrt()
                p_t_next = sqrt_ab_p * x0_hat + sqrt_om_p * eps_pred

                # 穩定性（若顯存夠、想做「可微短鏈」可移除 detach 或只留最後幾步不 detach）
                p_t = p_t_next.clamp(-1.0 + 1e-3, 1.0 - 1e-3)

            # === 短鏈結束後：用「最終 x0」對齊驗證口徑，計算 L_cnt_val 與 L_bg ===
            x0_like = x0_hat  # 顯存緊可用 x0_hat.detach()

            pf_val  = model.cond(*feats, x0_like)               # [B,N,cond*3]
            logit_v = model.conf_head(pf_val)                   # [B,N] or [B,N,1]
            if logit_v.dim() == 3 and logit_v.size(-1) == 1:
                logit_v = logit_v.squeeze(-1)
            prob_v  = torch.sigmoid(logit_v)                    # [B,N]

            pred_cnt_v = prob_v.sum(dim=1)                      # 短鏈口徑的軟和
            gt_cnt     = mask.sum(dim=1).float()
            L_cnt_val  = F.mse_loss(pred_cnt_v, gt_cnt)         # 校準總量（MSE）

            # 壓背景均值，避免 avg_p_rand 偏胖
            bgmask = (~mask).float()
            L_bg   = ((prob_v * bgmask).sum(1) / (bgmask.sum(1) + 1e-6)).mean()

            # 聚合 K 步（平均較穩）+ 加上兩個「短鏈口徑」loss
            loss = torch.stack(loss_steps).mean()
            Lex  = torch.stack(Lex_steps).mean()
            Lx0  = torch.stack(Lx0_steps).mean()
            Lcnt = torch.stack(Lcnt_steps).mean()

            loss = loss + lambda_cnt_val * L_cnt_val + lambda_bg * L_bg  # ### NEW

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

        bucket_loss  += float(loss)
        bucket_Lex   += float(Lex)
        bucket_Laux  += float(Lx0)
        bucket_Lcnt  += float(Lcnt)
        ### NEW: 記錄兩個新指標
        bucket_Lcnt_val += float(L_cnt_val)
        bucket_Lbg      += float(L_bg)

        bucket_k += 1

        if step % log_every == 0:
            msg = (f"[train-unroll] it={step:05d} "
                   f"loss={bucket_loss / bucket_k:.4f} "
                   f"Lex={bucket_Lex / bucket_k:.4f} "
                   f"{'Leps' if loss_mode == 'eps' else 'Lx0'}={bucket_Laux / bucket_k:.4f} "
                   f"Lcnt={bucket_Lcnt / bucket_k:.4f} "
                   f"Lcnt_val={bucket_Lcnt_val / bucket_k:.4f} "   # ### NEW
                   f"Lbg={bucket_Lbg / bucket_k:.4f} ")            # ### NEW
            print(msg)

            # reset bucket
            bucket_loss = bucket_Lex = bucket_Laux = bucket_Lcnt = 0.0
            bucket_Lcnt_val = bucket_Lbg = 0.0
            bucket_k = 0

    # 避免除以 0
    if epoch_step_cnt == 0:
        return 0.0
    return epoch_loss_sum / epoch_step_cnt


