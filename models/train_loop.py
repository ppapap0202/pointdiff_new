# train_loop.py
import logging
import torch
from torch.cuda.amp import autocast, GradScaler
from models.diffusion_utils import pixels_to_m11, forward_noisy, count_primary_loss, loss_exist_eps_balanced,loss_exist_x0_count
import time


@torch.no_grad()
@torch.no_grad()
def validate_one_epoch(model, data_loader, device, sched, signal_scale=1.0, T: int = 1000):
    import logging
    model.eval()

    # ---- 累積器（supervised loss 統計）----
    total_loss = 0.0
    n_steps = 0
    run_Lexist = run_Laux = 0.0   # Laux: 這裡代表 Lx0
    run_predC = run_gtC = 0.0

    # ---- 短步 DDIM 統計 ----
    total_mae, total_mse, total_imgs = 0.0, 0.0, 0

    # ---- 每張圖紀錄 ----
    per_image_records = []

    # 預先取出 abar
    abar_all = sched.abar.to(device=device)

    for images, points_pad, mask, metas in data_loader:
        images     = images.to(device, non_blocking=True)   # [B,C,H,W] in [0,1]
        points_pad = points_pad.to(device, non_blocking=True)  # [B,N,2] (pixels)
        mask       = mask.to(device, non_blocking=True)        # [B,N] (bool)

        B, C, H, W = images.shape

        # 影像只 encode 一次，供兩路徑共用
        feats = model.encode(images)

        # --- pixel -> [-1,1] 座標（若已是 [-1,1] 就跳過這步）---
        p0 = pixels_to_m11(points_pad, H, W)

        # --- 隨機 t 單步 supervised（x0 + count loss）---
        t_int = torch.randint(0, T, (B, 1), device=device, dtype=torch.long)  # [B,1]
        p_t, eps_true, abar_t = forward_noisy(p0, t_int, sched)               # p_t:[B,N,2], abar_t:[B,1]或[B,N]
        eps_pred, exist_logit = model.denoise(feats, p_t, t_int, abar_t=abar_t, clamp_eps=1e-6)

        # >>> 使用 x0 + count 損失（不要再丟 eps_true / lambda_eps 等 ε 專用參數） <<<
        loss, L_exist, L_x0, L_cnt, predC_mean, gtC_mean = loss_exist_x0_count(
            p_t=p_t, p0=p0, mask=mask, abar_t=abar_t,
            eps_pred=eps_pred, exist_logit=exist_logit,
            lambda_exist=1.0, lambda_x0=1.0, lambda_cnt=0.1
        )

        # ---- 累計 supervised loss 統計 ----
        total_loss += float(loss)
        n_steps    += 1
        run_Lexist += float(L_exist)
        run_Laux   += float(L_x0)
        run_predC  += float(predC_mean)
        run_gtC    += float(gtC_mean)

        # ---- 短步 DDIM 模擬（10步，可調 10~20）----
        steps = 10
        t_seq = torch.linspace(T - 1, 0, steps, device=device, dtype=torch.long)

        # 從均勻噪聲出發
        clamp_eps = 1e-3
        p_t_gen = torch.empty((B, mask.size(1), 2), device=device).uniform_(-1.0 + clamp_eps, 1.0 - clamp_eps)

        last_exist_logit = None
        for i, ti in enumerate(t_seq.tolist()):
            t_tensor = torch.full((B, 1), ti, device=device, dtype=torch.long)
            eps_hat, exist_logit_t = model.denoise(feats, p_t_gen, t_tensor, abar_t=abar_all[ti], clamp_eps=1e-6)
            last_exist_logit = exist_logit_t

            sqrt_ab_t  = (abar_all[ti].clamp(1e-6, 1.0)).sqrt()
            sqrt_om_t  = (1.0 - abar_all[ti]).clamp_min(0).sqrt()
            x0_hat     = (p_t_gen - sqrt_om_t * eps_hat) / sqrt_ab_t

            # 下一步的 abar
            abar_prev = abar_all[t_seq[i+1]] if i+1 < len(t_seq) else torch.tensor(1.0, device=device)
            sqrt_ab_prev = abar_prev.clamp(1e-6, 1.0).sqrt()
            sqrt_om_prev = (1.0 - abar_prev).clamp_min(0).sqrt()

            p_t_gen = sqrt_ab_prev * x0_hat + sqrt_om_prev * eps_hat
            p_t_gen = p_t_gen.clamp(min=-1.0 + clamp_eps, max=1.0 - clamp_eps)

        # 用 sampling 的存在機率估人數（而不是 supervised 的 exist_logit）
        exist_prob_sample = torch.sigmoid(last_exist_logit.detach())  # [B,N]
        pred_cnt = exist_prob_sample.sum(dim=1).cpu().numpy()         # [B]
        gt_cnt   = mask.sum(dim=1).float().cpu().numpy()              # [B]

        total_mae += float(abs(pred_cnt - gt_cnt).sum())
        total_mse += float(((pred_cnt - gt_cnt) ** 2).sum())
        total_imgs += len(gt_cnt)

        # 紀錄每張圖誤差（用 sampling 的結果）
        for i in range(B):
            per_image_records.append({
                "meta": metas[i] if isinstance(metas, (list, tuple)) else str(metas),
                "pred": float(pred_cnt[i]),
                "gt":   float(gt_cnt[i]),
                "abs_err": float(abs(pred_cnt[i] - gt_cnt[i])),
            })

    # ---- 輸出 summary ----
    if n_steps > 0:
        avg_loss  = total_loss / n_steps
        avg_Lexist= run_Lexist / n_steps
        avg_Lx0   = run_Laux   / n_steps
        avg_predC = run_predC  / n_steps
        avg_gtC   = run_gtC    / n_steps
    else:
        avg_loss = avg_Lexist = avg_Lx0 = avg_predC = avg_gtC = 0.0

    if total_imgs > 0:
        avg_mae  = total_mae / total_imgs
        avg_rmse = (total_mse / total_imgs) ** 0.5
    else:
        avg_mae = avg_rmse = 0.0

    logging.info(
        f"[val] loss={avg_loss:.4f} Lex={avg_Lexist:.4f} Lx0={avg_Lx0:.4f} "
        f"predCnt={avg_predC:.2f} gtCnt={avg_gtC:.2f} | MAE={avg_mae:.2f} RMSE={avg_rmse:.2f}"
    )

    if per_image_records:
        per_image_records.sort(key=lambda d: d["abs_err"], reverse=True)
        topk = per_image_records[:5]
        msg = " | ".join([f"pred={r['pred']:.1f} gt={r['gt']:.1f} err={r['abs_err']:.1f}" for r in topk])
        logging.info(f"[val-top5] {msg}")

    # 建議回傳 (val_loss, val_MAE) 方便挑 best_by_loss / best_by_mae
    return avg_loss, avg_mae




def train_one_epoch(
        model,
        data_loader,
        device,
        optimizer,
        scaler: GradScaler,
        sched,
        T: int = 1000,
        K: int = 10,  # unroll 步數（建議 5~20）
        loss_mode: str = "x0_count",  # "eps" 或 "x0_count"
        lambda_exist: float = 1.0,
        lambda_eps: float = 1.0,  # 只在 eps 模式用
        lambda_x0: float = 1.0,  # 只在 x0_count 模式用
        lambda_cnt: float = 0.1,  # 只在 x0_count 模式用
        log_every: int = 50,
        max_norm: float = 1.0
):
    """
    多步（短鏈）訓練：隨機取 t_start，從 p_{t_start} 開始 unroll K 步，每步都計 loss，最後平均。
    - model 需提供：
        feats = model.encode(images)
        eps_pred, exist_logit = model.denoise(feats, p_t, t_idx, abar_t=..., clamp_eps=...)
    - data_loader 輸出：(images[B,C,H,W], points_pad[B,N,2](pixels), mask[B,N], metas)
    - sched: CosineAbarSchedule，提供 .abar (tensor 長度 T)
    """


    model.train()

    # bucket 累計顯示
    loss_acc = 0.0
    Le_b = Laux_b = 0.0  # Laux = Leps 或 Lx0
    Lcnt_b = 0.0
    predC_b = gtC_b = 0.0
    k_bucket = 0

    for step, (images, points_pad, mask, metas) in enumerate(data_loader, start=1):
        images = images.to(device, non_blocking=True)  # [B,C,H,W]
        points_pad = points_pad.to(device, non_blocking=True)  # [B,N,2]
        mask = mask.to(device, non_blocking=True)  # [B,N]
        B, C, H, W = images.shape

        # encode 一次
        feats = model.encode(images)

        # pixel → [-1,1]（若你的資料已經是 [-1,1]，改為 p0 = points_pad 即可）
        p0 = pixels_to_m11(points_pad,H, W)  # [B,N,2]

        # ---- 隨機起點 t_start ----
        T_int = int(T)
        K_int = int(K)
        if T_int <= 1:
            raise ValueError(f"T must be > 1, got T={T_int}")
        K_eff = max(1, min(K_int, T_int - 1))
        low, high = K_eff, T_int
        if low >= high:  # 保底
            low = max(1, high - 1)
        t_start = torch.randint(low=low, high=high, size=(B, 1),
                                device=device, dtype=torch.long)  # [B,1]

        # 從真實 p0 加噪得到 p_{t_start}
        p_t, _, _ = forward_noisy(p0, t_start, sched)

        optimizer.zero_grad(set_to_none=True)
        total_loss_k = 0.0
        total_Lexist = total_Laux = total_Lcnt = 0.0
        total_predC = total_gtC = 0.0

        with autocast():
            for k in range(K_eff):
                # --- 當前時間步 ---
                t_cur = (t_start - k).clamp(min=0)  # [B,1]
                idx = t_cur.squeeze(-1)  # [B]
                abar_cur = sched.abar.index_select(0, idx).unsqueeze(-1).unsqueeze(-1)  # [B,1,1]

                # 預測
                eps_pred, exist_logit = model.denoise(feats, p_t, t_cur, abar_t=abar_cur, clamp_eps=1e-6)

                # ---- N 對齊：保證 [B,N,2] 與 [B,N,2] 一致 ----
                N_pred = eps_pred.size(1)


                loss_k, L_exist, L_x0, L_cnt, predC, gtC = loss_exist_x0_count(
                        p_t=p_t, p0=p0, mask=mask, abar_t=abar_cur,
                        eps_pred=eps_pred, exist_logit=exist_logit,
                        lambda_exist=lambda_exist,
                        lambda_x0=lambda_x0,
                        lambda_cnt=lambda_cnt
                    )
                L_aux = L_x0

                total_loss_k += loss_k
                total_Lexist += float(L_exist)
                total_Laux += float(L_aux)
                total_Lcnt += float(L_cnt)
                total_predC += float(predC)
                total_gtC += float(gtC)

                # --- 更新 p_t → p_{t-1} ---
                sqrt_ab_t = abar_cur.sqrt()
                sqrt_om_t = (1.0 - abar_cur).clamp_min(0).sqrt()
                x0_hat = (p_t - sqrt_om_t * eps_pred) / (sqrt_ab_t + 1e-6)

                t_prev = (t_cur - 1).clamp(min=0)
                idx_prev = t_prev.squeeze(-1)
                abar_prev = sched.abar.index_select(0, idx_prev).unsqueeze(-1).unsqueeze(-1)  # [B,1,1]
                sqrt_ab_p = abar_prev.sqrt()
                sqrt_om_p = (1.0 - abar_prev).clamp_min(0).sqrt()

                p_t = sqrt_ab_p * x0_hat + sqrt_om_p * eps_pred
                p_t = p_t.clamp(min=-1.0 + 1e-3, max=1.0 - 1e-3)

            # 平均 loss
            loss = total_loss_k / K_eff

        # 反傳
        scaler.scale(loss).backward()
        if max_norm is not None and max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()

        # bucket 累積
        loss_acc += float(loss)
        Le_b += total_Lexist / K_eff
        Laux_b += total_Laux / K_eff
        Lcnt_b += total_Lcnt / K_eff
        predC_b += total_predC / K_eff
        gtC_b += total_gtC / K_eff
        k_bucket += 1

        if step % log_every == 0:
            msg = (f"[train-unroll] it={step:05d} loss={loss_acc / k_bucket:.4f} "
                   f"Lex={Le_b / k_bucket:.4f} "
                   f"{'Leps' if loss_mode == 'eps' else 'Lx0'}={Laux_b / k_bucket:.4f} ")
            if loss_mode != "eps":
                msg += f"Lcnt={Lcnt_b / k_bucket:.4f} "
            msg += f"predCnt={predC_b / k_bucket:.2f} gtCnt={gtC_b / k_bucket:.2f}"
            print(msg)

            # reset
            loss_acc = Le_b = Laux_b = Lcnt_b = predC_b = gtC_b = 0.0
            k_bucket = 0

    return loss_acc
