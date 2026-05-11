# train_loop.py
import logging
import torch
from models.diffusion_utils import pixels_to_m11, forward_noisy, coverage_radius_hinge_loss
from models.pointdiff import sample_point_tokens, pool_local_tokens
import numpy as np
from collections import defaultdict


@torch.no_grad()
def batch_proposal_region_stats(pred_points_m11: torch.Tensor,
                                gt_points_m11: torch.Tensor,
                                gt_mask: torch.Tensor,
                                radius: float = 6.0,
                                H: int = 256,
                                W: int = 256):
    """
    Lightweight batch-level diagnostics:
    - prop_cov@r: fraction of GT that has at least one nearby proposal
    - multi@r: fraction of GT that has more than one nearby proposal
    """
    pred_pix = m11_to_pixels_batch(pred_points_m11, H, W)
    gt_pix = m11_to_pixels_batch(gt_points_m11, H, W)

    cover_rows = []
    multi_rows = []
    r2 = float(radius) * float(radius)

    for b in range(pred_pix.size(0)):
        gmask = gt_mask[b].bool()
        if not gmask.any():
            continue
        G = gt_pix[b, gmask]  # [Ng,2]
        P = pred_pix[b]       # [N,2]
        dist2 = ((G[:, None, :] - P[None, :, :]) ** 2).sum(dim=-1)  # [Ng,N]
        hit_counts = (dist2 <= r2).sum(dim=1).float()               # [Ng]
        cover_rows.append((hit_counts > 0).float().mean())
        multi_rows.append((hit_counts > 1).float().mean())

    if len(cover_rows) == 0:
        zero = pred_points_m11.new_tensor(0.0)
        return zero, zero

    return torch.stack(cover_rows).mean(), torch.stack(multi_rows).mean()
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
def validate_one_epoch(
        model,
        data_loader,
        device,
        sched,
        criterion,
        T: int = 1000,
        seed: int = 7113064165,
        hard_thresh: float = 0.0,
        ddim_steps: int = 50,
        nms_radius: float = 3.0,
        ddim_eta: float = 0.0,
        test_gate_mode: str = "prob_only",
):
    model.eval()
    torch.manual_seed(seed)

    # ---- supervised loss 統計 ----
    total_loss = 0.0
    n_steps = 0
    run_Lcnt = run_Lexist = run_Lx0 = run_Lbg = run_Leps = run_Lcov = run_Lcover = run_Ldup = 0.0

    # ---- 全圖聚合容器（overlap stride 專用）----
    pred_xy_full = defaultdict(list)   # img_idx -> [tensor(M,2), ...]  pixel coords in full image
    pred_sc_full = defaultdict(list)   # img_idx -> [tensor(M,), ...]
    gt_xy_full   = defaultdict(list)   # img_idx -> [tensor(K,2), ...]  pixel coords in full image

    # 固定 sampler 設定
    steps = int(ddim_steps)
    thr = float(hard_thresh)
    nms_r = float(nms_radius)
    eta = float(ddim_eta)
    gate_mode = str(test_gate_mode)
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
            t_fixed = 500  # 你想固定的t，0~T-1
            t_int = torch.full((B, 1), t_fixed, device=device, dtype=torch.long)
            p_t, _, abar_t_ = forward_noisy(p0, t_int, sched)

            if abar_t_.dim() == 1:
                abar_t = abar_t_.view(B, 1, 1)
            elif abar_t_.dim() == 2:
                abar_t = abar_t_.view(B, 1, 1)
            else:
                abar_t = abar_t_
            abar_t = abar_t.to(device=device)

            eps_pred, exist_logit, pro, pred_points_for_cls, pred_valid_mask = model.denoise(
                feats, p_t, t_int,
                abar_t=abar_t,
                clamp_eps=1e-6,
                cond_cache=cond_cache,
                need_exist=True,
            )

            if exist_logit is None:
                raise RuntimeError("validate supervised branch got None exist_logit")

            if exist_logit.dim() == 3 and exist_logit.size(-1) == 1:
                exist_logit = exist_logit.squeeze(-1)

            loss, L_exist, L_x0, L_cnt, L_bg, L_eps, L_cov, L_cover, L_dup = criterion(
                p_t=p_t,
                p0=p0,
                mask=mask,
                pro=pro,
                abar_t=abar_t,
                eps_pred=eps_pred,
                exist_logit=exist_logit,
                pred_points_for_cls=pred_points_for_cls,
                pred_valid_mask=pred_valid_mask,
                lambda_t=None,
                aux_weight=1.0,
            )

            total_loss += float(loss)
            n_steps += 1
            run_Lexist += float(L_exist)
            run_Lx0    += float(L_x0)
            run_Lcnt   += float(L_cnt)
            run_Leps   += float(L_eps)
            run_Lbg += float(L_bg)
            run_Lcov += float(L_cov)
            run_Lcover += float(L_cover)
            run_Ldup += float(L_dup)
            # ---------- (2) 多步 DDIM sampling：收集全圖候選點 ----------
            t_seq = torch.linspace(T - 1, 0, steps, device=device).round().long()
            t_seq = torch.unique_consecutive(t_seq)

            p_t_gen = torch.empty((B, N, 2), device=device).uniform_(-1.0 + clamp_eps, 1.0 - clamp_eps)

            exist_logit_x0 = None
            eps = 1e-12

            for si, ti in enumerate(t_seq.tolist()):
                ti = int(ti)
                t_tensor = torch.full((B, 1), ti, device=device, dtype=torch.long)
                abar_ti = abar_all[ti].view(1, 1, 1).expand(B, 1, 1)

                need_exist = (si == len(t_seq) - 1)
                eps_hat, exist_logit_t, pro, pred_points_for_cls, pred_valid_mask  = model.denoise(
                    feats, p_t_gen, t_tensor,
                    abar_t=abar_ti, clamp_eps=1e-6,
                    cond_cache=cond_cache,
                    need_exist=need_exist
                )
                if need_exist:
                    exist_logit_x0 = exist_logit_t
                if si + 1 < len(t_seq):
                    ti_prev = int(t_seq[si + 1].item())
                    abar_prev = abar_all[ti_prev].view(1, 1, 1).expand(B, 1, 1)
                    eta_step = eta
                else:
                    abar_prev = torch.ones((B, 1, 1), device=device)
                    eta_step = 0.0

                sqrt_ab_t = abar_ti.clamp(0.0, 1.0).add(eps).sqrt()
                sqrt_om_t = (1.0 - abar_ti).clamp_min(0.0).sqrt()
                x0_hat = (p_t_gen - sqrt_om_t * eps_hat) / sqrt_ab_t

                alpha_t = (abar_ti / (abar_prev + eps)).clamp_min(eps)
                sigma2 = (eta_step ** 2) * (
                    ((1.0 - abar_prev).clamp_min(0.0) / (1.0 - abar_ti).clamp_min(eps))
                    * (1.0 - alpha_t).clamp_min(0.0)
                )
                sigma = sigma2.clamp_min(0.0).sqrt()
                c_eps = (1.0 - abar_prev - sigma2).clamp_min(0.0).sqrt()

                p_t_gen = (abar_prev + eps).sqrt() * x0_hat + c_eps * eps_hat
                if eta_step > 0.0:
                    p_t_gen = p_t_gen + sigma * torch.randn_like(p_t_gen)
                p_t_gen = p_t_gen.clamp(min=-1.0 + clamp_eps, max=1.0 - clamp_eps)

            if exist_logit_x0 is None:
                raise RuntimeError("DDIM last step did not compute exist_logit_x0. Check need_exist logic.")

            if exist_logit_x0.dim() == 3 and exist_logit_x0.size(-1) == 1:
                exist_logit_x0 = exist_logit_x0.squeeze(-1)

            if exist_logit_x0.dim() == 3 and exist_logit_x0.size(-1) == 2:
                exist_prob_sample = exist_logit_x0.softmax(-1)[..., 1]  # [B,N] object prob
                if gate_mode == "argmax_only":
                    pos_mask_sample = exist_logit_x0.argmax(-1) == 1
                elif gate_mode == "argmax_or_prob":
                    pos_mask_sample = (exist_logit_x0.argmax(-1) == 1) | (exist_prob_sample > thr)
                else:
                    pos_mask_sample = exist_prob_sample > thr
            else:
                exist_prob_sample = torch.sigmoid(exist_logit_x0)  # [B,N]
                pos_mask_sample = exist_prob_sample > thr
                print("not cross entropy")
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
                cand = pos_mask_sample[b]
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
        avg_Lbg = run_Lbg / n_steps
        avg_Lcov = run_Lcov / n_steps
        avg_Lcover = run_Lcover / n_steps
        avg_Ldup = run_Ldup / n_steps
    else:
        avg_loss = avg_Lexist = avg_Lx0 = avg_Lcnt = avg_Leps = avg_Lbg = avg_Lcov = avg_Lcover = avg_Ldup = 0.0

    logging.info(
        f"[val] loss={avg_loss:.4f} Lex={avg_Lexist:.4f} Lx0={avg_Lx0:.4f} "
        f"Lcnt={avg_Lcnt:.4f} Leps={avg_Leps:.4f} Lbg={avg_Lbg:.4f} "
        f"Lcov={avg_Lcov:.4f} Lcover={avg_Lcover:.4f} Ldup={avg_Ldup:.4f} | "
        f"FULL-IMG hard: MAE={mae_hard_img:.2f} RMSE={rmse_hard_img:.2f} (N={len(img_ids)})"
    )

    return avg_loss, mae_hard_img






def train_one_epoch(
        model,
        data_loader,
        device,
        optimizer,
        criterion,
        scaler,
        sched,
        T: int = 1000,
        K: int = 10,                 # 保留介面相容，這版不使用
        log_every: int = 10,
        max_norm: float = 1.0,
        lambda_cnt_val: float = 0.00,
        lambda_rand_cover: float = 0.0,
        rand_cover_t_min: int = 700,
        rand_cover_t_max: int = 999,
        rand_cover_radius: float = 6.0,
):
    """
    單步 x0 訓練：
    - 每張圖隨機抽一個 t ∈ [0, T-1]
    - 從 p0 前向加噪得到 p_t
    - 模型只做一次 denoise
    - 以 Lx0 / Lex 為主，Leps 為輔助
    """
    import torch
    import torch.nn.functional as F
    from torch.cuda.amp import autocast

    model.train()

    epoch_loss_sum = torch.zeros((), device=device)
    epoch_step_cnt = 0

    bucket_loss     = torch.zeros((), device=device)
    bucket_Lex      = torch.zeros((), device=device)
    bucket_Lx0      = torch.zeros((), device=device)
    bucket_Lcnt     = torch.zeros((), device=device)
    bucket_Lcnt_val = torch.zeros((), device=device)
    bucket_Lbg      = torch.zeros((), device=device)
    bucket_Leps     = torch.zeros((), device=device)
    bucket_Lcov = torch.zeros((), device=device)
    bucket_Lcover = torch.zeros((), device=device)
    bucket_Lrandcover = torch.zeros((), device=device)
    bucket_Ldup = torch.zeros((), device=device)
    bucket_k = 0

    T_int = int(T)
    if T_int <= 1:
        raise ValueError(f"T must be > 1, got T={T_int}")

    for step, (images, points_pad, mask, metas) in enumerate(data_loader, start=1):
        images     = images.to(device, non_blocking=True)      # [B,C,H,W]
        points_pad = points_pad.to(device, non_blocking=True)  # [B,N,2]
        mask       = mask.to(device, non_blocking=True)        # [B,N]

        B, C, H, W = images.shape
        feats = model.encode(images)
        cond_cache = model.cond.precompute(*feats)

        # GT points in [-1, 1]
        p0 = pixels_to_m11(points_pad, H, W)

        # ---- 每張圖隨機抽一個 t ∈ [0, T-1] ----
        t_cur = torch.randint(
            low=0, high=T_int, size=(B, 1),
            device=device, dtype=torch.long
        )

        # 前向加噪：p0 -> p_t
        p_t, _, _ = forward_noisy(p0, t_cur, sched)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            L_rand_cover = torch.zeros((), device=device)
            rand_points_for_stats = None
            abar_cur = sched.get(t_cur).unsqueeze(-1)   # [B,1,1]

            eps_pred, exist_logit, pro, pred_points_for_cls, pred_valid_mask = model.denoise(
                feats, p_t, t_cur,
                abar_t=abar_cur,
                clamp_eps=1e-6,
                cond_cache=cond_cache,
                need_exist=True,
            )

            if exist_logit is None:
                raise RuntimeError("denoise returned None exist_logit")

            exist_logit = torch.clamp(exist_logit, -30.0, 30.0)



            loss, L_exist, L_x0, L_cnt, L_bg, Leps, Lcov, Lcover, Ldup = criterion(
                p_t=p_t,
                p0=p0,
                mask=mask,
                pro=pro,
                abar_t=abar_cur,
                eps_pred=eps_pred,
                exist_logit=exist_logit,
                pred_points_for_cls=pred_points_for_cls,
                pred_valid_mask=pred_valid_mask,
                lambda_t=None,
                aux_weight=1,
            )

            # ---- 額外 count validation-style loss ----
            cls_logits = exist_logit
            if cls_logits.dim() == 3 and cls_logits.size(-1) == 1:
                cls_logits = cls_logits.squeeze(-1)

            if cls_logits.dim() == 3 and cls_logits.size(-1) == 2:
                prob_obj = cls_logits.softmax(-1)[..., 1]
            else:
                prob_obj = torch.sigmoid(cls_logits)

            pred_cnt_v = prob_obj.sum(dim=1)
            gt_cnt = mask.sum(dim=1).float()
            L_cnt_val = F.smooth_l1_loss(pred_cnt_v, gt_cnt)

            loss = loss + lambda_cnt_val * L_cnt_val

            if lambda_rand_cover > 0:
                t_low = max(0, min(int(rand_cover_t_min), T_int - 1))
                t_high = max(t_low, min(int(rand_cover_t_max), T_int - 1))
                t_rand = torch.randint(
                    low=t_low,
                    high=t_high + 1,
                    size=(B, 1),
                    device=device,
                    dtype=torch.long,
                )
                p_rand = torch.empty_like(p0).uniform_(-1.0 + 1e-3, 1.0 - 1e-3)
                abar_rand = sched.get(t_rand).unsqueeze(-1)

                _, _, _, rand_x0_hat, _ = model.denoise(
                    feats,
                    p_rand,
                    t_rand,
                    abar_t=abar_rand,
                    clamp_eps=1e-6,
                    cond_cache=cond_cache,
                    need_exist=False,
                )
                if rand_x0_hat is None:
                    raise RuntimeError("random-start auxiliary branch did not return x0_hat")

                L_rand_cover = coverage_radius_hinge_loss(
                    pred_points_m11=rand_x0_hat,
                    gt_points_m11=p0,
                    gt_mask=mask,
                    pred_valid_mask=None,
                    H=H,
                    W=W,
                    cover_radius=float(rand_cover_radius),
                )
                loss = loss + float(lambda_rand_cover) * L_rand_cover
                rand_points_for_stats = rand_x0_hat

        scaler.scale(loss).backward()
        if max_norm is not None and max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss_sum += loss.detach()
        epoch_step_cnt += 1

        bucket_loss     += loss.detach()
        bucket_Lex      += L_exist.detach()
        bucket_Lx0      += L_x0.detach()
        bucket_Lcnt     += L_cnt.detach()
        bucket_Lcnt_val += L_cnt_val.detach()
        bucket_Lbg      += L_bg.detach()
        bucket_Leps     += Leps.detach()
        bucket_Lcov     += Lcov.detach()
        bucket_Lcover   += Lcover.detach()
        bucket_Lrandcover += L_rand_cover.detach()
        bucket_Ldup     += Ldup.detach()
        bucket_k += 1

        if step % log_every == 0:
            prop_cov, multi_ratio = batch_proposal_region_stats(
                pred_points_m11=pred_points_for_cls.detach() if pred_points_for_cls is not None else p_t.detach(),
                gt_points_m11=p0.detach(),
                gt_mask=mask.detach(),
                radius=6.0,
                H=H,
                W=W,
            )
            if lambda_rand_cover > 0 and rand_points_for_stats is not None:
                rand_prop_cov, rand_multi_ratio = batch_proposal_region_stats(
                    pred_points_m11=rand_points_for_stats.detach(),
                    gt_points_m11=p0.detach(),
                    gt_mask=mask.detach(),
                    radius=float(rand_cover_radius),
                    H=H,
                    W=W,
                )
            else:
                rand_prop_cov = torch.zeros((), device=device)
                rand_multi_ratio = torch.zeros((), device=device)
            inv_k = 1.0 / max(1, bucket_k)
            msg = (
                f"[train-x0] it={step:05d} "
                f"loss={(bucket_loss * inv_k).item():.4f} "
                f"Lex={(bucket_Lex * inv_k).item():.4f} "
                f"Lx0={(bucket_Lx0 * inv_k).item():.4f} "
                f"Lcnt={(bucket_Lcnt * inv_k).item():.4f} "
                f"Lcnt_val={(bucket_Lcnt_val * inv_k).item():.4f} "
                f"Lbg={(bucket_Lbg * inv_k).item():.4f} "
                f"Leps={(bucket_Leps * inv_k).item():.4f} "
                f"Lcov={(bucket_Lcov * inv_k).item():.4f} "
                f"Lcover={(bucket_Lcover * inv_k).item():.4f} "
                f"Lrandcover={(bucket_Lrandcover * inv_k).item():.4f} "
                f"Ldup={(bucket_Ldup * inv_k).item():.4f} "
                f"prop_cov@6={prop_cov.item():.4f} "
                f"multi@6={multi_ratio.item():.4f} "
                f"rand_prop_cov@6={rand_prop_cov.item():.4f} "
                f"rand_multi@6={rand_multi_ratio.item():.4f} "
            )
            print(msg)
            logging.info(msg)

            bucket_loss.zero_()
            bucket_Lex.zero_()
            bucket_Lx0.zero_()
            bucket_Lcnt.zero_()
            bucket_Lcnt_val.zero_()
            bucket_Lbg.zero_()
            bucket_Leps.zero_()
            bucket_Lcov.zero_()
            bucket_Lcover.zero_()
            bucket_Lrandcover.zero_()
            bucket_Ldup.zero_()
            bucket_k = 0

    if epoch_step_cnt == 0:
        return 0.0
    return (epoch_loss_sum / epoch_step_cnt).item()


