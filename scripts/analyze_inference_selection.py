"""Compare candidate-selection algorithms at inference time, no retraining.

The selector currently emits a global threshold decision (prob > hard_thresh).
Diagnosis says two different things are being conflated:

  * "is there a person here"  -> answerable; prior_density alone reaches 0.814 AUC
  * "which group member gets matched" -> not answerable; oracle ceiling 0.35 vs a
    0.31 random baseline, because maximum_bipartite_matching picks arbitrarily
    when several candidates sit inside one GT's radius

So "how many to keep" should come from the model (density), while "which one to
keep" can be decided by an algorithm without loss of correctness.

This script runs the normal DDIM inference once, caches every valid candidate
with its score and sampled density, then applies several selection rules to the
same cached candidates and reports full-image metrics for each. Includes oracle
variants that use the true GT count, which separate "count estimate is wrong"
from "score ranking is wrong".

Read-only: no checkpoint, config or training state is modified.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataset import build_dataset
from main import collate_points_padded, configure_trainable_params, load_model_state
from models import Diffusion_schedule, build_model
from models.diffusion_utils import forward_noisy, pixels_to_m11
from models.pointdiff import sample_selector_prior_features, selector_object_prob
from models.proposal_prior import build_mixed_x0_prior
from models.train_loop import full_image_matching_stats, m11_to_pixels_batch


# ---------------------------------------------------------------- selection ---
def greedy_nms(xy, score, radius):
    """Standard greedy NMS with one shared radius. Returns kept indices."""
    if xy.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(-score)
    kept = []
    for i in order:
        if not kept:
            kept.append(i)
            continue
        d2 = ((xy[kept] - xy[i]) ** 2).sum(axis=1)
        if (d2 >= radius * radius).all():
            kept.append(i)
    return np.asarray(kept, dtype=np.int64)


def greedy_nms_adaptive(xy, score, radii):
    """Greedy NMS where each candidate carries its own suppression radius.

    A point is suppressed by an already-kept point when it falls inside the
    larger of the two radii, so sparse regions clear a wide area and crowded
    regions stay tight.
    """
    if xy.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(-score)
    kept = []
    for i in order:
        if not kept:
            kept.append(i)
            continue
        d = np.sqrt(((xy[kept] - xy[i]) ** 2).sum(axis=1))
        r = np.maximum(radii[kept], radii[i])
        if (d >= r).all():
            kept.append(i)
    return np.asarray(kept, dtype=np.int64)


def density_to_radius(dens, alpha, r_min, r_max):
    """Mean spacing inside a P4 cell (4x4 px) is ~4/sqrt(count)."""
    d = np.clip(dens, 1e-3, None)
    return np.clip(alpha * 4.0 / np.sqrt(d), r_min, r_max)


def build_selectors(cfg):
    """name -> fn(xy, score, dens, k_density, k_oracle, gt) -> kept indices"""
    thr = float(cfg["thr"])
    nms_r = float(cfg["nms_radius"])
    eval_r = float(cfg["eval_radius"])
    sel = {}

    sweep = cfg.get("thr_sweep") or []
    if sweep:
        # Decomposition mode: only the current rule and the oracle-concentration
        # rule, evaluated at each threshold, so the recall loss can be split into
        # coverage / magnitude / dispersion without threshold as a confound.
        def _mk_plain(t):
            return lambda xy, s, d, kd, ko, gt: np.where(s > t)[0]

        def _mk_conc(t):
            def _f(xy, s, d, kd, ko, gt):
                if gt.shape[0] == 0 or xy.shape[0] == 0:
                    return np.zeros(0, dtype=np.int64)
                dist = np.sqrt(((xy[:, None, :] - gt[None, :, :]) ** 2).sum(-1))
                nearest = dist.argmin(1)
                dmin = dist.min(1)
                kept = []
                for gi in range(gt.shape[0]):
                    m = (nearest == gi) & (dmin <= eval_r)
                    if not m.any():
                        continue
                    idx = np.where(m)[0]
                    if s[idx].sum() > t:
                        kept.append(idx[np.argmax(s[idx])])
                return np.asarray(kept, dtype=np.int64)
            return _f

        for t in sweep:
            sel["A_thr%.2f" % t] = _mk_plain(t)
            sel["I_conc_thr%.2f" % t] = _mk_conc(t)
        return sel

    # ---- baselines ----
    sel["A_fixed_thr"] = lambda xy, s, d, kd, ko, gt: np.where(s > thr)[0]

    def _thr_nms(xy, s, d, kd, ko, gt):
        idx = np.where(s > thr)[0]
        return idx[greedy_nms(xy[idx], s[idx], nms_r)] if idx.size else idx
    sel["B_fixed_thr_nms%g" % nms_r] = _thr_nms

    def _topk_density(xy, s, d, kd, ko, gt):
        k = min(max(int(kd), 0), s.shape[0])
        return np.argsort(-s)[:k]
    sel["C_topk_density_count"] = _topk_density

    def _topk_oracle(xy, s, d, kd, ko, gt):
        k = min(max(int(ko), 0), s.shape[0])
        return np.argsort(-s)[:k]
    sel["F_ORACLE_topk_true_count"] = _topk_oracle

    # ---- (2) ORACLE mass concentration -------------------------------------
    # Group by true GT, hand the group's whole probability mass to its strongest
    # member, keep that member when the concentrated mass clears the threshold.
    # Upper bound on "what if the model could concentrate perfectly".
    def _oracle_concentrate(xy, s, d, kd, ko, gt):
        if gt.shape[0] == 0 or xy.shape[0] == 0:
            return np.zeros(0, dtype=np.int64)
        dist = np.sqrt(((xy[:, None, :] - gt[None, :, :]) ** 2).sum(-1))
        nearest = dist.argmin(1)
        dmin = dist.min(1)
        kept = []
        for gi in range(gt.shape[0]):
            m = (nearest == gi) & (dmin <= eval_r)
            if not m.any():
                continue
            idx = np.where(m)[0]
            if s[idx].sum() > thr:
                kept.append(idx[np.argmax(s[idx])])
        return np.asarray(kept, dtype=np.int64)
    sel["I_ORACLE_mass_concentrate"] = _oracle_concentrate

    # ---- (3) radius-based mass transfer, no GT ------------------------------
    # Greedy from the top: a kept point absorbs the probability of every
    # unclaimed neighbour inside `radius`, then those neighbours are removed.
    # Unlike NMS this moves mass rather than discarding it.
    def _mk_mass_transfer(radius):
        def _f(xy, s, d, kd, ko, gt):
            n = s.shape[0]
            if n == 0:
                return np.zeros(0, dtype=np.int64)
            order = np.argsort(-s)
            taken = np.zeros(n, dtype=bool)
            kept, mass = [], []
            for i in order:
                if taken[i]:
                    continue
                dd = np.sqrt(((xy - xy[i]) ** 2).sum(1))
                nb = (dd <= radius) & (~taken)
                mass.append(float(s[nb].sum()))
                taken[nb] = True
                kept.append(i)
            kept = np.asarray(kept, dtype=np.int64)
            mass = np.asarray(mass)
            return kept[mass > thr]
        return _f

    for r in cfg["transfer_radii"]:
        sel["J_mass_transfer_r%g" % r] = _mk_mass_transfer(r)

    # ---- (4) random bias: does breaking symmetry alone help? ----------------
    def _mk_random_bias(eps, seed):
        def _f(xy, s, d, kd, ko, gt):
            rng = np.random.default_rng(seed)
            return np.where(s + rng.normal(0.0, eps, size=s.shape[0]) > thr)[0]
        return _f

    for eps in cfg["bias_sigmas"]:
        sel["K_random_bias_sd%g" % eps] = _mk_random_bias(eps, int(cfg["bias_seed"]))

    return sel


def ddim_step(p_t, eps_pred, abar_t, abar_prev, eta):
    eps = 1e-12
    x0 = (p_t - (1 - abar_t).clamp_min(0).sqrt() * eps_pred) / abar_t.clamp(0, 1).add(eps).sqrt()
    alpha_t = (abar_t / (abar_prev + eps)).clamp_min(eps)
    sigma2 = (eta ** 2) * (((1 - abar_prev).clamp_min(0) / (1 - abar_t).clamp_min(eps))
                           * (1 - alpha_t).clamp_min(0))
    c_eps = (1 - abar_prev - sigma2).clamp_min(0).sqrt()
    out = (abar_prev + eps).sqrt() * x0 + c_eps * eps_pred
    if eta > 0.0:
        out = out + sigma2.clamp_min(0).sqrt() * torch.randn_like(out)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_batches", type=int, default=None)
    p.add_argument("--thr", type=float, default=None, help="Defaults to hard_thresh sweep midpoint 0.6.")
    p.add_argument("--nms_radius", type=float, default=None, help="Defaults to config nms_radius.")
    p.add_argument("--transfer_radii", default="2,4,6",
                   help="Radii for mass-transfer selection.")
    p.add_argument("--bias_sigmas", default="0.1,0.2,0.3",
                   help="Std-devs for the random-bias control.")
    p.add_argument("--bias_seed", type=int, default=17)
    p.add_argument("--thr_sweep", default=None,
                   help="Comma list of thresholds; switches to decomposition mode.")
    p.add_argument("--r_min", type=float, default=2.0)
    p.add_argument("--r_max", type=float, default=16.0)
    p.add_argument("--eval_radius", type=float, default=None)
    p.add_argument("--output_json", default=None)
    a = p.parse_args()

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    cfg["ckpt_path"] = a.ckpt_path
    cfg["resume_training"] = False
    cfg["batch_size"] = int(a.batch_size)
    cfg["val_num_workers"] = int(a.num_workers)
    args = SimpleNamespace(**cfg)
    eval_radius = float(a.eval_radius if a.eval_radius is not None
                        else getattr(args, "val_ddim_cover_radius", 6.0))
    thr = float(a.thr if a.thr is not None else 0.6)
    nms_radius = float(a.nms_radius if a.nms_radius is not None else getattr(args, "nms_radius", 4))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    _, val_data = build_dataset(args)
    lk = {"batch_size": int(args.batch_size), "shuffle": False, "num_workers": int(a.num_workers),
          "pin_memory": True, "collate_fn": collate_points_padded}
    if int(a.num_workers) > 0:
        lk["persistent_workers"] = True
        lk["prefetch_factor"] = 2
    loader = DataLoader(val_data, **lk)

    model = build_model(args, training=True).to(device)
    configure_trainable_params(model, args)
    ck = torch.load(args.ckpt_path, map_location=device)
    sd = ck["model_state"] if isinstance(ck, dict) and "model_state" in ck else ck
    inc = load_model_state(model, sd, shape_compatible_only=bool(
        getattr(args, "load_shape_compatible_only", False)))
    if inc.missing_keys or inc.unexpected_keys:
        print("[WARN] non-strict load | missing=%s unexpected=%s" % (inc.missing_keys, inc.unexpected_keys))
    model.eval()

    sched, _ = Diffusion_schedule(args.diffusion_T, device=device, signal_scale=args.signal_scale)
    abar_all = sched.abar.to(device)
    T = int(args.diffusion_T)
    start_t = (max(0, min(int(getattr(args, "proposal_prior_start_t", T - 1)), T - 1))
               if bool(getattr(args, "use_proposal_prior", False)) else T - 1)
    t_seq = torch.unique_consecutive(
        torch.linspace(start_t, 0, int(args.ddim_steps), device=device).round().long())
    eta = float(getattr(args, "ddim_eta", 0.0))
    ce = 1e-3

    cand_xy = defaultdict(list); cand_sc = defaultdict(list); cand_dn = defaultdict(list)
    gt_xy = defaultdict(list); img_dens_sum = defaultdict(float)

    with torch.no_grad():
        for bi, (images, pts_pad, mask, metas) in enumerate(tqdm(loader, desc="inference selection")):
            if a.max_batches is not None and bi >= int(a.max_batches):
                break
            images = images.to(device, non_blocking=True)
            pts_pad = pts_pad.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            B, _, H, W = images.shape
            N = pts_pad.size(1)

            feats = model.encode(images)
            cache = model.cond.precompute(*feats)
            prior_maps = None
            if bool(getattr(args, "use_proposal_prior", False)):
                occ, dens = model.predict_proposal_prior(feats)
                prior_maps = (occ, dens)

            # keep RNG aligned with validate_one_epoch
            p0 = pixels_to_m11(pts_pad, H, W)
            tf = torch.full((B, 1), 500, device=device, dtype=torch.long)
            ps, _, ab_s = forward_noisy(p0, tf, sched)
            if ab_s.dim() in (1, 2):
                ab_s = ab_s.view(B, 1, 1)
            model.denoise(feats, ps, tf, abar_t=ab_s.to(device), clamp_eps=1e-6,
                          cond_cache=cache, need_exist=True, selector_prior_maps=prior_maps)

            if prior_maps is not None:
                x0p, _ = build_mixed_x0_prior(
                    occ, dens, num_slots=N, clamp_eps=ce,
                    mode=str(getattr(args, "proposal_prior_mode", "occupancy")),
                    density_cell_capacity=int(getattr(args, "proposal_prior_cell_capacity", 2)))
                ti0 = torch.full((B, 1), int(t_seq[0].item()), device=device, dtype=torch.long)
                p_t, _, _ = forward_noisy(x0p, ti0, sched)
            else:
                p_t = torch.empty((B, N, 2), device=device).uniform_(-1 + ce, 1 - ce)

            logit = pts_o = vmask = None
            for si, ti in enumerate(t_seq.tolist()):
                tt = torch.full((B, 1), int(ti), device=device, dtype=torch.long)
                at = abar_all[int(ti)].view(1, 1, 1).expand(B, 1, 1)
                need = si == len(t_seq) - 1
                eps_hat, lg, _, pc, vm = model.denoise(feats, p_t, tt, abar_t=at, clamp_eps=1e-6,
                                                      cond_cache=cache, need_exist=need,
                                                      selector_prior_maps=prior_maps)
                if need:
                    logit, pts_o, vmask = lg, pc, vm
                ap = (abar_all[int(t_seq[si + 1].item())].view(1, 1, 1).expand(B, 1, 1)
                      if si + 1 < len(t_seq) else torch.ones((B, 1, 1), device=device))
                p_t = ddim_step(p_t, eps_hat, at, ap,
                                eta if si + 1 < len(t_seq) else 0.0).clamp(-1 + ce, 1 - ce)

            prob = selector_object_prob(logit.float())
            pf = sample_selector_prior_features(prior_maps, pts_o.detach(), vmask)
            pix = m11_to_pixels_batch(pts_o.detach(), H, W)

            for b in range(B):
                meta = metas[b] if isinstance(metas, (list, tuple)) else metas
                ii = int(meta["img_index"])
                top, left = float(meta["tile_top"]), float(meta["tile_left"])
                hf, wf = meta["orig_size"]

                g = pts_pad[b, mask[b]]
                if g.numel() > 0:
                    gg = g.clone(); gg[:, 0] += left; gg[:, 1] += top
                    gt_xy[ii].append(gg.cpu())

                v = vmask[b].bool()
                if v.any():
                    xy = pix[b, v].clone()
                    xy[:, 0] = (xy[:, 0] + left).clamp(0, float(wf) - 1)
                    xy[:, 1] = (xy[:, 1] + top).clamp(0, float(hf) - 1)
                    cand_xy[ii].append(xy.cpu())
                    cand_sc[ii].append(prob[b, v].cpu())
                    # expm1 undoes the log1p applied when sampling the prior map
                    cand_dn[ii].append(torch.expm1(pf[b, v, 1].float().clamp_min(0)).cpu()
                                       if pf is not None else torch.zeros(int(v.sum())))
                if prior_maps is not None:
                    img_dens_sum[ii] += float(dens[b].sum().item())

    # ------------------------------------------------------------- evaluate ---
    tr = [float(v) for v in str(a.transfer_radii).replace(";", ",").split(",") if v.strip()]
    bs = [float(v) for v in str(a.bias_sigmas).replace(";", ",").split(",") if v.strip()]
    sel_cfg = {"thr": thr, "nms_radius": nms_radius, "eval_radius": eval_radius,
               "transfer_radii": tr, "bias_sigmas": bs, "bias_seed": a.bias_seed,
               "thr_sweep": ([float(v) for v in str(a.thr_sweep).replace(";", ",").split(",")
                              if v.strip()] if a.thr_sweep else [])}
    selectors = build_selectors(sel_cfg)
    ids = sorted(set(list(gt_xy.keys()) + list(cand_xy.keys())))
    empty_gt_total = [0]
    gt_total_all = [0]
    acc = {n: {"abs_err": [], "sq_err": [], "gt": 0, "sel": 0, "tp": 0, "near": 0}
           for n in selectors}

    for ii in tqdm(ids, desc="scoring"):
        g = torch.cat(gt_xy[ii], 0).numpy() if gt_xy[ii] else np.zeros((0, 2), np.float32)
        if cand_xy[ii]:
            xy = torch.cat(cand_xy[ii], 0).numpy()
            s = torch.cat(cand_sc[ii], 0).numpy()
            d = torch.cat(cand_dn[ii], 0).numpy()
        else:
            xy = np.zeros((0, 2), np.float32); s = np.zeros(0, np.float32); d = np.zeros(0, np.float32)
        k_density = int(round(img_dens_sum.get(ii, 0.0)))
        k_oracle = int(g.shape[0])
        if g.shape[0] and xy.shape[0]:
            dmin_gt = np.sqrt(((g[:, None, :] - xy[None, :, :]) ** 2).sum(-1)).min(1)
            empty_gt_total[0] += int((dmin_gt > eval_radius).sum())
        else:
            empty_gt_total[0] += int(g.shape[0])
        gt_total_all[0] += int(g.shape[0])

        for name, fn in selectors.items():
            keep = fn(xy, s, d, k_density, k_oracle, g)
            pred = xy[keep] if keep.size else np.zeros((0, 2), np.float32)
            st = full_image_matching_stats(g, pred, eval_radius)
            n_sel = int(pred.shape[0])
            acc[name]["abs_err"].append(abs(n_sel - k_oracle))
            acc[name]["sq_err"].append((n_sel - k_oracle) ** 2)
            acc[name]["gt"] += k_oracle
            acc[name]["sel"] += n_sel
            acc[name]["tp"] += int(st["matched_gt"])
            acc[name]["near"] += int(st["nearby_proposals"])

    rows = {}
    for name, v in acc.items():
        gt = max(v["gt"], 1); sel = max(v["sel"], 1)
        rows[name] = {
            "mae": float(np.mean(v["abs_err"])),
            "rmse": float(np.sqrt(np.mean(v["sq_err"]))),
            "recall": float(v["tp"] / gt),
            "precision": float(v["tp"] / sel),
            "dup_per_gt": float(v["near"] / gt),
            "selected_per_gt": float(v["sel"] / gt),
            "total_selected": int(v["sel"]),
        }

    print("\nimages=%d  gt_total=%d  eval_radius=%.1f  thr=%.2f  nms_r=%.1f  transfer_radii=%s bias_sd=%s"
          % (len(ids), acc[list(acc)[0]]["gt"], eval_radius, thr, nms_radius, tr, bs))
    hdr = "%-44s %8s %8s %8s %8s %8s %8s" % ("selector", "MAE", "RMSE", "recall", "prec", "dup/gt", "sel/gt")
    print(hdr); print("-" * len(hdr))
    for name in sorted(rows, key=lambda n: rows[n]["mae"]):
        r = rows[name]
        print("%-44s %8.2f %8.2f %8.4f %8.4f %8.4f %8.4f"
              % (name, r["mae"], r["rmse"], r["recall"], r["precision"], r["dup_per_gt"], r["selected_per_gt"]))

    empty_rate = float(empty_gt_total[0] / max(gt_total_all[0], 1))
    decomposition = None
    if sel_cfg["thr_sweep"]:
        decomposition = {}
        print("")
        print("Recall loss decomposition (empty-group rate = %.4f, threshold independent)"
              % empty_rate)
        hdr2 = "%6s %10s %10s | %10s %10s %10s" % (
            "thr", "A_recall", "I_recall", "coverage", "magnitude", "dispersion")
        print(hdr2); print("-" * len(hdr2))
        for t in sel_cfg["thr_sweep"]:
            ar = rows["A_thr%.2f" % t]["recall"]
            ir = rows["I_conc_thr%.2f" % t]["recall"]
            cov = empty_rate
            mag = max(0.0, (1.0 - empty_rate) - ir)
            dis = max(0.0, ir - ar)
            decomposition["%.2f" % t] = {"A_recall": ar, "I_recall": ir,
                                         "loss_coverage": cov, "loss_magnitude": mag,
                                         "loss_dispersion": dis,
                                         "A_mae": rows["A_thr%.2f" % t]["mae"],
                                         "I_mae": rows["I_conc_thr%.2f" % t]["mae"],
                                         "A_precision": rows["A_thr%.2f" % t]["precision"]}
            print("%6.2f %10.4f %10.4f | %10.4f %10.4f %10.4f" % (t, ar, ir, cov, mag, dis))

    out = {"config": os.path.abspath(a.config), "ckpt_path": a.ckpt_path,
           "empty_group_rate": empty_rate, "decomposition": decomposition,
           "eval_radius_px": eval_radius, "thr": thr, "nms_radius": nms_radius,
           "transfer_radii": tr, "bias_sigmas": bs,
           "images": len(ids), "results": rows}
    text = json.dumps(out, indent=2, sort_keys=True)
    if a.output_json:
        dd = os.path.dirname(os.path.abspath(a.output_json))
        if dd:
            os.makedirs(dd, exist_ok=True)
        open(a.output_json, "w", encoding="utf-8").write(text)
        print("\nwrote", a.output_json)


if __name__ == "__main__":
    main()
