"""Measure the per-GT group probability mass that group_quota_confidence_loss acts on.

quota loss pulls sum(sigmoid(score)) inside each GT's group toward quota_target
(currently 1.0). Whether that loss has anything to push against depends on where
the mass actually sits today, which no existing metric reports: selected_per_gt
and dup@6 are counts after thresholding, while mass is a continuous pre-threshold
sum. Five candidates at 0.20 and one at 0.90 plus four at 0.10 have similar mass
but completely different recall.

Reports, for both the training branch (random start, what the loss optimises) and
the inference branch (DDIM, what the metrics measure):

  * mass distribution per GT group
  * fraction over / under / already inside the target band
  * mass split by local GT crowding, to test whether a constant quota_target=1.0
    is defensible or whether the target needs to scale with density

Read-only: no checkpoint, config or training state is modified.
"""

import argparse
import json
import os
import sys
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
from models.proposal_prior import build_mixed_x0_prior
from models.train_loop import m11_to_pixels_batch, object_score_from_logits


def describe(v, ps=(5, 25, 50, 75, 95)):
    a = np.asarray(v, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"count": 0}
    out = {"count": int(a.size), "mean": float(a.mean()), "std": float(a.std()),
           "min": float(a.min()), "max": float(a.max())}
    for p in ps:
        out["p%d" % p] = float(np.percentile(a, p))
    return out


@torch.no_grad()
def group_mass(pred_pix, gt_pix, prob, valid, radius, nearest_only=True):
    """Replicate the grouping inside group_quota_confidence_loss.

    Returns (mass, group_size) per GT that has at least one member.
    """
    vi = valid.nonzero(as_tuple=False).squeeze(1)
    if vi.numel() == 0 or gt_pix.size(0) == 0:
        return None, None
    d = torch.cdist(pred_pix[vi], gt_pix, p=2)          # [Nv, G]
    nearest = d.argmin(dim=1)
    p = prob[vi]
    masses, sizes = [], []
    for gi in range(gt_pix.size(0)):
        m = d[:, gi] <= radius
        if nearest_only:
            m = m & (nearest == gi)
        if not m.any():
            masses.append(0.0); sizes.append(0)
            continue
        masses.append(float(p[m].sum().item()))
        sizes.append(int(m.sum().item()))
    return masses, sizes


def ddim_step(p_t, eps, at, ap, eta):
    e = 1e-12
    x0 = (p_t - (1 - at).clamp_min(0).sqrt() * eps) / at.clamp(0, 1).add(e).sqrt()
    al = (at / (ap + e)).clamp_min(e)
    s2 = (eta ** 2) * (((1 - ap).clamp_min(0) / (1 - at).clamp_min(e)) * (1 - al).clamp_min(0))
    return (ap + e).sqrt() * x0 + (1 - ap - s2).clamp_min(0).sqrt() * eps


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--config", required=True)
    ap_.add_argument("--ckpt_path", required=True)
    ap_.add_argument("--batch_size", type=int, default=4)
    ap_.add_argument("--num_workers", type=int, default=2)
    ap_.add_argument("--max_batches", type=int, default=None)
    ap_.add_argument("--group_radius", type=float, default=None,
                     help="Defaults to rand_group_quota_radius.")
    ap_.add_argument("--quota_target", type=float, default=None,
                     help="Defaults to rand_group_quota_target.")
    ap_.add_argument("--crowd_radius", type=float, default=32.0,
                     help="Radius used to bucket GTs by local crowding.")
    ap_.add_argument("--output_json", default=None)
    a = ap_.parse_args()

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    cfg["ckpt_path"] = a.ckpt_path
    cfg["resume_training"] = False
    cfg["batch_size"] = int(a.batch_size)
    cfg["val_num_workers"] = int(a.num_workers)
    args = SimpleNamespace(**cfg)

    radius = float(a.group_radius if a.group_radius is not None
                   else getattr(args, "rand_group_quota_radius", 6.0))
    target = float(a.quota_target if a.quota_target is not None
                   else getattr(args, "rand_group_quota_target", 1.0))
    nearest_only = bool(getattr(args, "rand_group_quota_nearest_gt_only", True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    _, val_data = build_dataset(args)
    lk = {"batch_size": int(args.batch_size), "shuffle": False, "num_workers": int(a.num_workers),
          "pin_memory": True, "collate_fn": collate_points_padded}
    if int(a.num_workers) > 0:
        lk["persistent_workers"] = True; lk["prefetch_factor"] = 2
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
    abar = sched.abar.to(device)
    T = int(args.diffusion_T)
    t_low = max(0, min(int(getattr(args, "rand_cover_t_min", 50)), T - 1))
    t_high = max(t_low, min(int(getattr(args, "rand_cover_t_max", 50)), T - 1))
    start_t = (max(0, min(int(getattr(args, "proposal_prior_start_t", T - 1)), T - 1))
               if bool(getattr(args, "use_proposal_prior", False)) else T - 1)
    t_seq = torch.unique_consecutive(
        torch.linspace(start_t, 0, int(args.ddim_steps), device=device).round().long())
    ce = 1e-3

    bank = {"rand": {"mass": [], "size": []}, "ddim": {"mass": [], "size": []}, "crowd": []}
    n_tiles = 0
    imgs = set()

    with torch.no_grad():
        for bi, (images, pts, mask, metas) in enumerate(tqdm(loader, desc="group mass")):
            if a.max_batches is not None and bi >= int(a.max_batches):
                break
            images = images.to(device, non_blocking=True)
            pts = pts.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            B, _, H, W = images.shape
            N = pts.size(1)
            n_tiles += B
            for b in range(B):
                m = metas[b] if isinstance(metas, (list, tuple)) else metas
                imgs.add(int(m["img_index"]))

            feats = model.encode(images)
            cache = model.cond.precompute(*feats)
            pm = None
            if bool(getattr(args, "use_proposal_prior", False)):
                occ, dens = model.predict_proposal_prior(feats)
                pm = (occ, dens)
            p0 = pixels_to_m11(pts, H, W)

            # ---- training branch: prior -> noise at rand_cover_t -> one denoise ----
            t_rand = torch.randint(low=t_low, high=t_high + 1, size=(B, 1),
                                   device=device, dtype=torch.long)
            if pm is not None:
                x0p, _ = build_mixed_x0_prior(
                    occ, dens, num_slots=N, clamp_eps=ce,
                    mode=str(getattr(args, "proposal_prior_mode", "occupancy")),
                    density_cell_capacity=int(getattr(args, "proposal_prior_cell_capacity", 2)))
                p_rand, _, _ = forward_noisy(x0p, t_rand, sched)
            else:
                p_rand = torch.empty_like(p0).uniform_(-1 + ce, 1 - ce)
            _, r_logit, _, r_x0, r_valid = model.denoise(
                feats, p_rand, t_rand, abar_t=sched.get(t_rand).unsqueeze(-1),
                clamp_eps=1e-6, cond_cache=cache, need_exist=True, selector_prior_maps=pm)
            r_prob = torch.sigmoid(object_score_from_logits(torch.clamp(r_logit, -30, 30)).float())
            r_pix = m11_to_pixels_batch(r_x0.detach(), H, W)

            # ---- inference branch: full DDIM ----
            if pm is not None:
                x0p2, _ = build_mixed_x0_prior(
                    occ, dens, num_slots=N, clamp_eps=ce,
                    mode=str(getattr(args, "proposal_prior_mode", "occupancy")),
                    density_cell_capacity=int(getattr(args, "proposal_prior_cell_capacity", 2)))
                t0 = torch.full((B, 1), int(t_seq[0].item()), device=device, dtype=torch.long)
                p_t, _, _ = forward_noisy(x0p2, t0, sched)
            else:
                p_t = torch.empty((B, N, 2), device=device).uniform_(-1 + ce, 1 - ce)
            d_logit = d_x0 = d_valid = None
            for si, ti in enumerate(t_seq.tolist()):
                tt = torch.full((B, 1), int(ti), device=device, dtype=torch.long)
                at = abar[int(ti)].view(1, 1, 1).expand(B, 1, 1)
                need = si == len(t_seq) - 1
                eh, lg, _, pc, vm = model.denoise(feats, p_t, tt, abar_t=at, clamp_eps=1e-6,
                                                 cond_cache=cache, need_exist=need,
                                                 selector_prior_maps=pm)
                if need:
                    d_logit, d_x0, d_valid = lg, pc, vm
                apn = (abar[int(t_seq[si + 1].item())].view(1, 1, 1).expand(B, 1, 1)
                       if si + 1 < len(t_seq) else torch.ones((B, 1, 1), device=device))
                p_t = ddim_step(p_t, eh, at, apn, 0.0).clamp(-1 + ce, 1 - ce)
            d_prob = torch.sigmoid(object_score_from_logits(d_logit.float()))
            d_pix = m11_to_pixels_batch(d_x0.detach(), H, W)

            for b in range(B):
                gm = mask[b].bool()
                if not gm.any():
                    continue
                gp = pts[b, gm].float()
                gg = torch.cdist(gp, gp, p=2)
                crowd = ((gg <= float(a.crowd_radius)).sum(dim=1) - 1).cpu().numpy()

                rm, rs = group_mass(r_pix[b].float(), gp, r_prob[b].float(),
                                    r_valid[b].bool(), radius, nearest_only)
                dm, ds = group_mass(d_pix[b].float(), gp, d_prob[b].float(),
                                    d_valid[b].bool(), radius, nearest_only)
                if rm is None or dm is None:
                    continue
                bank["rand"]["mass"].extend(rm); bank["rand"]["size"].extend(rs)
                bank["ddim"]["mass"].extend(dm); bank["ddim"]["size"].extend(ds)
                bank["crowd"].extend(crowd.tolist())

    crowd = np.asarray(bank["crowd"], dtype=np.float64)
    out = {"config": os.path.abspath(a.config), "ckpt_path": a.ckpt_path,
           "group_radius": radius, "quota_target": target,
           "nearest_gt_only": nearest_only, "crowd_radius": a.crowd_radius,
           "tiles": n_tiles, "images": len(imgs), "gt_groups": int(crowd.size),
           "branches": {}}

    lo, hi = 0.8 * target, 1.2 * target
    for br in ("rand", "ddim"):
        mass = np.asarray(bank[br]["mass"], dtype=np.float64)
        size = np.asarray(bank[br]["size"], dtype=np.float64)
        d = {"mass": describe(mass), "group_size": describe(size),
             "frac_over_target": float((mass > target).mean()),
             "frac_under_target": float((mass < target).mean()),
             "frac_in_band_0.8_1.2": float(((mass >= lo) & (mass <= hi)).mean()),
             "frac_empty_group": float((size == 0).mean()),
             "mean_over_excess": float(np.clip(mass - target, 0, None).mean()),
             "mean_under_deficit": float(np.clip(target - mass, 0, None).mean())}
        buckets = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 10 ** 6)]
        d["by_crowding"] = {}
        for lo_c, hi_c in buckets:
            sel = (crowd >= lo_c) & (crowd <= hi_c)
            if sel.sum() == 0:
                continue
            d["by_crowding"]["gt_neighbours_%d_%s" % (lo_c, "inf" if hi_c > 10 ** 5 else hi_c)] = {
                "groups": int(sel.sum()), "mass_mean": float(mass[sel].mean()),
                "mass_median": float(np.median(mass[sel])),
                "group_size_mean": float(size[sel].mean())}
        out["branches"][br] = d

    print("\nimages=%d tiles=%d gt_groups=%d | group_radius=%.1f quota_target=%.2f nearest_only=%s"
          % (out["images"], out["tiles"], out["gt_groups"], radius, target, nearest_only))
    for br in ("rand", "ddim"):
        d = out["branches"][br]
        mm = d["mass"]
        print("\n[%s branch]  mass mean=%.3f median=%.3f  p5=%.3f p25=%.3f p75=%.3f p95=%.3f"
              % (br, mm["mean"], mm["p50"], mm["p5"], mm["p25"], mm["p75"], mm["p95"]))
        print("   over=%.3f  under=%.3f  in[0.8,1.2]=%.3f  empty_group=%.3f"
              % (d["frac_over_target"], d["frac_under_target"],
                 d["frac_in_band_0.8_1.2"], d["frac_empty_group"]))
        print("   mean excess(over)=%.3f  mean deficit(under)=%.3f  group_size mean=%.2f"
              % (d["mean_over_excess"], d["mean_under_deficit"], d["group_size"]["mean"]))
        print("   by crowding (GT neighbours within %.0fpx):" % a.crowd_radius)
        for k, v in d["by_crowding"].items():
            print("     %-26s groups=%-7d mass_mean=%.3f median=%.3f size=%.2f"
                  % (k, v["groups"], v["mass_mean"], v["mass_median"], v["group_size_mean"]))

    if a.output_json:
        dd = os.path.dirname(os.path.abspath(a.output_json))
        if dd:
            os.makedirs(dd, exist_ok=True)
        open(a.output_json, "w", encoding="utf-8").write(json.dumps(out, indent=2, sort_keys=True))
        print("\nwrote", a.output_json)


if __name__ == "__main__":
    main()
