"""Can the selector's representation separate near-GT slots from far background?

far background FP (a slot farther than the eval radius from every GT) is the
dominant error: 43% of errors at threshold 0.6, 72.7% at 0.5. Two loss
experiments aimed at it both failed, so the open question is whether the
information needed to make that call is present in the features at all.

For each valid candidate this measures how well a given feature set predicts
"within eval radius of some GT", via:

  * single-feature AUC (rank based, direction-agnostic)
  * a linear probe (logistic regression trained with torch), which upper-bounds
    what a linear head could extract from that representation

Comparing the pro / conf_feat probe against the selector's own score separates
two very different diagnoses:

  probe AUC >> selector AUC   -> information is there, the head or loss wastes it
  probe AUC ~= selector AUC   -> the representation itself lacks the information,
                                 and no amount of loss tuning will fix it

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
from models.pointdiff import (
    sample_selector_prior_features,
    selector_object_logit,
    selector_relative_geometry_features,
)
from models.proposal_prior import build_mixed_x0_prior
from models.train_loop import m11_to_pixels_batch


def auc_score(scores, labels):
    """Rank-based AUC (Mann-Whitney U). Returns None if a class is empty."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel().astype(bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.shape[0], dtype=np.float64)
    ranks[order] = np.arange(1, s.shape[0] + 1, dtype=np.float64)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(cnt.shape[0], dtype=np.float64)
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def linear_probe(x_tr, y_tr, x_te, y_te, epochs=300, lr=0.05, device="cpu"):
    """Balanced logistic regression. Returns train/test AUC."""
    mu = x_tr.mean(dim=0, keepdim=True)
    sd = x_tr.std(dim=0, keepdim=True).clamp_min(1e-6)
    xtr = ((x_tr - mu) / sd).to(device)
    xte = ((x_te - mu) / sd).to(device)
    ytr = y_tr.to(device).float()

    model = torch.nn.Linear(xtr.size(1), 1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pos_w = ((ytr.numel() - ytr.sum()) / ytr.sum().clamp_min(1.0)).clamp(0.05, 20.0)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    for _ in range(int(epochs)):
        opt.zero_grad()
        loss = lossf(model(xtr).squeeze(-1), ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        str_ = model(xtr).squeeze(-1).cpu().numpy()
        ste = model(xte).squeeze(-1).cpu().numpy()
    return auc_score(str_, y_tr.numpy()), auc_score(ste, y_te.numpy())


def ddim_step(p_t, eps_pred, abar_t, abar_prev, eta):
    eps = 1e-12
    sqrt_ab_t = abar_t.clamp(0.0, 1.0).add(eps).sqrt()
    sqrt_om_t = (1.0 - abar_t).clamp_min(0.0).sqrt()
    x0_hat = (p_t - sqrt_om_t * eps_pred) / sqrt_ab_t
    alpha_t = (abar_t / (abar_prev + eps)).clamp_min(eps)
    sigma2 = (eta ** 2) * (((1.0 - abar_prev).clamp_min(0.0) / (1.0 - abar_t).clamp_min(eps))
                           * (1.0 - alpha_t).clamp_min(0.0))
    sigma = sigma2.clamp_min(0.0).sqrt()
    c_eps = (1.0 - abar_prev - sigma2).clamp_min(0.0).sqrt()
    out = (abar_prev + eps).sqrt() * x0_hat + c_eps * eps_pred
    if eta > 0.0:
        out = out + sigma * torch.randn_like(out)
    return out


class FeatProbe:
    """Grab conf_feat (post-fusion, post-interaction) from the competition module."""

    def __init__(self, module):
        self.module = module
        self._orig = None
        self.enabled = False
        self.last = None

    def __enter__(self):
        if self.module is None:
            return self
        self._orig = self.module.forward

        def patched(logits, feat, points_m11, valid_mask):
            if self.enabled:
                self.last = feat.detach()
            return self._orig(logits, feat, points_m11, valid_mask)

        self.module.forward = patched
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.module is not None and self._orig is not None:
            try:
                del self.module.forward
            except AttributeError:
                self.module.forward = self._orig
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Default: run the whole loader.")
    parser.add_argument("--folds", type=int, default=5,
                        help="k-fold cross validation, folds drawn over images.")
    parser.add_argument("--split_seed", type=int, default=1234)
    parser.add_argument("--eval_radius", type=float, default=None)
    parser.add_argument("--probe_epochs", type=int, default=300)
    parser.add_argument("--output_json", default=None)
    a = parser.parse_args()

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    cfg["ckpt_path"] = a.ckpt_path
    cfg["resume_training"] = False
    cfg["batch_size"] = int(a.batch_size)
    cfg["val_num_workers"] = int(a.num_workers)
    args = SimpleNamespace(**cfg)
    eval_radius = float(a.eval_radius if a.eval_radius is not None
                        else getattr(args, "val_ddim_cover_radius", 6.0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    _, val_data = build_dataset(args)
    lk = {"batch_size": int(args.batch_size), "shuffle": False,
          "num_workers": int(a.num_workers), "pin_memory": True,
          "collate_fn": collate_points_padded}
    if int(a.num_workers) > 0:
        lk["persistent_workers"] = True
        lk["prefetch_factor"] = 2
    val_loader = DataLoader(val_data, **lk)

    model = build_model(args, training=True).to(device)
    configure_trainable_params(model, args)
    ck = torch.load(args.ckpt_path, map_location=device)
    sd = ck["model_state"] if isinstance(ck, dict) and "model_state" in ck else ck
    inc = load_model_state(model, sd, shape_compatible_only=bool(
        getattr(args, "load_shape_compatible_only", False)))
    if inc.missing_keys or inc.unexpected_keys:
        print(f"[WARN] non-strict load | missing={inc.missing_keys} unexpected={inc.unexpected_keys}")
    model.eval()

    sched, _ = Diffusion_schedule(args.diffusion_T, device=device, signal_scale=args.signal_scale)
    abar_all = sched.abar.to(device=device)
    T = int(args.diffusion_T)
    steps = int(args.ddim_steps)
    eta = float(getattr(args, "ddim_eta", 0.0))
    start_t = (max(0, min(int(getattr(args, "proposal_prior_start_t", T - 1)), T - 1))
               if bool(getattr(args, "use_proposal_prior", False)) else T - 1)
    t_seq = torch.unique_consecutive(torch.linspace(start_t, 0, steps, device=device).round().long())
    clamp_eps = 1e-3

    bank = {"pro": [], "conf": [], "relgeom": [], "prior": [], "score": [],
            "label": [], "img": [], "dist": []}

    with torch.no_grad(), FeatProbe(getattr(model, "selector_local_competition", None)) as fp:
        for bi, (images, points_pad, mask, metas) in enumerate(
                tqdm(val_loader, desc="near/far separability")):
            if a.max_batches is not None and bi >= int(a.max_batches):
                break
            images = images.to(device, non_blocking=True)
            points_pad = points_pad.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            B, _, H, W = images.shape
            N = points_pad.size(1)

            feats = model.encode(images)
            cache = model.cond.precompute(*feats)
            prior_maps = None
            if bool(getattr(args, "use_proposal_prior", False)):
                occ, dens = model.predict_proposal_prior(feats)
                prior_maps = (occ, dens)

            # Keep RNG aligned with validate_one_epoch.
            fp.enabled = False
            p0 = pixels_to_m11(points_pad, H, W)
            tf = torch.full((B, 1), 500, device=device, dtype=torch.long)
            pts_s, _, ab_s = forward_noisy(p0, tf, sched)
            if ab_s.dim() in (1, 2):
                ab_s = ab_s.view(B, 1, 1)
            model.denoise(feats, pts_s, tf, abar_t=ab_s.to(device), clamp_eps=1e-6,
                          cond_cache=cache, need_exist=True, selector_prior_maps=prior_maps)

            if prior_maps is not None:
                x0_prior, _ = build_mixed_x0_prior(
                    occ, dens, num_slots=N, clamp_eps=clamp_eps,
                    mode=str(getattr(args, "proposal_prior_mode", "occupancy")),
                    density_cell_capacity=int(getattr(args, "proposal_prior_cell_capacity", 2)))
                t_init = torch.full((B, 1), int(t_seq[0].item()), device=device, dtype=torch.long)
                p_t, _, _ = forward_noisy(x0_prior, t_init, sched)
            else:
                p_t = torch.empty((B, N, 2), device=device).uniform_(-1 + clamp_eps, 1 - clamp_eps)

            pro_x0 = pts_x0 = valid_x0 = logit_x0 = None
            for si, ti in enumerate(t_seq.tolist()):
                ti = int(ti)
                tt = torch.full((B, 1), ti, device=device, dtype=torch.long)
                abar_t = abar_all[ti].view(1, 1, 1).expand(B, 1, 1)
                need = si == len(t_seq) - 1
                fp.enabled = bool(need)
                eps_hat, lg, pro_out, pts_cls, vmask = model.denoise(
                    feats, p_t, tt, abar_t=abar_t, clamp_eps=1e-6, cond_cache=cache,
                    need_exist=need, selector_prior_maps=prior_maps)
                fp.enabled = False
                if need:
                    pro_x0, pts_x0, valid_x0, logit_x0 = pro_out, pts_cls, vmask, lg
                abar_prev = (abar_all[int(t_seq[si + 1].item())].view(1, 1, 1).expand(B, 1, 1)
                             if si + 1 < len(t_seq) else torch.ones((B, 1, 1), device=device))
                p_t = ddim_step(p_t, eps_hat, abar_t, abar_prev,
                                eta if si + 1 < len(t_seq) else 0.0).clamp(
                    min=-1 + clamp_eps, max=1 - clamp_eps)

            conf_all = fp.last
            score_all = selector_object_logit(logit_x0.float())
            rg_all = selector_relative_geometry_features(
                pts_x0.detach(), valid_x0,
                k=int(getattr(args, "selector_relgeom_k", 8)), H=H, W=W)
            pf_all = sample_selector_prior_features(prior_maps, pts_x0.detach(), valid_x0)
            cand_pix = m11_to_pixels_batch(pts_x0.detach(), H, W)

            for b in range(B):
                gm = mask[b].bool()
                vm = valid_x0[b].bool()
                if not gm.any() or not vm.any():
                    continue
                gt_pix = points_pad[b, gm].float()
                cp = cand_pix[b, vm].float()
                dmin = torch.cdist(cp, gt_pix, p=2).min(dim=1).values
                meta = metas[b] if isinstance(metas, (list, tuple)) else metas

                bank["pro"].append(pro_x0[b, vm].float().cpu())
                if conf_all is not None:
                    bank["conf"].append(conf_all[b, vm].float().cpu())
                bank["relgeom"].append(rg_all[b, vm].float().cpu())
                if pf_all is not None:
                    bank["prior"].append(pf_all[b, vm].float().cpu())
                bank["score"].append(score_all[b, vm].float().cpu())
                bank["label"].append((dmin <= eval_radius).cpu())
                bank["dist"].append(dmin.cpu())
                bank["img"].append(torch.full((int(vm.sum()),), int(meta["img_index"]),
                                              dtype=torch.long))

    cat = {k: (torch.cat(v, dim=0) if v else None) for k, v in bank.items()}
    y = cat["label"]
    img = cat["img"]
    n = int(y.numel())
    n_near = int(y.sum())
    print("\ncollected %d slots | near(<=%.1fpx)=%d (%.1f%%) far=%d"
          % (n, eval_radius, n_near, 100.0 * n_near / max(n, 1), n - n_near))

    # ---- k-fold cross validation, folds drawn over images ----
    # Images are shuffled with a fixed seed rather than split by sorted index, so
    # the folds do not inherit whatever ordering img_index encodes. Every slot of
    # an image lands in the same fold, so no image is ever on both sides.
    uniq = torch.unique(img)
    gsplit = torch.Generator().manual_seed(int(a.split_seed))
    shuffled = uniq[torch.randperm(uniq.numel(), generator=gsplit)]
    k = max(2, int(a.folds))
    fold_imgs = [shuffled[i::k] for i in range(k)]
    print("images=%d  folds=%d  images/fold=%s"
          % (uniq.numel(), k, [int(f.numel()) for f in fold_imgs]))

    rg = cat["relgeom"]
    sets = {"pro (conditioner output)": cat["pro"],
            "relgeom (18d geometry)": rg}
    if cat["conf"] is not None:
        sets["conf_feat (into head)"] = cat["conf"]
    if cat["prior"] is not None:
        sets["prior (occ+density)"] = cat["prior"]
        sets["pro + relgeom + prior"] = torch.cat(
            [cat["pro"], rg, cat["prior"]], dim=-1)

    single_defs = {"selector_score": cat["score"]}
    if cat["prior"] is not None:
        single_defs["prior_occupancy"] = cat["prior"][:, 0]
        single_defs["prior_density"] = cat["prior"][:, 1]
    single_defs["relgeom_centroid_dist"] = rg[:, 2]
    single_defs["relgeom_local_scale"] = rg[:, 15]
    single_defs["relgeom_nn1_dist"] = rg[:, 5]

    per_fold = {name: [] for name in sets}
    per_fold_single = {name: [] for name in single_defs}
    fold_rows = []

    for fi in range(k):
        te_mask = torch.isin(img, fold_imgs[fi])
        tr_mask = ~te_mask
        yt = y[te_mask].numpy()
        if yt.sum() == 0 or (~yt.astype(bool)).sum() == 0:
            print("  fold %d skipped (single class in test)" % fi)
            continue
        row = {"fold": fi,
               "test_images": int(fold_imgs[fi].numel()),
               "train_slots": int(tr_mask.sum()),
               "test_slots": int(te_mask.sum())}
        for name, feat in single_defs.items():
            v = auc_score(feat[te_mask].numpy(), yt)
            per_fold_single[name].append(v)
            row["single_" + name] = v
        for name, X in sets.items():
            _, te_auc = linear_probe(X[tr_mask], y[tr_mask], X[te_mask], y[te_mask],
                                     epochs=int(a.probe_epochs), device=device)
            per_fold[name].append(te_auc)
            row["probe_" + name] = te_auc
        fold_rows.append(row)
        print("  fold %d: test_imgs=%-3d test_slots=%-7d selector=%.4f  best_probe=%.4f"
              % (fi, row["test_images"], row["test_slots"],
                 row["single_selector_score"],
                 max(row["probe_" + n] for n in sets)))

    def agg(vals):
        v = [x for x in vals if x is not None]
        if not v:
            return None
        arr = np.asarray(v, dtype=np.float64)
        return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                "min": float(arr.min()), "max": float(arr.max()), "folds": int(arr.size)}

    probes = {name: {"dim": int(sets[name].size(1)), **agg(per_fold[name])} for name in sets}
    single = {name: agg(per_fold_single[name]) for name in single_defs}

    # Paired per-fold comparison against the selector, which removes fold-to-fold
    # difficulty as a confound.
    sel_folds = np.asarray(per_fold_single["selector_score"], dtype=np.float64)
    paired = {}
    for name in sets:
        pv = np.asarray(per_fold[name], dtype=np.float64)
        d = pv - sel_folds
        paired[name] = {
            "gap_mean": float(d.mean()),
            "gap_std": float(d.std(ddof=1)) if d.size > 1 else 0.0,
            "folds_where_probe_wins": int((d > 0).sum()),
            "folds_total": int(d.size),
        }

    best_name = max(probes, key=lambda nm: probes[nm]["mean"])
    best_probe = probes[best_name]["mean"]
    sel_te = single["selector_score"]["mean"]
    gp = paired[best_name]
    verdict = None
    if best_probe is not None and sel_te is not None:
        consistent = gp["folds_where_probe_wins"] == gp["folds_total"]
        margin_clear = gp["gap_mean"] > 2.0 * max(gp["gap_std"], 1e-9)
        if best_probe < 0.65:
            verdict = ("REPRESENTATION LACKS THE INFORMATION (best probe mean AUC < 0.65) "
                       "-> loss tuning cannot fix this; the conditioner/backbone must change")
        elif gp["gap_mean"] > 0.05 and consistent and margin_clear:
            verdict = ("INFORMATION IS PRESENT BUT UNUSED (probe beats selector in every fold, "
                       "gap > 2 sd) -> the head or the loss is wasting it")
        elif gp["gap_mean"] > 0.05:
            verdict = ("PROBE LEADS ON AVERAGE BUT NOT CONSISTENTLY ACROSS FOLDS "
                       "-> suggestive, needs more images before acting on it")
        else:
            verdict = ("SELECTOR ALREADY EXTRACTS WHAT IS THERE (probe ~= selector) "
                       "-> headroom requires a richer representation")

    result = {
        "config": os.path.abspath(a.config),
        "ckpt_path": a.ckpt_path,
        "eval_radius_px": eval_radius,
        "task": "predict: candidate is within eval_radius of some GT",
        "slots_total": n, "slots_near": n_near, "slots_far": n - n_near,
        "folds": k,
        "split_seed": int(a.split_seed),
        "images": int(uniq.numel()),
        "per_fold": fold_rows,
        "paired_gap_vs_selector": paired,
        "best_probe_name": best_name,
        "single_feature_auc": single,
        "linear_probe_auc": probes,
        "selector_score_test_auc": sel_te,
        "best_probe_test_auc": best_probe,
        "verdict": verdict,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print("\n" + text)
    if a.output_json:
        d = os.path.dirname(os.path.abspath(a.output_json))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(a.output_json, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
