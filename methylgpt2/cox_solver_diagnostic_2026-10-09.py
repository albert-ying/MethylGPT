#!/usr/bin/env python3
"""What differs between the two elastic-net Cox implementations?

Development participants only (9,866; outer fold 1 held out; endpoints: death and endocrine/metabolic).
Raw CpGs (49,156) with age and sex unpenalized, standardized with fitting-row statistics, as in the disease analyses.

1. Objective check on a small problem (25 distilled outputs): the GPU proximal-gradient solver run to convergence
   and scikit-survival CoxnetSurvivalAnalysis at the same alpha must give the same coefficients if the objectives
   agree; the alpha scaling between the two is identified empirically (factors 0.5, 1, 2).
2. Raw CpGs: GPU solver with 200 (as used), 2,000 and 20,000 iterations per penalty; for every penalty the
   objective value, number of nonzero CpG coefficients and held-out C-index.
3. Raw CpGs: scikit-survival Coxnet at the penalty the 200-iteration model selected in the disease analysis,
   with objective value, nonzero count, held-out C-index and correlation of held-out risks with each GPU fit.
   (Optional, --skip-sksurv-raw: a cold-start fit at the weakest penalty did not finish in a practical time; the
   comparison with scikit-survival at a common penalty is in cox_solver_common_fit_2026-10-09.py.)
Output: artifacts/gs_cox/cox_solver_diagnostic_2026-10-09.json (aggregate only).
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

from methylgpt2_lib import ARTIFACTS, cuda_preinit, log, require, write_json
import gpu_coxnet_2026_10_01 as gc

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("stk", HERE / "gs_stacked_2026-10-04.py")
stk = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(stk)
_spec2 = importlib.util.spec_from_file_location("tun", HERE / "gs_dev_tuning_2026-10-06.py")
tun = importlib.util.module_from_spec(_spec2); _spec2.loader.exec_module(tun)
G = ARTIFACTS / "gs_cox"
L1 = 0.5


def objective(X, time_, event, w, alpha, n_unpen, dev):
    Xt = torch.as_tensor(X, device=dev, dtype=torch.float32)
    order, tie_last, ev = gc.prepare_times(time_, event, dev)
    wt = torch.as_tensor(w, device=dev, dtype=torch.float32)
    nll = float(gc.breslow_nll(Xt @ wt, order, tie_last, ev))
    pen = wt[n_unpen:]
    return nll, nll + alpha * (L1 * float(pen.abs().sum()) + (1 - L1) / 2 * float((pen ** 2).sum()))


def sk_fit(X, time_, event, alpha, n_unpen):
    pf = np.ones(X.shape[1]); pf[:n_unpen] = 0
    m = CoxnetSurvivalAnalysis(l1_ratio=L1, alphas=[alpha], penalty_factor=pf, max_iter=100000, tol=1e-7, normalize=False, fit_baseline_model=False)
    m.fit(X.astype(np.float64), Surv.from_arrays(event, time_))
    return m.coef_[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--endpoints", default="date_of_death,Endocrine_and_Metabolic_Diseases")
    ap.add_argument("--iters", default="200,2000,20000")
    ap.add_argument("--skip-sksurv-raw", action="store_true", help="skip part 3 (hours per fit at the weakest penalty); the common-fit script covers it")
    a = ap.parse_args()
    cuda_preinit(); torch.backends.cuda.matmul.allow_tf32 = False
    dev = a.device; t0 = time.time()
    ids, B = tun.npz(tun.FEAT1, "train", "mt_bio")
    mp = require(next(stk.CACHE.glob(f"COMMON_SAMPLE_SPLIT_{stk.SPLIT}_train_all_gs_batches_all_probes_*.matrix.npz")))
    with np.load(mp, allow_pickle=False) as z:
        X = z["matrix"].astype(np.float32)
    pos = {s: i for i, s in enumerate(ids)}
    preds = pd.read_parquet(require(tun.PREDS)); preds = preds[preds.age_adjustment == "linear"].copy(); preds["dnam_id"] = preds.dnam_id.astype(str)
    res = {"objective": "(1/n) negative Breslow log partial likelihood + alpha * (0.5 * |w|_1 + 0.25 * |w|^2) on penalized coefficients", "endpoints": {}}
    for ep in a.endpoints.split(","):
        pe = preds[preds.endpoint == ep].reset_index(drop=True)
        idx = np.array([pos[s] for s in pe.dnam_id])
        ev, du = pe.event.astype(bool).to_numpy(), pe.duration.to_numpy(float)
        cov = pe[["age", "sex"]].to_numpy(np.float32)
        fit, held = np.flatnonzero(pe.outer_fold.to_numpy() != 1), np.flatnonzero(pe.outer_fold.to_numpy() == 1)
        cm, cs = stk.fit_scaler(cov[fit], dev)

        def design(F):
            fm, fs = stk.fit_scaler(F[idx[fit]], dev)
            return (np.hstack([stk.apply_scaler(cov[fit], cm, cs, dev), stk.apply_scaler(F[idx[fit]], fm, fs, dev)]),
                    np.hstack([stk.apply_scaler(cov[held], cm, cs, dev), stk.apply_scaler(F[idx[held]], fm, fs, dev)]))

        out = {}
        # ---- 1. objective / alpha-scale check on the 25 distilled outputs
        Xf, Xh = design(B)
        torch.manual_seed(0)
        alphas, path = gc.fit_path(Xf, du[fit], ev[fit], 2, dev=dev, iters=20000)
        k = 6
        w_gpu = path[k].cpu().numpy().astype(float)
        small = {"alpha": float(alphas[k]), "gpu_converged_objective": objective(Xf, du[fit], ev[fit], w_gpu, alphas[k], 2, dev)[1], "sksurv": {}}
        for fac in (0.5, 1.0, 2.0):
            w_sk = sk_fit(Xf, du[fit], ev[fit], float(alphas[k]) * fac, 2)
            small["sksurv"][str(fac)] = {"max_abs_coef_diff": float(np.abs(w_sk - w_gpu).max()),
                                         "objective_at_gpu_alpha": objective(Xf, du[fit], ev[fit], w_sk, alphas[k], 2, dev)[1]}
        out["distilled_outputs_objective_check"] = small
        log(f"{ep} small problem: " + ", ".join(f"alpha x{f}: max |dw| {v['max_abs_coef_diff']:.2e}" for f, v in small["sksurv"].items()))
        fac_best = float(min(small["sksurv"], key=lambda f: small["sksurv"][f]["max_abs_coef_diff"]))

        # ---- 2. raw CpGs: iterations per penalty
        Xf, Xh = design(X)
        gm = dict(np.load(require(G / f"factor_residual_2026-10-01/models/raw_cpg_gpu_{ep}_outer1.npz"), allow_pickle=False))
        k_sel = int(gm["alpha_index"])
        raw = {"selected_alpha_index_in_disease_analysis": k_sel, "by_iterations": {}}
        risks = {}
        for it in [int(v) for v in a.iters.split(",")]:
            torch.manual_seed(0)
            alphas, path = gc.fit_path(Xf, du[fit], ev[fit], 2, dev=dev, iters=it)
            rows = []
            for j, w in enumerate(path):
                wn = w.cpu().numpy().astype(float)
                nll, obj = objective(Xf, du[fit], ev[fit], wn, alphas[j], 2, dev)
                rows.append({"alpha_index": j, "alpha": float(alphas[j]), "neg_log_pl": nll, "objective": obj, "nonzero_cpgs": int((wn[2:] != 0).sum()),
                             "heldout_c": float(concordance_index_censored(ev[held], du[held], Xh @ wn)[0])})
            raw["by_iterations"][str(it)] = rows
            risks[it] = Xh @ path[k_sel].cpu().numpy().astype(float)
            r = rows[k_sel]
            log(f"{ep} raw CpGs, {it} iterations, alpha index {k_sel}: objective {r['objective']:.6f} nonzero {r['nonzero_cpgs']} held-out C {r['heldout_c']:.4f} ({time.time() - t0:.0f}s)")
        out["raw_cpgs"] = raw
        res["endpoints"][ep] = out
        write_json(G / "cox_solver_diagnostic_2026-10-09.json", res)
        if a.skip_sksurv_raw:
            continue
        # ---- 3. scikit-survival Coxnet at the same penalty
        t1 = time.time()
        w_sk = sk_fit(Xf, du[fit], ev[fit], float(alphas[k_sel]) * fac_best, 2)
        nll, obj = objective(Xf, du[fit], ev[fit], w_sk, alphas[k_sel], 2, dev)
        r_sk = Xh @ w_sk
        raw["sksurv_at_selected_alpha"] = {"alpha_scale_factor": fac_best, "neg_log_pl": nll, "objective": obj, "nonzero_cpgs": int((w_sk[2:] != 0).sum()),
                                           "heldout_c": float(concordance_index_censored(ev[held], du[held], r_sk)[0]), "fit_seconds": time.time() - t1,
                                           "heldout_risk_correlation_with_gpu": {str(it): float(np.corrcoef(r_sk, risks[it])[0, 1]) for it in risks}}
        s = raw["sksurv_at_selected_alpha"]
        log(f"{ep} raw CpGs, scikit-survival: objective {s['objective']:.6f} nonzero {s['nonzero_cpgs']} held-out C {s['heldout_c']:.4f}; risk r with GPU {s['heldout_risk_correlation_with_gpu']}")
        out["raw_cpgs"] = raw
        res["endpoints"][ep] = out
        write_json(G / "cox_solver_diagnostic_2026-10-09.json", res)
    res["elapsed_s"] = time.time() - t0
    write_json(G / "cox_solver_diagnostic_2026-10-09.json", res)


if __name__ == "__main__":
    main()
