#!/usr/bin/env python3
"""Agreement of the two elastic-net Cox implementations on a common fit.

For outer fold 1 of two endpoints (death; endocrine and metabolic disease), the stored original-protocol raw-CpG
model (scikit-survival CoxnetSurvivalAnalysis, run to its convergence tolerance; 49,156 CpGs, age and sex
unpenalized) is compared with the GPU proximal-gradient solver (FISTA, fixed step) at the SAME penalty, the SAME
fitting participants and the SAME standardization (taken from the stored model), started from zero and run for
200 (the number used per penalty in the disease analyses), 2,000 and 20,000 iterations.

Reported per fit (float64 evaluation): objective value, largest violation of the elastic-net optimality (KKT)
conditions in units of the l1 threshold alpha * l1, number of nonzero CpG coefficients, largest absolute coefficient
difference from the scikit-survival solution, correlation of held-out risks with those of the scikit-survival
solution, and held-out C-index.

Objective: (1/n) * negative Breslow log partial likelihood + alpha * (0.5 * |w|_1 + 0.25 * |w|^2) on the CpG
coefficients. Development participants only (9,866); aggregate output only:
artifacts/gs_cox/cox_solver_common_fit_2026-10-09.json
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
from sksurv.metrics import concordance_index_censored

from methylgpt_lib import ARTIFACTS, cuda_preinit, log, require, write_json
import gpu_coxnet_2026_10_01 as gc

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("stk", HERE / "gs_stacked_2026-10-04.py")
stk = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(stk)
_spec2 = importlib.util.spec_from_file_location("tun", HERE / "gs_dev_tuning_2026-10-06.py")
tun = importlib.util.module_from_spec(_spec2); _spec2.loader.exec_module(tun)
G = ARTIFACTS / "gs_cox"
V12 = Path("/Entropy/kying/tmp/methylgpt_publication_2026-09-28/results")
RAW_DIRS = [V12 / f"nested_raw_cpg_{p}_2026-09-28" for p in ("recovered_part1", "linear_part2", "recovered_part3", "linear_part4")]
L1, N_UNPEN = 0.5, 2


def evaluate(Xd, order, tie_last, ev, w, alpha):
    """Objective and KKT violation in float64. Xd: (n, p) float64 tensor; w: (p,) float64 tensor."""
    wv = w.clone().requires_grad_(True)
    nll = gc.breslow_nll(Xd @ wv, order, tie_last, ev)
    g = torch.autograd.grad(nll, wv)[0]
    pen = w[N_UNPEN:]
    obj = float(nll) + alpha * (L1 * float(pen.abs().sum()) + (1 - L1) / 2 * float((pen ** 2).sum()))
    s = g[N_UNPEN:] + alpha * (1 - L1) * pen                       # smooth part of the subgradient
    nz = pen != 0
    viol = torch.where(nz, (s + alpha * L1 * torch.sign(pen)).abs(), (s.abs() - alpha * L1).clamp_min(0))
    return {"neg_log_pl": float(nll), "objective": obj,
            "kkt_max_violation_over_l1_threshold": float(max(viol.max(), g[:N_UNPEN].abs().max()) / (alpha * L1)),
            "nonzero_cpgs": int(nz.sum())}


def fista(Xt, order, tie_last, ev, alpha, iters, checkpoints):
    n, p = Xt.shape
    pen = torch.ones(p, device=Xt.device); pen[:N_UNPEN] = 0
    torch.manual_seed(0)
    v = torch.randn(p, device=Xt.device)
    for _ in range(30):
        v = Xt.T @ (Xt @ v); v /= v.norm()
    step = 1.0 / (float((Xt @ v).norm() ** 2 / n) + alpha * (1 - L1))
    w = torch.zeros(p, device=Xt.device); y, tk, out = w.clone(), 1.0, {}
    for it in range(1, iters + 1):
        yv = y.clone().requires_grad_(True)
        g = torch.autograd.grad(gc.breslow_nll(Xt @ yv, order, tie_last, ev), yv)[0] + alpha * (1 - L1) * pen * y
        u = y - step * g
        wn = torch.sign(u) * torch.clamp(u.abs() - step * alpha * L1 * pen, min=0)
        tk1 = (1 + math.sqrt(1 + 4 * tk * tk)) / 2
        y = wn + ((tk - 1) / tk1) * (wn - w)
        w, tk = wn, tk1
        if it in checkpoints:
            out[it] = w.clone()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--endpoints", default="date_of_death,Endocrine_and_Metabolic_Diseases")
    ap.add_argument("--iters", default="200,2000,20000")
    a = ap.parse_args()
    cuda_preinit(); torch.backends.cuda.matmul.allow_tf32 = False
    dev = a.device; t0 = time.time(); its = [int(v) for v in a.iters.split(",")]
    mp = require(next(stk.CACHE.glob(f"COMMON_SAMPLE_SPLIT_{stk.SPLIT}_train_all_gs_batches_all_probes_*.matrix.npz")))
    ids = pd.read_parquet(mp.with_name(mp.name.replace(".matrix.npz", ".labels.parquet")), columns=["dnam_id"]).dnam_id.astype(str).to_numpy()
    with np.load(mp, allow_pickle=False) as z:
        X = z["matrix"].astype(np.float32)
    pos = {s: i for i, s in enumerate(ids)}
    preds = pd.read_parquet(require(tun.PREDS)); preds = preds[preds.age_adjustment == "linear"].copy(); preds["dnam_id"] = preds.dnam_id.astype(str)
    res = {"objective": "(1/n) negative Breslow log partial likelihood + alpha * (0.5 * |w|_1 + 0.25 * |w|^2) on CpG coefficients; age and sex unpenalized",
           "outer_fold": 1, "endpoints": {}}
    for ep in a.endpoints.split(","):
        rdir = [d for d in RAW_DIRS if (d / f"{ep}_linear_outer1_model.npz").exists()][0]
        rm = dict(np.load(require(rdir / f"{ep}_linear_outer1_model.npz"), allow_pickle=False))
        alpha = float(rm["alpha"])
        pe = preds[preds.endpoint == ep].set_index("dnam_id")
        fit_ids = rm["fit_ids"].astype(str)
        held_ids = pe.index[pe.outer_fold.to_numpy() == 1].to_numpy()
        assert not set(fit_ids) & set(held_ids) and set(fit_ids) <= set(pe.index), "stored fitting participants do not match the locked folds"

        def design(sel):
            F = X[[pos[s] for s in sel]].astype(np.float64)
            zf = (np.where(np.isfinite(F), F, rm["mean"]) - rm["mean"]) / rm["scale"]
            c = (pe.loc[sel, ["age", "sex"]].to_numpy(float) - rm["cov_mean"]) / rm["cov_scale"]
            return np.hstack([c, zf])

        Xf, Xh = design(fit_ids), design(held_ids)
        du, ev = pe.loc[fit_ids, "duration"].to_numpy(float), pe.loc[fit_ids, "event"].astype(bool).to_numpy()
        duh, evh = pe.loc[held_ids, "duration"].to_numpy(float), pe.loc[held_ids, "event"].astype(bool).to_numpy()
        order, tie_last, evs = gc.prepare_times(du, ev, dev)
        Xd = torch.as_tensor(Xf, device=dev, dtype=torch.float64)
        w_sk = rm["coefficient"].astype(np.float64)
        r_sk = Xh @ w_sk
        out = {"alpha": alpha, "n_fit": int(len(fit_ids)), "events_fit": int(ev.sum()), "n_heldout": int(len(held_ids)), "fits": {}}
        row = evaluate(Xd, order, tie_last, evs.double(), torch.as_tensor(w_sk, device=dev), alpha)
        row["heldout_c"] = float(concordance_index_censored(evh, duh, r_sk)[0])
        out["fits"]["scikit-survival Coxnet (stored original-protocol model)"] = row
        log(f"{ep} alpha {alpha:.5f} scikit-survival: objective {row['objective']:.6f} KKT {row['kkt_max_violation_over_l1_threshold']:.2e} nonzero {row['nonzero_cpgs']} held-out C {row['heldout_c']:.4f}")
        ws = fista(torch.as_tensor(Xf, device=dev, dtype=torch.float32), order, tie_last, evs, alpha, max(its), set(its))
        for it in its:
            w = ws[it].double()
            row = evaluate(Xd, order, tie_last, evs.double(), w, alpha)
            wn = w.cpu().numpy(); r = Xh @ wn
            row.update({"max_abs_coefficient_difference_from_scikit_survival": float(np.abs(wn - w_sk).max()),
                        "heldout_risk_correlation_with_scikit_survival": float(np.corrcoef(r, r_sk)[0, 1]),
                        "heldout_c": float(concordance_index_censored(evh, duh, r)[0])})
            out["fits"][f"GPU proximal gradient, {it} iterations"] = row
            log(f"{ep} GPU {it} iterations: objective {row['objective']:.6f} KKT {row['kkt_max_violation_over_l1_threshold']:.2e} nonzero {row['nonzero_cpgs']} "
                f"max |dw| {row['max_abs_coefficient_difference_from_scikit_survival']:.2e} risk r {row['heldout_risk_correlation_with_scikit_survival']:.5f} "
                f"held-out C {row['heldout_c']:.4f} ({time.time() - t0:.0f}s)")
        res["endpoints"][ep] = out
        del Xd
        write_json(G / "cox_solver_common_fit_2026-10-09.json", res)
    res["elapsed_s"] = time.time() - t0
    write_json(G / "cox_solver_common_fit_2026-10-09.json", res)


if __name__ == "__main__":
    main()
