#!/usr/bin/env python3
"""Large-gain analysis T: Generation Scotland (GS) risk separation of MethylGPT combined with one clock, against the clock
alone. Tables only. Analysis L's design and code applied unchanged to analysis Q's cross-fitted held-out risks.

Written on 9 Nov 2026, after the results of J to S and Q were known (Q: MethylGPT added to GrimAge2 +87%, to DunedinPACE
+58% in C-index gain on complete profiles). Q measured discrimination only; T asks how far the same cross-fitted risks
separate people who develop disease, on L's metrics, including Kaplan-Meier curves by risk decile. No model is refitted to
predict disease and no risk is recomputed. Every result is reported, whichever set it favours.

DATA
  Risks: artifacts/large_gain_2026-11-08_private/Q/private_crossfit_<scenario>.npz (Q's cross-fitted linear predictors,
  sets x endpoints x 9,866, endpoint-table order), scenarios full (complete profiles), hm27 and random_90 (simulated
  missingness on empirical profiles), depth_10, depth_5, depth_30, gauss_0.05 (simulated measurement noise on empirical
  profiles). Participants, events, follow-up, age, sex, outer folds and bootstrap positions: L's endpoint arrays
  (artifacts/large_gain_2026-10-30_private/L/private_endpoint_arrays.npz). Bootstrap: L's weights (seed 20261030, 2,000
  draws, one resample shared by every model, endpoint and scenario).

SETS (Q's, mean fill): QG_mean (MethylGPT combined + GrimAge2), QD_mean (MethylGPT combined + DunedinPACE), C1_mean
  (GrimAge2), C2_mean (DunedinPACE), C4_mean, C5_mean (raw CpGs), GV_mean, GG_mean, DV_mean, DG_mean (one clock plus one
  raw-CpG model), S1 (MethylGPT combined alone).

PREPARATION AND METRICS (L's, unchanged): within each outer fold, the risk is residualised on age and sex and standardised
  (L.prepare, including L's constant-risk test); deciles within fold. log hazard ratio per SD (Breslow Cox with age, sex
  and the prepared risk, stratified by fold), top-versus-bottom-decile log hazard ratio, top-decile event capture, 10-year
  absolute risk in the top and bottom deciles (Kaplan-Meier). H = mean over the 8 endpoints of the log hazard ratio per SD.
  Kaplan-Meier curves on L's yearly grid for QG_mean, QD_mean, C1_mean and C2_mean in full, hm27 and depth_10.

DECISION (complete profiles; two pre-specified contrasts, J's rule on H, no multiplicity adjustment)
  T_G: QG_mean against C1_mean.  T_D: QD_mean against C2_mean.
  PASS iff (H_a - H_b) / H_b >= 0.10 and the lower end of the 95% interval of H_a - H_b is > 0; not applicable if H_b <= 0.
SECONDARY (rule arithmetic for reference): the same contrasts in the other six scenarios; QG_mean against the better of
  GV_mean and GG_mean, QD_mean against the better of DV_mean and DG_mean; QG_mean and QD_mean against S1; every L metric per
  endpoint and as the mean over endpoints; BH q across endpoints (log hazard ratio per SD); leave one endpoint out on H.
  Intervals: percentile, L's pct and pboot; preparation held fixed in the draws (L's convention).

DISCLOSURE CONTROL: L's, unchanged (cells below 5 events or people at risk suppressed; KM rows released by L.release_km;
  abs-risk cells of a KM model use its KM release mask; contrasts blank if a cell is suppressed). Private copies keep every
  value; public tables are aggregates only.

CHECKS (abort on failure)
  1. L's endpoint arrays match J.load_preds (events, follow-up, folds, age, sex) for every endpoint.
  2. L.prepare applied to J's stored mgpt2_combined risk on complete profiles reproduces L's stored prepared risk and deciles.
  3. T's fit routine applied to L's stored prepared mgpt2_combined (complete profiles, every endpoint) reproduces L's stored
     point values and draws of every metric within 1e-10 and L's stored KM curves and counts (T's fit is L.task_fit with a
     configurable KM model list).
  4. Every Q linear predictor used is finite; the set list equals Q's.
  5. Privacy: no participant identifier pattern or identifier column in any public file; suppressed cells blank.

STAGES: run (prep, fits, checks, tables in one pass; refuses to run unless this script, L's, O's and Q's scripts are
committed unchanged). Run: python3 large_gain_T_gs_single_clock_risk_stratification_2026-11-09.py [--workers n]
"""
import argparse
import importlib.util
import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
DATE = "2026-11-09"
OUT = ART / "large_gain_2026-11-09/T"
PRIV = ART / "large_gain_2026-11-09_private/T"
QPRIV = ART / "large_gain_2026-11-08_private/Q"
SCRIPTS = {"L": HERE / "large_gain_L_gs_risk_stratification_2026-10-30.py",
           "O": HERE / "large_gain_O_gs_noise_risk_stratification_2026-11-08.py",
           "Q": HERE / "large_gain_Q_gs_single_clock_stacking_2026-11-08.py"}
SCENARIOS = ["full", "hm27", "random_90", "depth_10", "depth_5", "depth_30", "gauss_0.05"]
KM_SCEN = ["full", "hm27", "depth_10"]
SETS = ["QG_mean", "QD_mean", "C1_mean", "C2_mean", "C4_mean", "C5_mean", "GV_mean", "GG_mean", "DV_mean", "DG_mean", "S1"]
KM_SETS = ["QG_mean", "QD_mean", "C1_mean", "C2_mean"]
TOL = 1e-10


def load_module(name, path):
    if not path.exists():
        sys.exit(f"Cannot proceed - required data file not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


O = load_module("large_gain_O", SCRIPTS["O"])
L, J = O.L, O.J
ENDPOINTS = J.ENDPOINTS
N_BOOT, YEARS, YEAR, T10 = L.N_BOOT, L.YEARS, L.YEAR, L.T10


def require_committed():
    for p in [Path(__file__).resolve()] + list(SCRIPTS.values()):
        r = subprocess.run(["git", "-C", str(HERE), "status", "--porcelain", "--", str(p)], capture_output=True, text=True)
        t = subprocess.run(["git", "-C", str(HERE), "ls-files", "--error-unmatch", str(p)], capture_output=True, text=True)
        if r.stdout.strip() or t.returncode != 0:
            sys.exit(f"Cannot proceed - {p.name} must be committed unchanged before this analysis runs")


def fit_endpoint(X, G, models, ev, du, fold, age, sex, W, km_models):
    """L.task_fit's body for one endpoint, with a configurable KM model list."""
    cd = L.CoxData(ev, du, fold)
    nm = len(models)
    out = {k: np.full(nm, np.nan) for k in ("beta_pt", "tb_pt", "cap_pt", "rt_pt", "rb_pt")}
    out.update({k: np.full((nm, N_BOOT), np.nan) for k in ("beta_boot", "tb_boot", "cap_boot", "rt_boot", "rb_boot")})
    counts = np.zeros((nm, len(L.COUNT_NAMES)))
    for k in range(nm):
        x, g = X[k], G[k]
        top, bot, est = g == 2, g == 0, g >= 0
        counts[k] = [top.sum(), bot.sum(), (ev & est).sum(), (ev & top).sum(), (ev & bot).sum(), (ev & top & (du <= T10)).sum(),
                     (ev & bot & (du <= T10)).sum(), (top & (du >= T10)).sum(), (bot & (du >= T10)).sum(),
                     len(np.unique(fold[est])), est.sum()]
        if not est.any():
            continue
        Z = L.design(age, sex, x)
        b0 = L.cox_newton(cd, Z)[0]
        out["beta_pt"][k] = b0[2]
        out["beta_boot"][k] = L.cox_newton(cd, Z, W=W, beta0=b0)[:, 2]
        wt, wb = W[:, ev & top].sum(1), W[:, ev & bot].sum(1)
        if (ev & top).any() and (ev & bot).any():
            Z = L.design(age, sex, top.astype(float), bot.astype(float))
            b0 = L.cox_newton(cd, Z)[0]
            out["tb_pt"][k] = b0[2] - b0[3]
            bb = L.cox_newton(cd, Z, W=W, beta0=b0, valid=(wt > 0) & (wb > 0))
            out["tb_boot"][k] = bb[:, 2] - bb[:, 3]
        out["cap_pt"][k] = (ev & top).sum() / (ev & est).sum()
        out["cap_boot"][k] = wt / W[:, ev & est].sum(1)
        out["rt_pt"][k] = 1 - L.km_at(du[top], ev[top], np.array([T10]))[0, 0]
        out["rb_pt"][k] = 1 - L.km_at(du[bot], ev[bot], np.array([T10]))[0, 0]
        out["rt_boot"][k] = 1 - L.km_at(du[top], ev[top], np.array([T10]), W=W[:, top])[:, 0]
        out["rb_boot"][k] = 1 - L.km_at(du[bot], ev[bot], np.array([T10]), W=W[:, bot])[:, 0]
    if km_models:
        days = YEARS * YEAR
        na = len(km_models)
        Sp, Sb = np.full((na, 3, len(YEARS)), np.nan), np.full((na, 3, N_BOOT, len(YEARS)), np.nan)
        atr, cum, tot, ng = np.zeros((na, 3, len(YEARS))), np.zeros((na, 3, len(YEARS))), np.zeros((na, 3)), np.zeros((na, 3))
        for a, m in enumerate(km_models):
            g = G[models.index(m)]
            for b, gv in enumerate((2, 1, 0)):
                s = g == gv
                if not s.any():
                    continue
                Sp[a, b] = L.km_at(du[s], ev[s], days)[0]
                Sb[a, b] = L.km_at(du[s], ev[s], days, W=W[:, s])
                atr[a, b] = [(du[s] >= d).sum() for d in days]
                cum[a, b] = [(ev[s] & (du[s] <= d)).sum() for d in days]
                tot[a, b], ng[a, b] = ev[s].sum(), s.sum()
        out.update(km_S=Sp, km_boot=Sb, km_atrisk=atr, km_cum=cum, km_total=tot, km_n=ng)
    out["counts"] = counts
    return out


def task(args):
    scen, j = args
    fo = PRIV / f"private_fit_{scen}__{ENDPOINTS[j]}.npz"
    if fo.exists():
        return scen, j
    Tz = np.load(L.PRIV / "private_endpoint_arrays.npz")
    P = np.load(PRIV / f"private_prepared_{scen}.npz")
    W = np.load(L.PRIV / "private_boot_weights.npy", mmap_mode="r")[:, Tz["position"][j]].astype(np.float64)
    out = fit_endpoint(P["X"][:, j], P["G"][:, j], SETS, Tz["event"][j], Tz["duration"][j], Tz["fold"][j], Tz["age"][j],
                       Tz["sex"][j], W, KM_SETS if scen in KM_SCEN else [])
    tmp = fo.with_name(fo.stem + ".tmp.npz")
    np.savez(tmp, models=np.array(SETS), **out)
    tmp.replace(fo)
    return scen, j


def load_fit(scen):
    fits = [np.load(PRIV / f"private_fit_{scen}__{ep}.npz") for ep in ENDPOINTS]
    F = {"models": list(fits[0]["models"].astype(str))}
    for key in ("beta", "tb", "cap", "rt", "rb"):
        F[f"{key}_pt"] = np.stack([f[f"{key}_pt"] for f in fits], 1)
        F[f"{key}_boot"] = np.stack([f[f"{key}_boot"] for f in fits], 1)
    F["counts"] = np.stack([f["counts"] for f in fits], 1)
    if scen in KM_SCEN:
        F["km_models"] = KM_SETS
        for key in ("km_S", "km_boot", "km_atrisk", "km_cum", "km_total", "km_n"):
            F[key] = np.stack([f[key] for f in fits], 0)
    return F


def rule(va, vb, lo, decision):
    if not (np.isfinite(vb) and vb > 0):
        return "not applicable (H_b <= 0)" + ("" if decision else " (reference only)")
    ok = (va - vb) / vb >= 0.10 and lo > 0
    return ("PASS" if ok else "FAIL") if decision else ("meets (reference only)" if ok else "does not meet (reference only)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    require_committed()
    t0 = time.time()
    PRIV.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    checks = []
    chk = lambda name, ok, detail="": checks.append({"check": name, "passed": bool(ok), "detail": str(detail)})
    preds = J.load_preds()
    Tz = np.load(L.need(L.PRIV / "private_endpoint_arrays.npz"))
    for j, ep in enumerate(ENDPOINTS):
        pe = preds[ep]
        ok = (np.array_equal(Tz["event"][j], pe.event.astype(bool).to_numpy())
              and np.array_equal(Tz["duration"][j], pe.duration.to_numpy(float))
              and np.array_equal(Tz["fold"][j], pe.outer_fold.to_numpy(int))
              and np.array_equal(Tz["age"][j], pe.age.to_numpy(float)) and np.array_equal(Tz["sex"][j], pe.sex.to_numpy(float)))
        chk("1 endpoint arrays match J", ok, ep)
    # check 2: L.prepare reproduces L's prepared full mgpt2_combined
    A = J.assemble("full", preds)
    Lp = np.load(L.need(L.PRIV / "private_prepared_full.npz"))
    lm = list(Lp["models"].astype(str))
    for j, ep in enumerate(ENDPOINTS):
        g = A[A.endpoint == ep]
        x, gr, _ = L.prepare(g[J.OURS].to_numpy(float), Tz["age"][j], Tz["sex"][j], Tz["fold"][j])
        k = lm.index(J.OURS)
        chk("2 L.prepare reproduction", np.array_equal(x, Lp["X"][k, j]) and np.array_equal(gr, Lp["G"][k, j]), ep)
    # check 3: T's fit routine reproduces L's stored full fit for mgpt2_combined and L's KM
    Wall = np.load(L.PRIV / "private_boot_weights.npy", mmap_mode="r")
    for j, ep in enumerate(ENDPOINTS[:3]):
        k = lm.index(J.OURS)
        Lf = np.load(L.PRIV / f"private_fit_full__{ep}.npz")
        lfm = list(Lf["models"].astype(str))
        W = np.asarray(Wall[:, Tz["position"][j]], dtype=np.float64)
        km = [J.OURS, L.grimage_ref("full")]
        Xs = np.stack([Lp["X"][lm.index(m), j] for m in km])
        Gs = np.stack([Lp["G"][lm.index(m), j] for m in km])
        o = fit_endpoint(Xs, Gs, km, Tz["event"][j], Tz["duration"][j], Tz["fold"][j], Tz["age"][j], Tz["sex"][j], W, km)
        ok = True
        for key in ("beta", "tb", "cap", "rt", "rb"):
            for suf in ("pt", "boot"):
                ok &= np.allclose(o[f"{key}_{suf}"][0], Lf[f"{key}_{suf}"][lfm.index(J.OURS)], atol=TOL, rtol=0, equal_nan=True)
        ok &= np.allclose(o["km_S"], Lf["km_S"], atol=TOL, rtol=0, equal_nan=True)
        ok &= np.allclose(o["km_boot"], Lf["km_boot"], atol=TOL, rtol=0, equal_nan=True)
        ok &= np.array_equal(o["km_atrisk"], Lf["km_atrisk"]) and np.array_equal(o["km_cum"], Lf["km_cum"])
        ok &= np.array_equal(o["counts"][0], Lf["counts"][lfm.index(J.OURS)])
        chk("3 fit routine reproduces L", ok, ep)
    # preparation of Q's cross-fitted risks
    prows = []
    for scen in SCENARIOS:
        z = np.load(L.need(QPRIV / f"private_crossfit_{scen}.npz"))
        qs = list(z["sets"].astype(str))
        chk("4 Q set list", all(s in qs for s in SETS), scen)
        X = np.zeros((len(SETS), len(ENDPOINTS), L.N))
        G = np.zeros((len(SETS), len(ENDPOINTS), L.N), np.int8)
        for k, s in enumerate(SETS):
            lp = z["LP"][qs.index(s)]
            chk("4 finite predictors", np.isfinite(lp).all(), f"{scen} {s}")
            for j, ep in enumerate(ENDPOINTS):
                x, gr, rec = L.prepare(lp[j], Tz["age"][j], Tz["sex"][j], Tz["fold"][j])
                X[k, j], G[k, j] = x, gr
                prows += [{"scenario": scen, "endpoint": ep, "model": s, **r} for r in rec]
        L.atomic_npz(PRIV / f"private_prepared_{scen}.npz", X=X, G=G, models=np.array(SETS))
    ck = pd.DataFrame(checks)
    if not ck.passed.all():
        L.atomic_csv(OUT / f"T_checks_{DATE}.csv", ck)
        print(ck[~ck.passed].to_string())
        sys.exit("T checks failed before fitting")
    L.atomic_csv(OUT / f"T_preparation_{DATE}.csv", pd.DataFrame(prows))
    tasks = [(s, j) for s in SCENARIOS for j in range(len(ENDPOINTS))]
    with mp.get_context("fork").Pool(a.workers) as pool:
        for s, j in pool.imap_unordered(task, tasks):
            L.log(f"T {s} {ENDPOINTS[j]} done ({time.time() - t0:.0f} s)")
    # tables
    rows, per, loeo, krows, srows, results = [], [], [], [], [], {}
    for scen in SCENARIOS:
        F = load_fit(scen)
        ms = F["models"]
        ix = ms.index
        b, bb = F["beta_pt"], F["beta_boot"]
        allnot = ~np.isfinite(b)
        Hp, Hb = np.where(allnot, 0.0, b).mean(1), np.where(allnot[..., None], 0.0, bb).mean(1)
        for m in ms:
            lo, hi, _ = L.pct(Hb[ix(m)])
            srows.append({"scenario": scen, "model": m, "H": Hp[ix(m)], "H_ci95_low": lo, "H_ci95_high": hi,
                          "n_endpoints_not_estimable": int(allnot[ix(m)].sum())})
        rel = None
        if scen in KM_SCEN:
            rel = np.stack([np.stack([L.release_km(F["km_cum"][j, c], F["km_atrisk"][j, c], F["km_total"][j, c])
                                      for c in range(len(KM_SETS))]) for j in range(len(ENDPOINTS))])
            for c, m in enumerate(KM_SETS):
                for bi, gn in enumerate(L.GROUPS):
                    for y in range(len(YEARS)):
                        for j, ep in enumerate(ENDPOINTS):
                            lo, hi, _ = L.pct(F["km_boot"][j, c, bi, :, y])
                            krows.append({"scenario": scen, "endpoint": ep, "model": m, "group": gn, "year": int(YEARS[y]),
                                          "n_group": F["km_n"][j, c, bi], "survival": F["km_S"][j, c, bi, y],
                                          "survival_ci95_low": lo, "survival_ci95_high": hi,
                                          "n_at_risk": F["km_atrisk"][j, c, bi, y], "cum_events": F["km_cum"][j, c, bi, y],
                                          "suppressed": not bool(rel[j, c, bi, y])})
        sup, _ = O.suppression(F, scen, rel, KM_SETS if rel is not None else None)
        vals = O.metric_values(F)
        best = lambda c: max(c, key=lambda m: Hp[ix(m)])
        con = [("T_G: MethylGPT added to GrimAge2", "QG_mean", "C1_mean", True),
               ("T_D: MethylGPT added to DunedinPACE", "QD_mean", "C2_mean", True),
               ("MethylGPT + GrimAge2 vs raw-CpG model + GrimAge2", "QG_mean", best(["GV_mean", "GG_mean"]), False),
               ("MethylGPT + DunedinPACE vs raw-CpG model + DunedinPACE", "QD_mean", best(["DV_mean", "DG_mean"]), False),
               ("GrimAge2 added to MethylGPT", "QG_mean", "S1", False),
               ("DunedinPACE added to MethylGPT", "QD_mean", "S1", False)]
        results[scen] = {}
        for name, a_, b_, dec in con:
            ka, kb = ix(a_), ix(b_)
            for metric in L.METRICS:
                P, B = vals[metric]
                S = sup[metric]
                if metric == "log_hr_per_sd":
                    va, vb, d, nsup = Hp[ka], Hp[kb], Hb[ka] - Hb[kb], 0
                else:
                    va, vb, d = P[ka].mean(), P[kb].mean(), B[ka].mean(0) - B[kb].mean(0)
                    nsup = int((S[ka] | S[kb]).sum())
                lo, hi, nd = L.pct(d)
                isdec = dec and scen == "full" and metric == "log_hr_per_sd"
                row = {"scenario": scen, "contrast": name, "set_a": a_, "set_b": b_, "metric": metric, "decision_row": isdec,
                       "value_a": va, "value_b": vb, "difference": va - vb, "ci95_low": lo, "ci95_high": hi,
                       "p_boot": L.pboot(d), "n_draws_estimable": nd, "relative_gain": (va - vb) / vb if vb > 0 else np.nan,
                       "outcome": rule(va, vb, lo, isdec) if metric == "log_hr_per_sd" else "",
                       "suppressed": nsup > 0}
                rows.append(row)
                if metric == "log_hr_per_sd":
                    results[scen][name] = {k: v for k, v in row.items() if k not in ("scenario", "metric")}
                if dec:
                    pl = [L.pboot(B[ka, j] - B[kb, j]) for j in range(len(ENDPOINTS))]
                    okp = np.isfinite(pl)
                    q = np.full(len(pl), np.nan)
                    if metric == "log_hr_per_sd" and okp.any():
                        q[okp] = O.M.bh(np.asarray(pl)[okp])
                    for j, ep in enumerate(ENDPOINTS):
                        lo, hi, nd = L.pct(B[ka, j] - B[kb, j])
                        sp = bool(S[ka, j] or S[kb, j])
                        per.append({"scenario": scen, "contrast": name, "endpoint": ep, "metric": metric,
                                    "value_a": P[ka, j], "value_b": P[kb, j], "difference": P[ka, j] - P[kb, j],
                                    "ci95_low": lo, "ci95_high": hi, "p_boot": pl[j], "q_bh_across_endpoints": q[j],
                                    "n_draws_estimable": nd, "suppressed": sp})
            if dec:
                for j, ep in enumerate(ENDPOINTS):
                    keep = np.arange(len(ENDPOINTS)) != j
                    bl = np.where(allnot, 0.0, b)[:, keep].mean(1)
                    bbl = np.where(allnot[..., None], 0.0, bb)[:, keep].mean(1)
                    d = bbl[ka] - bbl[kb]
                    lo, hi, _ = L.pct(d)
                    loeo.append({"scenario": scen, "contrast": name, "left_out": ep, "H_a": bl[ka], "H_b": bl[kb],
                                 "difference": bl[ka] - bl[kb], "ci95_low": lo, "ci95_high": hi,
                                 "relative_gain": (bl[ka] - bl[kb]) / bl[kb] if bl[kb] > 0 else np.nan,
                                 "rule_arithmetic": rule(bl[ka], bl[kb], lo, False)})
    cdf, pdf, kdf = pd.DataFrame(rows), pd.DataFrame(per), pd.DataFrame(krows)
    L.atomic_csv(PRIV / "private_contrasts_unsuppressed.csv", cdf)
    L.atomic_csv(PRIV / "private_endpoint_contrasts_unsuppressed.csv", pdf)
    L.atomic_csv(PRIV / "private_km_unsuppressed.csv", kdf)
    blank = ["value_a", "value_b", "difference", "ci95_low", "ci95_high", "p_boot"]
    pc, pp, pk = cdf.copy(), pdf.copy(), kdf.copy()
    pc.loc[pc.suppressed, blank + ["relative_gain"]] = np.nan
    pp.loc[pp.suppressed, blank + ["q_bh_across_endpoints"]] = np.nan
    pk.loc[pk.suppressed, ["survival", "survival_ci95_low", "survival_ci95_high", "n_at_risk", "cum_events"]] = np.nan
    pk.loc[pk.n_group < L.MIN_CELL, "n_group"] = np.nan
    for df in (pc, pp):
        r_ = df.metric.str.startswith("abs_risk")
        df.loc[r_, ["value_a", "value_b", "difference", "ci95_low", "ci95_high"]] = \
            df.loc[r_, ["value_a", "value_b", "difference", "ci95_low", "ci95_high"]].round(4)
        df["hr_ratio"] = np.where(df.metric.str.startswith("log_hr"), np.exp(df.difference.to_numpy(float)), np.nan)
    pk[["survival", "survival_ci95_low", "survival_ci95_high"]] = pk[["survival", "survival_ci95_low", "survival_ci95_high"]].round(4)
    public = {f"T_summary_{DATE}.csv": pd.DataFrame(srows), f"T_contrasts_{DATE}.csv": pc,
              f"T_endpoint_contrasts_{DATE}.csv": pp, f"T_leave_one_endpoint_out_{DATE}.csv": pd.DataFrame(loeo),
              f"T_km_curves_{DATE}.csv": pk}
    for fn, df in public.items():
        ok = not any(c.lower() in L.ID_COLUMNS for c in df.columns) and L.ID_PATTERN.search(df.astype(str).to_csv(index=False)) is None
        chk("5 privacy", ok, fn)
    chk("5 suppressed cells blank", pc.loc[pc.suppressed, "difference"].isna().all() and pk.loc[pk.suppressed, "survival"].isna().all())
    ck = pd.DataFrame(checks)
    L.atomic_csv(OUT / f"T_checks_{DATE}.csv", ck)
    if not ck.passed.all():
        print(ck[~ck.passed].to_string())
        sys.exit("T checks failed; no result table written")
    for fn, df in public.items():
        L.atomic_csv(OUT / fn, df)
    L.write_json(OUT / f"T_results_{DATE}.json", {"written": "9 Nov 2026, after J to S and Q results were known",
                                                  "n_checks": int(len(ck)), "scenarios": results})
    print(pc[pc.metric == "log_hr_per_sd"].drop(columns=["metric"]).round(4).to_string())
    print(f"T done in {time.time() - t0:.0f} s; {len(ck)} checks pass")


if __name__ == "__main__":
    main()
