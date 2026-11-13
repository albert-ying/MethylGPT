#!/usr/bin/env python3
"""Large-gain analysis Q: Generation Scotland (GS) incident disease, MethylGPT combined with ONE clock at a time. Tables only.
Analysis P's design and code (cross-fitted ridge Cox on stored held-out risks), with every set holding at most one clock.

Written on 8 Nov 2026, after the results of J to S were known, to compare MethylGPT
with one clock at a time and never with the model that combines GrimAge2 and DunedinPACE. P's sets all contain both clocks,
so P cannot answer the single-clock question. No set of Q has been fitted before this commit; the sets C1, C2, C4, C5, S1 and
S0 are identical to P's and must reproduce P exactly (checks). Every result is reported, whichever set it favours.

DATA AND METHOD (P's, unchanged; P's functions are imported from the committed P script and called unchanged)
  9,866 GS development participants, 8 endpoints, J's locked outer folds; stored held-out risks of J (full, hm27,
  random_90) and M (depth_10, depth_5, depth_30, gauss_0.05; simulated measurement noise on empirical profiles; hm27 and
  random_90 are simulated missingness on empirical profiles). Within each endpoint and outer fold, the held-out participants
  are split into P's 10 inner folds (stratified by event, default_rng([20261108, endpoint, fold, flag])); a Breslow Cox model
  with age (standardised on the training inner folds) and sex unpenalised and the fold-standardised risks with a fixed ridge
  0.5 x sum of squared risk coefficients is fitted on 9 inner folds and applied to the 10th. Columns failing L's constant-
  risk test in a fold are dropped from that fold's fits (P's rule). No disease model is refitted; no risk is recomputed.

SETS (F = mean fill; in every scenario except full also F = knn)
  QG_F = MethylGPT combined + GrimAge2_F      QD_F = MethylGPT combined + DunedinPACE_F
  C1_F = GrimAge2_F    C2_F = DunedinPACE_F    C4_F = raw_cpg_v12_F    C5_F = raw_cpg_gpu_F            (as in P)
  GV_F = GrimAge2_F + raw_cpg_v12_F    GG_F = GrimAge2_F + raw_cpg_gpu_F
  DV_F = DunedinPACE_F + raw_cpg_v12_F    DG_F = DunedinPACE_F + raw_cpg_gpu_F
  S1 = MethylGPT combined alone (recalibrated)    S0 = age and sex only (as in P)
  Stored unstacked risks and J's age_sex are scored alongside (P's stored_for), as references.

METRIC: P's M = mean over the 8 endpoints of (pooled within-fold C-index of the cross-fitted linear predictor - J's stored
  age+sex C-index), J.fold_counts with tied tolerance 1e-8.

DECISION (complete profiles, scenario full; two pre-specified contrasts, each judged by J's rule, no multiplicity adjustment)
  D_G: QG_mean against C1_mean (MethylGPT added to GrimAge2).    D_D: QD_mean against C2_mean (added to DunedinPACE).
  PASS iff (M_a - M_b) / M_b >= 0.10 and the lower end of the 95% interval of M_a - M_b is > 0; not applicable if M_b <= 0.
  Every other scenario gives the same two contrasts as reference rows (rule arithmetic only).

SECONDARY CONTRASTS (every scenario, rule arithmetic for reference only)
  QG_mean against the better of GV_mean and GG_mean, and QD_mean against the better of DV_mean and DG_mean (a raw-CpG model
  added to the same clock instead of MethylGPT); QG_mean and QD_mean against the best single-risk set (C1, C2, C4, C5, either
  fill); QG_mean and QD_mean against S1 (the clock added to MethylGPT); QG_knn against C1_knn and QD_knn against C2_knn.
  For D_G and D_D in every scenario: per-endpoint differences with BH q across endpoints, leave one endpoint out, and the
  event-weighted mean.

INTERVALS: P's shared participant bootstrap (2,000 draws, seed 20261108; one resample per draw shared by every set,
  endpoint and scenario); cross-fitted risks held fixed; P's within-fold resample as the sensitivity interval. Percentile
  95% intervals; bootstrap P as J.

CHECKS (abort on failure)
  1. Q's inner folds equal P's stored inner folds exactly.
  2. Sets C1_F, C2_F, C4_F, C5_F, S1 and S0 reproduce P's stored cross-fitted linear predictors within 1e-12.
  3. Every fit converges and every linear predictor is finite.
  4. The shared bootstrap weights equal P's (SHA-256).
  5. Point and bootstrap C-indices of the sets shared with P reproduce P's stored arrays within 1e-12.
  6. No public file holds a participant identifier pattern or identifier column.

AMENDMENT 1 (8 Nov 2026, execution only, before any Q result was read): stage prep now creates the public output folder
before writing Q_preparation; the first prep run completed its cross-fitting and stopped at that write. No design change.

STAGES: prep (cross-fitting, checks 1 to 3), boot (checks 4 and 5), analyse (tables, check 6). Each refuses to run unless
this script and P's script are committed unchanged.
Run: python3 large_gain_Q_gs_single_clock_stacking_2026-11-08.py --stage prep|boot|analyse [--workers n]
"""
import argparse
import importlib.util
import json
import multiprocessing as mp
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
DATE = "2026-11-08"
OUTQ = ART / "large_gain_2026-11-08/Q"
PRIVQ = ART / "large_gain_2026-11-08_private/Q"
P_SCRIPT = HERE / "large_gain_P_gs_stacked_risks_2026-11-08.py"
TOL = 1e-12
ID_PATTERN = re.compile(r"\b\d{12}_R\d{2}C\d{2}\b")
ID_COLUMNS = {"dnam_id", "sample_id", "participant_id", "id", "sample", "participant", "barcode"}


def load_module(name, path):
    if not path.exists():
        sys.exit(f"Cannot proceed - required data file not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


P = load_module("large_gain_P", P_SCRIPT)
J, MM = P.J, P.MM
PRIVP = P.PRIV
ENDPOINTS, OURS, N_BOOT = P.ENDPOINTS, P.OURS, P.N_BOOT
SINGLE = {"C1": ["grimage2"], "C2": ["dunedinpace"], "C4": ["raw_cpg_v12"], "C5": ["raw_cpg_gpu"]}
PAIRS = {"GV": ["grimage2", "raw_cpg_v12"], "GG": ["grimage2", "raw_cpg_gpu"],
         "DV": ["dunedinpace", "raw_cpg_v12"], "DG": ["dunedinpace", "raw_cpg_gpu"]}
SHARED_WITH_P = lambda scen: [f"{c}_{F}" for F in P.fills(scen) for c in SINGLE] + ["S1", "S0"]


def q_sets_for(scen):
    s = {}
    for F in P.fills(scen):
        s[f"QG_{F}"] = ([OURS, f"grimage2_{F}"], "decision model (GrimAge2)" if F == "mean" else "secondary set (kNN fill)")
        s[f"QD_{F}"] = ([OURS, f"dunedinpace_{F}"], "decision model (DunedinPACE)" if F == "mean" else "secondary set (kNN fill)")
        for c, mem in SINGLE.items():
            s[f"{c}_{F}"] = ([f"{b}_{F}" for b in mem], "single-risk set")
        for c, mem in PAIRS.items():
            s[f"{c}_{F}"] = ([f"{b}_{F}" for b in mem], "clock plus raw-CpG model")
    s["S1"] = ([OURS], "MethylGPT combined alone (recalibrated)")
    s["S0"] = ([], "cross-fitted age and sex only")
    return s


# P's functions read these module globals at call time
P.sets_for = q_sets_for
P.PRIV = PRIVQ
P.OUTP = OUTQ
P.done_path = lambda stage: P.LOGS / f"large_gain_Q_{stage}_{DATE}.done"


def require_committed():
    for p in (Path(__file__).resolve(), P_SCRIPT):
        r = subprocess.run(["git", "-C", str(HERE), "status", "--porcelain", "--", str(p)], capture_output=True, text=True)
        t = subprocess.run(["git", "-C", str(HERE), "ls-files", "--error-unmatch", str(p)], capture_output=True, text=True)
        if r.stdout.strip() or t.returncode != 0:
            sys.exit(f"Cannot proceed - {p.name} must be committed unchanged before any result-producing stage runs")
    P.require_committed()


def write_checks(rows, stage):
    df = pd.DataFrame(rows)
    df.insert(0, "stage", stage)
    OUTQ.mkdir(parents=True, exist_ok=True)
    fn = OUTQ / f"Q_checks_{DATE}.csv"
    old = pd.read_csv(fn) if fn.exists() else pd.DataFrame()
    if len(old):
        old = old[old.stage != stage]
    P.atomic_csv(fn, pd.concat([old, df], ignore_index=True))
    if not df.passed.all():
        print(df[~df.passed].to_string())
        sys.exit(f"Q stage {stage}: checks failed")


# ------------------------------------------------------------------ stage prep
def stage_prep(workers):
    require_committed()
    t0 = time.time()
    PRIVQ.mkdir(parents=True, exist_ok=True)
    OUTQ.mkdir(parents=True, exist_ok=True)      # amendment 1 (execution only)
    preds = J.load_preds()
    inner, rows = {}, []
    ref = np.load(P.need(PRIVP / "private_inner_folds.npz"))
    for k, ep in enumerate(ENDPOINTS):
        pe = preds[ep]
        ev, fold = pe.event.astype(bool).to_numpy(), pe.outer_fold.to_numpy()
        a = np.full(len(pe), -1, np.int64)
        for f in sorted(np.unique(fold)):
            H = np.flatnonzero(fold == f)
            a[H] = P.inner_assignment(ev[H], k, int(f))
        inner[ep] = a
        rows.append({"check": "1 inner folds equal P", "passed": bool(np.array_equal(a, ref[ep])), "detail": ep})
    P.atomic_npz(PRIVQ / "private_inner_folds.npz", **inner)
    with mp.get_context("fork").Pool(min(workers, len(P.ALL_SCEN))) as pool:
        res = pool.map(P.crossfit_scenario, P.ALL_SCEN)
    prep = pd.DataFrame([r for rr, _ in res for r in rr])
    P.atomic_csv(OUTQ / f"Q_preparation_{DATE}.csv", prep)
    for _, info in res:
        rows.append({"check": "3 fits converged, finite predictors", "passed": info["not_converged"] == 0 and info["finite_lp"],
                     "detail": json.dumps(info)})
    for scen in P.ALL_SCEN:
        q = np.load(PRIVQ / f"private_crossfit_{scen}.npz")
        p = np.load(P.need(PRIVP / f"private_crossfit_{scen}.npz"))
        qs, ps = list(q["sets"].astype(str)), list(p["sets"].astype(str))
        for s in SHARED_WITH_P(scen):
            d = float(np.nanmax(np.abs(q["LP"][qs.index(s)] - p["LP"][ps.index(s)])))
            rows.append({"check": "2 shared sets reproduce P", "passed": d <= TOL, "detail": f"{scen} {s} max abs diff {d:.2e}"})
    write_checks(rows, "prep")
    P.done_marker("prep", {"seconds": round(time.time() - t0)})


# ------------------------------------------------------------------ stage boot
def boot_scenario(scen):
    fo = PRIVQ / "boot" / f"private_boot_{scen}.npz"
    if fo.exists():
        return scen
    t0 = time.time()
    preds = J.load_preds()
    ids = P.load_ids()
    pos = {s: i for i, s in enumerate(ids)}
    Wp = P.shared_weights()
    sens = P.within_fold_weights(preds)
    A = P.assemble(scen, preds)
    LPz = np.load(PRIVQ / f"private_crossfit_{scen}.npz")
    models = P.models_for(scen)
    point = np.zeros((len(models), len(ENDPOINTS)))
    bprim = np.zeros((len(models), len(ENDPOINTS), N_BOOT))
    bsens = np.zeros((len(models), len(ENDPOINTS), N_BOOT))
    for j, ep in enumerate(ENDPOINTS):
        g = A[A.endpoint == ep]
        ev, du = g.event.astype(bool).to_numpy(), g.duration.to_numpy(float)
        xr = np.array([pos[s] for s in g.dnam_id])
        R = P.risk_matrix(scen, A, ep, j, LPz)
        acc = {"p": [np.zeros(len(models)), np.zeros(len(models)), 0.0],
               "b": [np.zeros((len(models), N_BOOT)), np.zeros((len(models), N_BOOT)), np.zeros(N_BOOT)],
               "s": [np.zeros((len(models), N_BOOT)), np.zeros((len(models), N_BOOT)), np.zeros(N_BOOT)]}
        for f in sorted(g.outer_fold.unique()):
            s = np.flatnonzero(g.outer_fold.to_numpy() == f)
            out = P.fold_counts(ev[s], du[s], R[:, s], [None, Wp[:, xr[s]], sens[ep][f]])
            for key, (tot, arr) in zip(("p", "b", "s"), out):
                acc[key][0] += arr[..., 0]; acc[key][1] += arr[..., 1]; acc[key][2] = acc[key][2] + tot
        point[:, j] = J.cidx_from(*acc["p"])
        bprim[:, j] = J.cidx_from(acc["b"][0], acc["b"][1], acc["b"][2][None, :])
        bsens[:, j] = J.cidx_from(acc["s"][0], acc["s"][1], acc["s"][2][None, :])
        P.log(f"Q {scen}: {ep} done ({time.time() - t0:.0f} s)")
    tmp = fo.with_name(fo.stem + ".tmp.npz")
    np.savez(tmp, models=np.array(models), point=point, boot=bprim, boot_within_fold=bsens)
    tmp.replace(fo)
    return scen


def stage_boot(workers):
    require_committed()
    if not P.done_path("prep").exists():
        sys.exit("stage prep has not completed")
    t0 = time.time()
    rows = []
    Wp = P.shared_weights()
    ref = json.loads(P.need(PRIVP / "private_boot_weights.json").read_text())["sha256"]
    rows.append({"check": "4 bootstrap weights equal P", "passed": P.sha256_bytes(Wp.tobytes()) == ref, "detail": ref})
    del Wp
    (PRIVQ / "boot").mkdir(parents=True, exist_ok=True)
    with mp.get_context("fork").Pool(min(workers, len(P.ALL_SCEN))) as pool:
        pool.map(boot_scenario, P.ALL_SCEN)
    for scen in P.ALL_SCEN:
        q = np.load(PRIVQ / "boot" / f"private_boot_{scen}.npz")
        p = np.load(P.need(PRIVP / "boot" / f"private_boot_{scen}.npz"))
        qm, pm = list(q["models"].astype(str)), list(p["models"].astype(str))
        for s in SHARED_WITH_P(scen) + ["age_sex", OURS]:
            d = max(float(np.max(np.abs(q[k][qm.index(s)] - p[k][pm.index(s)]))) for k in ("point", "boot", "boot_within_fold"))
            rows.append({"check": "5 shared C-indices reproduce P", "passed": d <= TOL, "detail": f"{scen} {s} max abs diff {d:.2e}"})
    write_checks(rows, "boot")
    P.done_marker("boot", {"seconds": round(time.time() - t0)})


# ------------------------------------------------------------------ stage analyse
def pct(d, q):
    return float(np.percentile(d, q))


def rule(ma, mb, lo, decision):
    if not mb > 0:
        return "not applicable (M_b <= 0)" + ("" if decision else " (reference only)")
    ok = (ma - mb) / mb >= 0.10 and lo > 0
    return ("PASS" if ok else "FAIL") if decision else ("meets (reference only)" if ok else "does not meet (reference only)")


def stage_analyse():
    require_committed()
    if not P.done_path("boot").exists():
        sys.exit("stage boot has not completed")
    preds = J.load_preds()
    events = np.array([int(preds[ep].event.sum()) for ep in ENDPOINTS], float)
    summ, cons, per, loeo, ew = [], [], [], [], []
    res = {"settings": {"seed": P.SEED, "draws": N_BOOT, "inner_folds": P.N_INNER, "primary": "full",
                        "written": "8 Nov 2026, after J to S results were known; no Q set fitted before the design commit"},
           "scenarios": {}}
    for scen in P.ALL_SCEN:
        z = np.load(PRIVQ / "boot" / f"private_boot_{scen}.npz")
        models = list(z["models"].astype(str))
        Pt, B, S = z["point"], z["boot"], z["boot_within_fold"]
        ia = models.index("age_sex")
        dP, dB, dS = Pt - Pt[ia], B - B[ia], S - S[ia]
        M, MB, MS = dP.mean(1), dB.mean(1), dS.mean(1)
        ix = models.index
        sets = q_sets_for(scen)
        for m in models:
            if m == "age_sex":
                continue
            k = ix(m)
            summ.append({"scenario": scen, "label": P.label(scen), "model": m,
                         "members": "age, sex" + "".join(f", {c}" for c in sets[m][0]) if m in sets else m,
                         "role": sets[m][1] if m in sets else "unstacked stored risk", "M": M[k],
                         "M_ci95_low": pct(MB[k], 2.5), "M_ci95_high": pct(MB[k], 97.5)})
        best = lambda cands: max(cands, key=lambda m: M[ix(m)])
        singles = [f"{c}_{F}" for F in P.fills(scen) for c in SINGLE]
        con = [("D_G: MethylGPT added to GrimAge2", "QG_mean", "C1_mean", True),
               ("D_D: MethylGPT added to DunedinPACE", "QD_mean", "C2_mean", True),
               ("MethylGPT + GrimAge2 vs raw-CpG model + GrimAge2", "QG_mean", best(["GV_mean", "GG_mean"]), False),
               ("MethylGPT + DunedinPACE vs raw-CpG model + DunedinPACE", "QD_mean", best(["DV_mean", "DG_mean"]), False),
               ("MethylGPT + GrimAge2 vs best single-risk set", "QG_mean", best(singles), False),
               ("MethylGPT + DunedinPACE vs best single-risk set", "QD_mean", best(singles), False),
               ("GrimAge2 added to MethylGPT", "QG_mean", "S1", False),
               ("DunedinPACE added to MethylGPT", "QD_mean", "S1", False)]
        if scen != "full":
            con += [("MethylGPT added to GrimAge2 (kNN fill)", "QG_knn", "C1_knn", False),
                    ("MethylGPT added to DunedinPACE (kNN fill)", "QD_knn", "C2_knn", False)]
        res["scenarios"][scen] = {}
        for name, a, b, dec in con:
            ka, kb = ix(a), ix(b)
            d, ds = MB[ka] - MB[kb], MS[ka] - MS[kb]
            isdec = dec and scen == "full"
            row = {"scenario": scen, "label": P.label(scen), "contrast": name, "set_a": a, "set_b": b,
                   "decision_row": isdec, "M_a": M[ka], "M_b": M[kb], "difference": M[ka] - M[kb],
                   "ci95_low": pct(d, 2.5), "ci95_high": pct(d, 97.5), "ci95_low_within_fold": pct(ds, 2.5),
                   "ci95_high_within_fold": pct(ds, 97.5), "p_boot": J.pboot(d),
                   "relative_gain": (M[ka] - M[kb]) / M[kb] if M[kb] > 0 else np.nan,
                   "outcome": rule(M[ka], M[kb], pct(d, 2.5), isdec)}
            cons.append(row)
            res["scenarios"][scen][name] = {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                                            for k, v in row.items() if k not in ("label",)}
        for name, a, b, _ in con[:2]:
            ka, kb = ix(a), ix(b)
            p = [J.pboot(dB[ka, j] - dB[kb, j]) for j in range(len(ENDPOINTS))]
            q = MM.bh(np.array(p))
            for j, ep in enumerate(ENDPOINTS):
                d = dB[ka, j] - dB[kb, j]
                per.append({"scenario": scen, "contrast": name, "endpoint": ep, "events": int(events[j]),
                            "c_index_a": Pt[ka, j], "c_index_b": Pt[kb, j], "c_age_sex": Pt[ia, j],
                            "difference": dP[ka, j] - dP[kb, j], "ci95_low": pct(d, 2.5), "ci95_high": pct(d, 97.5),
                            "p_boot": p[j], "q_bh_across_endpoints": q[j]})
                keep = np.arange(len(ENDPOINTS)) != j
                Ml, MBl = dP[:, keep].mean(1), dB[:, keep].mean(1)
                dl = MBl[ka] - MBl[kb]
                loeo.append({"scenario": scen, "contrast": name, "left_out": ep, "M_a": Ml[ka], "M_b": Ml[kb],
                             "difference": Ml[ka] - Ml[kb], "ci95_low": pct(dl, 2.5), "ci95_high": pct(dl, 97.5),
                             "relative_gain": (Ml[ka] - Ml[kb]) / Ml[kb] if Ml[kb] > 0 else np.nan,
                             "rule_arithmetic": rule(Ml[ka], Ml[kb], pct(dl, 2.5), False)})
            wt = events / events.sum()
            Mw, MBw = (dP * wt).sum(1), (dB * wt[None, :, None]).sum(1)
            dw = MBw[ka] - MBw[kb]
            ew.append({"scenario": scen, "contrast": name, "M_a": Mw[ka], "M_b": Mw[kb], "difference": Mw[ka] - Mw[kb],
                       "ci95_low": pct(dw, 2.5), "ci95_high": pct(dw, 97.5),
                       "relative_gain": (Mw[ka] - Mw[kb]) / Mw[kb] if Mw[kb] > 0 else np.nan,
                       "rule_arithmetic": rule(Mw[ka], Mw[kb], pct(dw, 2.5), False)})
    files = {f"Q_summary_{DATE}.csv": summ, f"Q_contrasts_{DATE}.csv": cons, f"Q_endpoint_contrasts_{DATE}.csv": per,
             f"Q_leave_one_endpoint_out_{DATE}.csv": loeo, f"Q_event_weighted_{DATE}.csv": ew}
    rows = []
    for fn, r in files.items():
        df = pd.DataFrame(r)
        ok = not any(c.lower() in ID_COLUMNS for c in df.columns) and ID_PATTERN.search(df.astype(str).to_csv(index=False)) is None
        rows.append({"check": "6 privacy", "passed": bool(ok), "detail": fn})
    write_checks(rows, "analyse")
    for fn, r in files.items():
        P.atomic_csv(OUTQ / fn, pd.DataFrame(r))
    P.write_json(OUTQ / f"Q_results_{DATE}.json", res)
    P.done_marker("analyse")
    print(pd.DataFrame(cons).drop(columns=["label"]).round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["prep", "boot", "analyse"])
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    {"prep": lambda: stage_prep(a.workers), "boot": lambda: stage_boot(a.workers), "analyse": stage_analyse}[a.stage]()


if __name__ == "__main__":
    main()
