#!/usr/bin/env python3
"""Large-gain analysis U: Generation Scotland (GS) individual diseases, MethylGPT combined with ONE clock at a time.
Tables only. Analysis Q's design (P's cross-fitted ridge Cox and P's code) applied to analysis K's 13 primary-set diseases.

Written on 9 Nov 2026, after the results of J to T were known (K: MethylGPT alone against the best single comparator fails
over 13 diseases; Q: MethylGPT added to either clock passes over the 8 endpoints). U asks whether MethylGPT adds to each
clock across individual diseases. No disease model is refitted and no risk is recomputed: U combines K's stored held-out
risks. Every result is reported, whichever set it favours.

DATA
  K's 13 primary-set diseases (K_disease_table role "primary"), 9,866 GS development participants, settings full
  (complete profiles; primary), hm27 and random_90 (simulated missingness on empirical profiles, K's masks and fills).
  Labels and K's locked per-disease outer folds: artifacts/large_gain_2026-10-30_private/K/labels/private_labels_<d>.parquet
  (frame-cache order). Stored held-out risks: .../K/risks/private_risks_<d>__<setting>.parquet (row-aligned with labels).
  Age and sex: J.load_preds() (K's covariates()). The age_sex column is K's stored age+sex risk.

SETS (Q's; F = mean, and in hm27 and random_90 also F = knn)
  QG_F = MethylGPT combined + GrimAge2_F; QD_F = MethylGPT combined + DunedinPACE_F; C1_F GrimAge2; C2_F DunedinPACE;
  C4_F, C5_F raw CpGs (v12, dense fit); GV_F, GG_F, DV_F, DG_F one clock plus one raw-CpG model; S1 MethylGPT combined
  alone; S0 age and sex only.
METHOD (P's, unchanged code: P.fold_columns, P.crossfit_fold, P.cox_fit): within each disease and outer fold, 10 inner folds
  stratified by event, P.inner_assignment with endpoint index 100 + disease index (default_rng([20261108, 100 + d, fold,
  flag])); ridge Cox with age and sex unpenalised, fixed ridge 0.5 x sum of squared risk coefficients; columns failing L's
  constant-risk test in a fold are dropped from that fold.
METRIC: K's M = mean over the 13 diseases of (pooled within-fold C-index of the set - C-index of K's stored age+sex risk);
  J.fold_counts, tied tolerance 1e-8.

DECISION (setting full; two pre-specified contrasts, J's rule, no multiplicity adjustment)
  U_G: QG_mean against C1_mean.  U_D: QD_mean against C2_mean.
  PASS iff (M_a - M_b) / M_b >= 0.10 and the lower end of the 95% interval of M_a - M_b is > 0; not applicable if M_b <= 0.
SECONDARY (reference): the same in hm27 and random_90 (mean and kNN fill); MethylGPT + clock against the better raw-CpG +
  clock set; against S1; per disease with BH q across the 13 diseases; leave one disease out; event-weighted mean.
INTERVALS: P's shared participant bootstrap (P.shared_weights, seed 20261108, 2,000 draws, positions from the frame cache);
  cross-fitted risks fixed. Percentile 95% intervals; bootstrap P as J.

CHECKS (abort on failure)
  1. Labels are in frame-cache order (dnam_id equals P.load_ids()) and risks are row-aligned (row = position, same folds).
  2. The pooled C-index of K's stored mgpt2_combined, grimage2_mean and age_sex risks recomputed here equals K's stored
     point C-index within 1e-12 for every disease and setting.
  3. Every cross-fitted fit converges and every linear predictor is finite.
  4. Bootstrap weights equal P's (SHA-256).
  5. Privacy: no participant identifier pattern or identifier column in any public file.

AMENDMENT 1 (9 Nov 2026, before any U result was computed; the first run stopped in cross-fitting): in Depression outer
fold 1 all 12 incident events are in women, so the unpenalised sex coefficient has no finite maximum and P's Newton solve
is singular for every set, age and sex alone included. Rule: in any inner fit whose training events all share one sex, sex
is left out of the design for every set of that fit (it is not identifiable there); age, the ridge and everything else are
unchanged. Implemented by crossfit_fold_u, which is P.crossfit_fold with this rule; the number of affected fits is
recorded (U_fit_record). All sets of a fit are treated identically, so contrasts between sets stay like-for-like.

Run: python3 large_gain_U_gs_single_clock_stacking_diseases_2026-11-09.py [--workers n]
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
DATE = "2026-11-09"
OUT = ART / "large_gain_2026-11-09/U"
PRIV = ART / "large_gain_2026-11-09_private/U"
KPRIV = ART / "large_gain_2026-10-30_private/K"
KOUT = ART / "large_gain_2026-10-30/K"
SCRIPTS = {"P": HERE / "large_gain_P_gs_stacked_risks_2026-11-08.py", "Q": HERE / "large_gain_Q_gs_single_clock_stacking_2026-11-08.py"}
SETTINGS = ["full", "hm27", "random_90"]
ID_PATTERN = re.compile(r"\b\d{12}_R\d{2}C\d{2}\b")
ID_COLUMNS = {"dnam_id", "sample_id", "participant_id", "id", "sample", "participant", "barcode"}
TOL = 1e-12


def load_module(name, path):
    if not path.exists():
        sys.exit(f"Cannot proceed - required data file not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


P = load_module("large_gain_P", SCRIPTS["P"])
J = P.J
OURS, N_BOOT = P.OURS, P.N_BOOT
SINGLE = {"C1": ["grimage2"], "C2": ["dunedinpace"], "C4": ["raw_cpg_v12"], "C5": ["raw_cpg_gpu"]}
PAIRS = {"GV": ["grimage2", "raw_cpg_v12"], "GG": ["grimage2", "raw_cpg_gpu"],
         "DV": ["dunedinpace", "raw_cpg_v12"], "DG": ["dunedinpace", "raw_cpg_gpu"]}


def fills(s):
    return ["mean"] if s == "full" else ["mean", "knn"]


def sets_for(s):
    out = {}
    for F in fills(s):
        out[f"QG_{F}"] = [OURS, f"grimage2_{F}"]
        out[f"QD_{F}"] = [OURS, f"dunedinpace_{F}"]
        for c, mem in {**SINGLE, **PAIRS}.items():
            out[f"{c}_{F}"] = [f"{b}_{F}" for b in mem]
    out["S1"] = [OURS]
    out["S0"] = []
    return out


def diseases():
    t = pd.read_csv(P.need(KOUT / "K_disease_table_2026-10-30.csv"))
    return t.loc[t.role == "primary", "constituent"].tolist()


class G:
    def __init__(self, ev, du, fold, age, sex, R):
        self.ev, self.du, self.fold, self.age_v, self.sex_v, self.R = ev, du, fold, age, sex, R


def crossfit_fold_u(g, H, inner, z, est, sets):
    """P.crossfit_fold with amendment 1: sex is dropped from an inner fit whose training events all share one sex."""
    LP = np.full((len(sets), len(H)), np.nan)
    recs = []
    centre = (g.age_v[H].mean(), g.sex_v[H].mean())
    for j in range(P.N_INNER):
        tr, te = np.flatnonzero(inner != j), np.flatnonzero(inner == j)
        T, Te = H[tr], H[te]
        mu, sd = g.age_v[T].mean(), g.age_v[T].std()
        drop_sex = np.unique(g.sex_v[T][g.ev[T]]).size < 2
        keep = None
        ztr = {c: v[tr] for c, v in z.items()}
        zte = {c: v[te] for c, v in z.items()}
        for si, (name, mem) in enumerate(sets.items()):
            use = [c for c in mem if est[c]]
            Ztr = P.design(g, T, ztr, use, mu, sd)
            Zte = P.design(g, Te, zte, use, mu, sd, centre)
            pen = np.r_[0.0, 0.0, np.ones(len(use))]
            if drop_sex:
                keep = [0] + list(range(2, Ztr.shape[1]))
                Ztr, Zte, pen = Ztr[:, keep], Zte[:, keep], pen[keep]
            beta, it, conv, ll = P.cox_fit(Ztr, g.du[T], g.ev[T], pen)
            LP[si, te] = Zte @ beta
            recs.append({"inner_fold": j, "set": name, "n_train": len(T), "events_train": int(g.ev[T].sum()),
                         "columns_used": ";".join(use), "sex_dropped": bool(drop_sex), "iterations": it,
                         "converged": bool(conv)})
    return LP, recs


def require_committed():
    for p in [Path(__file__).resolve()] + list(SCRIPTS.values()):
        r = subprocess.run(["git", "-C", str(HERE), "status", "--porcelain", "--", str(p)], capture_output=True, text=True)
        t = subprocess.run(["git", "-C", str(HERE), "ls-files", "--error-unmatch", str(p)], capture_output=True, text=True)
        if r.stdout.strip() or t.returncode != 0:
            sys.exit(f"Cannot proceed - {p.name} must be committed unchanged before this analysis runs")
    P.require_committed()


def task(args):
    s, d_idx, d = args
    fo = PRIV / f"private_{s}__{d}.npz"
    if fo.exists():
        return s, d
    ids = P.load_ids()
    lab = pd.read_parquet(KPRIV / "labels" / f"private_labels_{d}.parquet")
    rk = pd.read_parquet(KPRIV / "risks" / f"private_risks_{d}__{s}.parquet")
    cov = J.load_preds()[J.ENDPOINTS[0]].set_index("dnam_id").loc[lab.dnam_id.astype(str)]
    ev, du, fold = lab.event.astype(bool).to_numpy(), lab.duration.to_numpy(float), lab.outer_fold.to_numpy()
    sets = sets_for(s)
    cols = list(dict.fromkeys(c for mem in sets.values() for c in mem))
    g = G(ev, du, fold, cov.age.to_numpy(float), cov.sex.to_numpy(float), {c: rk[c].to_numpy(float) for c in cols + ["age_sex"]})
    LP = np.full((len(sets), len(ev)), np.nan)
    fits = []
    for f in sorted(np.unique(fold)):
        H = np.flatnonzero(fold == f)
        inner = P.inner_assignment(ev[H], 100 + d_idx, int(f))
        z, est, _ = P.fold_columns(g, H, cols + ["age_sex"])
        lp, recs = crossfit_fold_u(g, H, inner, z, est, sets)
        LP[:, H] = lp
        fits += [{"setting": s, "disease": d, "outer_fold": int(f), **r} for r in recs]
    # C-index: sets, then stored references
    refs = ["age_sex", OURS, "grimage2_mean"]
    R = np.vstack([LP] + [rk[c].to_numpy(float)[None] for c in refs])
    Wp = P.shared_weights()
    acc = {"p": [np.zeros(len(R)), np.zeros(len(R)), 0.0],
           "b": [np.zeros((len(R), N_BOOT)), np.zeros((len(R), N_BOOT)), np.zeros(N_BOOT)]}
    pos = np.arange(len(ev))          # labels in frame-cache order (check 1)
    for f in sorted(np.unique(fold)):
        ss = np.flatnonzero(fold == f)
        res = P.fold_counts(ev[ss], du[ss], R[:, ss], [None, Wp[:, pos[ss]]])
        for key, (tot, arr) in zip(("p", "b"), res):
            acc[key][0] += arr[..., 0]; acc[key][1] += arr[..., 1]; acc[key][2] = acc[key][2] + tot
    point = J.cidx_from(*acc["p"])
    boot = J.cidx_from(acc["b"][0], acc["b"][1], acc["b"][2][None, :])
    fr = pd.DataFrame(fits)
    tmp = fo.with_name(fo.stem + ".tmp.npz")
    np.savez(tmp, models=np.array(list(sets) + refs), point=point, boot=boot, LP=LP,
             converged=bool(fr.converged.all()), finite=bool(np.isfinite(LP).all()), events=int(ev.sum()),
             n_fits=len(fr), n_sex_dropped=int(fr.sex_dropped.sum()))
    tmp.replace(fo)
    return s, d


def pct(x, q):
    return float(np.percentile(x, q))


def rule(ma, mb, lo, dec):
    if not mb > 0:
        return "not applicable (M_b <= 0)" + ("" if dec else " (reference only)")
    ok = (ma - mb) / mb >= 0.10 and lo > 0
    return ("PASS" if ok else "FAIL") if dec else ("meets (reference only)" if ok else "does not meet (reference only)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    require_committed()
    t0 = time.time()
    PRIV.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    checks = []
    chk = lambda n, ok, det="": checks.append({"check": n, "passed": bool(ok), "detail": str(det)})
    ds = diseases()
    ids = P.load_ids()
    for d in ds:
        lab = pd.read_parquet(P.need(KPRIV / "labels" / f"private_labels_{d}.parquet"))
        chk("1 labels in frame-cache order", np.array_equal(lab.dnam_id.astype(str).to_numpy(), ids), d)
        for s in SETTINGS:
            rk = pd.read_parquet(P.need(KPRIV / "risks" / f"private_risks_{d}__{s}.parquet"))
            chk("1 risks row-aligned", np.array_equal(rk.row.to_numpy(), np.arange(len(lab)))
                and np.array_equal(rk.outer_fold.to_numpy(), lab.outer_fold.to_numpy()), f"{d} {s}")
    ref = json.loads(P.need(P.PRIV / "private_boot_weights.json").read_text())["sha256"]
    chk("4 bootstrap weights equal P", P.sha256_bytes(P.shared_weights().tobytes()) == ref)
    ck = pd.DataFrame(checks)
    if not ck.passed.all():
        P.atomic_csv(OUT / f"U_checks_{DATE}.csv", ck)
        print(ck[~ck.passed].to_string())
        sys.exit("U checks failed before fitting")
    tasks = [(s, i, d) for s in SETTINGS for i, d in enumerate(ds)]
    with mp.get_context("fork").Pool(a.workers) as pool:
        for s, d in pool.imap_unordered(task, tasks):
            P.log(f"U {s} {d} done ({time.time() - t0:.0f} s)")
    kz = np.load(P.need(KPRIV / "private_bootstrap_cindex.npz"))
    kd = list(kz["diseases"].astype(str))
    summ, cons, per, loeo, ew, res, frec = [], [], [], [], [], {}, []
    for s in SETTINGS:
        Z = [np.load(PRIV / f"private_{s}__{d}.npz") for d in ds]
        frec += [{"setting": s, "disease": d, "fits": int(z["n_fits"]), "fits_with_sex_dropped": int(z["n_sex_dropped"])}
                 for d, z in zip(ds, Z)]
        models = list(Z[0]["models"].astype(str))
        Pt = np.stack([z["point"] for z in Z], 1)          # (m, d)
        B = np.stack([z["boot"] for z in Z], 1)            # (m, d, B)
        events = np.array([int(z["events"]) for z in Z], float)
        km = list(kz[f"{s}__models"].astype(str))
        for i, d in enumerate(ds):
            chk("3 converged and finite", bool(Z[i]["converged"]) and bool(Z[i]["finite"]), f"{s} {d}")
            for m in ("age_sex", OURS, "grimage2_mean"):
                dd = abs(Pt[models.index(m), i] - kz[f"{s}__point"][km.index(m), kd.index(d)])
                chk("2 stored C-index reproduces K", dd <= TOL, f"{s} {d} {m} {dd:.1e}")
        ia = models.index("age_sex")
        dP, dB = Pt - Pt[ia], B - B[ia]
        M, MB = dP.mean(1), dB.mean(1)
        ix = models.index
        for m in models:
            if m == "age_sex":
                continue
            summ.append({"setting": s, "model": m, "M": M[ix(m)], "M_ci95_low": pct(MB[ix(m)], 2.5), "M_ci95_high": pct(MB[ix(m)], 97.5)})
        best = lambda c: max(c, key=lambda m: M[ix(m)])
        con = [("U_G: MethylGPT added to GrimAge2", "QG_mean", "C1_mean", True),
               ("U_D: MethylGPT added to DunedinPACE", "QD_mean", "C2_mean", True),
               ("MethylGPT + GrimAge2 vs raw-CpG model + GrimAge2", "QG_mean", best(["GV_mean", "GG_mean"]), False),
               ("MethylGPT + DunedinPACE vs raw-CpG model + DunedinPACE", "QD_mean", best(["DV_mean", "DG_mean"]), False),
               ("GrimAge2 added to MethylGPT", "QG_mean", "S1", False),
               ("DunedinPACE added to MethylGPT", "QD_mean", "S1", False)]
        if s != "full":
            con += [("MethylGPT added to GrimAge2 (kNN fill)", "QG_knn", "C1_knn", False),
                    ("MethylGPT added to DunedinPACE (kNN fill)", "QD_knn", "C2_knn", False)]
        res[s] = {}
        for name, a_, b_, dec in con:
            ka, kb = ix(a_), ix(b_)
            d = MB[ka] - MB[kb]
            isdec = dec and s == "full"
            row = {"setting": s, "contrast": name, "set_a": a_, "set_b": b_, "decision_row": isdec, "M_a": M[ka], "M_b": M[kb],
                   "difference": M[ka] - M[kb], "ci95_low": pct(d, 2.5), "ci95_high": pct(d, 97.5), "p_boot": J.pboot(d),
                   "relative_gain": (M[ka] - M[kb]) / M[kb] if M[kb] > 0 else np.nan, "outcome": rule(M[ka], M[kb], pct(d, 2.5), isdec)}
            cons.append(row)
            res[s][name] = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in row.items()}
        for name, a_, b_, _ in con[:2] + (con[-2:] if s != "full" else []):
            ka, kb = ix(a_), ix(b_)
            p = [J.pboot(dB[ka, i] - dB[kb, i]) for i in range(len(ds))]
            q = P.MM.bh(np.array(p))
            for i, d in enumerate(ds):
                x = dB[ka, i] - dB[kb, i]
                per.append({"setting": s, "contrast": name, "disease": d, "events": int(events[i]), "c_index_a": Pt[ka, i],
                            "c_index_b": Pt[kb, i], "c_age_sex": Pt[ia, i], "difference": dP[ka, i] - dP[kb, i],
                            "ci95_low": pct(x, 2.5), "ci95_high": pct(x, 97.5), "p_boot": p[i], "q_bh_across_diseases": q[i]})
                keep = np.arange(len(ds)) != i
                Ml, MBl = dP[:, keep].mean(1), dB[:, keep].mean(1)
                y = MBl[ka] - MBl[kb]
                loeo.append({"setting": s, "contrast": name, "left_out": d, "M_a": Ml[ka], "M_b": Ml[kb], "difference": Ml[ka] - Ml[kb],
                             "ci95_low": pct(y, 2.5), "ci95_high": pct(y, 97.5),
                             "relative_gain": (Ml[ka] - Ml[kb]) / Ml[kb] if Ml[kb] > 0 else np.nan,
                             "rule_arithmetic": rule(Ml[ka], Ml[kb], pct(y, 2.5), False)})
            wt = events / events.sum()
            Mw, MBw = (dP * wt).sum(1), (dB * wt[None, :, None]).sum(1)
            y = MBw[ka] - MBw[kb]
            ew.append({"setting": s, "contrast": name, "M_a": Mw[ka], "M_b": Mw[kb], "difference": Mw[ka] - Mw[kb],
                       "ci95_low": pct(y, 2.5), "ci95_high": pct(y, 97.5),
                       "relative_gain": (Mw[ka] - Mw[kb]) / Mw[kb] if Mw[kb] > 0 else np.nan,
                       "rule_arithmetic": rule(Mw[ka], Mw[kb], pct(y, 2.5), False)})
    files = {f"U_summary_{DATE}.csv": summ, f"U_contrasts_{DATE}.csv": cons, f"U_disease_contrasts_{DATE}.csv": per,
             f"U_leave_one_disease_out_{DATE}.csv": loeo, f"U_event_weighted_{DATE}.csv": ew, f"U_fit_record_{DATE}.csv": frec}
    for fn, r in files.items():
        df = pd.DataFrame(r)
        ok = not any(c.lower() in ID_COLUMNS for c in df.columns) and ID_PATTERN.search(df.astype(str).to_csv(index=False)) is None
        chk("5 privacy", ok, fn)
    ck = pd.DataFrame(checks)
    P.atomic_csv(OUT / f"U_checks_{DATE}.csv", ck)
    if not ck.passed.all():
        print(ck[~ck.passed].to_string())
        sys.exit("U checks failed; no result table written")
    for fn, r in files.items():
        P.atomic_csv(OUT / fn, pd.DataFrame(r))
    P.write_json(OUT / f"U_results_{DATE}.json", {"written": "9 Nov 2026, after J to T results were known", "diseases": ds,
                                                  "n_checks": int(len(ck)), "settings": res})
    print(pd.DataFrame(cons).round(4).to_string())
    print(f"U done in {time.time() - t0:.0f} s; {len(ck)} checks pass")


if __name__ == "__main__":
    main()
