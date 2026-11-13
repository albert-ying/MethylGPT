#!/usr/bin/env python3
"""Large-gain analysis V: read-depth-aware MethylGPT against single-clock comparators in Generation Scotland (GS) under
simulated measurement noise on empirical profiles. Tables only; no model is refitted and no risk is recomputed.

Written on 9 Nov 2026, after the results of J to U were known. M, N and O pre-specified the depth-aware version of
MethylGPT combined (mgpt2_combined_depthaware: each observed CpG weighted by its read-count precision in the factor
posterior; in the Gaussian settings by the known noise variance) as a MethylGPT application outside their decisions, and
compared it with the locked MethylGPT and with the combined-clock comparator only. V tabulates it against the single-clock
comparator set of analysis R, from the per-draw statistics M and O stored with their own pre-specified bootstrap. N already
compared it with N's comparators (N_applications); V does not recompute N. Every row is reported, whichever model it favours;
rows are applications outside the decisions and carry J's rule arithmetic for reference only.

COMPARATORS (R's): best_single (R's choice per scenario), GrimAge2 and DunedinPACE at their better fill. M also: the best of
  M's added corpus-kNN-denoising comparators (outside M's decision), because at depths 1 and 2 they were the models above
  MethylGPT.
M (C-index gain above age and sex; M's private boot arrays, shared draws, seed 20261102): per scenario the mean over the
  8 endpoints, per endpoint with BH q, leave one endpoint out (best_single fixed and re-chosen).
O (L's metrics; O's private fit files, L's bootstrap seed 20261030): H and every L metric as the mean over endpoints, per
  endpoint log hazard ratio per SD with BH q, leave one endpoint out on H; Kaplan-Meier curves of the depth-aware model in
  O's KM scenarios (depth_10, depth_30, full) from O's stored arrays, with L's disclosure control.

AMENDMENT 1 (9 Nov 2026, after the first run): the best kNN-denoising comparator is chosen among single-clock and raw-CpG
models only; the first run let the combined GrimAge2+DunedinPACE kNN-denoising model be chosen in full, Gaussian SD 0.02 and
0.05 (R's rule excludes it). No other change; the first run's tables are replaced.

CHECKS (abort on failure)
  1. The depth-aware model's mean gain and the depth-aware-versus-locked interval reproduce M_summary and
     M_added_contrasts within 1e-12.
  2. The depth-aware model's H and the O_summary difference interval against it reproduce within 1e-12.
  3. best_single equals R's choice for every scenario.
  4. Privacy: no identifier pattern or column in public files; suppressed cells blank.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
DATE = "2026-11-09"
OUT = ART / "large_gain_2026-11-09/V"
PRIV = ART / "large_gain_2026-11-09_private/V"
TOL = 1e-12


def load_module(name, path):
    if not path.exists():
        sys.exit(f"Cannot proceed - required data file not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


S = load_module("large_gain_S", HERE / "large_gain_S_single_clock_endpoint_contrasts_2026-11-08.py")
O, L, M, J = S.O, S.L, S.M, S.J
ENDPOINTS, OURS, DA = J.ENDPOINTS, J.OURS, M.DA
pct, pct3, rule_ref, bh, need = S.pct, S.pct3, S.rule_ref, S.bh, S.need
CHECKS = []


def chk(name, ok, detail=""):
    CHECKS.append({"check": name, "passed": bool(ok), "detail": str(detail)})


def m_family():
    ms = pd.read_csv(need(ART / "large_gain_2026-11-02/M/M_summary_2026-11-02.csv"))
    ma = pd.read_csv(need(ART / "large_gain_2026-11-02/M/M_added_contrasts_2026-11-02.csv"))
    mean_rows, per, loeo = [], [], []
    for scen in ms.scenario.unique():
        z = np.load(need(ART / f"large_gain_2026-11-02_private/M/boot/private_boot_{scen}.npz"))
        models = list(z["models"].astype(str))
        if DA not in models:
            continue
        ia, io, kd = models.index("age_sex"), models.index(OURS), models.index(DA)
        dP, dB = z["point"] - z["point"][ia], z["boot"] - z["boot"][ia]
        Mp, MB = dP.mean(1), dB.mean(1)
        s_ = ms[ms.scenario == scen]
        chk("1 depth-aware mean gain", abs(Mp[kd] - s_.loc[s_.model == DA, "M_model"].iloc[0]) < TOL, scen)
        r = ma[(ma.scenario == scen) & (ma.contrast == "depth-aware vs standard MethylGPT combined")]
        if len(r):
            d = MB[kd] - MB[io]
            chk("1 depth-aware vs locked interval", abs(pct(d, 2.5) - r.ci95_low.iloc[0]) < TOL and abs(pct(d, 97.5) - r.ci95_high.iloc[0]) < TOL, scen)
        dec = s_.loc[s_.role == "decision comparator", "model"].tolist()
        single = [m for m in dec if not m.startswith(S.COMBINED)]
        cm = S.comparators(single, lambda m: Mp[models.index(m)])
        chk("3 best_single equals R", cm["best_single"] == S.r_best("M", scen), scen)
        added = [m for m in s_.loc[s_.role.str.startswith("added comparator"), "model"] if m in models
                 and not m.startswith(S.COMBINED)]      # amendment 1
        if added:
            cm["best_knn_denoised"] = max(added, key=lambda m: Mp[models.index(m)])
        cm["locked_MethylGPT"] = OURS
        for lab, c in cm.items():
            k = models.index(c)
            d = MB[kd] - MB[k]
            mean_rows.append({"scenario": scen, "contrast": lab, "comparator": c, "gain_depthaware": Mp[kd], "gain_comparator": Mp[k],
                              "difference": Mp[kd] - Mp[k], "ci95_low": pct(d, 2.5), "ci95_high": pct(d, 97.5), "p_boot": J.pboot(d),
                              "relative_gain": (Mp[kd] - Mp[k]) / Mp[k] if Mp[k] > 0 else np.nan,
                              "rule_arithmetic_for_reference": rule_ref(Mp[kd], Mp[k], pct(d, 2.5))})
            if lab == "locked_MethylGPT":
                continue
            p = [J.pboot(dB[kd, j] - dB[k, j]) for j in range(len(ENDPOINTS))]
            q = bh(p)
            for j, ep in enumerate(ENDPOINTS):
                x = dB[kd, j] - dB[k, j]
                per.append({"scenario": scen, "contrast": lab, "comparator": c, "endpoint": ep, "delta_c_depthaware": dP[kd, j],
                            "delta_c_comparator": dP[k, j], "difference": dP[kd, j] - dP[k, j], "ci95_low": pct(x, 2.5),
                            "ci95_high": pct(x, 97.5), "p_boot": p[j], "q_bh_across_endpoints": q[j]})
        kb = models.index(cm["best_single"])
        for j, ep in enumerate(ENDPOINTS):
            keep = np.arange(len(ENDPOINTS)) != j
            Ml, MBl = dP[:, keep].mean(1), dB[:, keep].mean(1)
            kr = models.index(max(single, key=lambda m: Ml[models.index(m)]))
            for kind, kk in (("fixed", kb), ("rechosen", kr)):
                x = MBl[kd] - MBl[kk]
                loeo.append({"scenario": scen, "left_out": ep, "comparator_kind": kind, "comparator": models[kk],
                             "gain_depthaware": Ml[kd], "gain_comparator": Ml[kk], "difference": Ml[kd] - Ml[kk],
                             "ci95_low": pct(x, 2.5), "ci95_high": pct(x, 97.5),
                             "rule_arithmetic_for_reference": rule_ref(Ml[kd], Ml[kk], pct(x, 2.5))})
    return pd.DataFrame(mean_rows), pd.DataFrame(per), pd.DataFrame(loeo)


def o_family():
    osum = pd.read_csv(need(O.OUT / "O_summary_2026-11-08.csv"))
    rows, per, loeo, krows = [], [], [], []
    for scen in O.SCENARIOS:
        F = O.load_fit(scen)
        ms = F["models"]
        if DA not in ms:
            continue
        io, kd = ms.index(OURS), ms.index(DA)
        b, bb = F["beta_pt"], F["beta_boot"]
        allnot = ~np.isfinite(b)
        Hp, Hb = np.where(allnot, 0.0, b).mean(1), np.where(allnot[..., None], 0.0, bb).mean(1)
        s_ = osum[osum.scenario == scen]
        r = s_[s_.model == DA].iloc[0]
        d = Hb[io] - Hb[kd]
        chk("2 depth-aware H and interval", abs(Hp[kd] - r.H_model) < TOL and abs(pct3(d)[0] - r.ci95_low) < TOL
            and abs(pct3(d)[1] - r.ci95_high) < TOL, scen)
        dec = s_.loc[s_.role == "decision comparator", "model"].tolist()
        single = [m for m in dec if not m.startswith(S.COMBINED)]
        cm = S.comparators(single, lambda m: Hp[ms.index(m)])
        chk("3 best_single equals R", cm["best_single"] == S.r_best("O", scen), scen)
        cm["locked_MethylGPT"] = OURS
        rel, kmm = None, None
        if scen in O.KM_SCENARIOS:
            kmm = [DA]
            c = F["km_models"].index(DA)
            rel = np.stack([np.stack([L.release_km(F["km_cum"][j, c], F["km_atrisk"][j, c], F["km_total"][j, c])])
                            for j in range(len(ENDPOINTS))])
            for bi, gn in enumerate(L.GROUPS):
                for y in range(len(L.YEARS)):
                    for j, ep in enumerate(ENDPOINTS):
                        lo, hi, _ = pct3(F["km_boot"][j, c, bi, :, y])
                        krows.append({"scenario": scen, "endpoint": ep, "model": DA, "group": gn, "year": int(L.YEARS[y]),
                                      "n_group": F["km_n"][j, c, bi], "survival": F["km_S"][j, c, bi, y], "survival_ci95_low": lo,
                                      "survival_ci95_high": hi, "n_at_risk": F["km_atrisk"][j, c, bi, y],
                                      "cum_events": F["km_cum"][j, c, bi, y], "suppressed": not bool(rel[j, 0, bi, y])})
        sup, _ = O.suppression(F, scen, rel, kmm)
        vals = O.metric_values(F)
        for lab, c in cm.items():
            k = ms.index(c)
            for metric in L.METRICS:
                P, B = vals[metric]
                Sm = sup[metric]
                if metric == "log_hr_per_sd":
                    va, vb, x, nsup = Hp[kd], Hp[k], Hb[kd] - Hb[k], 0
                else:
                    va, vb, x = P[kd].mean(), P[k].mean(), B[kd].mean(0) - B[k].mean(0)
                    nsup = int((Sm[kd] | Sm[k]).sum())
                lo, hi, nd = pct3(x)
                rows.append({"scenario": scen, "contrast": lab, "comparator": c, "metric": metric, "value_depthaware": va,
                             "value_comparator": vb, "difference": va - vb, "ci95_low": lo, "ci95_high": hi,
                             "p_boot": S.pboot_nan(x), "n_draws_estimable": nd,
                             "relative_gain": (va - vb) / vb if vb > 0 else np.nan,
                             "rule_arithmetic_for_reference": rule_ref(va, vb, lo) if metric == "log_hr_per_sd" else "",
                             "suppressed": nsup > 0})
            if lab == "locked_MethylGPT":
                continue
            P, B = vals["log_hr_per_sd"]
            p = [S.pboot_nan(B[kd, j] - B[k, j]) for j in range(len(ENDPOINTS))]
            q = bh(p)
            for j, ep in enumerate(ENDPOINTS):
                lo, hi, nd = pct3(B[kd, j] - B[k, j])
                per.append({"scenario": scen, "contrast": lab, "comparator": c, "endpoint": ep, "value_depthaware": P[kd, j],
                            "value_comparator": P[k, j], "difference": P[kd, j] - P[k, j], "ci95_low": lo, "ci95_high": hi,
                            "p_boot": p[j], "q_bh_across_endpoints": q[j], "n_draws_estimable": nd})
        kb = ms.index(cm["best_single"])
        for j, ep in enumerate(ENDPOINTS):
            keep = np.arange(len(ENDPOINTS)) != j
            bl = np.where(allnot, 0.0, b)[:, keep].mean(1)
            bbl = np.where(allnot[..., None], 0.0, bb)[:, keep].mean(1)
            kr = ms.index(max(single, key=lambda m: bl[ms.index(m)]))
            for kind, kk in (("fixed", kb), ("rechosen", kr)):
                x = bbl[kd] - bbl[kk]
                lo, hi, _ = pct3(x)
                loeo.append({"scenario": scen, "left_out": ep, "comparator_kind": kind, "comparator": ms[kk], "H_depthaware": bl[kd],
                             "H_comparator": bl[kk], "difference": bl[kd] - bl[kk], "ci95_low": lo, "ci95_high": hi,
                             "rule_arithmetic_for_reference": rule_ref(bl[kd], bl[kk], lo)})
    return pd.DataFrame(rows), pd.DataFrame(per), pd.DataFrame(loeo), pd.DataFrame(krows)


def main():
    mm, mp_, ml = m_family()
    om, op, ol, ok_ = o_family()
    OUT.mkdir(parents=True, exist_ok=True)
    PRIV.mkdir(parents=True, exist_ok=True)
    ck = pd.DataFrame(CHECKS)
    if not ck.passed.all():
        L.atomic_csv(OUT / f"V_checks_{DATE}.csv", ck)
        print(ck[~ck.passed].to_string())
        sys.exit("V checks failed; no result table written")
    L.atomic_csv(PRIV / "private_o_contrasts_unsuppressed.csv", om)
    L.atomic_csv(PRIV / "private_km_unsuppressed.csv", ok_)
    pom, pk = om.copy(), ok_.copy()
    pom.loc[pom.suppressed, ["value_depthaware", "value_comparator", "difference", "ci95_low", "ci95_high", "p_boot", "relative_gain"]] = np.nan
    rk = pom.metric.str.startswith("abs_risk")
    pom.loc[rk, ["value_depthaware", "value_comparator", "difference", "ci95_low", "ci95_high"]] = \
        pom.loc[rk, ["value_depthaware", "value_comparator", "difference", "ci95_low", "ci95_high"]].round(4)
    pk.loc[pk.suppressed, ["survival", "survival_ci95_low", "survival_ci95_high", "n_at_risk", "cum_events"]] = np.nan
    pk.loc[pk.n_group < L.MIN_CELL, "n_group"] = np.nan
    pk[["survival", "survival_ci95_low", "survival_ci95_high"]] = pk[["survival", "survival_ci95_low", "survival_ci95_high"]].round(4)
    public = {f"V_cindex_mean_contrasts_{DATE}.csv": mm, f"V_cindex_endpoint_contrasts_{DATE}.csv": mp_,
              f"V_cindex_leave_one_endpoint_out_{DATE}.csv": ml, f"V_hr_contrasts_{DATE}.csv": pom,
              f"V_hr_endpoint_contrasts_{DATE}.csv": op, f"V_hr_leave_one_endpoint_out_{DATE}.csv": ol, f"V_km_curves_{DATE}.csv": pk}
    for fn, df in public.items():
        ok = not any(c.lower() in L.ID_COLUMNS for c in df.columns) and L.ID_PATTERN.search(df.astype(str).to_csv(index=False)) is None
        chk("4 privacy", ok, fn)
    chk("4 suppressed cells blank", pom.loc[pom.suppressed, "difference"].isna().all() and pk.loc[pk.suppressed, "survival"].isna().all())
    ck = pd.DataFrame(CHECKS)
    L.atomic_csv(OUT / f"V_checks_{DATE}.csv", ck)
    if not ck.passed.all():
        print(ck[~ck.passed].to_string())
        sys.exit("V privacy checks failed")
    for fn, df in public.items():
        L.atomic_csv(OUT / fn, df)
    print(f"V: {len(ck)} checks pass")


if __name__ == "__main__":
    main()
