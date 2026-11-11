#!/usr/bin/env python3
"""Large-gain analysis S: per-endpoint, leave-one-endpoint-out and Kaplan-Meier tables of the Generation Scotland (GS)
analyses J, K, L, M and O against the single-clock comparators that analysis R selected. Tables only. No disease model or
Cox model is refitted and no risk is recomputed: S differences per-draw statistics that J, K, L, M and O stored with their
own pre-specified participant bootstrap. The only new computation is the Kaplan-Meier curve of comparators whose curves L
did not store (L's km_at, L's prepared deciles, L's bootstrap weights, unchanged).

Written on 8 Nov 2026 after the results of J to R were known, to compare MethylGPT
with one clock at a time. Like R, the comparator set is post hoc and is disclosed as such. Every row is reported, whichever
model it favours.

COMPARATORS (identical to R)
  Single set = each source analysis's decision comparators minus the models that combine GrimAge2 and DunedinPACE
  ("grimage2_dunedinpace_*"). Per scenario (K: setting): best_single = argmax of the scenario's primary metric over the
  single set (must equal R's choice, check 3); GrimAge2 = the better of grimage2_mean and grimage2_knn by the same metric;
  DunedinPACE likewise. Contrast labels: best_single, GrimAge2, DunedinPACE (a comparator can appear under two labels).

C-INDEX FAMILY: J (7 scenarios), M (9 scenarios), K (3 settings, statistic "mean" over the 13 primary-set diseases)
  Inputs (private, read-only): artifacts/large_gain_2026-10-28_private/J/private_bootstrap_cindex.npz,
  artifacts/large_gain_2026-11-02_private/M/boot/private_boot_<scenario>.npz,
  artifacts/large_gain_2026-10-30_private/K/private_bootstrap_cindex.npz: point C-index (model, endpoint) and the shared and
  within-fold bootstrap draws (model, endpoint, 2,000). Event counts from the committed per-endpoint tables.
  delta(model, e) = C(model, e) - C(age_sex, e). Per endpoint and comparator: delta_MethylGPT - delta_comparator, percentile
  95% interval over the shared draws, bootstrap P (J.pboot), Benjamini-Hochberg q across endpoints (M.bh) within scenario
  and contrast; each endpoint's own best within the single set. Leave one endpoint out: mean over the 7 remaining endpoints
  against best_single fixed and against the single-set best re-chosen on the 7; J's rule arithmetic for reference.
  Event-weighted mean (weights = endpoint events) against the single-set best re-chosen on that statistic. The decision rows
  stay R's; these tables are secondary.

HAZARD-RATIO FAMILY: L (hm27, full, random_90), O (9 scenarios)
  Inputs: L's and O's private per-endpoint fit files (L.load_fit, O.load_fit): point values and 2,000 shared draws of the
  log hazard ratio per SD (beta), top-versus-bottom-decile log hazard ratio, top-decile event capture and 10-year absolute
  risk in the top and bottom deciles, for every model. H = mean over the 8 endpoints of beta, with L's convention (an
  endpoint whose every fold is not estimable enters as 0). Per endpoint and as the mean over endpoints, every L metric,
  MethylGPT combined minus each comparator, with interval, bootstrap P and n_draws_estimable (L's pct and pboot); BH q
  across endpoints for log_hr_per_sd. Leave one endpoint out on H: best_single fixed and re-chosen.
  Kaplan-Meier: in L's KM scenarios (hm27, full) and O's (depth_10, depth_30, full), the released models are MethylGPT
  combined and the three comparators. O stored the curves of every decision comparator; L stored MethylGPT and GrimAge2
  with kNN fill only, so S computes the others exactly as L's task_fit does (top decile, middle 80%, bottom decile; yearly
  grid 0 to 17 years; shared weights).
  Disclosure control (L's, unchanged): counts below 5 suppressed; abs-risk cells of a released KM model use its KM release
  mask (release_km); a contrast is blank if either cell is suppressed; the private copy keeps every value.

CHECKS (S_checks; any failure stops the run before results are written)
  1. C-index family: every model's mean gain and every summary difference interval of J, K (mean) and M reproduce from the
     stored draws within 1e-12.
  2. HR family: every model's H and every summary difference interval of L and O reproduce within 1e-12.
  3. best_single equals R's best_single_comparator for every scenario.
  4. M's per-endpoint contrasts against M's own best reproduce M_endpoint_contrasts within 1e-12; O's and L's per-endpoint
     log_hr_per_sd contrasts against GrimAge2 (grimage2_knn; grimage2_mean in full) reproduce their private contrast tables
     within 1e-12.
  5. Kaplan-Meier curves recomputed by S for MethylGPT combined in L's hm27 and full equal L's stored curves within L's
     TOL_KM (1e-10), at-risk and event counts exactly.
  6. Privacy: no public file holds a participant identifier pattern or identifier column; suppressed cells are blank.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
DATE = "2026-11-08"
OUT = ART / "large_gain_2026-11-08/S"
PRIV = ART / "large_gain_2026-11-08_private/S"
COMBINED = "grimage2_dunedinpace_"
TOL = 1e-12


def load_module(name, path):
    if not path.exists():
        sys.exit(f"Cannot proceed - required data file not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


O = load_module("large_gain_O", HERE / "large_gain_O_gs_noise_risk_stratification_2026-11-08.py")
L, M, J = O.L, O.M, O.J
ENDPOINTS, OURS = J.ENDPOINTS, J.OURS
pct3, pboot_nan = L.pct, L.pboot


def need(p):
    p = Path(p)
    if not p.exists():
        sys.exit(f"Cannot proceed - required data file not found: {p}")
    return p


def pct(d, q):
    return float(np.percentile(d, q))


def rule_ref(v_m, v_b, lo):
    if not (np.isfinite(v_b) and v_b > 0):
        return "not applicable (comparator <= 0)"
    return "meets" if ((v_m - v_b) / v_b >= 0.10 and lo > 0) else "does not meet"


def bh(p):
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        q[ok] = M.bh(p[ok])
    return q


def comparators(single, metric_of):
    """{label: model} for best_single, GrimAge2, DunedinPACE (better fill)."""
    out = {"best_single": max(single, key=metric_of)}
    for lab, pre in (("GrimAge2", "grimage2_"), ("DunedinPACE", "dunedinpace_")):
        c = [m for m in single if m.startswith(pre)]
        if c:
            out[lab] = max(c, key=metric_of)
    return out


CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append({"check": name, "passed": bool(ok), "detail": str(detail)})


R = pd.read_csv(need(ART / "large_gain_2026-11-08/R/R_rule_outcome_2026-11-08.csv"))


def r_best(analysis, key):
    r = R[R.analysis == analysis]
    if analysis == "K":
        r = r[(r.setting == key) & (r.statistic == "mean")]
    else:
        r = r[r.scenario == key]
    return r.best_single_comparator.iloc[0]


# ------------------------------------------------------------------ C-index family
def cindex_sources():
    jz = np.load(need(ART / "large_gain_2026-10-28_private/J/private_bootstrap_cindex.npz"))
    js = pd.read_csv(need(ART / "large_gain_2026-10-28/J/J_summary_2026-10-28.csv"))
    ms = pd.read_csv(need(ART / "large_gain_2026-11-02/M/M_summary_2026-11-02.csv"))
    mc = pd.read_csv(need(ART / "large_gain_2026-11-02/M/M_endpoint_contrasts_2026-11-02.csv"))
    kz = np.load(need(ART / "large_gain_2026-10-30_private/K/private_bootstrap_cindex.npz"))
    ks = pd.read_csv(need(ART / "large_gain_2026-10-30/K/K_summary_2026-10-30.csv"))
    kc = pd.read_csv(need(ART / "large_gain_2026-10-30/K/K_cindex_per_disease_2026-10-30.csv"))
    ev8 = mc[mc.scenario == "full"].set_index("endpoint").loc[ENDPOINTS, "events"].to_numpy(float)
    diseases = list(kz["diseases"].astype(str))
    kev = kc.drop_duplicates("disease").set_index("disease").loc[diseases, "events"].astype(float).to_numpy()
    src = []
    for scen in sorted({k.split("__")[0] for k in jz.files}):
        src.append(("J", scen, list(jz[f"{scen}__models"].astype(str)), jz[f"{scen}__point"], jz[f"{scen}__boot"],
                    jz[f"{scen}__boot_within_fold"], js[js.scenario == scen], ENDPOINTS, ev8))
    for scen in ms.scenario.unique():
        z = np.load(need(ART / f"large_gain_2026-11-02_private/M/boot/private_boot_{scen}.npz"))
        src.append(("M", scen, list(z["models"].astype(str)), z["point"], z["boot"], z["boot_within_fold"],
                    ms[ms.scenario == scen], ENDPOINTS, ev8))
    for scen in ("full", "hm27", "random_90"):
        src.append(("K", scen, list(kz[f"{scen}__models"].astype(str)), kz[f"{scen}__point"], kz[f"{scen}__boot"], None,
                    ks[(ks.setting == scen) & (ks.statistic == "mean")], diseases, kev))
    return src, mc


def cindex_family():
    src, mc = cindex_sources()
    per, loeo, ew, mean_rows = [], [], [], []
    for a, scen, models, P, B, W, summ, eps, events in src:
        ia, io = models.index("age_sex"), models.index(OURS)
        dP, dB = P - P[ia], B - B[ia]
        Mp, MB = dP.mean(1), dB.mean(1)
        MW = (W - W[ia]).mean(1) if W is not None else None
        for r in summ.itertuples():
            k = models.index(r.model)
            check("1 C-index mean gain", abs(Mp[k] - r.M_model) < TOL, f"{a} {scen} {r.model}")
            if isinstance(getattr(r, "difference", None), float) and np.isfinite(r.difference) and r.model != OURS:
                d = MB[io] - MB[k]
                ok = abs(pct(d, 2.5) - r.ci95_low) < TOL and abs(pct(d, 97.5) - r.ci95_high) < TOL
                check("1 C-index summary interval", ok, f"{a} {scen} {r.model}")
        dec = summ.loc[summ.role == "decision comparator", "model"].tolist()
        single = [m for m in dec if not m.startswith(COMBINED)]
        cm = comparators(single, lambda m: Mp[models.index(m)])
        check("3 best_single equals R", cm["best_single"] == r_best(a, scen), f"{a} {scen} {cm['best_single']}")
        key = {"analysis": a, "scenario": scen}
        for lab, c in cm.items():
            k = models.index(c)
            d = MB[io] - MB[k]
            row = {**key, "contrast": lab, "comparator": c, "gain_methylgpt": Mp[io], "gain_comparator": Mp[k],
                   "difference": Mp[io] - Mp[k], "ci95_low": pct(d, 2.5), "ci95_high": pct(d, 97.5), "p_boot": J.pboot(d),
                   "relative_gain": (Mp[io] - Mp[k]) / Mp[k] if Mp[k] > 0 else np.nan,
                   "rule_arithmetic": rule_ref(Mp[io], Mp[k], pct(d, 2.5))}
            if MW is not None:
                dw = MW[io] - MW[k]
                row.update(ci95_low_within_fold=pct(dw, 2.5), ci95_high_within_fold=pct(dw, 97.5))
            mean_rows.append(row)
            p = [J.pboot(dB[io, j] - dB[k, j]) for j in range(len(eps))]
            q = bh(p)
            for j, ep in enumerate(eps):
                d = dB[io, j] - dB[k, j]
                per.append({**key, "endpoint": ep, "events": events[j], "contrast": lab, "comparator": c,
                            "c_methylgpt": P[io, j], "c_comparator": P[k, j], "c_age_sex": P[ia, j],
                            "delta_c_methylgpt": dP[io, j], "delta_c_comparator": dP[k, j],
                            "difference": dP[io, j] - dP[k, j], "ci95_low": pct(d, 2.5), "ci95_high": pct(d, 97.5),
                            "p_boot": p[j], "q_bh_across_endpoints": q[j]})
        si = [models.index(m) for m in single]
        own = [si[int(np.argmax(dP[si, j]))] for j in range(len(eps))]
        p_own = [J.pboot(dB[io, j] - dB[own[j], j]) for j in range(len(eps))]
        q_own = bh(p_own)
        for j, ep in enumerate(eps):
            d = dB[io, j] - dB[own[j], j]
            per.append({**key, "endpoint": ep, "events": events[j], "contrast": "own_best_single", "comparator": models[own[j]],
                        "c_methylgpt": P[io, j], "c_comparator": P[own[j], j], "c_age_sex": P[ia, j],
                        "delta_c_methylgpt": dP[io, j], "delta_c_comparator": dP[own[j], j],
                        "difference": dP[io, j] - dP[own[j], j], "ci95_low": pct(d, 2.5), "ci95_high": pct(d, 97.5),
                        "p_boot": p_own[j], "q_bh_across_endpoints": q_own[j]})
        kb = models.index(cm["best_single"])
        for j, ep in enumerate(eps):
            keep = np.arange(len(eps)) != j
            Ml, MBl = dP[:, keep].mean(1), dB[:, keep].mean(1)
            kr = models.index(max(single, key=lambda m: Ml[models.index(m)]))
            for kind, kk in (("fixed", kb), ("rechosen", kr)):
                d = MBl[io] - MBl[kk]
                loeo.append({**key, "left_out": ep, "comparator_kind": kind, "comparator": models[kk],
                             "gain_methylgpt": Ml[io], "gain_comparator": Ml[kk], "difference": Ml[io] - Ml[kk],
                             "ci95_low": pct(d, 2.5), "ci95_high": pct(d, 97.5), "p_boot": J.pboot(d),
                             "relative_gain": (Ml[io] - Ml[kk]) / Ml[kk] if Ml[kk] > 0 else np.nan,
                             "rule_arithmetic": rule_ref(Ml[io], Ml[kk], pct(d, 2.5))})
        wt = events / events.sum()
        Mw, MBw = (dP * wt).sum(1), (dB * wt[None, :, None]).sum(1)
        kw = models.index(max(single, key=lambda m: Mw[models.index(m)]))
        d = MBw[io] - MBw[kw]
        ew.append({**key, "statistic": "event-weighted mean", "comparator": models[kw], "gain_methylgpt": Mw[io],
                   "gain_comparator": Mw[kw], "difference": Mw[io] - Mw[kw], "ci95_low": pct(d, 2.5),
                   "ci95_high": pct(d, 97.5), "p_boot": J.pboot(d),
                   "relative_gain": (Mw[io] - Mw[kw]) / Mw[kw] if Mw[kw] > 0 else np.nan,
                   "rule_arithmetic": rule_ref(Mw[io], Mw[kw], pct(d, 2.5))})
        if a == "M":
            for r in mc[mc.scenario == scen].itertuples():
                j, k = ENDPOINTS.index(r.endpoint), models.index(r.best_comparator)
                d = dB[io, j] - dB[k, j]
                ok = (abs(dP[io, j] - dP[k, j] - r.difference_vs_best) < TOL and abs(pct(d, 2.5) - r.ci95_low_vs_best) < TOL
                      and abs(pct(d, 97.5) - r.ci95_high_vs_best) < TOL)
                check("4 M per-endpoint reproduction", ok, f"{scen} {r.endpoint}")
    return pd.DataFrame(mean_rows), pd.DataFrame(per), pd.DataFrame(loeo), pd.DataFrame(ew)


# ------------------------------------------------------------------ hazard-ratio family
def l_km(scen, models_needed):
    """KM arrays for models L did not store, computed as in L.task_fit."""
    T = np.load(need(L.PRIV / "private_endpoint_arrays.npz"))
    Pz = np.load(need(L.PRIV / f"private_prepared_{scen}.npz"))
    pm = list(Pz["models"].astype(str))
    Wall = np.load(need(L.PRIV / "private_boot_weights.npy"), mmap_mode="r")
    days = L.YEARS * L.YEAR
    nm, ny = len(models_needed), len(L.YEARS)
    out = {"km_S": np.full((8, nm, 3, ny), np.nan), "km_boot": np.full((8, nm, 3, L.N_BOOT, ny), np.nan),
           "km_atrisk": np.zeros((8, nm, 3, ny)), "km_cum": np.zeros((8, nm, 3, ny)), "km_total": np.zeros((8, nm, 3)),
           "km_n": np.zeros((8, nm, 3))}
    for j in range(len(ENDPOINTS)):
        ev, du = T["event"][j], T["duration"][j]
        W = np.asarray(Wall[:, T["position"][j]], dtype=np.float64)
        for a, m in enumerate(models_needed):
            g = Pz["G"][pm.index(m), j]
            for b, gv in enumerate((2, 1, 0)):
                s = g == gv
                if not s.any():
                    continue
                out["km_S"][j, a, b] = L.km_at(du[s], ev[s], days)[0]
                out["km_boot"][j, a, b] = L.km_at(du[s], ev[s], days, W=W[:, s])
                out["km_atrisk"][j, a, b] = [(du[s] >= d).sum() for d in days]
                out["km_cum"][j, a, b] = [(ev[s] & (du[s] <= d)).sum() for d in days]
                out["km_total"][j, a, b], out["km_n"][j, a, b] = ev[s].sum(), s.sum()
    return out


def hr_family():
    lsum = pd.read_csv(need(L.OUT / "L_summary_2026-10-30.csv"))
    osum = pd.read_csv(need(O.OUT / "O_summary_2026-11-08.csv"))
    lpc = pd.read_csv(need(L.PRIV / "private_contrasts_unsuppressed.csv"))
    opc = pd.read_csv(need(O.PRIV / "private_contrasts_unsuppressed.csv"))
    rows, loeo, kms = [], [], []
    for a, mod, scens, summ, pc, kmscen in (("L", L, L.SCENARIOS, lsum, lpc, L.KM_SCENARIOS),
                                            ("O", O, O.SCENARIOS, osum, opc, O.KM_SCENARIOS)):
        for scen in scens:
            F = mod.load_fit(scen)
            ms = F["models"]
            io = ms.index(OURS)
            b, bb = F["beta_pt"], F["beta_boot"]
            allnot = ~np.isfinite(b)
            Hp, Hb = np.where(allnot, 0.0, b).mean(1), np.where(allnot[..., None], 0.0, bb).mean(1)
            s_ = summ[summ.scenario == scen]
            for r in s_.itertuples():
                k = ms.index(r.model)
                d = Hb[io] - Hb[k]
                ok = (abs(Hp[k] - r.H_model) < TOL and abs(pct3(d)[0] - r.ci95_low) < TOL and abs(pct3(d)[1] - r.ci95_high) < TOL)
                check("2 HR summary reproduction", ok, f"{a} {scen} {r.model}")
            dec = s_.loc[s_.role == "decision comparator", "model"].tolist()
            single = [m for m in dec if not m.startswith(COMBINED)]
            cm = comparators(single, lambda m: Hp[ms.index(m)])
            check("3 best_single equals R", cm["best_single"] == r_best(a, scen), f"{a} {scen} {cm['best_single']}")
            # Kaplan-Meier arrays of the released models
            rel_models = list(dict.fromkeys([OURS] + list(cm.values())))
            km = None
            if scen in kmscen:
                if a == "O":
                    ci = [F["km_models"].index(m) for m in rel_models]
                    km = {k: F[k][:, ci] for k in ("km_S", "km_boot", "km_atrisk", "km_cum", "km_total", "km_n")}
                else:
                    stored = [OURS, L.grimage_ref(scen)]
                    km = l_km(scen, rel_models)
                    for a_, m in enumerate(rel_models):
                        if m in stored:
                            c_ = stored.index(m)
                            okS = np.allclose(km["km_S"][:, a_], F["km_S"][:, c_], atol=L.TOL_KM, rtol=0, equal_nan=True)
                            okB = np.allclose(km["km_boot"][:, a_], F["km_boot"][:, c_], atol=L.TOL_KM, rtol=0, equal_nan=True)
                            okC = (np.array_equal(km["km_atrisk"][:, a_], F["km_atrisk"][:, c_])
                                   and np.array_equal(km["km_cum"][:, a_], F["km_cum"][:, c_]))
                            check("5 L KM recomputation", okS and okB and okC, f"{scen} {m}")
                rel = np.stack([np.stack([L.release_km(km["km_cum"][j, a_], km["km_atrisk"][j, a_], km["km_total"][j, a_])
                                          for a_ in range(len(rel_models))]) for j in range(len(ENDPOINTS))])
                sup, why = O.suppression(F, scen, rel, rel_models)
                for a_, m in enumerate(rel_models):
                    for bi, gn in enumerate(L.GROUPS):
                        for y in range(len(L.YEARS)):
                            for j, ep in enumerate(ENDPOINTS):
                                lo, hi, _ = pct3(km["km_boot"][j, a_, bi, :, y])
                                ok = bool(rel[j, a_, bi, y])
                                kms.append({"analysis": a, "scenario": scen, "endpoint": ep, "model": m, "group": gn,
                                            "year": int(L.YEARS[y]), "n_group": km["km_n"][j, a_, bi],
                                            "survival": km["km_S"][j, a_, bi, y], "survival_ci95_low": lo,
                                            "survival_ci95_high": hi, "n_at_risk": km["km_atrisk"][j, a_, bi, y],
                                            "cum_events": km["km_cum"][j, a_, bi, y], "suppressed": not ok})
            else:
                sup, why = O.suppression(F, scen, None, None)
            vals = O.metric_values(F)
            for metric in L.METRICS:
                P, B = vals[metric]
                S = sup[metric]
                for lab, c in cm.items():
                    k = ms.index(c)
                    pl = [pboot_nan(B[io, j] - B[k, j]) for j in range(len(ENDPOINTS))]
                    ql = bh(pl) if metric == "log_hr_per_sd" else [np.nan] * len(ENDPOINTS)
                    for j, ep in enumerate(ENDPOINTS):
                        d = B[io, j] - B[k, j]
                        lo, hi, nd = pct3(d)
                        sp = bool(S[io, j] or S[k, j])
                        dp = P[io, j] - P[k, j]
                        rows.append({"analysis": a, "scenario": scen, "endpoint": ep, "metric": metric, "contrast": lab,
                                     "comparator": c, "value_methylgpt": P[io, j], "value_comparator": P[k, j],
                                     "difference": dp, "ci95_low": lo, "ci95_high": hi, "p_boot": pl[j],
                                     "q_bh_across_endpoints": ql[j], "n_draws_estimable": nd,
                                     "relative_gain": dp / P[k, j] if P[k, j] > 0 else np.nan, "suppressed": sp,
                                     "reason": "a cell of the contrast is suppressed" if sp else ""})
                    if metric == "log_hr_per_sd":
                        vo, vc, d = Hp[io], Hp[k], Hb[io] - Hb[k]
                        nsup = 0
                    else:
                        vo, vc, d = P[io].mean(), P[k].mean(), B[io].mean(0) - B[k].mean(0)
                        nsup = int((S[io] | S[k]).sum())
                    lo, hi, nd = pct3(d)
                    rows.append({"analysis": a, "scenario": scen, "endpoint": "mean_over_endpoints", "metric": metric,
                                 "contrast": lab, "comparator": c, "value_methylgpt": vo, "value_comparator": vc,
                                 "difference": vo - vc, "ci95_low": lo, "ci95_high": hi, "p_boot": pboot_nan(d),
                                 "n_draws_estimable": nd, "relative_gain": (vo - vc) / vc if vc > 0 else np.nan,
                                 "rule_arithmetic": rule_ref(vo, vc, lo) if metric == "log_hr_per_sd" else "",
                                 "suppressed": nsup > 0, "reason": f"{nsup} of 8 per-endpoint contrasts suppressed" if nsup else ""})
            # reproduction of the stored per-endpoint contrast against GrimAge2 (L's and O's grimage2 reference)
            gref = L.grimage_ref(scen)
            kg = ms.index(gref)
            vb = vals["log_hr_per_sd"][1]
            ref = pc[(pc.scenario == scen) & (pc.metric == "log_hr_per_sd") & (pc.contrast == "grimage2_knn")
                     & (pc.endpoint != "mean_over_endpoints")]
            for r in ref.itertuples():
                j = ENDPOINTS.index(r.endpoint)
                lo, hi, _ = pct3(vb[io, j] - vb[kg, j])
                ok = (r.comparator == gref and abs(lo - r.ci95_low) < TOL and abs(hi - r.ci95_high) < TOL)
                check("4 HR per-endpoint reproduction", ok, f"{a} {scen} {r.endpoint}")
            check("4 HR per-endpoint reproduction rows present", len(ref) == len(ENDPOINTS), f"{a} {scen} n={len(ref)}")
            kb = ms.index(cm["best_single"])
            for j, ep in enumerate(ENDPOINTS):
                keep = np.arange(len(ENDPOINTS)) != j
                bl = np.where(allnot, 0.0, b)[:, keep].mean(1)
                bbl = np.where(allnot[..., None], 0.0, bb)[:, keep].mean(1)
                kr = ms.index(max(single, key=lambda m: bl[ms.index(m)]))
                for kind, kk in (("fixed", kb), ("rechosen", kr)):
                    d = bbl[io] - bbl[kk]
                    lo, hi, _ = pct3(d)
                    loeo.append({"analysis": a, "scenario": scen, "left_out": ep, "comparator_kind": kind,
                                 "comparator": ms[kk], "H_methylgpt": bl[io], "H_comparator": bl[kk],
                                 "difference": bl[io] - bl[kk], "ci95_low": lo, "ci95_high": hi, "p_boot": pboot_nan(d),
                                 "relative_gain": (bl[io] - bl[kk]) / bl[kk] if bl[kk] > 0 else np.nan,
                                 "rule_arithmetic": rule_ref(bl[io], bl[kk], lo)})
    return pd.DataFrame(rows), pd.DataFrame(loeo), pd.DataFrame(kms)


def privacy_ok(df):
    if any(c.lower() in L.ID_COLUMNS for c in df.columns):
        return False
    txt = df.astype(str).to_csv(index=False)
    return L.ID_PATTERN.search(txt) is None


def main():
    cmean, cper, cloeo, cew = cindex_family()
    hrows, hloeo, kdf = hr_family()
    ck = pd.DataFrame(CHECKS)
    OUT.mkdir(parents=True, exist_ok=True)
    PRIV.mkdir(parents=True, exist_ok=True)
    if not ck.passed.all():
        L.atomic_csv(OUT / f"S_checks_{DATE}.csv", ck)
        print(ck[~ck.passed].to_string())
        sys.exit("S checks failed; no result table written")
    L.atomic_csv(PRIV / "private_hr_contrasts_unsuppressed.csv", hrows)
    L.atomic_csv(PRIV / "private_km_unsuppressed.csv", kdf)
    ph, pk = hrows.copy(), kdf.copy()
    ph.loc[ph.suppressed, ["value_methylgpt", "value_comparator", "difference", "ci95_low", "ci95_high", "p_boot",
                           "q_bh_across_endpoints", "relative_gain"]] = np.nan
    rk = ph.metric.str.startswith("abs_risk")
    ph.loc[rk, ["value_methylgpt", "value_comparator", "difference", "ci95_low", "ci95_high"]] = \
        ph.loc[rk, ["value_methylgpt", "value_comparator", "difference", "ci95_low", "ci95_high"]].round(4)
    ph["hr_ratio"] = np.where(ph.metric.str.startswith("log_hr"), np.exp(ph.difference.to_numpy(float)), np.nan)
    pk.loc[pk.suppressed, ["survival", "survival_ci95_low", "survival_ci95_high", "n_at_risk", "cum_events"]] = np.nan
    pk.loc[pk.n_group < L.MIN_CELL, "n_group"] = np.nan
    pk[["survival", "survival_ci95_low", "survival_ci95_high"]] = pk[["survival", "survival_ci95_low", "survival_ci95_high"]].round(4)
    public = {f"S_cindex_mean_contrasts_{DATE}.csv": cmean, f"S_cindex_endpoint_contrasts_{DATE}.csv": cper,
              f"S_cindex_leave_one_endpoint_out_{DATE}.csv": cloeo, f"S_cindex_event_weighted_{DATE}.csv": cew,
              f"S_hr_contrasts_{DATE}.csv": ph, f"S_hr_leave_one_endpoint_out_{DATE}.csv": hloeo,
              f"S_km_curves_{DATE}.csv": pk}
    for fn, df in public.items():
        check("6 privacy", privacy_ok(df), fn)
    supp_ok = ph.loc[ph.suppressed, "difference"].isna().all() and pk.loc[pk.suppressed, "survival"].isna().all()
    check("6 suppressed cells blank", supp_ok)
    ck = pd.DataFrame(CHECKS)
    L.atomic_csv(OUT / f"S_checks_{DATE}.csv", ck)
    if not ck.passed.all():
        print(ck[~ck.passed].to_string())
        sys.exit("S privacy checks failed; no result table written")
    for fn, df in public.items():
        L.atomic_csv(OUT / fn, df)
    res = {"analysis": "S", "written": "2026-11-08, after J to R results were known (post hoc single-clock comparator set)",
           "n_checks": int(len(ck)), "checks_passed": True}
    L.write_json(OUT / f"S_results_{DATE}.json", res)
    print(f"S: {len(ck)} checks pass; tables written to {OUT}")


if __name__ == "__main__":
    main()
