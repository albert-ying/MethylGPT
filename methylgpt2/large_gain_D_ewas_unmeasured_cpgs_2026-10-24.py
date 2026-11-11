#!/usr/bin/env python3
"""Large-gain analysis D: disease EWAS at CpGs that a reduced array does not measure (Generation Scotland).

Written on 24 Oct 2026 before running (large-gain analysis plan of 23 Oct 2026, analysis D). Every result is
reported, whichever method it favours.

Data and inclusion
  Generation Scotland (GS) development participants: the 9,866 training-subset participants of the locked
  incident-disease protocol (16,444-participant analysis set, none of the seven disease categories at baseline).
  Methylation: the frame-cache matrix of this protocol (COMMON_SAMPLE_SPLIT_full18859_common_task_a_incident_positive_
  duration_20260629_train_..._matrix.npz; beta values as released, 49,156 vocabulary columns, EPIC), as read by
  extract_gs_features_2026-10-01.py. Every GS CpG is observed in every participant (43,899 vocabulary CpGs); 5,257
  vocabulary CpGs are absent from the release.
  Endpoint: incident type 2 diabetes, which is the endocrine and metabolic disease endpoint of the locked protocol (its
  only constituent disease is Type_2_Diabetes; Methods, "Disease prediction in Generation Scotland"). Event, follow-up
  (days from the baseline month to first diagnosis, death or 1 October 2023), age (years) and sex (1 = female) come from
  the locked endpoint table artifacts/gs_cox/mgpt2_combined_2026-10-01/private_nested_predictions.parquet (endpoint
  Endocrine_and_Metabolic_Diseases, age_adjustment "linear"), cross-checked against the frame-cache labels. All 9,866
  participants enter; nobody is excluded.
  Smoking pack-years: absent from the GS release on Chronos (meta.csv, 2024-06-03_phenotypes.csv, the members of
  gs.tar.1.gz and the frame-cache labels were searched; large_gain_E_bmi_smoking_covariates_2026-10-24.py records the
  search). The smoking EWAS is therefore not run, and the README states "Cannot proceed - required data file not found".
Freezing of the analysis set
  --stage freeze writes the analysis-set table (aggregate counts, SHA-256 of the sorted participant IDs and of the
  observed-value pattern; no identifiers) and the per-CpG set table. Both are committed before any imputation or fit.

Settings (pre-specified; 27K primary)
  hm27   keep the vocabulary CpGs of the 27K array (artifacts/platform_vocab_masks_2026-10-01.npz "hm27": the 22,231 the
         27K array shares with EPIC; 21,577 are observed in GS). Target CpGs = GS-observed vocabulary CpGs that are not
         kept (22,322).
  hm450  keep "hm450" (47,826; 43,767 observed in GS). Target CpGs: 132.
  Each profile is reduced to its kept CpGs (all other vocabulary CpGs set to unobserved) and then filled.

Versions of the target CpGs
  full                measured EPIC values (the reference).
  MethylGPT           released bundle artifacts/release/methylgpt2-1.1 (checksums verified against
                      package/methylgpt2-1.1.checksums.sha256), package API MethylGPT.impute(reduced betas) with the
                      default method "factor": posterior factor scores from the kept CpGs and the reconstruction
                      mu + V s clipped to [0, 1]. In the reconstruction benchmark (heldout_platform_2026-10-01.json)
                      this reconstruction and the encoder's prediction have the same MSE to 5 decimals.
  mean filling        training-split CpG means (core mu of the bundle; the cpg_mean comparator of the reconstruction
                      benchmark). Constant across participants, so its per-CpG z is undefined.
  nearest neighbours  the kNN comparator of the reconstruction benchmark (knn_predict of
                      evaluate_heldout_platform_2026-10-01.py, called as knn_predict_lowmem of
                      si_reconstruction_diagnostics_2026-10-18.py): k = 10; reference = all 145,700 corpus
                      training-split profiles (float16; unobserved values filled with core mu); squared Euclidean
                      distance over the query's kept CpGs (TF32 for the search only); prediction = mean of the
                      neighbours' observed values (mean-filled value if none is observed). Query batch 1,024 and
                      reference chunk 8,192 change memory use only.

EWAS estimator (pre-specified)
  Per CpG, a Cox proportional hazards model of incident type 2 diabetes on the CpG value, age and sex (Breslow ties),
  fitted by maximum partial likelihood with batched Newton-Raphson on the GPU in float64. The CpG value, age and sex are
  standardized to mean 0 and SD 1 over the participants (weighted means and SDs in bootstrap draws). Start at 0, step
  halving when the log partial likelihood decreases, stop when the largest Newton step is below 1e-9 or after 50
  iterations. A CpG is not converged if its last step exceeds 1e-6 or its estimate is not finite; its z is then
  undefined (count reported). beta = log hazard ratio per SD of the CpG in that version; se from the inverse observed
  information; Wald z = beta / se; two-sided normal p. CpGs with SD < 1e-8 in a version have undefined z.
  Checks (reported): agreement with statsmodels PHReg(ties="breslow") for 30 CpGs; bootstrap weights against row
  duplication for one draw.

Primary metric
  Spearman correlation over the target CpGs between the per-CpG z of an imputed version and the z of the full data.
  Undefined z are set to 0 (no association) in every version; if every z of a version is undefined (mean filling), its
  correlation is 0 by rule.
Comparators: mean filling (correlation 0 by rule) and nearest neighbours. The best comparator is the comparator with the
  higher correlation, so nearest neighbours unless its correlation is <= 0. Both comparators are reported.
Decision rule (plan, "Decision rule": a result goes into a main figure only if it beats the best comparator in that
  panel by at least 10% on its pre-specified primary metric, and the 95% interval of the difference excludes zero;
  agreement metrics use the relative improvement): relative gain = (rho_MethylGPT - rho_best) / rho_best. Pass if the
  relative gain is >= 0.10 and the 95% interval of rho_MethylGPT - rho_best excludes zero (lower bound > 0); otherwise
  fail. The rule is applied to hm27 (primary; candidate Fig. 1 or Fig. 6 sub-panel) and to hm450 (second pre-specified
  setting) separately. If rho_best <= 0, the relative gain is undefined and the rule is reported as not applicable.
Interval: 1,000 bootstrap resamples of the 9,866 participants with replacement (numpy default_rng(20261024)), entered as
  frequency weights in every Cox fit. In each draw the full-data, MethylGPT and nearest-neighbour EWAS are refitted at the
  target CpGs and the correlations recomputed. Imputed values are not recomputed: each MethylGPT imputation uses only
  that participant's own CpGs and the fixed bundle, and nearest neighbours use the fixed external reference, so neither
  depends on the other participants. Percentile 95% intervals; bootstrap P = min(1, 2(k + 1)/(B + 1)) as in the Methods.

Secondary metrics
  (a) Recall of full-data hits. Hits: Benjamini-Hochberg q < 0.05 over the target CpGs of the setting. Recall = fraction
      of full-data hits that are hits of the imputed version with the same sign of z (mean filling: 0). Also the number
      of hits per version, precision (fraction of imputed-version hits that are full-data hits with the same sign) and
      bootstrap intervals from the same draws. Recall is undefined when the full data have no hit.
  (b) Pearson correlation of z, per version, with bootstrap intervals.
  (c) Relative gain against nearest neighbours: bootstrap interval of (rho_MethylGPT - rho_kNN) / rho_kNN.
  (d) Imputation accuracy at the target CpGs (MSE, MAE, pooled within-CpG R^2, median per-CpG Pearson r with the
      measured values) for MethylGPT, nearest neighbours and mean filling (descriptive).
  (e) Full-data EWAS over all 43,899 GS CpGs: number of BH q < 0.05 hits over all of them (descriptive).
Seeds: bootstrap 20261024; check CpGs default_rng(20261024). Resamples: 1,000.

Stages: --stage freeze (CPU), then commit; --stage impute (GPU); --stage fit (GPU); --stage analyze (CPU).
Outputs (aggregate and per-CpG summary statistics only): artifacts/large_gain_2026-10-24/D/ with README.md.
Participant-level intermediates (imputed values): artifacts/large_gain_2026-10-24_private/D/ (git-ignored, never
committed, stay on Chronos).
Run: python3 large_gain_D_ewas_unmeasured_cpgs_2026-10-24.py --stage {freeze,impute,fit,analyze} [--device cuda:1]
Amendment 24 Oct 2026, after the results (documentation only; no analysis change): the README gains a Notes section
  recording the run settings (fit with --batch 12000 after a first fit run with the default 2,048 was stopped after one
  draw for speed, with identical point estimates; NUMPY_MADVISE_HUGEPAGE=0 because huge-page requests stalled numpy
  allocations on Chronos; fit and analyze stages) and how to read the bootstrap intervals of the hit-based secondary metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from methylgpt2_lib import ARTIFACTS, Corpus, log, require, write_json

HERE = Path(__file__).resolve().parent
OUT = ARTIFACTS / "large_gain_2026-10-24/D"
PRIV = ARTIFACTS / "large_gain_2026-10-24_private/D"
CACHE = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/task_a_private_training_20260629/_frame_cache")
SPLIT = "full18859_common_task_a_incident_positive_duration_20260629"
ENDPOINT_TABLE = ARTIFACTS / "gs_cox/mgpt2_combined_2026-10-01/private_nested_predictions.parquet"
ENDPOINT = "Endocrine_and_Metabolic_Diseases"
MASKS = ARTIFACTS / "platform_vocab_masks_2026-10-01.npz"
BUNDLE = ARTIFACTS / "release/methylgpt2-1.1"
CHECKSUMS = HERE / "package/methylgpt2-1.1.checksums.sha256"
SETTINGS = ("hm27", "hm450")
PRIMARY = "hm27"
SEED = 20261024
N_BOOT = 1000
FDR = 0.05
COMPARATORS = ("mean filling", "nearest neighbours")
RULE = ("Pass if (rho_MethylGPT - rho_best) / rho_best >= 0.10 and the 95% bootstrap interval of rho_MethylGPT - rho_best "
        "excludes zero (lower bound > 0); rho = Spearman correlation of per-CpG Cox z with the full-data z at the target CpGs")


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def bh(p):
    """Benjamini-Hochberg q-values; NaN p stay NaN and are not counted."""
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.any():
        pp = p[ok]
        m = len(pp)
        o = np.argsort(pp)
        qq = np.empty(m)
        qq[o] = np.minimum.accumulate((pp[o] * m / np.arange(1, m + 1))[::-1])[::-1]
        q[ok] = np.minimum(qq, 1)
    return q


def boot_p(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    k = min((d <= 0).sum(), (d >= 0).sum())
    return float(min(1.0, 2 * (k + 1) / (len(d) + 1)))


def pct(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return [None, None]
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [float(lo), float(hi)]


# ---------------------------------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------------------------------
def load_gs():
    """Development participants: ids, beta matrix (n, 49,156), event, duration (days), age, sex (1 = female)."""
    mpath = require(next(CACHE.glob(f"COMMON_SAMPLE_SPLIT_{SPLIT}_train_all_gs_batches_all_probes_*.matrix.npz")))
    lpath = require(mpath.with_name(mpath.name.replace(".matrix.npz", ".labels.parquet")))
    lab = pd.read_parquet(lpath)
    ids = lab.dnam_id.astype(str).to_numpy()
    with np.load(mpath, allow_pickle=False) as z:
        X = z["matrix"].astype(np.float32)
    if X.shape != (len(ids), 49156):
        raise ValueError(f"unexpected GS matrix shape {X.shape}")
    ep = pd.read_parquet(require(ENDPOINT_TABLE))
    ep = ep[(ep.age_adjustment == "linear") & (ep.endpoint == ENDPOINT)].copy()
    ep["dnam_id"] = ep.dnam_id.astype(str)
    if len(ep) != len(ids) or ep.dnam_id.nunique() != len(ids) or set(ep.dnam_id) != set(ids):
        raise ValueError("endpoint table and frame-cache participants differ")
    ep = ep.set_index("dnam_id").loc[ids]
    checks = {"age_equal_to_frame_cache_labels": bool(np.array_equal(ep.age.to_numpy(float), lab.age.to_numpy(float))),
              "sex_equal_to_frame_cache_labels (1 = F)": bool(np.array_equal(ep.sex.to_numpy(float), (lab.sex.astype(str) == "F").to_numpy(float)))}
    if not all(checks.values()):
        raise ValueError(f"cross-check failed: {checks}")
    ev, du = ep.event.to_numpy(float), ep.duration.to_numpy(float)
    if not np.isin(ev, (0, 1)).all() or (du <= 0).any():
        raise ValueError("events must be 0/1 and durations positive")
    return {"ids": ids, "X": X, "event": ev, "duration": du, "age": ep.age.to_numpy(float), "sex": ep.sex.to_numpy(float),
            "matrix_path": str(mpath), "labels_path": str(lpath), "checks": checks}


def load_sets():
    s = pd.read_csv(require(OUT / "D_cpg_sets.csv.gz"))
    return s


# ---------------------------------------------------------------------------------------------------------------------
# stage 1: freeze
# ---------------------------------------------------------------------------------------------------------------------
def stage_freeze(a):
    OUT.mkdir(parents=True, exist_ok=True)
    g = load_gs()
    X = g["X"]
    M = np.load(require(MASKS))
    vocab = M["vocab"].astype(str)
    fin = np.isfinite(X)
    obs_all, obs_any = fin.all(0), fin.any(0)
    if not np.array_equal(obs_all, obs_any):
        raise ValueError("some GS CpGs are observed in only part of the participants; the frozen design assumes none")
    sets = pd.DataFrame({"cpg_id": vocab, "observed_in_gs": obs_all})
    for s in SETTINGS:
        sets[f"in_{s}_mask"] = M[s].astype(bool)
        sets[f"kept_{s}"] = M[s].astype(bool) & obs_all
        sets[f"target_{s}"] = obs_all & ~M[s].astype(bool)
    sets.to_csv(OUT / "D_cpg_sets.csv.gz", index=False)
    ids = np.sort(g["ids"])
    ev, du = g["event"], g["duration"]
    base = {"n_participants": int(len(ids)), "n_events_incident_t2d": int(ev.sum()),
            "n_female": int(g["sex"].sum()), "n_male": int((1 - g["sex"]).sum()),
            "age_mean": float(g["age"].mean()), "age_sd": float(g["age"].std(ddof=1)),
            "followup_days_median": float(np.median(du)), "followup_days_q25": float(np.percentile(du, 25)),
            "followup_days_q75": float(np.percentile(du, 75)),
            "n_vocabulary_cpgs": int(len(vocab)), "n_cpgs_observed_in_gs": int(obs_all.sum()),
            "sha256_sorted_participant_ids": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            "sha256_observed_pattern": hashlib.sha256(np.packbits(fin).tobytes()).hexdigest()}
    rows = []
    for s in SETTINGS:
        rows.append({"setting": s, "role": "primary" if s == PRIMARY else "second pre-specified setting",
                     **base, "n_cpgs_in_array_mask": int(sets[f"in_{s}_mask"].sum()), "n_kept_cpgs": int(sets[f"kept_{s}"].sum()),
                     "n_target_cpgs": int(sets[f"target_{s}"].sum())})
    pd.DataFrame(rows).to_csv(OUT / "D_frozen_analysis_set.csv", index=False)
    write_json(OUT / "D_frozen_analysis_set.json", {
        "script": Path(__file__).name, "written": time.strftime("%Y-%m-%d %H:%M:%S"),
        "inputs": {"matrix": g["matrix_path"], "matrix_mtime": time.ctime(Path(g["matrix_path"]).stat().st_mtime),
                   "labels": g["labels_path"], "endpoint_table": str(ENDPOINT_TABLE),
                   "endpoint_table_sha256": sha256_file(ENDPOINT_TABLE), "endpoint": ENDPOINT, "platform_masks": str(MASKS),
                   "platform_masks_sha256": sha256_file(MASKS)},
        "checks": g["checks"], "analysis_set": base, "settings": rows,
        "smoking_pack_years": "Cannot proceed - required data file not found: no smoking variable in the GS release files on Chronos "
                              "(see large_gain_E_bmi_smoking_covariates_2026-10-24.py and E/E_data_availability.csv)"})
    log(f"frozen: {base['n_participants']} participants, {base['n_events_incident_t2d']} incident T2D events; "
        + "; ".join(f"{r['setting']}: kept {r['n_kept_cpgs']}, targets {r['n_target_cpgs']}" for r in rows))


# ---------------------------------------------------------------------------------------------------------------------
# stage 2: imputation (GPU)
# ---------------------------------------------------------------------------------------------------------------------
def accuracy(pred, truth):
    """Imputation accuracy at target CpGs: pred and truth (n, m) float32; pred may be a constant row (mean filling)."""
    pred = np.broadcast_to(pred, truth.shape)
    sse = np.zeros(truth.shape[1]); sae = np.zeros(truth.shape[1]); sst = np.zeros(truth.shape[1]); r = np.full(truth.shape[1], np.nan)
    for j0 in range(0, truth.shape[1], 2048):
        y = truth[:, j0:j0 + 2048].astype(np.float64); p = np.asarray(pred[:, j0:j0 + 2048], np.float64)
        e = p - y
        sse[j0:j0 + 2048] = (e ** 2).sum(0); sae[j0:j0 + 2048] = np.abs(e).sum(0)
        yc = y - y.mean(0); pc = p - p.mean(0)
        sst[j0:j0 + 2048] = (yc ** 2).sum(0)
        den = np.sqrt((yc ** 2).sum(0) * (pc ** 2).sum(0))
        with np.errstate(invalid="ignore", divide="ignore"):
            r[j0:j0 + 2048] = np.where(den > 0, (yc * pc).sum(0) / den, np.nan)
    N = truth.size
    keep = sst > 1e-12
    return {"mse": float(sse.sum() / N), "mae": float(sae.sum() / N),
            "within_cpg_r2_pooled": float(1 - sse[keep].sum() / sst[keep].sum()),
            "median_per_cpg_pearson_r": float(np.nanmedian(r)) if np.isfinite(r).any() else None,
            "n_cpgs_with_defined_r": int(np.isfinite(r).sum()), "n_values": int(N)}


def stage_impute(a):
    import torch
    from methylgpt2_lib import cuda_preinit
    cuda_preinit()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device(a.device)
    t0 = time.time()
    PRIV.mkdir(parents=True, exist_ok=True)
    want = dict(reversed(l.split()) for l in require(CHECKSUMS).read_text().splitlines() if l.strip())
    bad = sorted(f for f in want if sha256_file(BUNDLE / f) != want[f])
    if bad:
        sys.exit(f"release bundle checksum mismatch: {bad}")
    log("release bundle checksums verified")
    g = load_gs()
    X = g["X"]
    sets = load_sets()
    vocab = sets.cpg_id.astype(str).to_numpy()
    info = {"bundle_checksums_verified": sorted(want), "settings": {}}
    sys.path.insert(0, str(HERE / "package/src"))
    from methylgpt2 import MethylGPT
    model = MethylGPT.from_pretrained(str(BUNDLE), device=str(dev))
    if model.vocab != list(vocab):
        raise ValueError("bundle vocabulary differs from the platform-mask vocabulary")
    mu = model.core.mu.cpu().numpy().astype(np.float32)
    mg = {}
    for s in SETTINGS:
        keep, tgt = sets[f"kept_{s}"].to_numpy(bool), sets[f"target_{s}"].to_numpy(bool)
        Xr = np.where(keep[None, :], X, np.nan).astype(np.float32)
        t1 = time.time()
        imp = model.impute(pd.DataFrame(Xr, columns=model.vocab), batch_size=8).to_numpy(np.float32)
        if not np.array_equal(imp[:, keep], X[:, keep]):
            raise ValueError("impute() changed observed values")
        mg[s] = np.ascontiguousarray(imp[:, tgt])
        info["settings"][s] = {"methylgpt_seconds": round(time.time() - t1, 1), "n_kept": int(keep.sum()), "n_targets": int(tgt.sum()),
                               "methylgpt_identical_to_measured_fraction": float((mg[s] == X[:, tgt]).mean())}
        log(f"{s}: MethylGPT imputation of {tgt.sum()} target CpGs from {keep.sum()} kept ({time.time() - t1:.0f} s)")
        del Xr, imp
    del model
    torch.cuda.empty_cache()

    spec = importlib.util.spec_from_file_location("recon_diag", HERE / "si_reconstruction_diagnostics_2026-10-18.py")
    rd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rd)              # knn_predict_lowmem (the reconstruction benchmark's kNN)
    free, total = torch.cuda.mem_get_info(dev)
    log(f"GPU free {free / 2**30:.1f} of {total / 2**30:.1f} GiB before loading the kNN reference")
    corpus = Corpus()
    train_rows = np.sort(corpus.rows("train"))
    Rraw = torch.empty((len(train_rows), X.shape[1]), dtype=torch.float16, device=dev)
    for i in range(0, len(train_rows), 8192):
        Rraw[i:i + 8192] = torch.from_numpy(corpus.beta[train_rows[i:i + 8192]]).to(dev)
    mu_h = torch.as_tensor(mu, device=dev).half()
    log(f"kNN reference loaded: {len(train_rows)} training-split profiles")
    Xt = torch.from_numpy(X).to(dev)
    obs = torch.isfinite(Xt)
    kn = {}
    for s in SETTINGS:
        keep_t = torch.from_numpy(sets[f"kept_{s}"].to_numpy(bool)).to(dev)
        ctx, masked = obs & keep_t[None, :], obs & ~keep_t[None, :]
        t1 = time.time()
        pr = rd.knn_predict_lowmem(Rraw, mu_h, Xt, ctx, masked, k=10, batch=1024, chunk=8192)
        kn[s] = np.stack(pr).astype(np.float32)
        tgt = sets[f"target_{s}"].to_numpy(bool)
        if kn[s].shape != (len(X), int(tgt.sum())):
            raise ValueError(f"kNN output shape {kn[s].shape}")
        info["settings"][s]["knn_seconds"] = round(time.time() - t1, 1)
        log(f"{s}: nearest-neighbour imputation done ({time.time() - t1:.0f} s)")
        del ctx, masked
    del Rraw, Xt, obs
    torch.cuda.empty_cache()

    acc = []
    for s in SETTINGS:
        tgt = sets[f"target_{s}"].to_numpy(bool)
        truth = np.ascontiguousarray(X[:, tgt])
        np.savez(PRIV / f"imputed_{s}.npz", methylgpt=mg[s], nearest_neighbours=kn[s], mean_filling=mu[tgt],
                 target_idx=np.flatnonzero(tgt), sha256_sorted_participant_ids=hashlib.sha256("\n".join(np.sort(g["ids"])).encode()).hexdigest())
        for name, p in (("MethylGPT", mg[s]), ("nearest neighbours", kn[s]), ("mean filling", mu[tgt][None, :])):
            acc.append({"setting": s, "version": name, "n_participants": len(X), "n_target_cpgs": int(tgt.sum()), **accuracy(p, truth)})
            log(f"{s} {name}: MSE {acc[-1]['mse']:.5f}, within-CpG R2 {acc[-1]['within_cpg_r2_pooled']:.3f}")
    pd.DataFrame(acc).to_csv(OUT / "D_imputation_accuracy.csv", index=False)
    info["elapsed_s"] = round(time.time() - t0, 1)
    write_json(OUT / "D_impute_record.json", info)


# ---------------------------------------------------------------------------------------------------------------------
# batched per-CpG Cox models
# ---------------------------------------------------------------------------------------------------------------------
class CoxBatch:
    """Cox models of (CpG, age, sex) for many CpGs at once; Breslow ties; float64 on the GPU; frequency weights."""

    def __init__(self, duration, event, age, sex, dev):
        import torch
        self.t = torch
        self.dev = dev
        self.order = np.argsort(-np.asarray(duration, float), kind="stable")
        neg = -np.asarray(duration, float)[self.order]
        end = np.searchsorted(neg, neg, side="right") - 1          # last row of each tie block: risk set = rows 0..end
        evpos = np.flatnonzero(np.asarray(event)[self.order] > 0)
        self.ev = torch.as_tensor(evpos, device=dev)
        self.g = torch.as_tensor(end[evpos], device=dev)
        self.age = torch.as_tensor(np.asarray(age, float)[self.order], dtype=torch.float64, device=dev)
        self.sex = torch.as_tensor(np.asarray(sex, float)[self.order], dtype=torch.float64, device=dev)
        self.n = len(neg)

    def sort(self, M):
        """Rows of a participant-ordered matrix in this object's time order, as a GPU tensor (float32 kept)."""
        return self.t.as_tensor(np.ascontiguousarray(M[self.order]), device=self.dev)

    def weights(self, w=None):
        w = np.ones(self.n) if w is None else np.asarray(w, float)
        return self.t.as_tensor(w[self.order], dtype=self.t.float64, device=self.dev)

    def _std(self, v, w, sw):
        t = self.t
        m = (w[:, None] * v).sum(0) / sw
        sd = t.sqrt((w[:, None] * (v - m) ** 2).sum(0) / (sw - 1))
        return m, sd

    def _eval(self, X, Z1, Z2, ev, g, we, th, grad=True):
        t = self.t
        eta = X * th[:, 0] + Z1[:, None] * th[:, 1] + Z2[:, None] * th[:, 2]
        c = eta.max(0).values
        r = t.exp(eta - c)
        r = r * self._w[:, None]
        S0 = t.cumsum(r, 0)[g]
        ll = (we[:, None] * (eta[ev] - c - t.log(S0))).sum(0)
        if not grad:
            return ll
        zs = (X, Z1[:, None], Z2[:, None])
        S1 = [t.cumsum(r * z, 0)[g] / S0 for z in zs]                      # risk-set means
        U = t.stack([(we[:, None] * (zs[k][ev] - S1[k])).sum(0) for k in range(3)], 1)
        I = t.empty((X.shape[1], 3, 3), dtype=t.float64, device=X.device)
        for k in range(3):
            for l in range(k, 3):
                S2 = t.cumsum(r * zs[k] * zs[l], 0)[g] / S0
                v = (we[:, None] * (S2 - S1[k] * S1[l])).sum(0)
                I[:, k, l] = v
                I[:, l, k] = v
        return ll, U, I

    def fit(self, Xs, w, max_iter=50, tol=1e-9, halvings=12):
        """Xs: (n, m) tensor in time order; w: (n,) float64 weights in time order.
        Returns numpy beta (log HR per SD), se, z, converged, sd of the CpG."""
        t = self.t
        self._w = w
        sw = w.sum()
        X = Xs.to(t.float64)
        mx, sd = self._std(X, w, sw)
        ok = sd > 1e-8
        X = (X - mx) / t.where(ok, sd, t.ones_like(sd))
        ma, sa = self._std(self.age[:, None], w, sw)
        ms, ss = self._std(self.sex[:, None], w, sw)
        Z1, Z2 = (self.age - ma) / sa, (self.sex - ms) / ss
        act = w[self.ev] > 0
        ev, g = self.ev[act], self.g[act]
        we = w[ev]
        m = X.shape[1]
        th = t.zeros((m, 3), dtype=t.float64, device=X.device)
        ll, U, I = self._eval(X, Z1, Z2, ev, g, we, th)
        last = t.full((m,), float("inf"), dtype=t.float64, device=X.device)
        it = 0
        for it in range(1, max_iter + 1):
            delta, info = t.linalg.solve_ex(I, U)
            delta = t.where((info == 0)[:, None] & t.isfinite(delta), delta, t.zeros_like(delta))
            step = t.ones(m, dtype=t.float64, device=X.device)
            for _ in range(halvings):
                ll_new = self._eval(X, Z1, Z2, ev, g, we, th + step[:, None] * delta, grad=False)
                bad = ~t.isfinite(ll_new) | (ll_new < ll - 1e-12 * ll.abs())
                if not bool(bad.any()):
                    break
                step = t.where(bad, step / 2, step)
            th = th + step[:, None] * delta
            last = (step[:, None] * delta).abs().max(1).values
            ll, U, I = self._eval(X, Z1, Z2, ev, g, we, th)
            if float(last.max()) < tol:
                break
        cov, info = t.linalg.inv_ex(I)
        se = t.sqrt(cov[:, 0, 0])
        beta = th[:, 0]
        z = beta / se
        conv = ok & (info == 0) & (last < 1e-6) & t.isfinite(z) & t.isfinite(se)
        nan = t.full_like(z, float("nan"))
        self.last_iterations = it
        return (t.where(conv, beta, nan).cpu().numpy(), t.where(conv, se, nan).cpu().numpy(), t.where(conv, z, nan).cpu().numpy(),
                conv.cpu().numpy(), sd.cpu().numpy())

    def fit_all(self, M, w, batch):
        """M: (n, P) tensor in time order (float32); fits every column in batches."""
        out = [self.fit(M[:, j:j + batch], w) for j in range(0, M.shape[1], batch)]
        return tuple(np.concatenate([o[k] for o in out]) for k in range(5))


def zfill(z):
    z = np.asarray(z, float)
    return np.where(np.isfinite(z), z, 0.0)


def compare(z_imp, z_full):
    """Primary and secondary agreement of an imputed version's z with the full-data z at the target CpGs."""
    from scipy.stats import pearsonr, spearmanr
    zi, zf = zfill(z_imp), zfill(z_full)
    allundef = not np.isfinite(np.asarray(z_imp, float)).any()
    if allundef or np.all(zi == zi[0]):
        rho, r = 0.0, 0.0
    else:
        rho, r = float(spearmanr(zi, zf)[0]), float(pearsonr(zi, zf)[0])
    return rho, r


def hits(z):
    from scipy.stats import norm
    z = np.asarray(z, float)
    p = np.where(np.isfinite(z), 2 * norm.sf(np.abs(z)), np.nan)
    q = bh(p)
    return np.isfinite(q) & (q < FDR), p, q


def recall_precision(z_imp, z_full):
    hf, _, _ = hits(z_full)
    hi, _, _ = hits(z_imp)
    sf, si = np.sign(zfill(z_full)), np.sign(zfill(z_imp))
    both = hf & hi & (sf == si)
    rec = float(both.sum() / hf.sum()) if hf.sum() > 0 else float("nan")
    prec = float(both.sum() / hi.sum()) if hi.sum() > 0 else float("nan")
    return {"n_hits_full": int(hf.sum()), "n_hits_imputed": int(hi.sum()), "n_hits_shared_same_sign": int(both.sum()),
            "recall": rec, "precision": prec}


# ---------------------------------------------------------------------------------------------------------------------
# stage 3: fits (GPU)
# ---------------------------------------------------------------------------------------------------------------------
def stage_fit(a):
    import torch
    from methylgpt2_lib import cuda_preinit
    from scipy.stats import norm
    cuda_preinit()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device(a.device)
    t0 = time.time()
    g = load_gs()
    X = g["X"]
    n = len(X)
    sets = load_sets()
    vocab = sets.cpg_id.astype(str).to_numpy()
    obs = sets.observed_in_gs.to_numpy(bool)
    cox = CoxBatch(g["duration"], g["event"], g["age"], g["sex"], dev)
    sha = hashlib.sha256("\n".join(np.sort(g["ids"])).encode()).hexdigest()
    imp = {}
    for s in SETTINGS:
        z = np.load(require(PRIV / f"imputed_{s}.npz"))
        if str(z["sha256_sorted_participant_ids"]) != sha or not np.array_equal(z["target_idx"], np.flatnonzero(sets[f"target_{s}"])):
            raise ValueError(f"imputed file for {s} does not match the frozen sets")
        imp[s] = {k: z[k] for k in ("methylgpt", "nearest_neighbours", "mean_filling")}
    checks = {"frozen_participant_sha256": sha}

    # ---- full-data EWAS over all GS CpGs (point estimate) -----------------------------------------------------------
    w1 = cox.weights()
    obs_idx = np.flatnonzero(obs)
    Mfull = cox.sort(X[:, obs_idx])
    t1 = time.time()
    bf, sf, zf, cf, sdf = cox.fit_all(Mfull, w1, a.batch)
    log(f"full-data EWAS: {len(obs_idx)} CpGs in {time.time() - t1:.0f} s; not converged {int((~cf).sum())}")
    full = pd.DataFrame({"cpg_id": vocab[obs_idx], "beta_log_hr_per_sd": bf, "se": sf, "z": zf,
                         "p": np.where(np.isfinite(zf), 2 * norm.sf(np.abs(zf)), np.nan), "sd_beta_value": sdf, "converged": cf})
    full["q_bh_all_gs_cpgs"] = bh(full.p.to_numpy())
    for s in SETTINGS:
        full[f"target_{s}"] = sets.set_index("cpg_id").loc[full.cpg_id, f"target_{s}"].to_numpy(bool)
    pos = {c: i for i, c in enumerate(full.cpg_id)}

    # ---- checks of the estimator ------------------------------------------------------------------------------------
    from statsmodels.duration.hazard_regression import PHReg   # statsmodels.api fails to import with scipy 1.16
    rng = np.random.default_rng(SEED)
    tg = np.flatnonzero(sets["target_hm27"].to_numpy(bool))
    pick = list(rng.choice(tg, 25, replace=False))
    top = full.loc[full.target_hm27].assign(a=lambda d: d.z.abs()).sort_values("a", ascending=False).cpg_id.head(5)
    pick += [int(np.flatnonzero(vocab == c)[0]) for c in top]
    dz, db, dse = [], [], []
    for j in pick:
        exog = np.column_stack([X[:, j], g["age"], g["sex"]]).astype(float)
        exog = (exog - exog.mean(0)) / exog.std(0, ddof=1)
        r = PHReg(g["duration"], exog, status=g["event"].astype(int), ties="breslow").fit()
        k = pos[vocab[j]]
        dz.append(abs(r.params[0] / r.bse[0] - zf[k])); db.append(abs(r.params[0] - bf[k])); dse.append(abs(r.bse[0] - sf[k]))
    checks["statsmodels_phreg_30_cpgs"] = {"max_abs_diff_z": float(max(dz)), "max_abs_diff_beta": float(max(db)), "max_abs_diff_se": float(max(dse))}
    log(f"check against statsmodels PHReg (30 CpGs): max |dz| {max(dz):.2e}, |dbeta| {max(db):.2e}, |dse| {max(dse):.2e}")
    cnt = np.bincount(np.random.default_rng(SEED + 1).integers(0, n, n), minlength=n)
    cols = np.array(pick[:30])
    rep = np.repeat(np.arange(n), cnt)
    cox_dup = CoxBatch(g["duration"][rep], g["event"][rep], g["age"][rep], g["sex"][rep], dev)
    zw = cox.fit(cox.sort(X[:, cols]), cox.weights(cnt))[2]
    zd = cox_dup.fit(cox_dup.sort(X[rep][:, cols]), cox_dup.weights())[2]
    checks["weights_vs_row_duplication_30_cpgs_max_abs_diff_z"] = float(np.nanmax(np.abs(zw - zd)))
    log(f"check frequency weights against row duplication: max |dz| {checks['weights_vs_row_duplication_30_cpgs_max_abs_diff_z']:.2e}")
    del cox_dup

    # ---- per-setting point estimates ------------------------------------------------------------------------------
    per_cpg, point = {}, {}
    gpu = {}
    for s in SETTINGS:
        tmask = sets[f"target_{s}"].to_numpy(bool)
        tid = vocab[tmask]
        fi = np.array([pos[c] for c in tid])
        df = pd.DataFrame({"cpg_id": tid, "setting": s})
        for col, v in (("beta", bf), ("se", sf), ("z", zf), ("sd_beta_value", sdf), ("converged", cf)):
            df[f"full_{col}"] = v[fi]
        df["full_p"] = np.where(np.isfinite(df.full_z), 2 * norm.sf(np.abs(df.full_z)), np.nan)
        df["full_q_bh_targets"] = bh(df.full_p.to_numpy())
        gpu[s] = {"MethylGPT": cox.sort(imp[s]["methylgpt"]), "nearest neighbours": cox.sort(imp[s]["nearest_neighbours"])}
        pt = {"n_target_cpgs": int(tmask.sum())}
        for name, key in (("MethylGPT", "methylgpt"), ("nearest neighbours", "nearest_neighbours")):
            b_, s_, z_, c_, sd_ = cox.fit_all(gpu[s][name], w1, a.batch)
            tag = "methylgpt" if name == "MethylGPT" else "knn"
            df[f"{tag}_beta"], df[f"{tag}_se"], df[f"{tag}_z"], df[f"{tag}_sd_beta_value"], df[f"{tag}_converged"] = b_, s_, z_, sd_, c_
            df[f"{tag}_p"] = np.where(np.isfinite(z_), 2 * norm.sf(np.abs(z_)), np.nan)
            df[f"{tag}_q_bh_targets"] = bh(df[f"{tag}_p"].to_numpy())
            rho, r = compare(z_, df.full_z.to_numpy())
            pt[name] = {"spearman_rho": rho, "pearson_r": r, "n_not_converged": int((~c_).sum()), **recall_precision(z_, df.full_z.to_numpy())}
        # mean filling: constant values, so every z is undefined (SD 0)
        mf = imp[s]["mean_filling"]
        df["mean_filling_value"] = mf
        df["mean_filling_z"] = np.nan
        pt["mean filling"] = {"spearman_rho": 0.0, "pearson_r": 0.0, "n_not_converged": int(tmask.sum()),
                              "note": "constant imputed values; z undefined at every CpG; correlation 0 and recall 0 by rule",
                              **{**recall_precision(np.full(tmask.sum(), np.nan), df.full_z.to_numpy()), "recall": 0.0}}
        pt["full_not_converged"] = int((~df.full_converged).sum())
        per_cpg[s], point[s] = df, pt
        log(f"{s}: rho MethylGPT {pt['MethylGPT']['spearman_rho']:.4f}, nearest neighbours {pt['nearest neighbours']['spearman_rho']:.4f}; "
            f"full-data hits {pt['MethylGPT']['n_hits_full']}, recall MethylGPT {pt['MethylGPT']['recall']:.3f}, kNN {pt['nearest neighbours']['recall']:.3f}")
        df.to_csv(OUT / f"D_ewas_per_cpg_{s}.csv.gz", index=False, float_format="%.6g")
    full.to_csv(OUT / "D_ewas_full_data_all_gs_cpgs.csv.gz", index=False, float_format="%.6g")
    write_json(OUT / "D_point_estimates.json", {"checks": checks, "point": point,
                                                "full_data_all_cpgs": {"n_cpgs": int(len(full)), "n_not_converged": int((~full.converged).sum()),
                                                                       "n_hits_bh_all_gs_cpgs": int((full.q_bh_all_gs_cpgs < FDR).sum())}})

    # ---- participant bootstrap --------------------------------------------------------------------------------------
    union = np.flatnonzero(np.logical_or.reduce([sets[f"target_{s}"].to_numpy(bool) for s in SETTINGS]))
    uni_pos = {j: i for i, j in enumerate(union)}
    sub = {s: np.array([uni_pos[j] for j in np.flatnonzero(sets[f"target_{s}"].to_numpy(bool))]) for s in SETTINGS}
    Mu = cox.sort(X[:, union])
    del Mfull
    torch.cuda.empty_cache()
    rngb = np.random.default_rng(SEED)
    draws = {s: [] for s in SETTINGS}
    tb = time.time()
    for b in range(N_BOOT):
        cntb = np.bincount(rngb.integers(0, n, n), minlength=n)
        w = cox.weights(cntb)
        zfu = cox.fit_all(Mu, w, a.batch)[2]
        for s in SETTINGS:
            zfs = zfu[sub[s]]
            row = {"draw": b, "full_not_converged": int((~np.isfinite(zfs)).sum())}
            for name in ("MethylGPT", "nearest neighbours"):
                zb = cox.fit_all(gpu[s][name], w, a.batch)[2]
                tag = "methylgpt" if name == "MethylGPT" else "knn"
                rho, r = compare(zb, zfs)
                rp = recall_precision(zb, zfs)
                row.update({f"{tag}_rho": rho, f"{tag}_pearson": r, f"{tag}_not_converged": int((~np.isfinite(zb)).sum()),
                            f"{tag}_recall": rp["recall"], f"{tag}_precision": rp["precision"], f"{tag}_n_hits": rp["n_hits_imputed"],
                            "n_hits_full": rp["n_hits_full"]})
            row["mean_filling_rho"] = 0.0
            draws[s].append(row)
        if (b + 1) % 25 == 0 or b == 0:
            el = time.time() - tb
            log(f"bootstrap draw {b + 1}/{N_BOOT} ({el / (b + 1):.1f} s per draw); hm27 rho MGPT {draws['hm27'][-1]['methylgpt_rho']:.3f} "
                f"kNN {draws['hm27'][-1]['knn_rho']:.3f}")
    for s in SETTINGS:
        pd.DataFrame(draws[s]).to_csv(OUT / f"D_bootstrap_draws_{s}.csv", index=False, float_format="%.6g")
    write_json(OUT / "D_fit_record.json", {"checks": checks, "n_boot": N_BOOT, "seed": SEED, "batch": a.batch,
                                           "elapsed_s": round(time.time() - t0, 1), "device": a.device})
    log(f"fit stage done ({time.time() - t0:.0f} s)")


# ---------------------------------------------------------------------------------------------------------------------
# stage 4: analysis and README
# ---------------------------------------------------------------------------------------------------------------------
def stage_analyze(a):
    pe = json.loads(require(OUT / "D_point_estimates.json").read_text())
    fr = pd.read_csv(require(OUT / "D_frozen_analysis_set.csv")).set_index("setting")
    rule_rows, sec = [], []
    res = {"script": Path(__file__).name, "rule": RULE, "settings": {}}
    for s in SETTINGS:
        pt = pe["point"][s]
        d = pd.read_csv(require(OUT / f"D_bootstrap_draws_{s}.csv"))
        rho_m, rho_k = pt["MethylGPT"]["spearman_rho"], pt["nearest neighbours"]["spearman_rho"]
        comp = {"mean filling": 0.0, "nearest neighbours": rho_k}
        best = max(comp, key=comp.get) if rho_k != 0.0 else "nearest neighbours"
        rho_b = comp[best]
        bcol = "knn_rho" if best == "nearest neighbours" else "mean_filling_rho"
        diff = d.methylgpt_rho - d[bcol]
        dci = pct(diff)
        if rho_b > 0:
            rel = (rho_m - rho_b) / rho_b
            rel_d = (d.methylgpt_rho - d[bcol]) / d[bcol]
            rel_ci = pct(rel_d)
            passed = bool(rel >= 0.10 and dci[0] is not None and dci[0] > 0)
            outcome = "pass" if passed else "fail"
        else:
            rel, rel_ci, passed, outcome = None, [None, None], False, "not applicable (rho_best <= 0)"
        rule_rows.append({"analysis": "D", "setting": s, "role": "primary" if s == PRIMARY else "second pre-specified setting",
                          "endpoint": "incident type 2 diabetes (Cox, age and sex adjusted)",
                          "primary_metric": "Spearman rho of per-CpG z, imputed vs full data, at target CpGs",
                          "n_participants": int(fr.loc[s, "n_participants"]), "n_events": int(fr.loc[s, "n_events_incident_t2d"]),
                          "n_target_cpgs": int(fr.loc[s, "n_target_cpgs"]),
                          "methylgpt_value": rho_m, "methylgpt_ci95_low": pct(d.methylgpt_rho)[0], "methylgpt_ci95_high": pct(d.methylgpt_rho)[1],
                          "best_comparator": best, "best_comparator_value": rho_b,
                          "best_comparator_ci95_low": pct(d[bcol])[0], "best_comparator_ci95_high": pct(d[bcol])[1],
                          "other_comparator": "mean filling" if best != "mean filling" else "nearest neighbours",
                          "other_comparator_value": 0.0 if best != "mean filling" else rho_k,
                          "difference": rho_m - rho_b, "difference_ci95_low": dci[0], "difference_ci95_high": dci[1],
                          "bootstrap_p_difference": boot_p(diff), "relative_gain": rel, "relative_gain_ci95_low": rel_ci[0],
                          "relative_gain_ci95_high": rel_ci[1], "outcome": outcome, "pass": passed,
                          "n_bootstrap": int(len(d)), "interval": "percentile, 1,000 participant bootstrap draws (seed 20261024)",
                          "rule": RULE})
        for tag, name in (("methylgpt", "MethylGPT"), ("knn", "nearest neighbours")):
            p = pt[name]
            for met, col in (("spearman_rho", "rho"), ("pearson_r", "pearson"), ("recall", "recall"), ("precision", "precision"),
                             ("n_hits_imputed", "n_hits")):
                ci = pct(d[f"{tag}_{col}"])
                sec.append({"setting": s, "version": name, "metric": met, "estimate": p[met], "ci95_low": ci[0], "ci95_high": ci[1],
                            "n_bootstrap_defined": int(np.isfinite(d[f"{tag}_{col}"]).sum())})
            sec.append({"setting": s, "version": name, "metric": "n_not_converged", "estimate": p["n_not_converged"], "ci95_low": None,
                        "ci95_high": None, "n_bootstrap_defined": None})
        mf = pt["mean filling"]
        for met in ("spearman_rho", "pearson_r", "recall"):
            sec.append({"setting": s, "version": "mean filling", "metric": met, "estimate": mf[met], "ci95_low": None, "ci95_high": None,
                        "n_bootstrap_defined": None, "note": "z undefined (constant values); value by rule"})
        sec.append({"setting": s, "version": "full data", "metric": "n_hits_bh_targets", "estimate": pt["MethylGPT"]["n_hits_full"],
                    "ci95_low": pct(d.n_hits_full)[0], "ci95_high": pct(d.n_hits_full)[1], "n_bootstrap_defined": int(len(d))})
        rd = d.methylgpt_recall - d.knn_recall
        sec.append({"setting": s, "version": "MethylGPT minus nearest neighbours", "metric": "recall difference",
                    "estimate": (pt["MethylGPT"]["recall"] - pt["nearest neighbours"]["recall"]) if pt["MethylGPT"]["n_hits_full"] > 0 else None,
                    "ci95_low": pct(rd)[0], "ci95_high": pct(rd)[1], "n_bootstrap_defined": int(np.isfinite(rd).sum()),
                    "bootstrap_p": boot_p(rd) if np.isfinite(rd).any() else None})
        rk = (d.methylgpt_rho - d.knn_rho) / d.knn_rho
        sec.append({"setting": s, "version": "MethylGPT vs nearest neighbours", "metric": "relative gain in rho",
                    "estimate": (rho_m - rho_k) / rho_k if rho_k > 0 else None, "ci95_low": pct(rk)[0], "ci95_high": pct(rk)[1],
                    "n_bootstrap_defined": int(np.isfinite(rk).sum())})
        res["settings"][s] = {"rule": rule_rows[-1], "point": pt}
    rt = pd.DataFrame(rule_rows)
    rt.to_csv(OUT / "D_rule_outcome.csv", index=False)
    st = pd.DataFrame(sec)
    st.to_csv(OUT / "D_secondary_metrics.csv", index=False)
    res["full_data_all_cpgs"] = pe["full_data_all_cpgs"]
    res["checks"] = pe["checks"]
    write_json(OUT / "D_results.json", res)
    for r in rule_rows:
        log(f"{r['setting']} ({r['role']}): rho MethylGPT {r['methylgpt_value']:.4f} vs {r['best_comparator']} {r['best_comparator_value']:.4f}; "
            f"difference {r['difference']:.4f} [{r['difference_ci95_low']:.4f}, {r['difference_ci95_high']:.4f}]; relative gain "
            f"{r['relative_gain'] if r['relative_gain'] is None else round(r['relative_gain'], 4)}; {r['outcome']}")
    write_readme()


def write_readme():
    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "README.md")

    def nrows(p):
        if p.suffix == ".json":
            return "JSON"
        return str(len(pd.read_csv(p)))

    desc = {
        "D_frozen_analysis_set.csv": ("Frozen analysis set, one row per setting (aggregate counts, SHA-256 of the sorted participant IDs and of "
                                      "the observed-value pattern; no identifiers)", "--stage freeze"),
        "D_frozen_analysis_set.json": ("Same, with input paths, checksums and modification times, cross-checks and the smoking-data statement", "--stage freeze"),
        "D_cpg_sets.csv.gz": ("Per vocabulary CpG: observed_in_gs, in_<setting>_mask (array membership), kept_<setting> (CpGs left in the reduced "
                              "profile), target_<setting> (GS-observed CpGs the reduced array does not measure)", "--stage freeze"),
        "D_impute_record.json": ("Bundle checksum verification and timings of the imputation stage", "--stage impute"),
        "D_imputation_accuracy.csv": ("Secondary (d): imputation accuracy at the target CpGs per setting and version: mse, mae, within_cpg_r2_pooled, "
                                      "median_per_cpg_pearson_r, n_cpgs_with_defined_r, n_values", "--stage impute"),
        "D_ewas_full_data_all_gs_cpgs.csv.gz": ("Full-data EWAS of incident type 2 diabetes at every GS CpG: cpg_id, beta_log_hr_per_sd, se, z, p, "
                                                "sd_beta_value, converged, q_bh_all_gs_cpgs, target_hm27, target_hm450", "--stage fit"),
        "D_ewas_per_cpg_hm27.csv.gz": ("Per target CpG of the 27K setting: full_*, methylgpt_* and knn_* (beta = log HR per SD, se, z, p, "
                                       "q_bh_targets, sd_beta_value, converged), mean_filling_value and mean_filling_z (undefined)", "--stage fit"),
        "D_ewas_per_cpg_hm450.csv.gz": ("Same for the 450K setting", "--stage fit"),
        "D_point_estimates.json": ("Point estimates per setting and version (rho, Pearson r, hits, recall, precision, non-converged counts) and the "
                                   "estimator checks", "--stage fit"),
        "D_bootstrap_draws_hm27.csv": ("One row per participant-bootstrap draw (aggregate statistics only): rho, Pearson r, recall, precision, "
                                       "hits and non-converged counts for MethylGPT and nearest neighbours, full-data hits", "--stage fit"),
        "D_bootstrap_draws_hm450.csv": ("Same for the 450K setting", "--stage fit"),
        "D_fit_record.json": ("Checks, seed, number of draws, timing of the fit stage", "--stage fit"),
        "D_rule_outcome.csv": ("Rule-outcome table: primary metric for MethylGPT, best comparator and value, difference with 95% interval, "
                               "relative gain, pass or fail, per setting", "--stage analyze"),
        "D_secondary_metrics.csv": ("Secondary metrics with bootstrap intervals: setting, version, metric, estimate, ci95_low, ci95_high, "
                                    "n_bootstrap_defined, note, bootstrap_p", "--stage analyze"),
        "D_results.json": ("All results of the analysis stage", "--stage analyze"),
    }
    lines = ["# D. Disease EWAS at CpGs the reduced array does not measure (Generation Scotland)", "",
             f"Source script: `{Path(__file__).name}` (stages freeze, impute, fit, analyze). Pre-specification, estimator, rule and "
             "seeds are in its docstring.", "",
             "Inputs (on Chronos only): GS frame-cache matrix and labels of the training subset "
             f"(`{CACHE}/COMMON_SAMPLE_SPLIT_{SPLIT}_train_*`), the locked endpoint table "
             f"`artifacts/gs_cox/mgpt2_combined_2026-10-01/private_nested_predictions.parquet` (endpoint {ENDPOINT} = incident type 2 "
             "diabetes), `artifacts/platform_vocab_masks_2026-10-01.npz`, the released bundle `artifacts/release/methylgpt2-1.1` and the "
             "corpus training split (`artifacts/corpus_2026-09-30`, kNN reference).", "",
             "Only aggregate tables and per-CpG summary statistics are here. Imputed participant-level values stay in "
             "`artifacts/large_gain_2026-10-24_private/D/` on Chronos and are not committed.", "",
             "Smoking pack-years EWAS: Cannot proceed - required data file not found: no smoking variable (pack-years, ever or current "
             "smoking) in the Generation Scotland release files on Chronos. Searched: "
             "/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/meta.csv, "
             "/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/2024-06-03_phenotypes.csv, "
             "/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/gs.tar.1.gz (members gs/age.csv, "
             "gs/2024-06-03_phenotypes_lastfollow202310.csv, gs/batches/split_file_1-19.csv.gz), and the frame-cache labels "
             f"({CACHE}/*.labels.parquet). See ../E/E_data_availability.csv.", "",
             "Notes", "",
             "- Run settings: `--stage fit --batch 12000` (CpGs per batched Cox fit; memory and speed only). A first fit run with the "
             "default batch of 2,048 was stopped after one bootstrap draw because it was too slow on the shared GPU; its point "
             "estimates agree with the reported ones to the 6 significant digits written. The fit and analyze stages ran with "
             "`NUMPY_MADVISE_HUGEPAGE=0` (the impute stage without it), because numpy's transparent huge-page requests stalled large allocations on Chronos; "
             "it changes speed only.",
             "- Estimator checks (D_point_estimates.json): batched Cox fits agree with statsmodels PHReg (Breslow) within 6e-14 in "
             "z for 30 CpGs, and bootstrap frequency weights agree with row duplication within 6e-14.",
             "- Bootstrap intervals of the hit-based secondary metrics (number of hits, recall, precision) are not sampling "
             "intervals of the point estimates. In a participant bootstrap the model-based standard errors ignore the "
             "resampling, so null z-scores have about twice the variance, and BH hits are far more numerous in the draws "
             "(full data, 27K targets: 373 hits in the data, 95% range 649 to 6,185 in the draws). The primary interval, of the "
             "difference in Spearman correlation between the two imputed versions, compares versions within each draw. In the "
             "27K setting the bootstrap median of the MethylGPT correlation lies below its point estimate (0.62 against 0.67) "
             "and that of nearest neighbours slightly above it (0.33 against 0.31).",
             "- The 450K setting has only 132 target CpGs (EPIC-only vocabulary CpGs). There MethylGPT imputation had a higher "
             "MSE than nearest neighbours (D_imputation_accuracy.csv) although its z-scores agreed better with the full data.",
             "",
             "| File | Contents and columns | Rows | Stage |", "|---|---|---:|---|"]
    for p in files:
        d, st = desc.get(p.name, ("", ""))
        lines.append(f"| `{p.name}` | {d} | {nrows(p)} | {st} |")
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    log(f"README written ({len(files)} files)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["freeze", "impute", "fit", "analyze"], required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--batch", type=int, default=2048, help="CpGs per batched Cox fit (memory only)")
    a = ap.parse_args()
    {"freeze": stage_freeze, "impute": stage_impute, "fit": stage_fit, "analyze": stage_analyze}[a.stage](a)


if __name__ == "__main__":
    main()
