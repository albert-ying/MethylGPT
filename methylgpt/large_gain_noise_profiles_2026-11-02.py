#!/usr/bin/env python3
"""Shared input of large-gain analyses M and N: SIMULATED MEASUREMENT NOISE ON EMPIRICAL PROFILES. Every profile is a
measured Generation Scotland (GS) development profile (the 9,866 training-subset participants of the locked
incident-disease protocol, frame-cache order, as in analysis J); the noise is simulated with the generator below. No
outcome, model or held-out participant is read. Validation and test participants are not read.

Written on 2 Nov 2026, after analyses J, K and L. Why: in J, MethylGPT's advantage over the clocks grew as the deployed
input lost CpGs (complete profiles +7.5%, 27K +66%), and K showed that the C-index gap on complete or 27K inputs does
not widen on individual diseases. Measurement noise is the other way a deployed input degrades, and low-depth bisulfite
or enzymatic sequencing is the cheapest route to cohort-scale methylation. This choice of setting was made after those
results and is recorded as such in M's and N's docstrings and in the results file.

COLUMNS: the union U of the 49,156 vocabulary columns (frame-cache order) and then the 940 non-vocabulary clock CpGs
  (clock-matrix order; J's clock matrix, 1,195 cg CpGs of GrimAgeV2 and DunedinPACE). A vocabulary clock CpG takes the draw
  of its vocabulary column, so the noisy vocabulary and clock matrices agree at those CpGs.
TRUTH: beta = the GS value clipped to [0, 1]. A CpG absent from GS stays NaN in every scenario (reads 0).
SCENARIOS (fixed here, before any model sees a noisy profile)
  depth_10     PRIMARY for M and N. Simulated bisulfite read sampling: reads n ~ Poisson(10), methylated reads
               m ~ Binomial(n, beta), observed value m / n, NaN where n = 0. Mean depth 10 is the per-CpG calling
               threshold of analysis I and lies within the range of low-pass whole-genome and enzymatic methyl
               sequencing.
  depth_1, depth_2, depth_5, depth_30   the same generator at mean depth 1, 2, 5 and 30 (secondary).
  gauss_0.02, gauss_0.05, gauss_0.10    simulated array measurement noise: clip(beta + sigma * e, 0, 1), e ~ N(0, 1)
               (secondary).
RNG: scenario code c (depth_L: L; gauss_s: 100 + 100 s); rows in blocks of 512 in frame-cache order; block b draws from
  default_rng([20261102, c, b]): depth draws n over the block's (rows, |U|) array, then m; gauss draws e. The block size
  is part of the generator.
OUTPUTS
  Private (inside the git-ignored artifacts/, never committed, stay on Chronos): artifacts/large_gain_2026-11-02_private/
  noise/private_noisy_<scenario>.npz with vocab (float32, 9,866 x 49,156), clock (float32, 9,866 x 1,195, clock-matrix
  order) and, for depth scenarios, reads_vocab and reads_clock (uint8).
  Public: artifacts/large_gain_2026-11-02/noise/noise_summary_2026-11-02.csv (per scenario: mean reads, fraction of
  GS-observed vocabulary CpGs with no read, RMSE and mean within-participant Pearson r of the noisy against the measured
  values, SHA-256 of every private array), noise_checks_2026-11-02.csv and README.md. Aggregates only.
CHECKS (abort on failure): participant order of the clock matrix equals the frame cache; measured vocabulary clock CpGs
  equal their vocabulary columns before noise and after noise; NaN pattern of absent CpGs unchanged; reads below 256.
Run: /usr/bin/python large_gain_noise_profiles_2026-11-02.py   (the script must be committed unchanged first)
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
DATE = "2026-11-02"
OUT = ART / "large_gain_2026-11-02/noise"
PRIV = ART / "large_gain_2026-11-02_private/noise"
J_SCRIPT = HERE / "large_gain_J_gs_missing_cpgs_knn_2026-10-28.py"
SEED = 20261102
BLOCK = 512
SCENARIOS = {"depth_10": ("depth", 10.0), "depth_1": ("depth", 1.0), "depth_2": ("depth", 2.0), "depth_5": ("depth", 5.0),
             "depth_30": ("depth", 30.0), "gauss_0.02": ("gauss", 0.02), "gauss_0.05": ("gauss", 0.05), "gauss_0.10": ("gauss", 0.10)}
PRIMARY = "depth_10"
LABEL = "simulated measurement noise on empirical profiles"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def committed():
    rel = str(Path(__file__).resolve().relative_to(HERE))
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=HERE, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=HERE).returncode == 0
    return tracked and clean


def load_j():
    spec = importlib.util.spec_from_file_location("J", J_SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["J"] = m
    spec.loader.exec_module(m)
    return m


def code(scen):
    kind, v = SCENARIOS[scen]
    return int(v) if kind == "depth" else int(round(100 + 100 * v))


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    if not committed():
        sys.exit("Cannot proceed - the script must be committed unchanged before it runs")
    J = load_j()
    OUT.mkdir(parents=True, exist_ok=True)
    PRIV.mkdir(parents=True, exist_ok=True)
    ids, X = J.load_gs()
    cids, cpgs, C = J.load_clock_matrix()
    vocab = np.load(J.need(J.MASKS))["vocab"].astype(str)        # vocabulary column names, as J reads them
    if len(vocab) != X.shape[1]:
        sys.exit("vocabulary length differs from the frame cache")
    checks = []

    def check(name, ok, note=""):
        checks.append({"check": name, "pass": bool(ok), "note": note})
        if not ok:
            pd.DataFrame(checks).to_csv(OUT / f"noise_checks_{DATE}.csv", index=False)
            sys.exit(f"check failed: {name} {note}")

    check("clock-matrix participant order equals the frame cache", np.array_equal(cids, ids))
    invoc, ic, vj = J.clock_vocab_index(cpgs, vocab)
    both = np.isfinite(C[:, ic]) & np.isfinite(X[:, vj])
    check("measured vocabulary clock CpGs equal their vocabulary columns before noise",
          np.array_equal(C[:, ic][both], X[:, vj][both]) and np.array_equal(np.isfinite(C[:, ic]), np.isfinite(X[:, vj])),
          f"{len(ic)} vocabulary clock CpGs")
    inv = np.flatnonzero(~invoc)
    U = np.hstack([X, C[:, inv]])
    present = np.isfinite(U)
    truth = np.clip(np.where(present, U, 0.0), 0.0, 1.0).astype(np.float64)
    nv = X.shape[1]
    rows = []
    for scen, (kind, v) in SCENARIOS.items():
        t0 = time.time()
        noisy = np.full(U.shape, np.nan, np.float32)
        reads = np.zeros(U.shape, np.uint8) if kind == "depth" else None
        for b, s in enumerate(range(0, len(U), BLOCK)):
            e = slice(s, min(s + BLOCK, len(U)))
            rng = np.random.default_rng([SEED, code(scen), b])
            t, p = truth[e], present[e]
            if kind == "depth":
                n = rng.poisson(v, size=t.shape)
                m = rng.binomial(n, t)
                if n.max() > 255:
                    sys.exit(f"{scen}: reads above 255")
                n = np.where(p, n, 0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    val = np.where(n > 0, m / np.maximum(n, 1), np.nan)
                reads[e] = n.astype(np.uint8)
            else:
                val = np.clip(t + v * rng.standard_normal(size=t.shape), 0.0, 1.0)
                val = np.where(p, val, np.nan)
            noisy[e] = val.astype(np.float32)
        nvoc = noisy[:, :nv]
        nclk = np.empty(C.shape, np.float32)
        nclk[:, ic] = nvoc[:, vj]
        nclk[:, inv] = noisy[:, nv:]
        check(f"{scen}: absent CpGs stay NaN", not np.isfinite(nvoc[~present[:, :nv]]).any() and not np.isfinite(nclk[~np.isfinite(C)]).any())
        check(f"{scen}: vocabulary clock CpGs equal their vocabulary columns after noise",
              np.array_equal(np.nan_to_num(nclk[:, ic], nan=-1), np.nan_to_num(nvoc[:, vj], nan=-1)))
        arrays = {"vocab": nvoc, "clock": nclk}
        if reads is not None:
            rc = np.zeros(C.shape, np.uint8)
            rc[:, ic] = reads[:, :nv][:, vj]
            rc[:, inv] = reads[:, nv:]
            arrays.update(reads_vocab=reads[:, :nv], reads_clock=rc)
        tmp = PRIV / f"private_noisy_{scen}.tmp.npz"
        np.savez(tmp, **arrays)
        tmp.replace(PRIV / f"private_noisy_{scen}.npz")
        pv = present[:, :nv]
        obs = np.isfinite(nvoc) & pv
        diff = (nvoc - X)[obs]
        xr, nr = np.where(obs, X, np.nan), np.where(obs, nvoc, np.nan)
        xc, nc = xr - np.nanmean(xr, 1, keepdims=True), nr - np.nanmean(nr, 1, keepdims=True)
        r = np.nansum(xc * nc, 1) / np.sqrt(np.nansum(xc ** 2, 1) * np.nansum(nc ** 2, 1))
        row = {"scenario": scen, "role": "primary" if scen == PRIMARY else "secondary", "kind": kind, "parameter": v, "label": LABEL,
               "participants": len(U), "vocabulary_cpgs_measured_in_gs_per_participant": float(pv.sum(1).mean()),
               "fraction_measured_vocabulary_values_with_no_read": float(1 - obs.sum() / pv.sum()),
               "mean_reads_measured_vocabulary": float(reads[:, :nv][pv].mean()) if reads is not None else np.nan,
               "rmse_noisy_vs_measured": float(np.sqrt(np.mean(diff.astype(np.float64) ** 2))),
               "mean_within_participant_pearson_r": float(np.mean(r)), "seconds": round(time.time() - t0)}
        row.update({f"sha256_{k}": sha(a) for k, a in arrays.items()})
        rows.append(row)
        log(f"{scen}: no-read fraction {row['fraction_measured_vocabulary_values_with_no_read']:.4f}, RMSE {row['rmse_noisy_vs_measured']:.4f}, "
            f"r {row['mean_within_participant_pearson_r']:.4f} ({row['seconds']} s)")
    pd.DataFrame(rows).to_csv(OUT / f"noise_summary_{DATE}.csv", index=False)
    pd.DataFrame(checks).to_csv(OUT / f"noise_checks_{DATE}.csv", index=False)
    (OUT / "README.md").write_text(
        "# Shared noisy GS profiles for analyses M and N\n\n"
        f"Script: `large_gain_noise_profiles_2026-11-02.py` (generator and scenarios in its docstring). {LABEL.capitalize()}: measured GS development "
        "profiles (9,866 participants) with noise drawn by the stated generator. Primary scenario depth_10.\n\n"
        f"- `noise_summary_{DATE}.csv`: per scenario, mean reads, fraction of measured values with no read, RMSE and mean within-participant "
        "Pearson r of the noisy against the measured values, SHA-256 of each private array.\n"
        f"- `noise_checks_{DATE}.csv`: {len(checks)} checks, all pass.\n\n"
        "Participant-level noisy profiles and read counts stay in `artifacts/large_gain_2026-11-02_private/noise/` on Chronos and are never committed.\n")
    (PRIV / "done_noise.json").write_text(json.dumps({"scenarios": list(SCENARIOS), "time": time.strftime("%Y-%m-%d %H:%M:%S")}))
    log("done")


if __name__ == "__main__":
    main()
