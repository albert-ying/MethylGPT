#!/usr/bin/env python3
"""Agreement of the 25 distilled clock outputs with full-array ClockBase values when CpGs are missing.

Figure 5a reports, for complete profiles, the correlation between each distilled output and the value ClockBase
computed from the sample's full array, on test-split profiles (datasets not used for training). This script repeats
that comparison after removing CpGs with the masks of the age scenarios (30-90% removed at random, seed
20261001 + percentage; CpGs of the 27K, 450K and EPIC v2 arrays from platform_vocab_masks_2026-10-01.npz) and
reports, per output and scenario, Pearson r and the median absolute error in units of the training standard
deviation (IQR / 1.349) and in the output's own units. Intervals for the median r over outputs resample datasets.
Model: the released clock-stage model (runs/multitask_bio_2026-10-01/final.pt = bundle weights.pt).
The complete-profile scenario must reproduce runs/multitask_bio_2026-10-01/results.json.
Output: artifacts/eval/distilled_clocks_missing_cpgs_2026-10-10.{json,csv}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from methylgpt2_lib import ARTIFACTS, CORPUS_DIR, Corpus, cuda_preinit, log, require, write_json
from methylgpt2_model import BioHeads, embed_batch, load_checkpoint

RUN = ARTIFACTS / "runs/multitask_bio_2026-10-01"
SCEN = ["all", "random_30", "random_50", "random_70", "random_90", "hm450", "epic_v2", "hm27"]


def degrade(X, scen, plat):
    X = X.copy()
    if scen.startswith("random_"):
        p = int(scen.split("_")[1]) / 100
        X[np.random.default_rng(20261001 + int(p * 100)).random(X.shape) < p] = np.nan
    elif scen != "all":
        X[:, ~plat[scen]] = np.nan
    return X


def corr_cols(P, T):
    """Pearson r per column over rows where the target is finite (NaN if fewer than 50 rows)."""
    out = np.full(P.shape[1], np.nan)
    for j in range(P.shape[1]):
        ok = np.isfinite(T[:, j])
        if ok.sum() > 50 and np.std(T[ok, j]) > 0 and np.std(P[ok, j]) > 0:
            out[j] = np.corrcoef(P[ok, j], T[ok, j])[0, 1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--draws", type=int, default=1000)
    a = ap.parse_args()
    cuda_preinit()
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device(a.device)
    try:
        ck = torch.load(require(RUN / "final.pt"), map_location=dev, weights_only=False)
        stored = json.loads(require(RUN / "results.json").read_text())["biomarker_r2_heldout"]
        plat = np.load(require(ARTIFACTS / "platform_vocab_masks_2026-10-01.npz"))
        clocks = pd.read_parquet(require(CORPUS_DIR / "clock_targets_2026-10-01.parquet"))
    except OSError as e:
        raise SystemExit(f"Cannot proceed - required data file not found: {e}")
    corpus = Corpus()
    prof = corpus.profiles
    assert (clocks.id.to_numpy() == prof.id.to_numpy()).all()
    names = list(ck["bio_names"])
    med, scale = np.asarray(ck["bio_median"], float), np.asarray(ck["bio_scale"], float)
    Bz = np.clip((clocks[names].to_numpy(float) - med) / scale, -6, 6)
    rows = np.flatnonzero((prof.split == "test").to_numpy() & np.isfinite(Bz).any(1))
    T = Bz[rows]
    groups = prof.group.to_numpy()[rows]
    log(f"test profiles with at least one clock target: {len(rows)} from {len(set(groups))} datasets")

    model, core, _ = load_checkpoint(Path(ck["ckpt"]), ck.get("config", {}).get("core"), dev)
    model.load_state_dict(ck["model"]); model.eval()
    heads = BioHeads(core.rank + model.latents.shape[1], len(ck["tissues"]) + 1, len(names)).to(dev)
    heads.load_state_dict(ck["heads"]); heads.eval()

    @torch.no_grad()
    def predict(X, batch=32):
        out = []
        for i in range(0, len(X), batch):
            x = torch.from_numpy(X[i:i + batch]).to(dev)
            out.append(heads(embed_batch(model, core, x, torch.isfinite(x)))[3].float().cpu().numpy())
        return np.concatenate(out)

    X0 = corpus.get(rows)
    by_group = [np.flatnonzero(groups == g) for g in sorted(set(groups))]
    res = {"model": "runs/multitask_bio_2026-10-01/final.pt (released clock-stage model, run 1)", "profiles": int(len(rows)),
           "datasets": int(len(by_group)), "bootstrap": f"{a.draws} draws resampling datasets, seed 20261010", "scenarios": {}}
    table = []
    for scen in SCEN:
        X = degrade(X0, scen, plat)
        P = predict(X)
        r = corr_cols(P, T)
        err = np.abs(P - T)
        if scen == "all":
            worst = max(abs(r[j] - stored[nm]["pearson_r"]) for j, nm in enumerate(names) if nm in stored)
            assert worst < 2e-3, f"complete-profile correlations differ from the stored results by {worst}"
            log(f"complete profiles reproduce the stored correlations (largest difference {worst:.1e})")
            r_all = r
        rng = np.random.default_rng(20261010)
        boot, boot_drop = [], []
        for _ in range(a.draws):
            k = np.concatenate([by_group[i] for i in rng.integers(0, len(by_group), len(by_group))])
            rb = corr_cols(P[k], T[k])
            boot.append(np.nanmedian(rb))
        per = {}
        for j, nm in enumerate(names):
            ok = np.isfinite(T[:, j])
            per[nm] = {"n": int(ok.sum()), "pearson_r": float(r[j]), "median_abs_error_sd_units": float(np.median(err[ok, j])),
                       "median_abs_error_clock_units": float(np.median(err[ok, j]) * scale[j])}
            table.append({"scenario": scen, "output": nm, **per[nm]})
        res["scenarios"][scen] = {"observed_fraction_of_vocabulary": float(np.isfinite(X).mean()),
                                  "median_r": float(np.nanmedian(r)), "median_r_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                                  "min_r": float(np.nanmin(r)), "min_r_output": names[int(np.nanargmin(r))], "max_r": float(np.nanmax(r)),
                                  "outputs_with_r_at_least_0.9": int((r >= 0.9).sum()), "largest_decrease_in_r_from_complete": float(np.nanmax(r_all - r)),
                                  "largest_decrease_output": names[int(np.nanargmax(r_all - r))],
                                  "median_abs_error_sd_units_median_over_outputs": float(np.median([v["median_abs_error_sd_units"] for v in per.values()])),
                                  "outputs": per}
        s = res["scenarios"][scen]
        log(f"{scen}: observed {s['observed_fraction_of_vocabulary']:.2f}; median r {s['median_r']:.3f} [{s['median_r_ci95'][0]:.3f}, {s['median_r_ci95'][1]:.3f}]; "
            f"range {s['min_r']:.3f} ({s['min_r_output']}) to {s['max_r']:.3f}; largest decrease {s['largest_decrease_in_r_from_complete']:.3f} ({s['largest_decrease_output']}); "
            f"median abs error {s['median_abs_error_sd_units_median_over_outputs']:.3f} SD")
    write_json(ARTIFACTS / "eval/distilled_clocks_missing_cpgs_2026-10-10.json", res)
    pd.DataFrame(table).to_csv(ARTIFACTS / "eval/distilled_clocks_missing_cpgs_2026-10-10.csv", index=False)


if __name__ == "__main__":
    main()
