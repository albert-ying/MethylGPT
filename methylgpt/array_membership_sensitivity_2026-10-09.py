#!/usr/bin/env python3
"""Array-membership audit and sensitivity of the array scenarios.

The array masks of the reported analyses (artifacts/platform_vocab_masks_2026-10-01.npz) were built from the
Methyl27_Loci and Methyl450_Loci flags of the Illumina EPIC v1 manifest, so "27K" and "450K" denote the vocabulary
CpGs that these arrays share with EPIC v1 (22,231 and 47,826 CpGs). The arrays' full vocabulary content is larger:
22,287 CpGs for the 27K array (probe list of GPL8490, from the public 27K dataset GSE49908) and 49,020 for the
450K array (Zhou-lab HM450 hg38 manifest). This script
1. records the membership audit (counts, overlaps, the six tutorial profiles);
2. recomputes age on the AltumAge test set for the 27K and 450K scenarios with the reported masks (to confirm
   the reported numbers) and with the full membership: MethylGPT (three-run mean, package), the elastic net with
   training-mean filling and with factor-model imputation;
3. counts measured values of Generation Scotland at the CpGs that differ between the two definitions (its EPIC
   array does not carry them, so its scenario results cannot change).
Aggregate output: artifacts/eval/array_membership_audit_2026-10-09.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from methylgpt_lib import ARTIFACTS, Finv, age_scores, cuda_preinit, load_altumage_split, log, require, write_json

HERE = Path(__file__).resolve().parent
CACHE = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/task_a_private_training_20260629/_frame_cache")
SPLIT = "full18859_common_task_a_incident_positive_duration_20260629"


def main():
    cuda_preinit(); torch.backends.cuda.matmul.allow_tf32 = False
    old = np.load(require(ARTIFACTS / "platform_vocab_masks_2026-10-01.npz"))
    new = np.load(require(ARTIFACTS / "platform_vocab_masks_full_membership_2026-10-09.npz"))
    vocab = old["vocab"].astype(str)
    ex = pd.read_csv(require(HERE / "package/examples/GSE54848_example_betas.csv"), index_col=0)
    exc = np.isin(vocab, ex.columns.to_numpy())
    res = {"sources": {"hm450": "Zhou-lab HM450 hg38 manifest (InfiniumAnnotationV1, Anno/HM450/HM450.hg38.manifest.tsv.gz, downloaded 9 Oct 2026)",
                       "hm27": "probe identifiers of the public 27K dataset GSE49908 (GPL8490), ClockBase matrix",
                       "epic_v1": "Illumina Infinium MethylationEPIC v1.0 B5 manifest", "epic_v2": "Zhou-lab EPICv2 hg38 manifest",
                       "reported_masks": "Methyl27_Loci and Methyl450_Loci flags of the EPIC v1 manifest, i.e. CpGs shared with EPIC v1"},
           "vocabulary_cpgs": int(len(vocab)),
           "membership": {k: {"reported_mask": int(old[k].sum()), "full_membership": int(new[k].sum()),
                              "reported_is_subset": bool((~old[k] | new[k]).all()),
                              "reported_equals_full_and_epic_v1": bool((old[k] == (new[k] & old["epic_v1"])).all())} for k in ("hm27", "hm450")},
           "epic_v1": int(old["epic_v1"].sum()), "epic_v2": int(old["epic_v2"].sum()),
           "tutorial_profiles_gse54848": {"vocabulary_cpgs_in_file": int(exc.sum()), "of_which_on_450k_full_membership": int((exc & new["hm450"]).sum()),
                                          "450k_vocabulary_cpgs_absent_from_the_geo_series_matrix": int((new["hm450"] & ~exc).sum()),
                                          "finite_values_per_profile": [int(v) for v in ex.reindex(columns=vocab).notna().sum(1)]}}
    log(f"membership: {res['membership']}; tutorial: {res['tutorial_profiles_gse54848']}")

    # Generation Scotland: measured values at the CpGs that differ
    gs = {}
    for split in ("train", "val", "test"):
        mp = require(next(CACHE.glob(f"COMMON_SAMPLE_SPLIT_{SPLIT}_{split}_all_gs_batches_all_probes_*.matrix.npz")))
        with np.load(mp, allow_pickle=False) as z:
            X = z["matrix"]
        gs[split] = {k: int(np.isfinite(X[:, new[k] & ~old[k]]).sum()) for k in ("hm27", "hm450")}
        del X
    res["generation_scotland_finite_values_at_cpgs_added_by_full_membership"] = gs
    log(f"GS: {gs}")

    # age on the AltumAge test set
    sys.path.insert(0, str(HERE / "package/src"))
    from methylgpt import MethylGPT
    model = MethylGPT.from_pretrained(ARTIFACTS / "release/methylgpt2-1.1", device="cuda:0")
    ids, X, y = load_altumage_split("test")
    z = np.load(require(ARTIFACTS / "eval/baseline_corpus_age_clean_rerun_2026-10-01.enet_model.npz"))
    enet = lambda T: Finv(((T - z["mu"]) / z["sd"]) @ z["w"] * float(z["ys"]) + float(z["ym"]))
    age = {}
    for scen in ("hm27", "hm450"):
        for name, masks in (("reported_mask", old), ("full_membership", new)):
            Xs = X.copy(); Xs[:, ~masks[scen]] = np.nan
            df = pd.DataFrame(Xs, columns=vocab)
            r = {"cpgs_kept": int(masks[scen].sum()), "observed_fraction": float(np.isfinite(Xs).mean()),
                 "MethylGPT": age_scores(model.predict_age(df).to_numpy(), y)["medae"],
                 "elastic net, training-mean filling": age_scores(enet(np.where(np.isfinite(Xs), Xs, z["mu"])), y)["medae"],
                 "elastic net, factor-model imputation": age_scores(enet(model.impute(df).to_numpy(np.float32)), y)["medae"]}
            age[f"{scen} | {name}"] = r
            log(f"age {scen} {name}: {r}")
    res["age_altumage_test_medae"] = age
    write_json(ARTIFACTS / "eval/array_membership_audit_2026-10-09.json", res)


if __name__ == "__main__":
    main()
