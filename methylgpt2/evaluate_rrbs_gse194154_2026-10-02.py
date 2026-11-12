#!/usr/bin/env python3
"""MethylGPT on sequencing data: whole-blood RRBS (GSE194154).

110 pre-surgery whole-blood samples (ages 18-88), hg38 RRBS calls mapped to vocabulary CpGs
through the Zhou EPIC v2 hg38 manifest; a CpG is observed when it has >= 10 reads (median
10,678 of 49,156 vocabulary CpGs per sample). Built by artifacts/external/GSE194154/scripts.
MethylGPT uses the observed CpGs directly. The ElasticNet age model trained on the same corpus
labels fills missing CpGs with training means. Published clocks (BioLearn, integrative env)
need values at all of their CpGs, so missing clock CpGs are filled with the corpus whole-blood
array mean, the usual way to apply array clocks to RRBS. Reported for all samples and without
GSM5829631, whose genotype-free sex signals contradict its reported sex.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from methylgpt2_lib import ARTIFACTS, age_scores, cuda_preinit, Finv, log, require, write_json

HERE = Path(__file__).resolve().parent
D = ARTIFACTS / "external/GSE194154"
BIOLEARN_PY = "/home/kying/anaconda3/envs/integrative/bin/python"
FLAGGED = "GSM5829631"


def main():
    cuda_preinit()
    sys.path.insert(0, str(HERE / "package/src"))
    from methylgpt2 import MethylGPT
    betas = pd.read_parquet(require(D / "betas_cov10.parquet"))
    samples = pd.read_csv(require(D / "samples.csv")).set_index("gsm").loc[betas.index]
    age = samples["age"].astype(float).to_numpy()
    log(f"{len(betas)} samples, {betas.shape[1]} CpGs; observed per sample median {int(betas.notna().sum(1).median())}")
    model = MethylGPT.from_pretrained(ARTIFACTS / "release/methylgpt2-1.0", device="cuda:0")
    pred = model.predict(betas)
    out = pd.DataFrame({"gsm": betas.index, "age": age, "sex": samples["sex"].to_numpy(),
                        "observed_cpgs": betas.notna().sum(1).to_numpy(), "MethylGPT": pred["age"].to_numpy(),
                        "MethylGPT p_male": pred["p_male"].to_numpy()})
    for c in ("GrimAge2", "DunedinPACE", "PhenoAge"):
        out[f"MethylGPT {c} (distilled)"] = pred[c].to_numpy()

    vocab = model.vocab
    X = betas.reindex(columns=vocab).to_numpy(np.float32)
    z = np.load(require(ARTIFACTS / "eval/baseline_corpus_age_clean_rerun_2026-10-01.enet_model.npz"))
    T = np.where(np.isfinite(X), X, z["mu"])
    out["ElasticNet (same labels)"] = Finv(((T - z["mu"]) / z["sd"]) @ z["w"] * float(z["ys"]) + float(z["ym"]))

    ref = pd.read_csv(require(ARTIFACTS / "external/corpus_whole_blood_mean_beta.csv")).set_index("cpg").blood_array_mean_beta
    full = betas.reindex(columns=vocab).T                      # CpG x sample
    tmp = D / "biolearn_input.csv"
    full.to_csv(tmp)
    code = (
        "import pandas as pd, numpy as np\n"
        "from biolearn.data_library import GeoData\nfrom biolearn.model_gallery import ModelGallery\n"
        f"b = pd.read_csv('{tmp}', index_col=0)\n"
        f"ref = pd.read_csv('{ARTIFACTS / 'external/corpus_whole_blood_mean_beta.csv'}').set_index('cpg').blood_array_mean_beta\n"
        "g = ModelGallery(); out = pd.DataFrame(index=b.columns); cov = {}\n"
        "for n in ['Horvathv1', 'Hannum', 'PhenoAge', 'Horvathv2', 'Lin']:\n"
        "    m = g.get(n)\n"
        "    cpgs = [c for c in m.clock.methylation_sites() if c.startswith('cg')]\n"
        "    bb = b.copy()\n"
        "    if cpgs is not None:\n"
        "        miss = [c for c in cpgs if c not in bb.index]\n"
        "        bb = pd.concat([bb, pd.DataFrame(np.nan, index=miss, columns=bb.columns)])\n"
        "        cov[n] = float(bb.loc[cpgs].notna().mean().mean())\n"
        "    fill = ref.reindex(bb.index)\n"
        "    bb = bb.apply(lambda col: col.fillna(fill)).fillna(0.5)\n"
        "    p = m.predict(GeoData(pd.DataFrame(index=bb.columns), bb))\n"
        "    out[n] = p.iloc[:, 0].reindex(bb.columns)\n"
        f"out.to_csv('{D / 'biolearn_clocks.csv'}')\n"
        "import json; print(json.dumps(cov))\n")
    r = subprocess.run([BIOLEARN_PY, "-c", code], capture_output=True, text=True)
    if r.returncode != 0:
        log(r.stderr[-2000:])
        raise SystemExit("BioLearn scoring failed")
    clock_cov = json.loads(r.stdout.strip().splitlines()[-1])
    tmp.unlink()
    ck = pd.read_csv(D / "biolearn_clocks.csv", index_col=0).reindex(betas.index)
    names = {"Horvathv1": "Horvath 2013", "Hannum": "Hannum", "PhenoAge": "PhenoAge", "Horvathv2": "Horvath skin & blood", "Lin": "Lin"}
    for k, v in names.items():
        out[v] = ck[k].to_numpy()
    out.to_csv(ARTIFACTS / "eval/rrbs_gse194154_2026-10-02.csv", index=False)

    res = {"n_samples": int(len(out)), "age_range": [float(age.min()), float(age.max())],
           "observed_vocab_cpgs_median": float(out.observed_cpgs.median()),
           "clock_cpg_coverage_mean_fraction": clock_cov, "age": {}}
    keep = out.gsm != FLAGGED

    def recalibrated(pred, y, seed=20261002):
        """5-fold cross-validated linear recalibration (age ~ a + b * prediction), as for any
        clock moved to a new platform; each sample is scored by a fit on the other folds."""
        from sklearn.model_selection import KFold
        pred, y = np.asarray(pred, float), np.asarray(y, float)
        cal = np.empty(len(y))
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(pred):
            b, a = np.polyfit(pred[tr], y[tr], 1)
            cal[te] = a + b * pred[te]
        return cal

    for col in ["MethylGPT", "ElasticNet (same labels)"] + list(names.values()):
        res["age"][col] = {"all": age_scores(out[col], out.age), f"without_{FLAGGED}": age_scores(out[col][keep], out.age[keep]),
                           "mean_bias_years": float(np.mean(out[col] - out.age)),
                           "recalibrated_5fold": age_scores(recalibrated(out[col], out.age), out.age),
                           f"recalibrated_5fold_without_{FLAGGED}": age_scores(recalibrated(out[col][keep], out.age[keep]), out.age[keep])}
    from sklearn.model_selection import KFold
    base = np.empty(len(out))
    for tr, te in KFold(5, shuffle=True, random_state=20261002).split(out):
        base[te] = out.age.to_numpy()[tr].mean()
    res["age"]["fold-mean age (no methylation)"] = {"recalibrated_5fold": age_scores(base, out.age)}
    log(f"fold-mean age baseline: MedAE {res['age']['fold-mean age (no methylation)']['recalibrated_5fold']['medae']:.2f} "
        f"MAE {res['age']['fold-mean age (no methylation)']['recalibrated_5fold']['mae']:.2f}")
    sex_known = out.sex.isin(["M", "F", "male", "female", "Male", "Female"])
    is_male = out.sex.astype(str).str.lower().str.startswith("m")
    acc = float(((out["MethylGPT p_male"] > 0.5) == is_male)[sex_known].mean())
    res["sex_accuracy_vs_reported"] = {"all": acc, "n": int(sex_known.sum())}
    write_json(ARTIFACTS / "eval/rrbs_gse194154_2026-10-02.json", res)
    for col, v in res["age"].items():
        if "all" not in v:
            continue
        log(f"{col}: MedAE {v['all']['medae']:.2f} MAE {v['all']['mae']:.2f} r {v['all']['pearson_r']:.3f} bias {v['mean_bias_years']:+.1f}; "
            f"recalibrated MedAE {v['recalibrated_5fold']['medae']:.2f} MAE {v['recalibrated_5fold']['mae']:.2f}; "
            f"without flagged r {v[f'without_{FLAGGED}']['pearson_r']:.3f}")
    log(f"sex accuracy {acc:.3f}; clock CpG coverage {clock_cov}")


if __name__ == "__main__":
    main()
