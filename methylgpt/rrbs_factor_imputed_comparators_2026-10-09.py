#!/usr/bin/env python3
"""RRBS (GSE194154): comparators with factor-model imputation.

In evaluate_rrbs_gse194154_2026-10-05.py the elastic net fills missing vocabulary CpGs with training means and the
published clocks with the whole-blood array mean, while MethylGPT uses the corpus-fitted factor model. Here the
comparators receive the same imputation: missing vocabulary CpGs are filled with the factor model's reconstruction
from each sample's observed CpGs (methylgpt package, impute(); observed values unchanged). Clock CpGs outside the
49,156-CpG vocabulary (3-39% of each clock's CpGs) are not in the RRBS matrix and are set to 0.5, as before.
Same samples, ages and fivefold recalibration (seed 20261002) as the main RRBS analysis.
Output: artifacts/eval/rrbs_factor_imputed_comparators_2026-10-09.{json,csv}
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from methylgpt_lib import ARTIFACTS, Finv, age_scores, cuda_preinit, log, require, write_json

HERE = Path(__file__).resolve().parent
D = ARTIFACTS / "external/GSE194154"
BIOLEARN_PY = "/home/kying/anaconda3/envs/integrative/bin/python"
FLAGGED = "GSM5829631"


def recalibrated(pred, y, seed=20261002):
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    cal = np.empty(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(pred):
        b, a = np.polyfit(pred[tr], y[tr], 1)
        cal[te] = a + b * pred[te]
    return cal


def main():
    cuda_preinit()
    sys.path.insert(0, str(HERE / "package/src"))
    from methylgpt import MethylGPT
    betas = pd.read_parquet(require(D / "betas_cov10.parquet"))
    samples = pd.read_csv(require(D / "samples.csv")).set_index("gsm").loc[betas.index]
    age = samples["age"].astype(float).to_numpy()
    model = MethylGPT.from_pretrained(ARTIFACTS / "release/methylgpt2-1.1", device="cuda:1")
    full = model.impute(betas)                                    # factor-model reconstruction for unobserved CpGs
    assert np.isfinite(full.to_numpy()).all() and list(full.columns) == list(model.vocab)
    main_csv = pd.read_csv(require(ARTIFACTS / "eval/rrbs_gse194154_2026-10-05.csv")).set_index("gsm").loc[betas.index]
    z = np.load(require(ARTIFACTS / "eval/baseline_corpus_age_clean_rerun_2026-10-01.enet_model.npz"))
    out = pd.DataFrame({"gsm": betas.index, "age": age, "MethylGPT": main_csv["MethylGPT"].to_numpy(),
                        "ElasticNet (training-mean fill)": main_csv["ElasticNet (same labels)"].to_numpy()})
    out["ElasticNet (factor-model imputation)"] = Finv(((full.to_numpy(np.float32) - z["mu"]) / z["sd"]) @ z["w"] * float(z["ys"]) + float(z["ym"]))
    tmp = D / "biolearn_input_factor_imputed.csv"
    full.T.to_csv(tmp)
    code = (
        "import pandas as pd, numpy as np, json\n"
        "from biolearn.data_library import GeoData\nfrom biolearn.model_gallery import ModelGallery\n"
        f"b = pd.read_csv('{tmp}', index_col=0)\n"
        "g = ModelGallery(); out = pd.DataFrame(index=b.columns); cov = {}\n"
        "for n in ['Horvathv1', 'Hannum', 'PhenoAge', 'Horvathv2', 'Lin']:\n"
        "    m = g.get(n)\n"
        "    cpgs = [c for c in m.clock.methylation_sites() if c.startswith('cg')]\n"
        "    miss = [c for c in cpgs if c not in b.index]\n"
        "    cov[n] = 1 - len(miss) / len(cpgs)\n"
        "    bb = pd.concat([b, pd.DataFrame(0.5, index=miss, columns=b.columns)])\n"
        "    p = m.predict(GeoData(pd.DataFrame(index=bb.columns), bb))\n"
        "    out[n] = p.iloc[:, 0].reindex(bb.columns)\n"
        f"out.to_csv('{D / 'biolearn_clocks_factor_imputed.csv'}')\n"
        "print(json.dumps(cov))\n")
    r = subprocess.run([BIOLEARN_PY, "-c", code], capture_output=True, text=True)
    if r.returncode != 0:
        log(r.stderr[-2000:]); raise SystemExit("BioLearn scoring failed")
    in_vocab = json.loads(r.stdout.strip().splitlines()[-1])
    tmp.unlink()
    ck = pd.read_csv(D / "biolearn_clocks_factor_imputed.csv", index_col=0).reindex(betas.index)
    names = {"Horvathv1": "Horvath 2013", "Hannum": "Hannum", "PhenoAge": "PhenoAge", "Horvathv2": "Horvath skin & blood", "Lin": "Lin"}
    for k, v in names.items():
        out[f"{v} (training-mean fill)"] = main_csv[v].to_numpy()
        out[f"{v} (factor-model imputation)"] = ck[k].to_numpy()
    out.to_csv(ARTIFACTS / "eval/rrbs_factor_imputed_comparators_2026-10-09.csv", index=False)
    keep = (out.gsm != FLAGGED).to_numpy()
    res = {"n_samples": int(len(out)), "fraction_of_clock_cpgs_in_vocabulary": in_vocab, "age": {}}
    for col in [c for c in out.columns if c not in ("gsm", "age")]:
        p = out[col].to_numpy(float)
        res["age"][col] = {"all": age_scores(p, age), f"without_{FLAGGED}": age_scores(p[keep], age[keep]), "mean_bias_years": float(np.mean(p - age)),
                           "recalibrated_5fold": age_scores(recalibrated(p, age), age)}
        v = res["age"][col]
        log(f"{col}: r {v['all']['pearson_r']:.3f} bias {v['mean_bias_years']:+.1f} recalibrated MAE {v['recalibrated_5fold']['mae']:.2f} MedAE {v['recalibrated_5fold']['medae']:.2f}")
    write_json(ARTIFACTS / "eval/rrbs_factor_imputed_comparators_2026-10-09.json", res)


if __name__ == "__main__":
    main()
