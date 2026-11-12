#!/usr/bin/env python3
"""RRBS (GSE194154): published clocks computed from every clock CpG the sequencing measured.

The earlier RRBS analyses gave the published clocks only the CpGs of the 49,156-CpG vocabulary and set clock CpGs
outside it to 0.5. Here each clock (BioLearn: Horvathv1, Hannum, PhenoAge, Horvathv2, Lin) receives all of its
CpGs that the RRBS call files cover with at least 10 reads, in or outside the vocabulary:

1. Clock CpGs are mapped to GRCh38 with the public Zhou-lab HM450 manifest
   (https://github.com/zhou-lab/InfiniumAnnotationV1, Anno/HM450/HM450.hg38.manifest.tsv.gz, downloaded 9 Oct 2026;
   CpG_beg is the 0-based start of the CG dinucleotide, as in the EPIC v2 manifest used for the vocabulary).
2. Calls are extracted from the raw per-sample files with the functions of
   artifacts/external/GSE194154/scripts/03_build_matrix.py (both strands summed; observed if >= 10 reads).
   Values for CpGs also present in betas_cov10.parquet are compared with that matrix.
3. Unmeasured clock CpGs are filled in two ways:
   "array means": the whole-blood array mean of the CpG (vocabulary CpGs: mean of 8,000 whole-blood training
       profiles, artifacts/external/corpus_whole_blood_mean_beta.csv; other clock CpGs: mean over the 656
       whole-blood 450K profiles of the public dataset GSE40279);
   "factor model": vocabulary CpGs from MethylGPT's factor-model reconstruction (package impute() on the
       vocabulary matrix), other clock CpGs from the same whole-blood array mean.
Same samples, ages, flagged sample and fivefold recalibration (seed 20261002) as the main RRBS analysis.
Outputs: artifacts/eval/rrbs_clocks_all_measured_cpgs_2026-10-09.{json,csv} and
artifacts/external/clock_cpg_whole_blood_reference_2026-10-09.csv
"""
from __future__ import annotations

import importlib
import json
import multiprocessing as mp
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from methylgpt2_lib import ARTIFACTS, age_scores, cuda_preinit, log, require, write_json

HERE = Path(__file__).resolve().parent
D = ARTIFACTS / "external/GSE194154"
MANIFEST = ARTIFACTS / "external/manifests/HM450.hg38.manifest.tsv.gz"
GSE40279 = Path("/Entropy/kying/GEO_data/GEO_hs_dnam/processed_data/GSE40279.csv.gz")
BIOLEARN_PY = "/home/kying/anaconda3/envs/integrative/bin/python"
FLAGGED = "GSM5829631"
CLOCKS = {"Horvathv1": "Horvath 2013", "Hannum": "Hannum", "PhenoAge": "PhenoAge", "Horvathv2": "Horvath skin & blood", "Lin": "Lin"}
MIN_COV = 10


def recalibrated(pred, y, seed=20261002):
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    cal = np.empty(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(pred):
        b, a = np.polyfit(pred[tr], y[tr], 1)
        cal[te] = a + b * pred[te]
    return cal


def biolearn(code):
    r = subprocess.run([BIOLEARN_PY, "-c", code], capture_output=True, text=True)
    if r.returncode != 0:
        log(r.stderr[-2000:]); raise SystemExit("BioLearn step failed")
    return r.stdout.strip().splitlines()[-1]


def main():
    try:
        for p in (MANIFEST, GSE40279, D / "betas_cov10.parquet", D / "samples.csv", ARTIFACTS / "external/corpus_whole_blood_mean_beta.csv"):
            require(p)
    except OSError as e:
        sys.exit(f"Cannot proceed - required data file not found: {e}")
    sys.path.insert(0, str(D / "scripts"))             # spawned workers import the module by this name
    b3 = importlib.import_module("03_build_matrix")

    # 1. clock CpGs
    sites = json.loads(biolearn(
        "import json\nfrom biolearn.model_gallery import ModelGallery\ng = ModelGallery()\n"
        f"print(json.dumps({{n: list(g.get(n).clock.methylation_sites()) for n in {list(CLOCKS)!r}}}))"))
    union = sorted({c for v in sites.values() for c in v})
    cg = [c for c in union if c.startswith("cg")]
    man = pd.read_csv(MANIFEST, sep="\t", usecols=["CpG_chrm", "CpG_beg", "CpG_end", "Probe_ID"], dtype={"CpG_chrm": str})
    man = man[man.Probe_ID.isin(set(cg))].drop_duplicates("Probe_ID")
    man = man[man.CpG_chrm.isin(list(b3.CHR2I)) & ((man.CpG_end - man.CpG_beg) == 2)].reset_index(drop=True)
    keys = (man.CpG_chrm.map(b3.CHR2I).astype(np.int64) * int(b3.SCALE) + man.CpG_beg.astype(np.int64) + 1).to_numpy()
    assert not pd.Index(keys).has_duplicates
    order = np.argsort(keys, kind="stable")
    log(f"clock CpGs: {len(union)} ({len(cg)} cg probes), with GRCh38 CpG coordinates {len(man)}")

    # 2. extract calls
    samples = pd.read_csv(D / "samples.csv", dtype={"patient_id": str})
    tasks = [(g, str(require(D / lf))) for g, lf in zip(samples.gsm, samples.local_file)]
    M = np.zeros((len(tasks), len(man))); C = np.zeros((len(tasks), len(man)), np.int64)
    row = {g: i for i, (g, _) in enumerate(tasks)}
    with ProcessPoolExecutor(max_workers=24, mp_context=mp.get_context("spawn"), initializer=b3.init_worker,
                             initargs=(keys[order], order)) as ex:
        for k, (gsm, col, m, cov, _) in enumerate(ex.map(b3.process_sample, tasks), 1):
            M[row[gsm], col] = m; C[row[gsm], col] = cov
            if k % 20 == 0 or k == len(tasks):
                log(f"processed {k}/{len(tasks)} samples")
    obs = C >= MIN_COV
    with np.errstate(invalid="ignore", divide="ignore"):
        meas = pd.DataFrame(np.where(obs, M / np.where(C > 0, C, 1), np.nan), index=[g for g, _ in tasks], columns=man.Probe_ID.to_numpy())
    betas = pd.read_parquet(D / "betas_cov10.parquet")
    meas = meas.loc[betas.index]
    both = [c for c in meas.columns if c in betas.columns]
    a, b = meas[both].to_numpy(), betas[both].to_numpy(float)
    agree = {"cpgs_in_both": len(both), "max_abs_difference_where_both_observed": float(np.nanmax(np.abs(a - b))),
             "observed_pattern_identical": bool((np.isfinite(a) == np.isfinite(b)).all())}
    log(f"agreement with the vocabulary matrix: {agree}")

    # 3. reference means and the two fillings
    vocab = pd.read_csv(ARTIFACTS / "external/corpus_whole_blood_mean_beta.csv").set_index("cpg").blood_array_mean_beta
    with tempfile.TemporaryDirectory(dir="/Entropy/kying/tmp") as td:
        ids = Path(td) / "ids.txt"; ids.write_text("\n".join(cg) + "\n")
        out = Path(td) / "rows.csv"
        subprocess.run(f"zcat {GSE40279} | head -n 1 > {out}; zcat {GSE40279} | grep -F -w -f {ids} >> {out}", shell=True, check=True)
        g4 = pd.read_csv(out, index_col=0)
    g4 = g4.loc[[c for c in g4.index if c in set(cg)]].apply(pd.to_numeric, errors="coerce")
    g4_mean = g4.mean(axis=1)
    shared = [c for c in g4_mean.index if c in vocab.index and np.isfinite(vocab[c])]
    ref = pd.DataFrame({"cpg": cg}).set_index("cpg")
    ref["in_vocabulary"] = ref.index.isin(vocab.index)
    ref["corpus_whole_blood_mean"] = vocab.reindex(ref.index)
    ref["gse40279_whole_blood_mean"] = g4_mean.reindex(ref.index)
    ref["reference"] = ref.corpus_whole_blood_mean.where(ref.in_vocabulary & ref.corpus_whole_blood_mean.notna(), ref.gse40279_whole_blood_mean)
    ref.to_csv(ARTIFACTS / "external/clock_cpg_whole_blood_reference_2026-10-09.csv")
    ref_check = {"gse40279_profiles": int(g4.shape[1]), "clock_cpgs_with_gse40279_mean": int(g4_mean.notna().sum()),
                 "clock_cpgs_without_reference": int(ref.reference.isna().sum()),
                 "pearson_r_gse40279_vs_corpus_mean_on_shared_cpgs": float(np.corrcoef(g4_mean[shared], vocab[shared])[0, 1]), "shared_cpgs": len(shared)}
    log(f"reference: {ref_check}")

    cuda_preinit()
    sys.path.insert(0, str(HERE / "package/src"))
    from methylgpt2 import MethylGPT
    model = MethylGPT.from_pretrained(ARTIFACTS / "release/methylgpt2-1.1", device="cuda:0")
    imp = model.impute(betas)                                             # vocabulary CpGs, observed values unchanged
    X = meas.reindex(columns=cg)                                          # measured values; NaN if unmeasured or unmapped
    fill_mean = X.apply(lambda col: col.fillna(ref.reference[col.name]))
    fac = imp.reindex(columns=cg)
    fill_fac = X.where(X.notna(), fac).apply(lambda col: col.fillna(ref.reference[col.name]))
    cov = {}
    for n, v in sites.items():
        cl = [c for c in v if c.startswith("cg")]
        cov[CLOCKS[n]] = {"clock_cpgs": len(v), "cg_probes": len(cl), "with_grch38_coordinates": int(sum(c in set(man.Probe_ID) for c in cl)),
                          "in_vocabulary": int(sum(c in vocab.index for c in cl)),
                          "mean_fraction_measured": float(X[cl].notna().to_numpy().mean()),
                          "mean_fraction_measured_or_factor_imputed": float((X[cl].notna() | fac[cl].notna()).to_numpy().mean()),
                          "without_reference_mean": int(ref.reference.reindex(cl).isna().sum())}
    log(f"coverage: {cov}")

    # 4. BioLearn clocks on both inputs
    with tempfile.TemporaryDirectory(dir="/Entropy/kying/tmp") as td:
        for nm, df in (("mean", fill_mean), ("factor", fill_fac)):
            df.T.to_csv(Path(td) / f"{nm}.csv")
        biolearn(
            "import pandas as pd, numpy as np\nfrom biolearn.data_library import GeoData\nfrom biolearn.model_gallery import ModelGallery\n"
            "g = ModelGallery()\n"
            f"for nm in ('mean', 'factor'):\n"
            f"    b = pd.read_csv('{td}/' + nm + '.csv', index_col=0)\n"
            "    out = pd.DataFrame(index=b.columns)\n"
            f"    for n in {list(CLOCKS)!r}:\n"
            "        m = g.get(n)\n"
            "        miss = [c for c in m.clock.methylation_sites() if c not in b.index]\n"
            "        bb = pd.concat([b, pd.DataFrame(0.5, index=miss, columns=b.columns)]).fillna(0.5)\n"
            "        out[n] = m.predict(GeoData(pd.DataFrame(index=bb.columns), bb)).iloc[:, 0].reindex(bb.columns)\n"
            f"    out.to_csv('{td}/pred_' + nm + '.csv')\n"
            "print('ok')\n")
        pm = pd.read_csv(Path(td) / "pred_mean.csv", index_col=0).reindex(betas.index)
        pf = pd.read_csv(Path(td) / "pred_factor.csv", index_col=0).reindex(betas.index)

    age = pd.read_csv(D / "samples.csv").set_index("gsm").loc[betas.index, "age"].astype(float).to_numpy()
    out = pd.DataFrame({"gsm": betas.index, "age": age})
    res = {"n_samples": int(len(out)), "agreement_with_vocabulary_matrix": agree, "reference_means": ref_check, "clock_cpg_coverage": cov,
           "non_cg_or_unreferenced_clock_sites_set_to_0.5": {CLOCKS[n]: int(sum((not c.startswith("cg")) or not np.isfinite(ref.reference.get(c, np.nan)) for c in v)) for n, v in sites.items()},
           "age": {}}
    keep = out.gsm != FLAGGED
    for n, lab in CLOCKS.items():
        for suffix, pr in ((" (array means)", pm), (" (factor model)", pf)):
            col = lab + suffix
            out[col] = pr[n].to_numpy()
            cal = recalibrated(out[col], out.age)
            res["age"][col] = {"all": age_scores(out[col], out.age), f"without_{FLAGGED}": age_scores(out[col][keep], out.age[keep]),
                               "mean_bias_years": float(np.mean(out[col] - out.age)), "recalibrated_5fold": age_scores(cal, out.age)}
            s = res["age"][col]
            log(f"{col}: r {s['all']['pearson_r']:.3f} bias {s['mean_bias_years']:+.1f} MAE {s['all']['mae']:.2f}; recalibrated MAE {s['recalibrated_5fold']['mae']:.2f}")
    out.to_csv(ARTIFACTS / "eval/rrbs_clocks_all_measured_cpgs_2026-10-09.csv", index=False)
    write_json(ARTIFACTS / "eval/rrbs_clocks_all_measured_cpgs_2026-10-09.json", res)


if __name__ == "__main__":
    main()
