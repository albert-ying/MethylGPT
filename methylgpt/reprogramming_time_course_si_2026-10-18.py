#!/usr/bin/env python3
"""Reprogramming time course for the Supplementary Information (GSE54848).

Data: GSE54848 (Illumina 450K, GPL13534), all 30 methylation profiles as read by evaluate_reprogramming_2026-10-01.py:
dermal fibroblasts (day 0, n = 3), reprogramming intermediates (days 3-28, SOX2-EGFP+ on day 3, TRA-1-60+ from day 7),
iPSCs after sorting and passages 1-3 (nominal days 35, 42, 49), and the pluripotent references H9 ESC (n = 2) and the
201B7 iPSC line (n = 1). GSE54848 is not in the MethylGPT corpus (checked below on dataset and GSM identifiers).

Pre-specified primary quantity (descriptive; written before running)
  For each method and run: predicted age at day 0 (mean of the 3 fibroblast replicates, with each replicate) and the
  plateau from day 20 on (mean over every profile from day 20 to day 49, n = 14, and the range of the day means), and
  the MethylGPT predictions for the ESC/iPSC references (each profile and their mean). Methods: MethylGPT run 1
  (weights.pt), run 2 (weights_run2.pt), run 3 (weights_run3.pt) and their mean (the released age ensemble); the
  elastic net trained on the same age labels; Horvath 2013 and Horvath skin & blood (BioLearn, all 450K CpGs).
  No intervals: one dataset with 2 or 3 replicates per day, so every replicate is reported instead.

Secondary (descriptive)
  * day-15 replicates for every method and their difference;
  * PCA fitted on these 30 profiles for vocabulary-CpG raw beta values (CpGs observed in all 30 profiles; no
    imputation), latent scores (input-layer posterior s, natural scale) and the full 2,048-dimensional embedding of
    each run; PC1/PC2 variance explained; covariance PCA (centred, unscaled);
  * Euclidean distance of every profile to the centroid of the three ESC/iPSC references in the full representation
    space, also as a fraction of the mean day-0 distance; the references' own distances to their centroid give the
    scale of variation among pluripotent lines.
  No correlation tests, no bootstrap, no colouring by predicted age. No GrimAge2, PhenoAge or DunedinPACE.

Checks: release checksums against package/methylgpt-1.1.checksums.sha256; the mean of the three runs against the
per-sample ensemble values in age_ensemble_2026-10-04.json (i_reprogramming_gse54848); run 1 against
reprogramming_gse54848_2026-10-01.csv; elastic net against the ensemble json; BioLearn clocks against the 2026-10-01 csv.
Donor age: searched in the local GEO metadata (metadata.csv and GSE54848_family.soft.gz); no internet.

Outputs
  artifacts/eval/reprogramming_time_course_si_2026-10-18.csv        predicted age per profile and method
  artifacts/eval/reprogramming_time_course_si_2026-10-18.pca.csv    PC1/PC2 and distances per representation and profile
  artifacts/eval/reprogramming_time_course_si_2026-10-18.json       primary and secondary quantities, checks, provenance
Run: CUDA_VISIBLE_DEVICES=1 python reprogramming_time_course_si_2026-10-18.py --device cuda:0   (GPU 1 only)
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from methylgpt_lib import ARTIFACTS, CORPUS_DIR, Finv, cuda_preinit, log, require, write_json

HERE = Path(__file__).resolve().parent
EVAL = ARTIFACTS / "eval"
D = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/dataset/GSE54848")
BUNDLE = ARTIFACTS / "release/methylgpt2-1.1"
CHECKSUMS = HERE / "package/methylgpt-1.1.checksums.sha256"
BIOLEARN_PY = "/home/kying/anaconda3/envs/integrative/bin/python"
TMP = Path("/Entropy/kying/tmp/si_reruns_2026-10-18/reprogramming_tmp")
DAY = {"p1": 35, "p2": 42, "p3": 49}       # as evaluate_reprogramming_2026-10-01.py (7 days after sorting / passage)
RUN_FILES = {"MethylGPT run 1": "weights.pt", "MethylGPT run 2": "weights_run2.pt", "MethylGPT run 3": "weights_run3.pt"}
RUNS = list(RUN_FILES)
MEAN = "MethylGPT mean of runs"
EN = "Elastic net (same labels)"
H13, HSB = "Horvath 2013", "Horvath skin & blood"
METHODS = RUNS + [MEAN, EN, H13, HSB]
OUT = EVAL / "reprogramming_time_course_si_2026-10-18"


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def verify_checksums():
    res = {}
    for line in require(CHECKSUMS).read_text().split("\n"):
        if not line.strip():
            continue
        digest, name = line.split()
        got = sha256(require(BUNDLE / name))
        res[name] = got == digest
        if got != digest:
            sys.exit(f"checksum mismatch for {BUNDLE / name}")
    return res


def donor_age_search():
    """Look for a donor age (or any age/sex/donor field) in the local GEO metadata of GSE54848."""
    meta = pd.read_csv(require(D / "metadata.csv"))
    soft = require(D / "GSE54848_family.soft.gz")
    pat = re.compile(r"\b(age|aged|years?|yr|y\.o\.|donor|male|female|sex|gender|neonatal|newborn|adult)\b", re.I)
    hits, n_header = [], 0
    with gzip.open(soft, "rt", errors="replace") as f:
        for line in f:
            if not line.startswith(("!", "^")):
                continue
            n_header += 1
            if line.startswith(("!Sample_data_processing", "!Sample_extract_protocol", "!Sample_label_protocol")):
                continue        # protocol text (reagent names); searched separately below for numbers with 'year'
            if pat.search(line):
                hits.append(line.strip()[:300])
    year_hits = []
    with gzip.open(soft, "rt", errors="replace") as f:
        for line in f:
            if line.startswith(("!", "^")) and re.search(r"\d+\s*-?\s*(years?|yrs?|y\.o\.)", line, re.I):
                year_hits.append(line.strip()[:300])
    meta_cols = [c for c in meta.columns if pat.search(c.replace("_", " ").replace(".", " "))]
    meta_vals = sorted({str(v)[:200] for c in meta.columns for v in meta[c].dropna().astype(str) if re.search(r"\d+\s*-?\s*(years?|yrs?)", v, re.I)})
    return {"files_searched": [str(D / "metadata.csv"), str(soft)], "soft_header_lines_searched": n_header,
            "soft_lines_matching_age_sex_donor_terms": hits, "soft_lines_with_number_and_year": year_hits,
            "metadata_columns_matching": meta_cols, "metadata_values_with_number_and_year": meta_vals,
            "series_matrix_file_present": any(D.glob("*series_matrix*")),
            "result": ("donor age not reported in the local GEO metadata"
                       if not (year_hits or meta_cols or meta_vals or any(re.search(r"\bage\b", h, re.I) for h in hits)) else
                       "possible age field found; inspect hits")}


def biolearn_clocks(m):
    TMP.mkdir(parents=True, exist_ok=True)
    tmp = TMP / "gse54848_betas_2026-10-18.csv"
    m.to_csv(tmp)
    code = (
        "import pandas as pd\n"
        "from biolearn.data_library import GeoData\nfrom biolearn.model_gallery import ModelGallery\n"
        f"b = pd.read_csv('{tmp}', index_col=0)\n"
        "meta = pd.DataFrame(index=b.columns)\n"
        "g = ModelGallery(); out = pd.DataFrame(index=b.columns)\n"
        "for n in ['Horvathv1', 'Horvathv2']:\n"
        "    p = g.get(n, imputation_method='averaging').predict(GeoData(meta, b))\n"
        "    out[n] = p.iloc[:, 0].reindex(b.columns)\n"
        f"out.to_csv('{tmp.with_suffix('.clocks.csv')}')\n")
    try:
        subprocess.run([BIOLEARN_PY, "-c", code], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        sys.exit(f"BioLearn failed: {e.stderr[-2000:]}")
    ck = pd.read_csv(tmp.with_suffix(".clocks.csv"), index_col=0)
    tmp.unlink()
    tmp.with_suffix(".clocks.csv").unlink()
    return ck


def pca(Z):
    """Covariance PCA (centred, unscaled) on the rows of Z; returns scores, variance fractions."""
    Zc = Z - Z.mean(0)
    U, S, _ = np.linalg.svd(Zc, full_matrices=False)
    var = S ** 2 / np.sum(S ** 2)
    return U * S, var


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()
    t0 = time.time()
    res = {"script": Path(__file__).name, "dataset": "GSE54848 (GPL13534, 450K)",
           "primary_quantity": "predicted age at day 0 (mean of 3 replicates) and the plateau from day 20 on (mean of the 14 "
                               "profiles from day 20 to 49; range of day means) for each method and run; MethylGPT "
                               "predictions for the ESC/iPSC references. Descriptive, no intervals."}

    log("verifying release checksums")
    res["checksums_ok"] = verify_checksums()
    res["donor_age"] = donor_age_search()
    log(f"donor age: {res['donor_age']['result']}")

    # ---- data, as evaluate_reprogramming_2026-10-01.py -----------------------------------------------------
    meta = pd.read_csv(require(D / "metadata.csv"))
    meta = meta[meta.platform_id == "GPL13534"].reset_index(drop=True)
    gsm = meta.geo_accession.tolist()
    m = pd.read_csv(require(D / "expression_matrix.csv"), usecols=["ID_REF"] + gsm, index_col=0)
    m = m[m.index.astype(str).str.startswith("cg")].astype(float)
    t = meta.title.astype(str)
    day = np.array([float(re.match(r"Day(\d+)_", s).group(1)) if s.startswith("Day") else DAY.get(s.split("_")[0], np.nan) for s in t])
    group = np.where(t.str.startswith("H9"), "ESC", np.where(t == "201B7", "iPSC reference", "time course"))
    out = pd.DataFrame({"gsm": gsm, "title": t, "cell_type": meta["characteristics_ch1.0.cell type"], "day": day, "group": group})
    log(f"{len(gsm)} methylation profiles, {len(m)} CpG rows; groups {pd.Series(group).value_counts().to_dict()}")
    prof = pd.read_parquet(CORPUS_DIR / "profiles.parquet", columns=["id", "group"])
    res["corpus_membership"] = {"profiles_with_these_gsm_ids": int(prof.id.astype(str).isin(gsm).sum()),
                                "profiles_in_group_GSE54848": int(prof.group.astype(str).str.split(" ").str[0].eq("GSE54848").sum())}

    vocab = np.array((BUNDLE / "vocab.txt").read_text().split())
    X = m.reindex(vocab).T.to_numpy(np.float32)
    X = np.where(np.isfinite(X), np.clip(X, 0, 1), np.nan).astype(np.float32)
    out["vocab_cpgs_observed"] = np.isfinite(X).sum(1)
    betas = pd.DataFrame(X, index=gsm, columns=vocab)

    # ---- MethylGPT: per-run age, latent scores and per-run embeddings via the release API -------------------
    cuda_preinit()
    dev = torch.device(a.device)
    sys.path.insert(0, str(HERE / "package/src"))
    from methylgpt import MethylGPT
    mg = MethylGPT.from_pretrained(BUNDLE, device=dev)
    assert [mm.name for mm in mg.age_members] == ["run1", "run2", "run3"]
    runs = mg.predict_age_runs(betas)
    for col, r in zip(RUNS, ["run1", "run2", "run3"]):
        out[col] = runs[r].to_numpy()
    out[MEAN] = out[RUNS].mean(axis=1)
    S = mg.factor_scores(betas).to_numpy(np.float64)
    emb = {}
    with torch.no_grad():
        x = torch.from_numpy(X).to(dev)
        s = mg.core.posterior(x, torch.isfinite(x).float())
        for col, mem in zip(RUNS, mg.age_members):
            emb[col] = mg._encode(x, mem.net, s)[2].float().cpu().numpy().astype(np.float64)
    assert np.allclose(s.cpu().numpy(), S)
    del mg
    torch.cuda.empty_cache()

    # ---- elastic net (same labels), as age_ensemble_2026-10-04.py ------------------------------------------
    zen = np.load(require(EVAL / "baseline_corpus_age_clean_rerun_2026-10-01.enet_model.npz"))
    T = np.where(np.isfinite(X), X, zen["mu"])
    out[EN] = Finv(((T - zen["mu"]) / zen["sd"]) @ zen["w"] * float(zen["ys"]) + float(zen["ym"]))

    # ---- Horvath clocks (BioLearn, all 450K CpGs) ------------------------------------------------------------
    log("BioLearn Horvath 2013 and skin & blood")
    ck = biolearn_clocks(m).reindex(gsm)
    out[H13], out[HSB] = ck.Horvathv1.to_numpy(), ck.Horvathv2.to_numpy()

    # ---- checks against existing outputs ---------------------------------------------------------------------
    ens = pd.DataFrame(json.loads(require(EVAL / "age_ensemble_2026-10-04.json").read_text())["i_reprogramming_gse54848"]["per_sample"]).set_index("gsm").loc[gsm]
    old = pd.read_csv(require(EVAL / "reprogramming_gse54848_2026-10-01.csv")).set_index("gsm").loc[gsm]
    res["checks"] = {
        "mean_of_runs_vs_age_ensemble_json_max_abs_diff_years": float(np.max(np.abs(out[MEAN].to_numpy() - ens["ensemble"].to_numpy()))),
        "run_vs_age_ensemble_json_max_abs_diff_years": {c: float(np.max(np.abs(out[c].to_numpy() - ens[k].to_numpy())))
                                                         for c, k in zip(RUNS, ["run 1", "seed 2", "seed 3"])},
        "elastic_net_vs_age_ensemble_json_max_abs_diff_years": float(np.max(np.abs(out[EN].to_numpy() - ens["elastic net"].to_numpy()))),
        "run1_vs_reprogramming_2026-10-01_csv_max_abs_diff_years": float(np.max(np.abs(out[RUNS[0]].to_numpy() - old["MethylGPT-2"].to_numpy()))),
        "horvath2013_vs_2026-10-01_csv_max_abs_diff_years": float(np.max(np.abs(out[H13].to_numpy() - old["Horvath 2013"].to_numpy()))),
        "horvath_skin_blood_vs_2026-10-01_csv_max_abs_diff_years": float(np.max(np.abs(out[HSB].to_numpy() - old["Horvath skin & blood"].to_numpy()))),
        "note": "age_ensemble_2026-10-04.json per-sample values are rounded to 4 decimals"}
    log(f"checks: {res['checks']}")
    out.to_csv(OUT.with_suffix(".csv"), index=False)

    # ---- primary and secondary age quantities -----------------------------------------------------------------
    tc = out[out.group == "time course"]
    refs = out[out.group != "time course"]
    res["n_per_day"] = {str(int(k)): int(v) for k, v in tc.groupby("day").size().items()}
    res["n_references"] = {"ESC (H9)": int((out.group == "ESC").sum()), "iPSC reference (201B7)": int((out.group == "iPSC reference").sum())}
    prim, day_means = {}, {}
    for c in METHODS:
        dm = tc.groupby("day")[c].mean()
        late = tc[tc.day >= 20][c]
        d15 = tc.loc[tc.day == 15, c].to_numpy()
        prim[c] = {"day0_mean": float(dm.loc[0.0]), "day0_replicates": [float(v) for v in tc.loc[tc.day == 0, c]],
                   "plateau_day_ge20_mean_of_profiles": float(late.mean()), "plateau_n_profiles": int(len(late)),
                   "plateau_day_means_range": [float(dm[dm.index >= 20].min()), float(dm[dm.index >= 20].max())],
                   "plateau_profiles_range": [float(late.min()), float(late.max())],
                   "day0_minus_plateau": float(dm.loc[0.0] - late.mean()),
                   "references": {ti: float(v) for ti, v in zip(refs.title, refs[c])},
                   "references_mean": float(refs[c].mean()),
                   "esc_mean": float(out.loc[out.group == "ESC", c].mean()),
                   "day15_replicates": [float(v) for v in d15], "day15_abs_difference": float(abs(d15[0] - d15[1]))}
        day_means[c] = {str(int(k)): float(v) for k, v in dm.items()}
    res["primary_predicted_age"] = prim
    res["day_means"] = day_means
    res["between_run_sd_of_plateau_means"] = float(np.std([prim[c]["plateau_day_ge20_mean_of_profiles"] for c in RUNS], ddof=1))

    # ---- PCA and distance to the ESC/iPSC centroid -------------------------------------------------------------
    full = np.isfinite(X).all(0)
    reps = {"Raw beta values": X[:, full].astype(np.float64), "Latent scores": S,
            **{f"Embedding, {c.replace('MethylGPT ', '')}": emb[c] for c in RUNS}}
    is_ref = (out.group != "time course").to_numpy()
    is_d0 = (out.day == 0).to_numpy()
    rows, pres = [], {}
    for name, Z in reps.items():
        sc, var = pca(Z)
        if sc[is_d0, 0].mean() > 0:          # orientation only: fibroblasts on the left
            sc[:, 0] *= -1
        if sc[is_ref, 1].mean() < 0:
            sc[:, 1] *= -1
        cen = Z[is_ref].mean(0)
        dist = np.linalg.norm(Z - cen, axis=1)
        d0 = dist[is_d0].mean()
        for i in range(len(out)):
            rows.append({"representation": name, "gsm": gsm[i], "title": t[i], "day": day[i], "group": group[i],
                         "PC1": sc[i, 0], "PC2": sc[i, 1], "distance_to_reference_centroid": dist[i],
                         "distance_fraction_of_day0": dist[i] / d0})
        dd = pd.Series(dist / d0).groupby(out.day.where(~is_ref)).mean()
        pres[name] = {"n_features": int(Z.shape[1]), "pc1_variance_explained": float(var[0]), "pc2_variance_explained": float(var[1]),
                      "day0_mean_distance": float(d0),
                      "distance_fraction_of_day0_by_day": {str(int(k)): float(v) for k, v in dd.items()},
                      "reference_distance_fraction_of_day0": {ti: float(v) for ti, v in zip(out.title[is_ref], dist[is_ref] / d0)},
                      "day_ge20_distance_fraction_mean": float((dist / d0)[(out.day >= 20).to_numpy()].mean())}
    res["pca"] = pres
    res["raw_beta_cpgs_observed_in_all_30_profiles"] = int(full.sum())
    # The embedding is [s / sqrt(lam) (1,536, identical across runs), pooled encoder tokens (512, run-specific)]:
    # share of the embedding's total variance over these 30 profiles carried by the run-specific block, and the
    # distance of that block alone to the reference centroid (descriptive; explains why the runs' PCAs coincide).
    blk = {}
    K = S.shape[1]
    for c in RUNS:
        Z = emb[c]
        assert np.allclose(Z[:, :K], emb[RUNS[0]][:, :K], atol=1e-4)
        v = Z.var(0)
        tb = Z[:, K:]
        dist = np.linalg.norm(tb - tb[is_ref].mean(0), axis=1)
        dist = dist / dist[is_d0].mean()
        blk[c] = {"encoder_token_block_dims": int(tb.shape[1]), "encoder_token_block_variance_share": float(v[K:].sum() / v.sum()),
                  "encoder_token_block_distance_fraction_of_day0_by_day": {str(int(k)): float(x) for k, x in
                                                                           pd.Series(dist).groupby(out.day.where(~is_ref)).mean().items()},
                  "encoder_token_block_reference_distance_fraction_of_day0": {ti: float(x) for ti, x in zip(out.title[is_ref], dist[is_ref])}}
    res["embedding_blocks"] = blk
    pd.DataFrame(rows).to_csv(OUT.with_suffix(".pca.csv"), index=False)
    res["outputs"] = [str(OUT.with_suffix(".csv")), str(OUT.with_suffix(".pca.csv")), str(OUT.with_suffix(".json"))]
    res["elapsed_s"] = time.time() - t0
    write_json(OUT.with_suffix(".json"), res)
    for c in METHODS:
        p = prim[c]
        log(f"{c:28s} day0 {p['day0_mean']:6.2f}  plateau {p['plateau_day_ge20_mean_of_profiles']:6.2f} "
            f"[{p['plateau_day_means_range'][0]:.2f}, {p['plateau_day_means_range'][1]:.2f}]  refs {p['references_mean']:6.2f}  "
            f"d15 {p['day15_replicates']}")
    for k, v in pres.items():
        log(f"{k:24s} PC1 {v['pc1_variance_explained']:.3f} PC2 {v['pc2_variance_explained']:.3f} "
            f"d>=20 frac {v['day_ge20_distance_fraction_mean']:.3f}")
    log(f"done in {res['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
