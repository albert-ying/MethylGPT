#!/usr/bin/env python3
"""Supplementary analysis: genomic enrichment of the CpGs MethylGPT uses for age.

Written on 21 Oct 2026, before running. The original submission's attention analysis was removed (the rebuilt encoder
has no CpG-level attention). This script asks the following questions of the released model instead: which CpGs
drive MethylGPT's age output, are they enriched for PRC2 targets and transcription-factor binding sites, in which
direction does their methylation change with age, and which gene-ontology terms do they carry (gometh,
missMethyl). Every result is reported.

Attribution (stage attribute, GPU). For each of the three released training runs (runs/multitask_bio_*/final.pt),
gradient x input of the age output: the gradient of the (Horvath-transformed, standardized) age output with respect to
each observed beta value, times the beta value minus the training mean. Gradients flow through the input layer's
posterior, the token features and the encoder (the no-grad decorators of the inference path are bypassed; weights are
unchanged). Profiles: 2,000 held-out test-split profiles with metadata age, drawn with seed 20261021. A CpG's score is
its mean absolute attribution over the profiles in which it is observed, averaged over the three runs. Check: the
forward pass reproduces the saved held-out age predictions of each run.
Direction (stage direction, CPU): within-dataset Pearson correlation of each CpG's beta value with age in 10,000
training-split profiles with a clean age (seed 20261021), values and ages centred within dataset.

Sets (vocabulary CpGs on the EPIC v1.0 B5 manifest, n = 47,962, are the background)
  PRC2 targets   CpGs annotated (UCSC RefGene) to a gene of BENPORATH_PRC2_TARGETS (MSigDB; 650 genes; copied to
                 artifacts/external/BENPORATH_PRC2_TARGETS.txt from revision_round3_20260703/revision_analyses/61_prc2_enrichment)
  TFBS           CpGs with any ENCODE TFBS_NAME in the EPIC v1.0 B5 manifest
  CpG island, enhancer, DNase hypersensitivity: artifacts/cpg_annotation_2026-10-01.parquet

PRE-SPECIFIED PRIMARY: enrichment of PRC2-target CpGs among the 1,000 CpGs with the highest attribution score against
the rest of the background, Fisher's exact test (two-sided) with the conditional maximum-likelihood odds ratio and its
95% interval.
Secondary (all reported): TFBS, CpG island, shore, enhancer and DNase hypersensitivity enrichment; the PRC2 test
stratified by CpG-island relation (Mantel-Haenszel odds ratio); the fraction of top CpGs whose methylation rises with
age, overall and among PRC2 targets, against the background; the same tests for the top 500 and 2,000; gometh
(missMethyl, GO and KEGG, array.type EPIC, prior.prob TRUE, background = the 47,962 CpGs) with the number of terms at
FDR < 0.05 and the top 20 terms; the same tests for the 1,000 CpGs most correlated with age in the training profiles
(univariate reference).
Energy distances (stage energy, CPU): in set A (2,371 profiles, six tissues, 43 test datasets),
energy distance between tissues and between sexes on the first 50 principal components (energy_distance() of
benchmark_representations_2026-10-01.py) for latent scores and for raw beta values filled with training means, with
complete profiles, 27K-array CpGs and 70% of CpGs removed (protocol of benchmark_representations_missing_2026-10-02.py);
latent minus raw with a paired 95% interval from resampling datasets (1,000 draws, seed 20261021).

Outputs: artifacts/si_enrichment_2026-10-21/ (attribution and direction arrays), artifacts/eval/si_enrichment_attribution_2026-10-21.json/.csv
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import time
import types
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from methylgpt2_lib import ARTIFACTS, CORPUS_DIR, log, require, write_json

HERE = Path(__file__).resolve().parent
EVAL = ARTIFACTS / "eval"
OUT = ARTIFACTS / "si_enrichment_2026-10-21"
RUNS = {"run 1": "runs/multitask_bio_2026-10-01/final.pt", "seed 2": "runs/multitask_bio_seed20261002/final.pt",
        "seed 3": "runs/multitask_bio_seed20261003/final.pt"}
PRC2_SRC = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/revision_round3_20260703/revision_analyses/61_prc2_enrichment/BENPORATH_PRC2_TARGETS.txt")
PRC2 = ARTIFACTS / "external/BENPORATH_PRC2_TARGETS.txt"
MANIFEST = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/infinium-methylationepic-v-1-0-b5-manifest-file-csv.zip")
SEED = 20261021


def _module(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def heldout_rows():
    from methylgpt2_lib import Corpus
    corpus = Corpus()
    lab = pd.read_parquet(require(CORPUS_DIR / "profiles_age_clean_2026-10-01.parquet"))
    ho = np.flatnonzero((corpus.profiles.split == "test").to_numpy() & np.isfinite(lab.age_meta.to_numpy(float)))
    return corpus, lab, ho


def attribute(device):
    import torch
    from methylgpt2_lib import cuda_preinit, Finv
    from methylgpt2_model import BioHeads, Heads, load_checkpoint, embed_batch
    cuda_preinit()
    dev = torch.device(device)
    OUT.mkdir(parents=True, exist_ok=True)
    corpus, lab, ho = heldout_rows()
    rows = np.sort(np.random.default_rng(SEED).choice(ho, 2000, replace=False))
    X = corpus.get(rows)
    zsaved = np.load(require(EVAL / "age_revision_2026-10-01.predictions.npz"))
    zseeds = {"seed 2": np.load(require(EVAL / "age_seed2_scenarios_2026-10-04.predictions.npz")),
              "seed 3": np.load(require(EVAL / "age_seed3_scenarios_2026-10-04.predictions.npz"))}
    saved_key = {"run 1": (zsaved, "MethylGPT|heldout"), "seed 2": (zseeds["seed 2"], "MethylGPT seed 2|heldout"),
                 "seed 3": (zseeds["seed 3"], "MethylGPT seed 3|heldout")}
    pos = np.searchsorted(ho, rows)
    t0 = time.time()
    res = {"rows": rows}
    checks = {}
    for name, rel in RUNS.items():
        ck = torch.load(require(ARTIFACTS / rel), map_location=dev, weights_only=False)
        model, core, _ = load_checkpoint(Path(ck["ckpt"]), ck.get("config", {}).get("core"), dev)
        model.load_state_dict(ck["model"]); model.eval()
        d_in = core.rank + model.latents.shape[1]
        heads = (BioHeads(d_in, len(ck["tissues"]) + 1, len(ck["bio_names"])) if "bio_names" in ck else Heads(d_in, len(ck["tissues"]) + 1)).to(dev)
        heads.load_state_dict(ck["heads"]); heads.eval()
        # bypass the no-grad decorators of the inference path (weights unchanged)
        for obj in {id(core): core, id(model.core): model.core}.values():
            obj.posterior = types.MethodType(type(obj).posterior.__wrapped__, obj)
            obj.predict = types.MethodType(type(obj).predict.__wrapped__, obj)
        model.inputs = types.MethodType(type(model).inputs.__wrapped__, model)
        for mod in (model, heads, core):
            for prm in mod.parameters():
                prm.requires_grad_(False)
        mu = core.mu.detach()
        S = np.zeros(X.shape[1]); N = np.zeros(X.shape[1]); signed = np.zeros(X.shape[1]); pred = np.empty(len(X))
        for i in range(0, len(X), 4):
            xb = torch.from_numpy(X[i:i + 4]).to(dev)
            ctx = torch.isfinite(xb)
            x = torch.where(ctx, xb, mu).requires_grad_(True)
            out = heads(embed_batch(model, core, x, ctx))[0]
            out.sum().backward()
            a = (x.grad * (x.detach() - mu) * ctx).cpu().numpy()
            S += np.abs(a).sum(0); signed += a.sum(0); N += ctx.cpu().numpy().sum(0)
            pred[i:i + 4] = Finv(out.detach().float().cpu().numpy() * ck["fs"] + ck["fm"])
        zs, key = saved_key[name]
        checks[f"{name}_max_abs_diff_vs_saved_prediction"] = float(np.max(np.abs(pred - zs[key][pos])))
        res[f"score|{name}"] = S / np.maximum(N, 1); res[f"signed|{name}"] = signed / np.maximum(N, 1); res[f"n_observed|{name}"] = N
        log(f"{name}: attribution done ({time.time() - t0:.0f} s); check {checks[f'{name}_max_abs_diff_vs_saved_prediction']:.2e}")
        del model, core, heads, ck
        torch.cuda.empty_cache()
    np.savez(OUT / "attribution.npz", **res)
    write_json(OUT / "attribution_checks.json", checks)


def direction():
    OUT.mkdir(parents=True, exist_ok=True)
    from methylgpt2_lib import Corpus
    corpus = Corpus()
    lab = pd.read_parquet(require(CORPUS_DIR / "profiles_age_clean_2026-10-01.parquet"))
    age = lab.age_clean.to_numpy(float)
    pool = np.flatnonzero((corpus.profiles.split == "train").to_numpy() & np.isfinite(age))
    rows = np.sort(np.random.default_rng(SEED).choice(pool, 10000, replace=False))
    X = corpus.get(rows).astype(np.float64)
    g = corpus.profiles.group.to_numpy()[rows].astype(str)
    y = age[rows]
    df = pd.DataFrame(X); m = df.groupby(g).transform("mean").to_numpy()
    Xc = X - m
    yc = y - pd.Series(y).groupby(g).transform("mean").to_numpy()
    fin = np.isfinite(Xc)
    Xc = np.where(fin, Xc, 0.0)
    num = (Xc * yc[:, None] * fin).sum(0)
    den = np.sqrt((Xc ** 2).sum(0) * ((yc[:, None] ** 2) * fin).sum(0))
    r = num / np.where(den > 0, den, np.nan)
    np.savez(OUT / "direction.npz", r=r, n=fin.sum(0), rows=rows)
    log(f"direction written; finite r for {np.isfinite(r).sum()} CpGs")


def manifest_tfbs(cpgs):
    with zipfile.ZipFile(require(MANIFEST)) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            m = pd.read_csv(f, skiprows=7, usecols=["Name", "TFBS_NAME"], dtype=str, low_memory=False)
    m = m[m.Name.str.startswith("cg", na=False)].drop_duplicates("Name").set_index("Name")
    return m.reindex(cpgs).TFBS_NAME.notna().to_numpy()


def fisher(top, inset):
    from scipy.stats import fisher_exact
    from statsmodels.stats.contingency_tables import Table2x2
    a = int((top & inset).sum()); b = int((top & ~inset).sum()); c = int((~top & inset).sum()); d = int((~top & ~inset).sum())
    t = Table2x2(np.array([[a, b], [c, d]]) + (0.5 if min(a, b, c, d) == 0 else 0))
    return {"top_in_set": a, "top": a + b, "background_in_set": c, "background": c + d, "fraction_top": a / max(a + b, 1),
            "fraction_background": c / max(c + d, 1), "odds_ratio": float(t.oddsratio), "ci95": [float(v) for v in t.oddsratio_confint()],
            "p": float(fisher_exact([[a, b], [c, d]])[1])}


def mantel_haenszel(top, inset, strata):
    from statsmodels.stats.contingency_tables import StratifiedTable
    tabs = []
    for s in np.unique(strata):
        k = strata == s
        tb = np.array([[(top & inset & k).sum(), (top & ~inset & k).sum()], [(~top & inset & k).sum(), (~top & ~inset & k).sum()]], float)
        if (tb.sum(1) > 0).all() and (tb.sum(0) > 0).all():
            tabs.append(tb + 0.5 * (tb.min() == 0))
    st = StratifiedTable(tabs)
    return {"odds_ratio": float(st.oddsratio_pooled), "ci95": [float(v) for v in st.oddsratio_pooled_confint()], "p": float(st.test_null_odds().pvalue)}


def gometh(sig, allc, tag):
    d = OUT / f"gometh_{tag}"; d.mkdir(parents=True, exist_ok=True)
    pd.Series(sig).to_csv(d / "sig.txt", index=False, header=False); pd.Series(allc).to_csv(d / "all.txt", index=False, header=False)
    script = f"""
suppressMessages(library(missMethyl))
sig <- readLines("{d}/sig.txt"); allc <- readLines("{d}/all.txt")
for (col in c("GO", "KEGG")) {{
  r <- gometh(sig.cpg = sig, all.cpg = allc, collection = col, array.type = "EPIC", prior.prob = TRUE)
  r <- r[order(r$P.DE), ]
  write.csv(r, file.path("{d}", paste0(col, ".csv")))
}}
"""
    (d / "run.R").write_text(script)
    p = subprocess.run(["Rscript", str(d / "run.R")], capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"gometh failed: {p.stderr[-2000:]}")
    out = {}
    for col in ("GO", "KEGG"):
        r = pd.read_csv(d / f"{col}.csv", index_col=0)
        term = "TERM" if "TERM" in r else ("Description" if "Description" in r else r.columns[0])
        out[col] = {"terms_fdr_0.05": int((r.FDR < 0.05).sum()), "terms_tested": int(len(r)),
                    "top20": [{"id": str(i), "term": str(row[term]), "n": int(row["N"]), "de": int(row["DE"]), "p": float(row["P.DE"]), "fdr": float(row["FDR"])}
                              for i, row in r.head(20).iterrows()]}
    return out


def energy():
    import torch
    from sklearn.decomposition import PCA
    from methylgpt2_lib import Corpus, PPCACore
    br = _module("benchmark_representations", "benchmark_representations_2026-10-01.py")
    corpus = Corpus(); prof = corpus.profiles
    plat = np.load(require(ARTIFACTS / "platform_vocab_masks_2026-10-01.npz"))
    core = PPCACore.load(require(ARTIFACTS / "ppca_core_meanfilled_2026-09-30.npz"), rank=1536)
    mu = core.mu.numpy()
    rows_a = np.load(require(EVAL / "representations_setA_2026-10-01.npz"))["rows"]
    Xa = corpus.get(np.sort(rows_a))[np.argsort(np.argsort(rows_a))]
    ga = prof.group.to_numpy()[rows_a].astype(str); tissue = prof.tissue.to_numpy()[rows_a].astype(str); sex = prof.sex.to_numpy()[rows_a]
    res = {}
    rng = np.random.default_rng(SEED)
    uniq = np.unique(ga); idx = [np.flatnonzero(ga == u) for u in uniq]
    draws = [np.concatenate([idx[i] for i in rng.integers(0, len(uniq), len(uniq))]) for _ in range(1000)]
    for scen in ("all", "hm27", "random_70"):
        X = Xa.copy()
        if scen == "hm27":
            X[:, ~plat["hm27"]] = np.nan
        elif scen == "random_70":
            X[np.random.default_rng(20261071).random(X.shape) < 0.7] = np.nan
        raw = np.where(np.isfinite(X), X, mu[None, :])
        with torch.no_grad():
            S = np.concatenate([core.posterior(torch.from_numpy(X[i:i + 64]), torch.from_numpy(np.isfinite(X[i:i + 64])).float()).numpy() for i in range(0, len(X), 64)])
        P = {"latent": PCA(50, random_state=0).fit_transform(S - S.mean(0)), "raw": PCA(50, random_state=0).fit_transform(raw - raw.mean(0))}
        for task, labels, keep in (("tissue", tissue, np.ones(len(tissue), bool)), ("sex", sex, np.isin(sex, ["M", "F"]))):
            v = {k: br.energy_distance(Pk[keep], labels[keep]) for k, Pk in P.items()}
            diffs = []
            for t in draws:
                t = t[keep[t]]
                diffs.append(br.energy_distance(P["latent"][t], labels[t]) - br.energy_distance(P["raw"][t], labels[t]))
            res[f"{scen}|{task}"] = {"latent": v["latent"], "raw": v["raw"], "difference": v["latent"] - v["raw"],
                                     "ci95": [float(x) for x in np.percentile(diffs, [2.5, 97.5])], "profiles": int(keep.sum()),
                                     "datasets": int(len(np.unique(ga[keep])))}
            log(f"energy {scen} {task}: {res[f'{scen}|{task}']}")
    write_json(OUT / "energy.json", res)


def analyse():
    ann = pd.read_parquet(require(ARTIFACTS / "cpg_annotation_2026-10-01.parquet"))
    if not PRC2.exists():
        PRC2.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(require(PRC2_SRC), PRC2)
    prc2_genes = set(l.strip() for l in PRC2.read_text().split("\n") if l.strip())
    A = np.load(require(OUT / "attribution.npz")); dz = np.load(require(OUT / "direction.npz"))
    score = np.mean([A[f"score|{r}"] for r in RUNS], axis=0)
    r_age = dz["r"]
    bg = ann.in_epic_v1.to_numpy() & (np.mean([A[f"n_observed|{r}"] for r in RUNS], axis=0) > 0)
    genes = ann.genes.fillna("").astype(str).str.split(";")
    sets = {"PRC2 targets": genes.map(lambda gs: any(g in prc2_genes for g in gs)).to_numpy(),
            "TFBS": manifest_tfbs(ann.cpg.to_numpy()),
            "CpG island": (ann.island_relation == "Island").to_numpy(),
            "CpG shore": ann.island_relation.astype(str).str.contains("Shore").to_numpy(),
            "enhancer": ann.enhancer.fillna(False).to_numpy(bool),
            "DNase hypersensitivity": ann.dnase_hs.fillna(False).to_numpy(bool)}
    island = ann.island_relation.fillna("OpenSea").astype(str).str.replace("N_|S_", "", regex=True).to_numpy()
    res = {"background_cpgs": int(bg.sum()), "prc2_genes": len(prc2_genes), "prc2_list_sha256": hashlib.sha256(PRC2.read_bytes()).hexdigest(),
           "checks": json.loads((OUT / "attribution_checks.json").read_text()), "rankings": {}}
    rows = []
    for rank_name, val in (("MethylGPT attribution", score), ("univariate age correlation (reference)", np.abs(r_age))):
        v = np.where(bg & np.isfinite(val), val, -np.inf)
        order = np.argsort(-v)
        blk = {}
        for n in (500, 1000, 2000):
            top = np.zeros(len(v), bool); top[order[:n]] = True
            b = {k: fisher(top[bg], s[bg]) for k, s in sets.items()}
            b["PRC2 targets, stratified by CpG-island relation"] = mantel_haenszel(top[bg], sets["PRC2 targets"][bg], island[bg])
            up = (r_age > 0)
            b["methylation rises with age"] = {"top": float(np.mean(up[top & bg])), "background": float(np.mean(up[bg & ~top])),
                                               "top_prc2": float(np.mean(up[top & bg & sets["PRC2 targets"]])),
                                               "background_prc2": float(np.mean(up[bg & ~top & sets["PRC2 targets"]])),
                                               "fisher": fisher(top[bg], up[bg])}
            if n == 1000:
                b["gometh"] = gometh(ann.cpg.to_numpy()[top], ann.cpg.to_numpy()[bg], ("attr" if rank_name.startswith("MethylGPT") else "univ"))
            blk[str(n)] = b
            for k, f in b.items():
                if isinstance(f, dict) and "odds_ratio" in f:
                    rows.append({"ranking": rank_name, "top": n, "set": k, **{kk: (str(vv) if isinstance(vv, list) else vv) for kk, vv in f.items()}})
        res["rankings"][rank_name] = blk
    res["overlap_top1000_attribution_vs_univariate"] = int(len(set(np.argsort(-np.where(bg, score, -np.inf))[:1000]) &
                                                               set(np.argsort(-np.where(bg & np.isfinite(r_age), np.abs(r_age), -np.inf))[:1000])))
    res["primary"] = {"definition": "PRC2-target CpGs among the 1,000 CpGs with the highest MethylGPT age attribution, Fisher's exact test against the rest of the EPIC v1 vocabulary",
                      **res["rankings"]["MethylGPT attribution"]["1000"]["PRC2 targets"]}
    en = OUT / "energy.json"
    res["energy_distances"] = json.loads(en.read_text()) if en.exists() else "not run"
    pd.DataFrame(rows).to_csv(EVAL / "si_enrichment_attribution_2026-10-21.csv", index=False)
    write_json(EVAL / "si_enrichment_attribution_2026-10-21.json", res)
    log(f"primary {res['primary']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["attribute", "direction", "energy", "analyse"], required=True)
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()
    {"attribute": lambda: attribute(a.device), "direction": direction, "energy": energy, "analyse": analyse}[a.stage]()


if __name__ == "__main__":
    main()
