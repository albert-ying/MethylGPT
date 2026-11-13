#!/usr/bin/env python3
"""Supplementary analyses of Fig. 3a batch effects, Fig. 5b attention and gene sets.

Written on 22 Oct 2026, before running. Every result is reported, whichever model it favours.

H  Harmonization across array generations. Set A (2,371 profiles, six tissues, 43 test datasets). Each profile is
   represented from all its CpGs and from the vocabulary CpGs of the 27K, 450K or EPIC v2 array, or with 50%, 70% or 90%
   of its CpGs removed at random (seed 20261022): latent scores computed from the remaining CpGs (input layer
   ppca_core_meanfilled_2026-09-30.npz), raw beta values with missing CpGs filled with training means, and the first 50
   principal components of raw beta values (PCA fitted on the complete profiles, applied to the filled reduced profiles).
   Retrieval: cosine similarity, after centring on the mean of the complete representations, between each reduced
   profile and the complete representations of all 2,371 profiles; top-1 accuracy (the nearest complete profile is the
   profile itself) and the cosine similarity with itself. Cross-array agreement: the same with the 450K-array (or
   27K-array) representation as query and the EPIC v2-array representations of all profiles as targets.
   PRIMARY (H): top-1 retrieval accuracy with 27K-array CpGs, latent scores minus raw beta values, paired 95% interval
   resampling datasets (1,000 draws, seed 20261022).
A  Attribution follow-up, from artifacts/si_enrichment_2026-10-21/attribution.npz (gradient x input of the released age
   output, three runs, 2,000 held-out profiles). (A1) Overlap of the 1,000 CpGs with the highest attribution with the
   CpGs of published age clocks in the vocabulary (Horvath multi-tissue, Horvath skin and blood, Hannum, PhenoAge; BioLearn
   coefficient files), Fisher's exact test against the EPIC v1 vocabulary background, for each clock and their union.
   (A2) Age-specific attribution: the same gradient x input on held-out profiles younger than 30 years and 60 years or
   older (up to 1,000 each, seed 20261022); Spearman correlation of the young and old CpG scores, overlap of their top
   1,000, and enrichment (PRC2 targets, TFBS, CpG island) of the 1,000 CpGs whose attribution rises most from young to old.
   PRIMARY (A): Fisher odds ratio for the union of the four clocks' CpGs among the top 1,000 attribution CpGs.
G  Gene sets: why gometh found no term at FDR < 0.05. (G1) gometh on the top 1,000 attribution CpGs split by whether
   methylation rises or falls with age (within-dataset correlation in training profiles); (G2) gometh with the whole
   EPIC array as background (gometh's default) instead of the vocabulary; (G3) gometh on the top 2,000 and 5,000;
   (G4) a rank-based test over all genes: gene score = mean attribution of the gene's EPIC vocabulary CpGs, fgsea
   (scoreType "pos", gene sets of 15 to 500 genes, multilevel) on MSigDB Hallmark, GO biological process, KEGG (legacy)
   and Reactome (msigdbr). Diagnostics: genes covered by the top 1,000 and by the background, and the fraction of top
   CpGs annotated to a gene, against the background.
   PRIMARY (G): number of Hallmark gene sets at FDR < 0.05 in G4.

Stages: --stage harmonize (GPU), --stage attribution_age (GPU), --stage genesets (R), --stage analyse.
Outputs: artifacts/si_followup_2026-10-22/ and artifacts/eval/si_followup_2026-10-22.json/.csv
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

from methylgpt_lib import ARTIFACTS, CORPUS_DIR, log, require, write_json

HERE = Path(__file__).resolve().parent
EVAL = ARTIFACTS / "eval"
OUT = ARTIFACTS / "si_followup_2026-10-22"
ENR = ARTIFACTS / "si_enrichment_2026-10-21"
CLOCKS = {"Horvath multi-tissue": "Horvath1.csv", "Horvath skin and blood": "Horvath2.csv", "Hannum": "Hannum.csv", "PhenoAge": "PhenoAge.csv"}
CLOCK_DIR = Path("/home/kying/anaconda3/lib/python3.12/site-packages/biolearn/data")
SEED = 20261022
SCEN = ["hm27", "hm450", "epic_v2", "random_50", "random_70", "random_90"]


def _module(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def boot_diff(a, b, groups, draws=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups); idx = [np.flatnonzero(groups == g) for g in uniq]
    vals = []
    for _ in range(draws):
        t = np.concatenate([idx[i] for i in rng.integers(0, len(uniq), len(uniq))])
        vals.append(a[t].mean() - b[t].mean())
    return [float(v) for v in np.percentile(vals, [2.5, 97.5])]


def harmonize(device):
    import torch
    from sklearn.decomposition import PCA
    from methylgpt_lib import Corpus, PPCACore
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)
    corpus = Corpus(); prof = corpus.profiles
    plat = np.load(require(ARTIFACTS / "platform_vocab_masks_2026-10-01.npz"))
    core = PPCACore.load(require(ARTIFACTS / "ppca_core_meanfilled_2026-09-30.npz"), rank=1536).to(dev)
    mu = core.mu.cpu().numpy()
    rows = np.load(require(EVAL / "representations_setA_2026-10-01.npz"))["rows"]
    Xa = corpus.get(np.sort(rows))[np.argsort(np.argsort(rows))]
    g = prof.group.to_numpy()[rows].astype(str)

    def reduce(s):
        X = Xa.copy()
        if s.startswith("random_"):
            p = int(s.split("_")[1]) / 100
            X[np.random.default_rng(SEED + int(p * 100)).random(X.shape) < p] = np.nan
        elif s != "all":
            X[:, ~plat[s]] = np.nan
        return X

    def latent(X):
        with torch.no_grad():
            return np.concatenate([core.posterior(torch.from_numpy(X[i:i + 64]).to(dev), torch.from_numpy(np.isfinite(X[i:i + 64])).to(dev).float()).cpu().numpy()
                                   for i in range(0, len(X), 64)])

    reps = {}
    for s in ["all"] + SCEN:
        X = reduce(s)
        reps[("latent", s)] = latent(X)
        reps[("raw", s)] = np.where(np.isfinite(X), X, mu[None, :])
        log(f"representations {s} done")
    pca = PCA(50, random_state=0).fit(reps[("raw", "all")])
    for s in ["all"] + SCEN:
        reps[("raw_pca50", s)] = pca.transform(reps[("raw", s)])

    def retrieval(Q, T):
        c = T.mean(0)
        Qn = (Q - c); Qn /= np.linalg.norm(Qn, axis=1, keepdims=True)
        Tn = (T - c); Tn /= np.linalg.norm(Tn, axis=1, keepdims=True)
        S = Qn @ Tn.T
        top1 = (S.argmax(1) == np.arange(len(S))).astype(float)
        selfcos = np.diag(S).copy()
        rank = (S > np.diag(S)[:, None]).sum(1) + 1
        return top1, selfcos, rank

    res = {}
    for kind in ("latent", "raw", "raw_pca50"):
        for s in SCEN:
            top1, sc, rk = retrieval(reps[(kind, s)], reps[(kind, "all")])
            res[f"{kind}|{s}|to_complete"] = {"top1": top1, "selfcos": sc, "rank": rk}
        for q in ("hm450", "hm27"):
            top1, sc, rk = retrieval(reps[(kind, q)], reps[(kind, "epic_v2")])
            res[f"{kind}|{q}|to_epic_v2"] = {"top1": top1, "selfcos": sc, "rank": rk}
    np.savez(OUT / "harmonize.npz", groups=g, **{f"{k}|{m}": v[m] for k, v in res.items() for m in v})
    log("harmonize written")


def attribution_age(device):
    import torch
    from methylgpt_lib import Corpus, cuda_preinit, Finv
    from methylgpt_model import BioHeads, Heads, load_checkpoint, embed_batch
    se = _module("si_enrichment_attribution", "si_enrichment_attribution_2026-10-21.py")
    cuda_preinit()
    dev = torch.device(device)
    OUT.mkdir(parents=True, exist_ok=True)
    corpus = Corpus()
    lab = pd.read_parquet(require(CORPUS_DIR / "profiles_age_clean_2026-10-01.parquet"))
    age = lab.age_meta.to_numpy(float)
    ho = np.flatnonzero((corpus.profiles.split == "test").to_numpy() & np.isfinite(age))
    rng = np.random.default_rng(SEED)
    groups = {"young": ho[age[ho] < 30], "old": ho[age[ho] >= 60]}
    groups = {k: np.sort(rng.choice(v, min(1000, len(v)), replace=False)) for k, v in groups.items()}
    out = {f"rows|{k}": v for k, v in groups.items()}
    for name, rel in se.RUNS.items():
        ck = torch.load(require(ARTIFACTS / rel), map_location=dev, weights_only=False)
        model, core, _ = load_checkpoint(Path(ck["ckpt"]), ck.get("config", {}).get("core"), dev)
        model.load_state_dict(ck["model"]); model.eval()
        d_in = core.rank + model.latents.shape[1]
        heads = (BioHeads(d_in, len(ck["tissues"]) + 1, len(ck["bio_names"])) if "bio_names" in ck else Heads(d_in, len(ck["tissues"]) + 1)).to(dev)
        heads.load_state_dict(ck["heads"]); heads.eval()
        for obj in {id(core): core, id(model.core): model.core}.values():
            obj.posterior = types.MethodType(type(obj).posterior.__wrapped__, obj)
            obj.predict = types.MethodType(type(obj).predict.__wrapped__, obj)
        model.inputs = types.MethodType(type(model).inputs.__wrapped__, model)
        for mod in (model, heads, core):
            for prm in mod.parameters():
                prm.requires_grad_(False)
        mu = core.mu.detach()
        for gname, rows in groups.items():
            X = corpus.get(rows)
            S = np.zeros(X.shape[1]); N = np.zeros(X.shape[1])
            for i in range(0, len(X), 4):
                xb = torch.from_numpy(X[i:i + 4]).to(dev)
                ctx = torch.isfinite(xb)
                x = torch.where(ctx, xb, mu).requires_grad_(True)
                heads(embed_batch(model, core, x, ctx))[0].sum().backward()
                a = (x.grad * (x.detach() - mu) * ctx).cpu().numpy()
                S += np.abs(a).sum(0); N += ctx.cpu().numpy().sum(0)
            out[f"score|{gname}|{name}"] = S / np.maximum(N, 1)
            log(f"{name} {gname}: done")
        del model, core, heads, ck
        torch.cuda.empty_cache()
    np.savez(OUT / "attribution_age.npz", **out)


def genesets():
    OUT.mkdir(parents=True, exist_ok=True)
    ann = pd.read_parquet(require(ARTIFACTS / "cpg_annotation_2026-10-01.parquet"))
    A = np.load(require(ENR / "attribution.npz")); dz = np.load(require(ENR / "direction.npz"))
    runs = [k.split("|", 1)[1] for k in A.files if k.startswith("score|")]
    score = np.mean([A[f"score|{r}"] for r in runs], axis=0)
    bg = ann.in_epic_v1.to_numpy() & (np.mean([A[f"n_observed|{r}"] for r in runs], axis=0) > 0)
    order = np.argsort(-np.where(bg, score, -np.inf))
    cpg = ann.cpg.to_numpy()
    d = OUT / "genesets"; d.mkdir(parents=True, exist_ok=True)
    top1000 = order[:1000]
    up = dz["r"] > 0
    lists = {"top1000_up": cpg[top1000][up[top1000]], "top1000_down": cpg[top1000][~up[top1000]], "top2000": cpg[order[:2000]],
             "top5000": cpg[order[:5000]], "top1000": cpg[top1000]}
    for k, v in lists.items():
        pd.Series(v).to_csv(d / f"{k}.txt", index=False, header=False)
    pd.Series(cpg[bg]).to_csv(d / "background.txt", index=False, header=False)
    genes = ann.genes.fillna("").astype(str).str.split(";")
    rows = [(gname, score[i]) for i in np.flatnonzero(bg) for gname in set(genes[i]) if gname]
    gs = pd.DataFrame(rows, columns=["gene", "score"]).groupby("gene").score.mean()
    gs.to_csv(d / "gene_scores.csv")
    diag = {"top1000_cpgs_with_gene": float(np.mean([bool(genes[i][0]) for i in top1000])),
            "background_cpgs_with_gene": float(np.mean([bool(genes[i][0]) for i in np.flatnonzero(bg)])),
            "genes_top1000": int(len({x for i in top1000 for x in genes[i] if x})), "genes_background": int(len(gs)),
            "top1000_up": int(up[top1000].sum()), "top1000_down": int((~up[top1000]).sum())}
    write_json(d / "diagnostics.json", diag)
    script = f"""
suppressMessages({{library(missMethyl); library(fgsea); library(msigdbr)}})
d <- "{d}"
bg <- readLines(file.path(d, "background.txt"))
go1 <- function(sig, all, tag) {{
  for (col in c("GO", "KEGG")) {{
    r <- gometh(sig.cpg = sig, all.cpg = all, collection = col, array.type = "EPIC", prior.prob = TRUE)
    r <- r[order(r$P.DE), ]
    write.csv(r, file.path(d, paste0("gometh_", tag, "_", col, ".csv")))
  }}
}}
go1(readLines(file.path(d, "top1000_up.txt")), bg, "top1000_up")
go1(readLines(file.path(d, "top1000_down.txt")), bg, "top1000_down")
go1(readLines(file.path(d, "top1000.txt")), NULL, "top1000_arraybg")
go1(readLines(file.path(d, "top2000.txt")), bg, "top2000")
go1(readLines(file.path(d, "top5000.txt")), bg, "top5000")
gs <- read.csv(file.path(d, "gene_scores.csv"))
stats <- setNames(gs$score, gs$gene)
get_sets <- function(coll, sub = NULL) {{
  m <- tryCatch(if (is.null(sub)) msigdbr(species = "Homo sapiens", collection = coll) else msigdbr(species = "Homo sapiens", collection = coll, subcollection = sub),
                error = function(e) if (is.null(sub)) msigdbr(species = "Homo sapiens", category = coll) else msigdbr(species = "Homo sapiens", category = coll, subcategory = sub))
  split(m$gene_symbol, m$gs_name)
}}
colls <- list(Hallmark = c("H", NA), GO_BP = c("C5", "GO:BP"), KEGG_legacy = c("C2", "CP:KEGG_LEGACY"), Reactome = c("C2", "CP:REACTOME"))
set.seed({SEED})
for (nm in names(colls)) {{
  cc <- colls[[nm]]
  sets <- get_sets(cc[1], if (is.na(cc[2])) NULL else cc[2])
  r <- fgseaMultilevel(pathways = sets, stats = stats, scoreType = "pos", minSize = 15, maxSize = 500)
  r <- r[order(r$pval), ]
  r$leadingEdge <- sapply(r$leadingEdge, function(x) paste(head(x, 15), collapse = ";"))
  write.csv(as.data.frame(r), file.path(d, paste0("fgsea_", nm, ".csv")), row.names = FALSE)
}}
"""
    (d / "run.R").write_text(script)
    p = subprocess.run(["Rscript", str(d / "run.R")], capture_output=True, text=True)
    (d / "run.log").write_text(p.stdout + "\n" + p.stderr)
    if p.returncode != 0:
        raise SystemExit(f"R failed: {p.stderr[-3000:]}")
    log("genesets written")


def analyse():
    from scipy.stats import spearmanr
    se = _module("si_enrichment_attribution", "si_enrichment_attribution_2026-10-21.py")
    res = {}
    # H
    Hz = np.load(require(OUT / "harmonize.npz")); g = Hz["groups"]
    hres = {}
    for key in sorted({k.rsplit("|", 1)[0] for k in Hz.files if k != "groups"}):
        hres[key] = {"top1": float(Hz[f"{key}|top1"].mean()), "selfcos_median": float(np.median(Hz[f"{key}|selfcos"])),
                     "rank_median": float(np.median(Hz[f"{key}|rank"]))}
    for key in [k for k in hres if k.startswith("latent|")]:
        rest = key.split("|", 1)[1]
        for comp in ("raw", "raw_pca50"):
            a, b = Hz[f"{key}|top1"], Hz[f"{comp}|{rest}|top1"]
            hres[key][f"top1_minus_{comp}"] = float(a.mean() - b.mean())
            hres[key][f"top1_minus_{comp}_ci95"] = boot_diff(a, b, g)
    res["H_harmonization"] = hres
    res["primary_H"] = {"definition": "top-1 retrieval of the complete profile from 27K-array CpGs, latent minus raw beta values (set A, 2,371 profiles, 43 datasets)",
                        "latent": hres["latent|hm27|to_complete"]["top1"], "raw": hres["raw|hm27|to_complete"]["top1"],
                        "difference": hres["latent|hm27|to_complete"]["top1_minus_raw"], "ci95": hres["latent|hm27|to_complete"]["top1_minus_raw_ci95"]}
    # A1
    ann = pd.read_parquet(require(ARTIFACTS / "cpg_annotation_2026-10-01.parquet"))
    A = np.load(require(ENR / "attribution.npz"))
    runs = [k.split("|", 1)[1] for k in A.files if k.startswith("score|")]
    score = np.mean([A[f"score|{r}"] for r in runs], axis=0)
    bg = ann.in_epic_v1.to_numpy() & (np.mean([A[f"n_observed|{r}"] for r in runs], axis=0) > 0)
    order = np.argsort(-np.where(bg, score, -np.inf)); top = np.zeros(len(score), bool); top[order[:1000]] = True
    cpg = ann.cpg.to_numpy()
    sets = {}
    for name, f in CLOCKS.items():
        c = pd.read_csv(require(CLOCK_DIR / f)).CpGmarker.astype(str)
        sets[name] = np.isin(cpg, c.to_numpy())
    sets["union of four clocks"] = np.any(list(sets.values()), axis=0)
    a1 = {k: {**se.fisher(top[bg], s[bg]), "clock_cpgs_in_background": int(s[bg].sum())} for k, s in sets.items()}
    res["A1_clock_overlap"] = a1
    res["primary_A"] = {"definition": "union of Horvath multi-tissue, Horvath skin and blood, Hannum and PhenoAge CpGs among the top 1,000 attribution CpGs", **a1["union of four clocks"]}
    # A2
    za = np.load(require(OUT / "attribution_age.npz"))
    young = np.mean([za[f"score|young|{r}"] for r in runs], axis=0); old = np.mean([za[f"score|old|{r}"] for r in runs], axis=0)
    rho = spearmanr(young[bg], old[bg]).correlation
    ty = set(np.argsort(-np.where(bg, young, -np.inf))[:1000]); to = set(np.argsort(-np.where(bg, old, -np.inf))[:1000])
    rise = np.where(bg, np.log((old + 1e-12) / (young + 1e-12)), -np.inf)
    trise = np.zeros(len(score), bool); trise[np.argsort(-rise)[:1000]] = True
    prc2 = set(l.strip() for l in (ARTIFACTS / "external/BENPORATH_PRC2_TARGETS.txt").read_text().split("\n") if l.strip())
    genes = ann.genes.fillna("").astype(str).str.split(";")
    s2 = {"PRC2 targets": genes.map(lambda gs_: any(x in prc2 for x in gs_)).to_numpy(), "TFBS": se.manifest_tfbs(cpg),
          "CpG island": (ann.island_relation == "Island").to_numpy()}
    res["A2_age_specific"] = {"n_young": int(len(za["rows|young"])), "n_old": int(len(za["rows|old"])), "spearman_young_old": float(rho),
                              "top1000_overlap_young_old": int(len(ty & to)),
                              "rising_top1000_enrichment": {k: se.fisher(trise[bg], s[bg]) for k, s in s2.items()},
                              "top1000_old_enrichment": {k: se.fisher(np.isin(np.arange(len(score)), list(to))[bg], s[bg]) for k, s in s2.items()}}
    # G
    d = OUT / "genesets"
    G = {"diagnostics": json.loads((d / "diagnostics.json").read_text()), "gometh": {}, "fgsea": {}}
    for f in sorted(d.glob("gometh_*.csv")):
        r = pd.read_csv(f, index_col=0)
        term = "TERM" if "TERM" in r else ("Description" if "Description" in r else r.columns[0])
        G["gometh"][f.stem] = {"fdr_0.05": int((r.FDR < 0.05).sum()), "tested": int(len(r)),
                               "top10": [{"term": str(x[term]), "de": int(x["DE"]), "n": int(x["N"]), "fdr": float(x["FDR"])} for _, x in r.head(10).iterrows()]}
    for f in sorted(d.glob("fgsea_*.csv")):
        r = pd.read_csv(f)
        G["fgsea"][f.stem] = {"fdr_0.05": int((r.padj < 0.05).sum()), "tested": int(len(r)),
                              "top10": [{"pathway": str(x.pathway), "size": int(x["size"]), "NES": float(x.NES), "padj": float(x.padj)} for _, x in r.head(10).iterrows()]}
    res["G_genesets"] = G
    res["primary_G"] = {"definition": "Hallmark gene sets at FDR < 0.05, fgsea on gene-level attribution", **{k: G["fgsea"]["fgsea_Hallmark"][k] for k in ("fdr_0.05", "tested")}}
    rows = [{"analysis": "H", "key": k, **{kk: (str(vv) if isinstance(vv, list) else vv) for kk, vv in v.items()}} for k, v in hres.items()]
    rows += [{"analysis": "A1", "key": k, **{kk: (str(vv) if isinstance(vv, list) else vv) for kk, vv in v.items()}} for k, v in a1.items()]
    pd.DataFrame(rows).to_csv(EVAL / "si_followup_2026-10-22.csv", index=False)
    write_json(EVAL / "si_followup_2026-10-22.json", res)
    log(f"primaries H {res['primary_H']} | A {res['primary_A']['odds_ratio']:.2f} {res['primary_A']['ci95']} | G {res['primary_G']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["harmonize", "attribution_age", "genesets", "analyse"], required=True)
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()
    {"harmonize": lambda: harmonize(a.device), "attribution_age": lambda: attribution_age(a.device), "genesets": genesets, "analyse": analyse}[a.stage]()


if __name__ == "__main__":
    main()
