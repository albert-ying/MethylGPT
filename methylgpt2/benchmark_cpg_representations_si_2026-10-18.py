#!/usr/bin/env python3
"""Supplementary benchmark: CpG representations, DNase hypersensitivity and baselines (18 Oct 2026).

Extends benchmark_cpg_representations_2026-10-01.py (main Fig. 2, Supplementary Table 3) for
the Supplementary Information. Everything is computed on methylgpt2; no result of the
originally submitted model is used.

PRE-SPECIFIED PRIMARY ANALYSIS (fixed before the first run)
  Metric: balanced accuracy of 10-nearest-neighbour classification (uniform votes, ties to the
    first class in sorted order as in scikit-learn), cosine distance on the column-standardized
    representation, out of fold in 5-fold cross-validation with folds grouped by 1 Mb genomic
    blocks (chromosome, floor(position / 10^6)). Blocks are assigned to folds once (seed 20261018,
    folds balanced for CpG count) and the same folds serve every representation and annotation.
  Annotations: relation to CpG island (island, shore, shelf, open sea), enhancer (450K or FANTOM5)
    and DNase hypersensitivity, all defined only for EPIC v1 CpGs (47,962; others NA); chromosome X
    versus autosomes (chromosome Y excluded), reported in its own column because it reflects sex
    dosage in the corpus.
  Primary comparisons: input-layer loadings minus raw_pca64, and input-layer loadings minus the
    baseline features, for each annotation. Baseline features: mean beta, SD and probe type
    (Infinium I or II) of each CpG; mean and SD over the observed values of the same 2,000 training
    profiles as raw_profiles. The baseline is classified by kNN with Euclidean distance on the three
    standardized features (cosine distance is degenerate in three dimensions).
  Uncertainty: 95% percentile intervals from 2,000 bootstrap resamples of 1 Mb blocks applied to the
    fixed out-of-fold predictions, paired across representations (same resampled blocks). No P values.

SECONDARY (all reported)
  - Baseline-adjusted accuracy: kNN restricted to CpGs of the same stratum (mean-beta decile x probe
    type, 20 strata); within-stratum balanced accuracy over classes with >= 20 CpGs in the stratum,
    averaged over strata with >= 2 such classes, weighted by stratum size. Also: kNN on the
    representation after residualizing every feature on indicators of mean-beta decile x probe type
    and of SD decile.
  - DNase within each CpG-island stratum (kNN restricted to the stratum): balanced accuracy and AUC.
  - Island, enhancer and DNase on autosomes only (chromosome X excluded).
  - One-vs-rest AUC per class from out-of-fold vote fractions; confusion matrices; distribution of
    vote fractions for the four island classes (threshold question).
  - Cosine silhouette (50 PCs, 10,000 CpGs), mean pairwise energy distance between classes (50 PCs
    scaled to unit total variance, 2,000 CpGs per class), ARI and NMI of k-means (k = number of
    classes, unit-normalized 50 PCs) against the annotation.
  - Ungrouped 5-fold stratified CV as in v13 (StratifiedKFold, seed 20261001), for continuity.
  - Paired differences of every representation from the loadings.

REPRESENTATIONS (one vector per vocabulary CpG, 49,156)
  loadings          input-layer loadings V * sqrt(lambda) (1,536), as in the existing script
  loadings_natural  the same without column standardization (the geometry of the Fig. 2a UMAP)
  encoder_input     released encoder input weights for the observation mask (dense_w; the v13
                    "encoder input weights"), 2,048
  dense_c, dense_e  released encoder input weights for centred values and for input-layer residuals
  q_emb             decoder query embedding (512)
  query_full        full decoder query, q_emb + q_load(V sqrt(lambda)) (512)
  h_full            decoder state h = dec2(dec1(q, z), z), full query, self-supervised checkpoint
                    (encoder and decoder trained together), mean over 2,000 held-out profiles
  h_qemb            the same with q_emb alone as the query
  h_random          the same architecture at random initial weights (seed 20261018), full query
  h_release         released encoder with its decoder, full query (the decoder is identical to the
                    self-supervised one; the encoder was fine-tuned for age and clocks)
  raw_profiles      beta values in 2,000 random training profiles (training-mean filled)
  raw_heldout       beta values in the same 2,000 held-out profiles used for h (training-mean filled)
  raw_pca64         first 64 principal components of raw_profiles
  baseline          mean beta, SD and probe type (Euclidean kNN)
  random            Gaussian vectors (1,536)
Held-out profiles: 2,000 drawn at random (seed 20261018) from the test split (datasets never used
for training). The h representations are cached in artifacts/eval/cpg_decoder_states_2026-10-18.npz.

DNase source: EPIC v1.0 B5 manifest (DNase_Hypersensitivity_NAME), ENCODE DNase I hypersensitive
sites summarised across cell types by Illumina, not tissue-specific. The manifest's
DNase_Hypersensitivity_Evidence_Count is added as an ordinal label only if it takes more than one
value among vocabulary CpGs.

Run (GPU 1 only):
  CUDA_VISIBLE_DEVICES=1 python benchmark_cpg_representations_si_2026-10-18.py --device cuda:0
  python benchmark_cpg_representations_si_2026-10-18.py --umap      # illustration only, CPU
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")  # GPU 0 runs pretraining jobs

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from methylgpt2_lib import ARTIFACTS, Corpus, PPCACore, cuda_preinit, log, require, write_json
from methylgpt2_model import MethylGPT2, load_checkpoint

SEED = 20261018
K = 10
N_FOLDS = 5
BLOCK_BP = 1_000_000
N_BOOT = 2000
MIN_CLASS_IN_STRATUM = 20
EPIC_ZIP = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/infinium-methylationepic-v-1-0-b5-manifest-file-csv.zip")
HM450 = Path("/Entropy/kying/09_shallowBS_is_all_you_need/data/HM450.hg19.manifest.tsv.gz")
RELEASE = ARTIFACTS / "release/methylgpt2-1.1"
SSL_CKPT = ARTIFACTS / "runs/stopped_cf_dense_2026-09-30/best.pt"
STATES = ARTIFACTS / "eval/cpg_decoder_states_2026-10-18.npz"
OUT_JSON = ARTIFACTS / "eval/cpg_representation_si_2026-10-18.json"
OUT_CSV = ARTIFACTS / "eval/cpg_representation_si_st3_rows_2026-10-18.csv"
OUT_FIG = ARTIFACTS / "eval/cpg_representation_si_figdata_2026-10-18.npz"
OUT_UMAP = ARTIFACTS / "eval/cpg_umap_si_2026-10-18.npz"
UMAP_GRID = [(15, 0.1), (15, 0.3), (30, 0.1), (30, 0.3), (50, 0.1), (50, 0.3)]

REP_NAMES = {
    "loadings": "Input-layer loadings",
    "loadings_natural": "Input-layer loadings, not standardized",
    "encoder_input": "Encoder input weights, observation mask",
    "dense_c": "Encoder input weights, centred values",
    "dense_e": "Encoder input weights, input-layer residuals",
    "q_emb": "Decoder query embedding",
    "query_full": "Decoder query, embedding plus projected loadings",
    "h_full": "Decoder state h, full query",
    "h_qemb": "Decoder state h, query embedding only",
    "h_random": "Decoder state h, randomly initialized",
    "h_release": "Decoder state h, released encoder",
    "raw_profiles": "Raw beta values, 2,000 training profiles",
    "raw_heldout": "Raw beta values, 2,000 held-out profiles",
    "raw_pca64": "Raw beta values, 64 PCs",
    "baseline": "Baseline: mean beta, SD, probe type",
    "random": "Random vectors",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_release():
    """Checksums of the released bundle against package/methylgpt2-1.1.checksums.sha256."""
    manifest = require(ARTIFACTS.parent / "package/methylgpt2-1.1.checksums.sha256")
    want = dict(line.split()[::-1] for line in manifest.read_text().split("\n") if line.strip())
    out = {}
    for name in ("config.json", "core.npz", "weights.pt"):
        got = sha256(require(RELEASE / name))
        if got != want[name]:
            raise SystemExit(f"checksum mismatch for {RELEASE / name}: {got} != {want[name]}")
        out[name] = got
    out["ssl_best.pt (no published checksum)"] = sha256(require(SSL_CKPT))
    log(f"checksums verified: {out}")
    return out


# ---------------------------------------------------------------------------------------- labels
def annotation():
    """Vocabulary annotation plus probe type and DNase evidence count from the manifests."""
    ann = pd.read_parquet(require(ARTIFACTS / "cpg_annotation_2026-10-01.parquet"))
    with zipfile.ZipFile(require(EPIC_ZIP)) as z:
        raw = z.read("infinium-methylationepic-v-1-0-b5-manifest-file.csv").decode()
    cols = ["Name", "CHR", "Infinium_Design_Type", "DNase_Hypersensitivity_NAME", "DNase_Hypersensitivity_Evidence_Count"]
    epic = pd.read_csv(io.StringIO(raw[raw.index("IlmnID,Name,"):].split("[Controls]")[0]), usecols=cols, low_memory=False)
    epic = epic.drop_duplicates("Name").set_index("Name").reindex(ann.cpg)
    hm = pd.read_csv(require(HM450), sep="\t", usecols=["Probe_ID", "type"]).drop_duplicates("Probe_ID").set_index("Probe_ID").reindex(ann.cpg)
    if not (epic.DNase_Hypersensitivity_NAME.notna().to_numpy() == ann.dnase_hs.to_numpy()).all():
        raise SystemExit("DNase flag in the annotation parquet disagrees with the manifest")
    ann["probe_type"] = epic.Infinium_Design_Type.fillna(hm["type"]).to_numpy()
    ann["dnase_evidence_count"] = epic.DNase_Hypersensitivity_Evidence_Count.to_numpy()
    return ann


def label_table(ann):
    epic = ann.in_epic_v1.to_numpy()
    labels = {
        "island_relation": np.where(epic, ann.island_relation.astype(object), None),
        "enhancer": np.where(epic, ann.enhancer.astype(str), None),
        "dnase_hs": np.where(epic, ann.dnase_hs.astype(str), None),
        "x_chromosome": np.where(ann.chr == "Y", None, np.where(ann.chr == "X", "X", "autosome")),
    }
    ev = ann.dnase_evidence_count[epic]
    distinct = sorted(float(v) for v in pd.unique(ev.dropna()))
    note = {"dnase_evidence_count_values_in_vocabulary": [float(v) for v in distinct],
            "dnase_evidence_count_annotated": int(ev.notna().sum()), "dnase_evidence_count_na_epic": int(ev.isna().sum())}
    if len(distinct) > 1:
        labels["dnase_evidence"] = np.where(epic, ann.dnase_evidence_count.fillna(0).astype(int).astype(str), None)
        note["ordinal_label_added"] = True
    else:
        note["ordinal_label_added"] = False
        note["reason"] = (f"DNase_Hypersensitivity_Evidence_Count equals {distinct} for every annotated vocabulary CpG and is "
                          "missing otherwise, so it carries no information beyond the binary flag")
    return labels, note


# ------------------------------------------------------------------------------- representations
def std(Z):
    Z = np.asarray(Z, np.float64)
    return (Z - Z.mean(0)) / np.where(Z.std(0) > 1e-8, Z.std(0), 1.0)


@torch.no_grad()
def decoder_states(model, core, X, S, dev, query="full", B=8, T=12288):
    """Mean over profiles of h = dec2(dec1(q, z), z) for every vocabulary CpG (deployment mode: all
    observed CpGs are context). X: (n, P) betas with NaN; S: (n, K) posterior latent scores."""
    P = core.V.shape[0]
    lam_sqrt = torch.sqrt(core.lam)
    acc = torch.zeros(P, model.q_emb.shape[1], dtype=torch.float64, device=dev)
    for i in range(0, len(X), B):
        x = torch.as_tensor(X[i:i + B], device=dev)
        obs = torch.isfinite(x)
        s = torch.as_tensor(S[i:i + B], device=dev)
        with torch.autocast("cuda", enabled=False):
            xs, c, e = model.inputs(x, obs, s)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model.encode(xs, c, e, obs, s)
            for t0 in range(0, P, T):
                idx = torch.arange(t0, min(P, t0 + T), device=dev)
                q = model.q_emb[idx]
                if query == "full":
                    q = q + model.q_load((core.V[idx] * lam_sqrt).to(z.dtype))
                h = model.dec1(q[None].expand(x.shape[0], -1, -1), z)
                h = model.dec2(h, z)
                acc[idx] += h.float().sum(0).double()
        if (i // B) % 50 == 0:
            log(f"  decoder states {query}: {i + len(x)}/{len(X)} profiles")
    return (acc / len(X)).float().cpu().numpy()


def release_model(core, dev):
    cfg = json.loads((RELEASE / "config.json").read_text())
    a = cfg["architecture"]
    net = MethylGPT2(core, d_tok=a["d_tok"], d=a["d"], heads=a["heads"], n_latent=a["latents"], n_score_tokens=a["score_tokens"],
                     n_self=a["self_layers"], d_read=a["d_read"], encoder=a["encoder"], d_dense=a["d_dense"],
                     n_dense_tokens=a["dense_tokens"]).to(dev)
    state = torch.load(RELEASE / "weights.pt", map_location=dev, weights_only=True)
    res = net.load_state_dict(state["encoder"], strict=False)
    if res.unexpected_keys or any(not k.startswith("core.") for k in res.missing_keys):
        raise SystemExit(f"release weights do not match the architecture: {res}")
    net.eval()
    return net


def compute_states(corpus, core_np, dev):
    if STATES.exists():
        z = np.load(STATES)
        log(f"decoder states from cache {STATES}")
        return {k: z[k].astype(np.float32) for k in ("h_full", "h_qemb", "h_random", "h_release")}, z["heldout_rows"]
    rng = np.random.default_rng(SEED)
    rows = np.sort(rng.choice(corpus.rows("test"), 2000, replace=False))
    X = corpus.get(rows)
    ssl, core, _ = load_checkpoint(SSL_CKPT, None, dev)
    if not torch.allclose(core.V[:, :5].cpu(), torch.as_tensor(core_np["V"][:, :5])):
        raise SystemExit("self-supervised checkpoint core differs from the released core")
    S = np.zeros((len(X), core.rank), np.float32)
    with torch.no_grad():
        for i in range(0, len(X), 16):
            x = torch.as_tensor(X[i:i + 16], device=dev)
            S[i:i + 16] = core.posterior(x, torch.isfinite(x).float()).cpu().numpy()
    log("posterior latent scores computed for 2,000 held-out profiles")
    out = {"h_full": decoder_states(ssl, core, X, S, dev, "full"),
           "h_qemb": decoder_states(ssl, core, X, S, dev, "qemb")}
    del ssl
    torch.cuda.empty_cache()
    torch.manual_seed(SEED)
    rnd, rcore, _ = load_checkpoint(SSL_CKPT, None, dev, random_init=True)
    out["h_random"] = decoder_states(rnd, rcore, X, S, dev, "full")
    del rnd
    torch.cuda.empty_cache()
    rel = release_model(core, dev)
    out["h_release"] = decoder_states(rel, core, X, S, dev, "full")
    del rel
    torch.cuda.empty_cache()
    np.savez(STATES, heldout_rows=rows, **{k: v.astype(np.float32) for k, v in out.items()})  # float32: |h| ~ 20 with per-CpG SD ~ 0.02
    log(f"wrote {STATES}")
    return out, rows


def representations(ann, dev):
    rng = np.random.default_rng(20261001)  # the existing script's seed for raw rows and random vectors
    core_np = np.load(require(RELEASE / "core.npz"))
    V, lam, mu = core_np["V"][:, :1536].astype(np.float32), core_np["lam"][:1536].astype(np.float32), core_np["mu"].astype(np.float32)
    reps = {"loadings": V * np.sqrt(lam)[None, :]}
    reps["loadings_natural"] = reps["loadings"]
    enc = torch.load(require(RELEASE / "weights.pt"), map_location="cpu", weights_only=True)["encoder"]
    reps["encoder_input"] = enc["dense_w.weight"].float().T.numpy()
    reps["dense_c"] = enc["dense_c.weight"].float().T.numpy()
    reps["dense_e"] = enc["dense_e.weight"].float().T.numpy()
    reps["q_emb"] = enc["q_emb"].float().numpy()
    with torch.no_grad():
        reps["query_full"] = (enc["q_emb"].float() + torch.as_tensor(reps["loadings"]) @ enc["q_load.weight"].float().T).numpy()
    del enc
    corpus = Corpus()
    rows = np.sort(rng.choice(corpus.rows("train"), 2000, replace=False))
    X = corpus.get(rows)
    obs_mean = np.nanmean(X, 0)
    obs_sd = np.nanstd(X, 0)
    X = np.where(np.isfinite(X), X, mu[None, :])
    reps["raw_profiles"] = X.T.copy()
    reps["raw_pca64"] = PCA(64, random_state=0).fit_transform(std(X.T))
    reps["random"] = rng.standard_normal((len(ann), 1536))
    del X
    states, held = compute_states(corpus, core_np, dev)
    reps.update(states)
    Xh = corpus.get(held)
    reps["raw_heldout"] = np.where(np.isfinite(Xh), Xh, mu[None, :]).T.copy()
    del Xh
    probe = (ann.probe_type.to_numpy() == "I").astype(float)
    reps["baseline"] = np.stack([obs_mean, obs_sd, probe], 1)
    base = {"mean_beta": obs_mean, "sd_beta": obs_sd, "probe_type": ann.probe_type.to_numpy(),
            "train_rows": rows, "heldout_rows": held}
    return reps, base


# --------------------------------------------------------------------------------------- kNN core
def prep(name, Z, dev):
    """Representation as used for kNN: column-standardized (except loadings_natural); rows unit-normalized
    for cosine distance; the baseline stays Euclidean."""
    Zs = np.asarray(Z, np.float64) if name == "loadings_natural" else std(Z)
    T = torch.as_tensor(Zs, dtype=torch.float32, device=dev)
    if name != "baseline":
        T = T / T.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return T, Zs


@torch.no_grad()
def neighbours(T, elig, fold, strata=None, euclid=False, k=K, chunk=4096):
    """Out-of-fold k nearest neighbours of every eligible CpG among eligible CpGs of the other folds
    (and of the same stratum when strata is given). Returns (P, k) global indices, -1 where absent."""
    P = T.shape[0]
    nn_idx = np.full((P, k), -1, np.int64)
    st = np.zeros(P, np.int64) if strata is None else np.asarray(strata)
    sq = (T * T).sum(1) if euclid else None
    for sv in np.unique(st[elig]):
        for f in range(N_FOLDS):
            q = np.flatnonzero(elig & (st == sv) & (fold == f))
            r = np.flatnonzero(elig & (st == sv) & (fold != f))
            if len(q) == 0 or len(r) == 0:
                continue
            kk = min(k, len(r))
            rt = torch.as_tensor(r, device=T.device)
            R = T[rt]
            for c0 in range(0, len(q), chunk):
                qc = q[c0:c0 + chunk]
                Q = T[torch.as_tensor(qc, device=T.device)]
                sim = Q @ R.T
                if euclid:
                    sim = 2 * sim - sq[rt][None, :]
                nn_idx[qc, :kk] = r[sim.topk(kk, dim=1).indices.cpu().numpy()]
    return nn_idx


def votes_from(nn_idx, codes, n_cls):
    """Vote counts (P, n_cls) and number of neighbours (P,)."""
    ok = nn_idx >= 0
    lab = np.where(ok, codes[np.clip(nn_idx, 0, None)], -1)
    v = np.stack([(lab == c).sum(1) for c in range(n_cls)], 1).astype(np.int16)
    return v, ok.sum(1)


def encode(y):
    ok = pd.notna(y)
    cls = [str(c) for c in sorted(set(np.asarray(y)[ok].astype(str)))]
    codes = np.full(len(y), -1, np.int64)
    codes[ok] = np.searchsorted(cls, np.asarray(y)[ok].astype(str))
    return codes, cls


# --------------------------------------------------------------------------------------- metrics
class Boot:
    """Block bootstrap over 1 Mb blocks; one weight matrix shared by every metric (paired)."""

    def __init__(self, block, n_boot=N_BOOT, seed=SEED):
        self.ids, self.b = np.unique(block, return_inverse=True)
        self.nb = len(self.ids)
        self.W = np.random.default_rng(seed + 1).multinomial(self.nb, np.full(self.nb, 1.0 / self.nb), size=n_boot).astype(np.float64)

    def counts(self, keys, n_keys, mask):
        """Per-block counts of integer keys (n_blocks, n_keys) over mask."""
        flat = self.b[mask] * n_keys + keys[mask]
        return np.bincount(flat, minlength=self.nb * n_keys).reshape(self.nb, n_keys).astype(np.float64)


def bacc_from_cm(cm, cls_ok=None):
    """cm (..., C, C) true x predicted counts; balanced accuracy over classes with counts (or cls_ok)."""
    rows = cm.sum(-1)
    diag = np.diagonal(cm, axis1=-2, axis2=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = diag / rows
    use = rows > 0 if cls_ok is None else np.broadcast_to(cls_ok, rec.shape) & (rows > 0)
    return np.nansum(np.where(use, rec, 0), -1) / np.maximum(use.sum(-1), 1)


def ci(boot):
    boot = np.asarray(boot, float)
    boot = boot[np.isfinite(boot)]
    return [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))] if len(boot) else [None, None]


def predictions(v):
    return v.argmax(1)


def knn_metrics(bs, v, nnc, codes, n_cls, elig, strata=None):
    """Point estimate and bootstrap draws of balanced accuracy (pooled, or stratum-adjusted when strata
    is given), plus confusion matrix."""
    m = elig & (codes >= 0) & (nnc > 0)
    pred = predictions(v)
    if strata is None:
        Cb = bs.counts(codes * n_cls + pred, n_cls * n_cls, m)
        cm = Cb.sum(0).reshape(n_cls, n_cls)
        draws = bacc_from_cm((bs.W @ Cb).reshape(-1, n_cls, n_cls))
        return {"est": float(bacc_from_cm(cm)), "draws": draws, "cm": cm, "n": int(m.sum())}
    sv, sidx = np.unique(strata[m], return_inverse=True)
    S = len(sv)
    keys = np.zeros(len(codes), np.int64)
    keys[m] = sidx * n_cls * n_cls + codes[m] * n_cls + pred[m]
    Cb = bs.counts(keys, S * n_cls * n_cls, m)
    cm = Cb.sum(0).reshape(S, n_cls, n_cls)
    cls_ok = cm.sum(-1) >= MIN_CLASS_IN_STRATUM          # (S, C), fixed from the full data
    st_ok = cls_ok.sum(1) >= 2

    def agg(cms):
        b = bacc_from_cm(cms, cls_ok)                     # (..., S)
        w = cms.sum((-1, -2))
        w = np.where(st_ok, w, 0)
        return (b * w).sum(-1) / w.sum(-1)

    draws = agg((bs.W @ Cb).reshape(-1, S, n_cls, n_cls))
    per = bacc_from_cm(cm, cls_ok)
    return {"est": float(agg(cm)), "draws": draws, "n": int(m.sum()), "n_strata_used": int(st_ok.sum()),
            "per_stratum": {str(s): {"bacc": float(per[i]), "n": int(cm[i].sum()), "used": bool(st_ok[i]),
                                     "classes_used": int(cls_ok[i].sum())} for i, s in enumerate(sv)}}


def auc_metrics(bs, v, nnc, codes, n_cls, elig, cls_list):
    """One-vs-rest AUC from vote fractions (all eligible CpGs have k neighbours in the pooled analyses)."""
    m = elig & (codes >= 0) & (nnc == K)
    out = {}
    targets = range(n_cls) if n_cls > 2 else [cls_list.index("X" if "X" in cls_list else "True")]
    for c in targets:
        keys = (codes == c).astype(np.int64) * (K + 1) + v[:, c].astype(np.int64)
        Hb = bs.counts(keys, 2 * (K + 1), m)

        def auc(H):
            neg, pos = H[..., :K + 1], H[..., K + 1:]
            below = np.cumsum(neg, -1) - neg
            return (pos * (below + 0.5 * neg)).sum(-1) / (pos.sum(-1) * neg.sum(-1))

        draws = auc(bs.W @ Hb)
        out[cls_list[c]] = {"auc": float(auc(Hb.sum(0))), "ci95": ci(draws), "draws": draws}
    return out


def energy(P, y, rng, n=2000):
    P = P / np.sqrt((P ** 2).sum(1).mean())
    cats = sorted(set(y))
    idx = {c: rng.choice(np.flatnonzero(y == c), min(n, (y == c).sum()), replace=False) for c in cats}
    T = torch.as_tensor(P)
    w = {c: torch.cdist(T[i], T[i]).mean().item() for c, i in idx.items()}
    out = [2 * torch.cdist(T[idx[a]], T[idx[b]]).mean().item() - w[a] - w[b] for k, a in enumerate(cats) for b in cats[k + 1:]]
    return float(np.mean(out))


@torch.no_grad()
def pcs50(Zs, dev):
    """First 50 principal components (exact, from the eigendecomposition of the covariance on GPU)."""
    if Zs.shape[1] <= 50:
        return np.asarray(Zs, np.float64)
    T = torch.as_tensor(Zs, dtype=torch.float64, device=dev)
    T = T - T.mean(0)
    w, U = torch.linalg.eigh(T.T @ T)
    P = (T @ U[:, -50:].flip(1)).cpu().numpy()
    del T, U
    return P


@torch.no_grad()
def silhouette_cosine(P, y, dev):
    """Mean cosine silhouette (as sklearn.metrics.silhouette_score with metric="cosine")."""
    T = torch.as_tensor(P, dtype=torch.float64, device=dev)
    T = T / T.norm(dim=1, keepdim=True).clamp_min(1e-12)
    D = (1 - T @ T.T).clamp_min(0)
    D.fill_diagonal_(0)
    yt = torch.as_tensor(y, device=dev)
    cats = torch.unique(yt)
    M = torch.stack([D[:, yt == c].sum(1) for c in cats], 1)
    n = torch.stack([(yt == c).sum() for c in cats]).double()
    own = (yt[:, None] == cats[None, :])
    a = (M * own).sum(1) / ((n[None, :] - 1) * own).sum(1).clamp_min(1)
    b = torch.where(own, torch.full_like(M, float("inf")), M / n[None, :]).min(1).values
    s = (b - a) / torch.maximum(a, b)
    s = torch.where((n[None, :] * own).sum(1) > 1, s, torch.zeros_like(s))
    return float(s.mean())


def geometry_metrics(P50, codes, cls, elig, rng, dev):
    ok = elig & (codes >= 0)
    y = codes[ok]
    sub = rng.choice(np.flatnonzero(ok), 10000, replace=False)
    Pn = P50[ok] / np.maximum(np.linalg.norm(P50[ok], axis=1, keepdims=True), 1e-12)
    km = KMeans(len(cls), n_init=4, random_state=0).fit_predict(Pn)
    return {"silhouette": silhouette_cosine(P50[sub], codes[sub], dev),
            "energy_distance": energy(P50[ok], y, rng),
            "kmeans_ari": float(adjusted_rand_score(y, km)), "kmeans_nmi": float(normalized_mutual_info_score(y, km))}


def residualize(Zs, design):
    """Residuals of every column of Zs on the design (least squares through an orthonormal basis)."""
    Q, _ = np.linalg.qr(design)
    return Zs - Q @ (Q.T @ Zs)


# ---------------------------------------------------------------------------------------------- main
def run(a):
    cuda_preinit()
    dev = torch.device(a.device)
    checks = verify_release()
    ann = annotation()
    labels, ev_note = label_table(ann)
    log(f"DNase evidence count: {ev_note}")
    enc = {k: encode(y) for k, y in labels.items()}
    reps, base = representations(ann, dev)
    P = len(ann)
    chr_ = ann.chr.astype(str).to_numpy()
    pos = ann.pos.to_numpy()
    block = np.array([f"{c}:{int(p // BLOCK_BP)}" for c, p in zip(chr_, pos)])
    # folds: blocks permuted and cut into five groups of near-equal CpG count
    bid, binv, bcount = np.unique(block, return_inverse=True, return_counts=True)
    perm = np.random.default_rng(SEED).permutation(len(bid))
    cum = np.cumsum(bcount[perm])
    fold_of_block = np.empty(len(bid), np.int64)
    fold_of_block[perm] = np.minimum((cum - bcount[perm] / 2) * N_FOLDS // cum[-1], N_FOLDS - 1).astype(np.int64)
    fold = fold_of_block[binv]
    bs = Boot(block)
    epic = ann.in_epic_v1.to_numpy()
    xauto = chr_ != "Y"
    auto_epic = epic & (chr_ != "X") & (chr_ != "Y")
    mean_dec = np.searchsorted(np.quantile(base["mean_beta"], np.linspace(0, 1, 11)[1:-1]), base["mean_beta"], side="right")
    sd_dec = np.searchsorted(np.quantile(base["sd_beta"], np.linspace(0, 1, 11)[1:-1]), base["sd_beta"], side="right")
    ptype = (base["probe_type"] == "I").astype(np.int64)
    strata_base = mean_dec * 2 + ptype
    isl_codes, isl_cls = enc["island_relation"]
    design = np.concatenate([np.eye(20)[strata_base], np.eye(10)[sd_dec][:, 1:]], 1)
    epic_labels = [k for k in labels if k != "x_chromosome"]
    res = {"n_cpgs": P, "n_epic_v1": int(epic.sum()), "n_blocks": int(bs.nb), "n_boot": N_BOOT, "k": K, "folds": N_FOLDS,
           "fold_sizes": np.bincount(fold).tolist(), "checksums": checks, "dnase_evidence_note": ev_note,
           "label_counts": {k: dict(zip(c, np.bincount(enc[k][0][enc[k][0] >= 0]).tolist())) for k, (_, c) in enc.items()},
           "probe_type_counts": pd.Series(base["probe_type"]).value_counts(dropna=False).to_dict(),
           "heldout_profiles": {"n": int(len(base["heldout_rows"])), "datasets": None},
           "results": {}}
    corpus_prof = pd.read_parquet(ARTIFACTS / "corpus_2026-09-30/profiles.parquet", columns=["group", "split"])
    res["heldout_profiles"]["datasets"] = int(corpus_prof.group.iloc[base["heldout_rows"]].nunique())
    res["training_profiles_raw"] = {"n": int(len(base["train_rows"])),
                                    "datasets": int(corpus_prof.group.iloc[base["train_rows"]].nunique())}
    res["dnase_prevalence_by_island"] = {s: float(ann.dnase_hs[epic & (ann.island_relation == s)].mean()) for s in isl_cls}
    res["dnase_prevalence_by_island_n"] = {s: int((epic & (ann.island_relation == s)).sum()) for s in isl_cls}
    qv = np.asarray(reps["q_emb"], np.float64).var(0).sum()
    lv = (np.asarray(reps["query_full"], np.float64) - np.asarray(reps["q_emb"], np.float64)).var(0).sum()
    res["decoder_query_variance"] = {"q_emb_total_variance": float(qv), "projected_loadings_total_variance": float(lv),
                                     "ratio": float(qv / lv)}
    draws = {}
    figdata = {}
    for name, Z in reps.items():
        T, Zs = prep(name, Z, dev)
        eu = name == "baseline"
        blk = {}
        # primary: grouped folds
        nnE = neighbours(T, epic, fold, euclid=eu)
        nnX = neighbours(T, xauto, fold, euclid=eu)
        for lab in labels:
            codes, cls = enc[lab]
            nn_ = nnX if lab == "x_chromosome" else nnE
            el = xauto if lab == "x_chromosome" else epic
            v, nnc = votes_from(nn_, codes, len(cls))
            km = knn_metrics(bs, v, nnc, codes, len(cls), el)
            au = auc_metrics(bs, v, nnc, codes, len(cls), el, cls)
            draws[(name, "grouped", lab)] = km["draws"]
            draws[(name, "auc", lab)] = {c: d.pop("draws") for c, d in au.items()}
            blk[lab] = {"grouped_cv": {"balanced_accuracy": km["est"], "ci95": ci(km["draws"]), "n": km["n"],
                                       "confusion": km["cm"].astype(int).tolist(), "classes": cls},
                        "auc_one_vs_rest": au}
            if lab == "island_relation":
                ok = (codes >= 0) & epic
                hist = np.zeros((len(cls), len(cls), K + 1), np.int64)
                for t in range(len(cls)):
                    for c in range(len(cls)):
                        hist[t, c] = np.bincount(v[ok & (codes == t), c], minlength=K + 1)
                figdata[f"{name}__island_vote_hist"] = hist
                figdata[f"{name}__island_confusion"] = km["cm"]
        # baseline-adjusted: kNN within mean-beta decile x probe type
        nnEs = neighbours(T, epic, fold, strata_base, euclid=eu)
        nnXs = neighbours(T, xauto, fold, strata_base, euclid=eu)
        for lab in labels:
            codes, cls = enc[lab]
            nn_, el = (nnXs, xauto) if lab == "x_chromosome" else (nnEs, epic)
            v, nnc = votes_from(nn_, codes, len(cls))
            km = knn_metrics(bs, v, nnc, codes, len(cls), el, strata_base)
            draws[(name, "adjusted", lab)] = km["draws"]
            blk[lab]["within_mean_decile_probe_type"] = {"balanced_accuracy": km["est"], "ci95": ci(km["draws"]), "n": km["n"],
                                                         "n_strata_used": km["n_strata_used"], "per_stratum": km["per_stratum"]}
        # residualized on the baseline
        if name != "baseline":
            Tr = torch.as_tensor(residualize(Zs, design), dtype=torch.float32, device=dev)
            Tr = Tr / Tr.norm(dim=1, keepdim=True).clamp_min(1e-12)
        else:
            Tr = torch.as_tensor(residualize(Zs, design), dtype=torch.float32, device=dev)
        nnEr = neighbours(Tr, epic, fold, euclid=eu)
        nnXr = neighbours(Tr, xauto, fold, euclid=eu)
        del Tr
        for lab in labels:
            codes, cls = enc[lab]
            nn_, el = (nnXr, xauto) if lab == "x_chromosome" else (nnEr, epic)
            v, nnc = votes_from(nn_, codes, len(cls))
            km = knn_metrics(bs, v, nnc, codes, len(cls), el)
            draws[(name, "residualized", lab)] = km["draws"]
            blk[lab]["residualized_on_baseline"] = {"balanced_accuracy": km["est"], "ci95": ci(km["draws"]), "n": km["n"]}
        # DNase (and evidence label if any) within each island stratum
        nnI = neighbours(T, epic, fold, isl_codes, euclid=eu)
        for lab in [l for l in epic_labels if l.startswith("dnase")]:
            codes, cls = enc[lab]
            v, nnc = votes_from(nnI, codes, len(cls))
            km = knn_metrics(bs, v, nnc, codes, len(cls), epic, isl_codes)
            draws[(name, "within_island", lab)] = km["draws"]
            per = {}
            for si, sname in enumerate(isl_cls):
                ms = epic & (isl_codes == si)
                kms = knn_metrics(bs, v, nnc, codes, len(cls), ms)
                aus = auc_metrics(bs, v, nnc, codes, len(cls), ms, cls)
                draws[(name, f"within_island_{sname}", lab)] = kms["draws"]
                per[sname] = {"balanced_accuracy": kms["est"], "ci95": ci(kms["draws"]), "n": kms["n"],
                              "auc": {c: {"auc": d["auc"], "ci95": d["ci95"]} for c, d in aus.items()}}
            blk[lab]["within_island_stratum"] = {"balanced_accuracy_weighted": km["est"], "ci95": ci(km["draws"]), "per_stratum": per}
        # autosomes only
        nnA = neighbours(T, auto_epic, fold, euclid=eu)
        for lab in epic_labels:
            codes, cls = enc[lab]
            v, nnc = votes_from(nnA, codes, len(cls))
            km = knn_metrics(bs, v, nnc, codes, len(cls), auto_epic)
            draws[(name, "autosomes", lab)] = km["draws"]
            blk[lab]["autosomes_only"] = {"balanced_accuracy": km["est"], "ci95": ci(km["draws"]), "n": km["n"]}
        # ungrouped stratified CV as in v13 (fold assignment reproduces the existing script)
        for lab in labels:
            codes, cls = enc[lab]
            el = xauto if lab == "x_chromosome" else epic
            ok = el & (codes >= 0)
            f_ung = np.full(P, -1, np.int64)
            for fi, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=20261001).split(np.zeros(ok.sum()), codes[ok])):
                f_ung[np.flatnonzero(ok)[te]] = fi
            nnU = neighbours(T, ok, f_ung, euclid=eu)
            v, nnc = votes_from(nnU, codes, len(cls))
            km = knn_metrics(bs, v, nnc, codes, len(cls), ok)
            blk[lab]["ungrouped_cv_v13"] = {"balanced_accuracy": km["est"], "n": km["n"]}
        del T
        torch.cuda.empty_cache()
        # geometry (secondary)
        P50 = pcs50(Zs, dev)
        for lab in labels:
            codes, cls = enc[lab]
            el = xauto if lab == "x_chromosome" else epic
            blk[lab]["geometry"] = geometry_metrics(P50, codes, cls, el, np.random.default_rng(20261001), dev)
        res["results"][name] = blk
        log(f"{name}: " + "; ".join(f"{l} {b['grouped_cv']['balanced_accuracy']:.3f} adj {b['within_mean_decile_probe_type']['balanced_accuracy']:.3f}"
                                    f" ungrouped {b['ungrouped_cv_v13']['balanced_accuracy']:.3f}" for l, b in blk.items()))
        write_json(OUT_JSON, res)
    # paired differences from the loadings, and named contrasts
    contrasts = {f"{r} - loadings": (r, "loadings") for r in reps if r != "loadings"}
    contrasts.update({"h_full - h_random": ("h_full", "h_random"), "h_full - h_qemb": ("h_full", "h_qemb"),
                      "h_full - query_full": ("h_full", "query_full"), "h_release - h_full": ("h_release", "h_full"),
                      "raw_profiles - raw_pca64": ("raw_profiles", "raw_pca64"), "raw_heldout - raw_profiles": ("raw_heldout", "raw_profiles")})
    res["primary_comparisons"] = {}
    res["differences"] = {}
    for analysis in ("grouped", "adjusted", "residualized", "autosomes", "within_island"):
        res["differences"][analysis] = {}
        for cname, (r1, r2) in contrasts.items():
            for lab in labels:
                if (r1, analysis, lab) not in draws:
                    continue
                k1 = {"grouped": "grouped_cv", "adjusted": "within_mean_decile_probe_type", "residualized": "residualized_on_baseline",
                      "autosomes": "autosomes_only", "within_island": "within_island_stratum"}[analysis]
                e1 = res["results"][r1][lab][k1]
                e2 = res["results"][r2][lab][k1]
                key = "balanced_accuracy_weighted" if analysis == "within_island" else "balanced_accuracy"
                d = draws[(r1, analysis, lab)] - draws[(r2, analysis, lab)]
                res["differences"][analysis].setdefault(cname, {})[lab] = {"difference": e1[key] - e2[key], "ci95": ci(d)}
    for cname in ("raw_pca64 - loadings", "baseline - loadings"):
        res["primary_comparisons"]["loadings minus " + cname.split(" - ")[0]] = {
            lab: {"difference": -v["difference"], "ci95": [-v["ci95"][1], -v["ci95"][0]]}
            for lab, v in res["differences"]["grouped"][cname].items()}
    write_json(OUT_JSON, res)
    np.savez(OUT_FIG, island_classes=np.array(isl_cls), **figdata)
    write_csv(res)
    log(f"wrote {OUT_JSON}, {OUT_CSV}, {OUT_FIG}")


def write_csv(res):
    rows = []
    resampled = f"1 Mb genomic blocks ({res['n_blocks']:,} blocks, {res['n_boot']:,} draws)"
    an = {"grouped_cv": "kNN, 5-fold CV grouped by 1 Mb blocks (primary)",
          "within_mean_decile_probe_type": "kNN within mean-beta decile x probe type",
          "residualized_on_baseline": "kNN after residualizing on baseline",
          "autosomes_only": "kNN, grouped CV, autosomes only",
          "ungrouped_cv_v13": "kNN, ungrouped stratified CV (v13 protocol)"}
    for r, blk in res["results"].items():
        for lab, b in blk.items():
            for key, desc in an.items():
                if key in b:
                    e = b[key]
                    rows.append({"analysis": desc, "representation": REP_NAMES[r], "representation_key": r, "annotation": lab,
                                 "stratum": "all", "metric": "balanced accuracy", "estimate": e["balanced_accuracy"],
                                 "ci95_low": (e.get("ci95") or [None, None])[0], "ci95_high": (e.get("ci95") or [None, None])[1],
                                 "n_cpgs": e["n"], "resampled": resampled if e.get("ci95") else ""})
            for c, d in b["auc_one_vs_rest"].items():
                rows.append({"analysis": an["grouped_cv"], "representation": REP_NAMES[r], "representation_key": r, "annotation": lab,
                             "stratum": "all", "metric": f"one-vs-rest AUC, class {c}", "estimate": d["auc"], "ci95_low": d["ci95"][0],
                             "ci95_high": d["ci95"][1], "n_cpgs": b["grouped_cv"]["n"], "resampled": resampled})
            for mk, mv in b["geometry"].items():
                rows.append({"analysis": "geometry (50 PCs)", "representation": REP_NAMES[r], "representation_key": r, "annotation": lab,
                             "stratum": "all", "metric": mk, "estimate": mv, "ci95_low": None, "ci95_high": None,
                             "n_cpgs": b["grouped_cv"]["n"], "resampled": ""})
            if "within_island_stratum" in b:
                w = b["within_island_stratum"]
                rows.append({"analysis": "kNN within CpG-island stratum", "representation": REP_NAMES[r], "representation_key": r,
                             "annotation": lab, "stratum": "weighted mean", "metric": "balanced accuracy",
                             "estimate": w["balanced_accuracy_weighted"], "ci95_low": w["ci95"][0], "ci95_high": w["ci95"][1],
                             "n_cpgs": b["grouped_cv"]["n"], "resampled": resampled})
                for s, d in w["per_stratum"].items():
                    rows.append({"analysis": "kNN within CpG-island stratum", "representation": REP_NAMES[r], "representation_key": r,
                                 "annotation": lab, "stratum": s, "metric": "balanced accuracy", "estimate": d["balanced_accuracy"],
                                 "ci95_low": d["ci95"][0], "ci95_high": d["ci95"][1], "n_cpgs": d["n"], "resampled": resampled})
                    for c, dd in d["auc"].items():
                        rows.append({"analysis": "kNN within CpG-island stratum", "representation": REP_NAMES[r], "representation_key": r,
                                     "annotation": lab, "stratum": s, "metric": f"AUC, class {c}", "estimate": dd["auc"],
                                     "ci95_low": dd["ci95"][0], "ci95_high": dd["ci95"][1], "n_cpgs": d["n"], "resampled": resampled})
    for analysis, block in res["differences"].items():
        for cname, labs in block.items():
            r1, r2 = cname.split(" - ")
            for lab, d in labs.items():
                rows.append({"analysis": f"paired difference ({analysis})", "representation": f"{REP_NAMES[r1]} minus {REP_NAMES[r2]}",
                             "representation_key": cname, "annotation": lab, "stratum": "all", "metric": "difference in balanced accuracy",
                             "estimate": d["difference"], "ci95_low": d["ci95"][0], "ci95_high": d["ci95"][1], "n_cpgs": None,
                             "resampled": resampled})
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)


def run_umap():
    """UMAPs for illustration: loadings (scaled, as in Fig. 2a), h (full query) and raw beta values
    (2,000 training profiles, column-standardized as for kNN); cosine distance, same 15,000 CpGs as Fig. 2."""
    import umap
    idx = np.sort(np.random.default_rng(20261002).choice(49156, 15000, replace=False))
    core_np = np.load(require(RELEASE / "core.npz"))
    L = core_np["V"][:, :1536] * np.sqrt(core_np["lam"][:1536])[None, :]
    h = np.load(require(STATES))["h_full"].astype(np.float32)
    rng = np.random.default_rng(20261001)
    corpus = Corpus()
    rows = np.sort(rng.choice(corpus.rows("train"), 2000, replace=False))
    X = corpus.get(rows)
    X = np.where(np.isfinite(X), X, core_np["mu"][None, :])
    inputs = {"loadings": L[idx], "h_full": std(h)[idx], "raw_profiles": std(X.T)[idx]}
    out = {"idx": idx, "grid": np.array(UMAP_GRID)}
    for name, Z in inputs.items():
        for nn_, md in UMAP_GRID:
            out[f"{name}__{nn_}__{md}"] = umap.UMAP(n_neighbors=nn_, min_dist=md, metric="cosine", random_state=0).fit_transform(Z)
            log(f"UMAP {name} n_neighbors {nn_} min_dist {md}")
    np.savez(OUT_UMAP, **out)
    log(f"wrote {OUT_UMAP}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--umap", action="store_true")
    a = ap.parse_args()
    with threadpool_limits(32):  # shared 256-core server: oversubscribed BLAS threads are very slow
        if a.umap:
            run_umap()
        else:
            run(a)
