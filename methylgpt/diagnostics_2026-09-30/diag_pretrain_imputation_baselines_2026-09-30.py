#!/usr/bin/env python3
"""Diagnostic (30 Sep 2026): does the January 2025 MethylGPT checkpoint beat simple
matrix-completion baselines on its own masked-value pretraining task?

Empirical inputs only: the 34 shuffled pretraining parquet blocks (public EWAS Data Hub /
ClockBase beta profiles) and the January checkpoint. No simulated data.

Design
- Split: the pretraining driver's file-level rule (first floor(33*0.9)=29 files in
  os.listdir order = train, remaining 5 = validation). The January run's own listdir
  order is not recorded, so query profiles may have been seen by MethylGPT during
  pretraining; that can only favour MethylGPT.
- Reference: `--n-ref-files` training blocks (5,000 profiles each).
- Query: `--n-query` profiles from the first validation block; 30% of each profile's
  observed CpGs are masked with a fixed seed.
- Regime A (random held-out profiles): reference may contain other profiles of the
  same dataset, as in the pretraining validation split.
- Regime B (held-out datasets): reference rows sharing any dataset with the query set
  are removed before fitting the baselines.
- Methods: reference CpG mean; kNN (k=10) on unmasked observed CpGs; probabilistic-PCA
  projection (ranks 32/128/256) fitted on the reference; MethylGPT MLM head and MVC
  (GEPC) head on a paired subset (observed-token policy, BF16 autocast on CUDA).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT")
PARQUET_DIR = PROJECT / "processed_dataset/processed_type3_parquet_shuffled"
QC_TABLE = PROJECT / "QCed_samples_type3.csv"
DEV = Path("/Entropy/kying/tmp/methylgpt_publication_2026-09-28")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_block(path: Path):
    try:
        table = pq.read_table(path)
    except Exception as exc:
        raise RuntimeError(f"Cannot read parquet block {path}: {exc}") from exc
    ids = np.asarray(table.column("id").to_pylist(), dtype=object)
    col = table.column("data").combine_chunks()
    offsets = col.offsets.to_numpy()
    lengths = np.diff(offsets)
    if len(np.unique(lengths)) != 1:
        raise ValueError(f"Ragged methylation rows in {path}")
    flat = col.values.to_numpy(zero_copy_only=False)
    X = flat.reshape(len(ids), int(lengths[0])).astype(np.float32)
    return ids, X


def dataset_map():
    qc = pd.read_csv(QC_TABLE, usecols=["dataset_name", "sample_name"])
    groups = qc.groupby("sample_name")["dataset_name"].apply(lambda s: frozenset(s.astype(str)))
    return groups.to_dict()


def chunked_nanmean(X, block=4096):
    """Column-wise nanmean with float64 accumulation (multithreaded torch, chunked)."""
    import torch
    out = np.empty(X.shape[1], np.float64)
    for j in range(0, X.shape[1], block):
        c = torch.from_numpy(np.ascontiguousarray(X[:, j:j + block]))
        fin = torch.isfinite(c)
        tot = torch.where(fin, c, torch.zeros((), dtype=c.dtype)).sum(0, dtype=torch.float64)
        cnt = fin.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[j:j + block] = (tot / cnt).numpy()
    return out


def metrics(pred, truth, cpg_index, min_per_cpg=20):
    pred = pred.astype(np.float64); truth = truth.astype(np.float64)
    res = pred - truth
    out = {"n_targets": int(len(truth)), "mse": float(np.mean(res ** 2)),
           "mae": float(np.mean(np.abs(res))),
           "mae_clipped": float(np.mean(np.abs(np.clip(pred, 0, 1) - truth)))}
    # Within-CpG R2 across query profiles (variation between individuals at a CpG).
    P = int(cpg_index.max()) + 1
    n = np.bincount(cpg_index, minlength=P).astype(np.float64)
    sy = np.bincount(cpg_index, weights=truth, minlength=P)
    sy2 = np.bincount(cpg_index, weights=truth ** 2, minlength=P)
    sse = np.bincount(cpg_index, weights=res ** 2, minlength=P)
    with np.errstate(invalid="ignore", divide="ignore"):
        sst = sy2 - sy ** 2 / n
    keep = (n >= min_per_cpg) & (sst > 1e-12)
    r2 = 1 - sse[keep] / sst[keep]
    out["within_cpg_r2_median"] = float(np.median(r2)) if keep.any() else None
    out["within_cpg_r2_pooled"] = float(1 - sse[keep].sum() / sst[keep].sum()) if keep.any() else None
    out["n_cpgs_within"] = int(keep.sum())
    return out


def knn_predict(Xq_in, Wq, Rf, R, mu, masks, k):
    """Squared Euclidean distance over each query's unmasked observed CpGs."""
    import torch
    Xt = torch.from_numpy(Xq_in)
    Wt = torch.from_numpy(Wq)
    Rt = torch.from_numpy(Rf)
    d = (Xt ** 2).sum(1, keepdim=True) - 2 * Xt @ Rt.T + Wt @ (Rt ** 2).T
    nbrs = torch.topk(d, k, dim=1, largest=False).indices.numpy()
    preds = []
    for q, m in enumerate(masks):
        cols = np.flatnonzero(m)
        vals = R[nbrs[q]][:, cols]
        with np.errstate(invalid="ignore"):
            p = np.nanmean(vals, axis=0)
        bad = ~np.isfinite(p)
        p[bad] = mu[cols][bad]
        preds.append(p.astype(np.float32))
    return preds


def ppca_fit(Rf, mu, rank):
    import torch
    Rc = torch.from_numpy(Rf)
    Rc = Rc - torch.from_numpy(mu.astype(np.float32))[None, :]
    n = Rc.shape[0]
    U, S, V = torch.svd_lowrank(Rc, q=rank + 16, niter=6)
    S, V = S[:rank], V[:, :rank]
    total = sum(float((Rc[i:i + 4096].double() ** 2).sum()) for i in range(0, n, 4096)) / (n - 1)
    lam = (S.double() ** 2 / (n - 1))
    sigma2 = max((total - float(lam.sum())) / (Rc.shape[1] - rank), 1e-8)
    return V.numpy().astype(np.float64), lam.numpy(), sigma2


def ppca_predict_all(Xq, obs_in, masks, mu, V, lam, sigma2, ranks):
    """Posterior-mean PPCA scores from each query's unmasked observed CpGs (torch/MKL)."""
    import torch
    Vt = torch.from_numpy(np.ascontiguousarray(V))
    mu_t = torch.from_numpy(mu)
    lam_t = torch.from_numpy(lam)
    out = {r: [] for r in ranks}
    for q in range(len(Xq)):
        o = torch.from_numpy(np.flatnonzero(obs_in[q]))
        Vo = Vt.index_select(0, o)
        y = torch.from_numpy(Xq[q].astype(np.float64)).index_select(0, o) - mu_t.index_select(0, o)
        G = Vo.T @ Vo / sigma2
        b = Vo.T @ y / sigma2
        cols = torch.from_numpy(np.flatnonzero(masks[q]))
        Vm = Vt.index_select(0, cols)
        mm = mu_t.index_select(0, cols)
        for r in ranks:
            A = G[:r, :r] + torch.diag(1.0 / lam_t[:r])
            z = torch.linalg.solve(A, b[:r])
            out[r].append((mm + Vm[:, :r] @ z).numpy().astype(np.float32))
    return out


def methylgpt_predict(Xq, masks, device, batch=4):
    import torch
    sys.path.insert(0, str(DEV))
    from source.encoder_survival import load_pretrained_encoder
    encoder, vocab, probes, info = load_pretrained_encoder(PROJECT, dropout=0.0)
    if device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(0.08, int(device.split(":")[1]))
    encoder.to(device).eval()
    token_ids = np.arange(len(probes)) + 3
    mlm, mvc, cls = [], [], []
    for start in range(0, len(Xq), batch):
        rows = range(start, min(start + batch, len(Xq)))
        # Observed-token policy: originally missing CpGs are not tokens (equivalent to
        # the pretraining key-padding mask); masked CpGs keep their IDs with value -1.
        for q in rows:
            obs = np.flatnonzero(np.isfinite(Xq[q]))
            genes = np.concatenate([[1], token_ids[obs]])[None, :]
            vals = np.concatenate([[0.0], Xq[q, obs]]).astype(np.float32)[None, :]
            m = masks[q][obs]
            vals[0, 1:][m] = -1.0
            g = torch.as_tensor(genes, device=device)
            v = torch.as_tensor(vals, device=device)
            with torch.inference_mode(), torch.autocast(device.split(":")[0], dtype=torch.bfloat16,
                                                        enabled=device.startswith("cuda")):
                ctx = encoder._encode(g, v, src_key_padding_mask=None).float()
                gene_embs = encoder.cur_gene_token_embs.float()
            with torch.inference_mode():
                p_mlm = encoder.decoder(ctx)["pred"][0, 1:]
                p_mvc = encoder.mvc_decoder(ctx[:, 0, :], gene_embs)["pred"][0, 1:]
            encoder.cur_gene_token_embs = None
            sel = np.flatnonzero(m)
            mlm.append(p_mlm[sel].cpu().numpy().astype(np.float32))
            mvc.append(p_mvc[sel].cpu().numpy().astype(np.float32))
            cls.append(ctx[0, 0].cpu().numpy())
        log(f"MethylGPT {min(start + batch, len(Xq))}/{len(Xq)}")
    return mlm, mvc, info


def gather(preds, masks, Xq, subset):
    p = np.concatenate([preds[q] for q in subset])
    t = np.concatenate([Xq[q, np.flatnonzero(masks[q])] for q in subset])
    j = np.concatenate([np.flatnonzero(masks[q]) for q in subset])
    return p, t, j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ref-files", type=int, default=4)
    ap.add_argument("--n-query", type=int, default=600)
    ap.add_argument("--n-model", type=int, default=200)
    ap.add_argument("--mask-ratio", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--ranks", default="32,128,256")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-model", action="store_true")
    ap.add_argument("--valid-prefixes", default="",
                    help="comma-separated block hash prefixes forming the validation split; "
                         "default reproduces the driver's current os.listdir rule")
    ap.add_argument("--query-prefix", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    if not PARQUET_DIR.is_dir():
        sys.exit(f"Cannot proceed - required data file not found: {PARQUET_DIR}")
    files = os.listdir(PARQUET_DIR)
    if len(files) != 34:
        sys.exit(f"Expected 34 parquet blocks, found {len(files)}")
    if a.valid_prefixes:
        prefixes = [f"block_csv_data_{x}" for x in a.valid_prefixes.split(",")]
        valid_files = [f for f in files if any(f.startswith(p) for p in prefixes)]
        if len(valid_files) != len(prefixes):
            sys.exit(f"Validation prefixes matched {len(valid_files)} of {len(prefixes)} blocks")
        train_files = [f for f in files if f not in valid_files]
        if a.query_prefix:
            q = [f for f in valid_files if f.startswith(f"block_csv_data_{a.query_prefix}")]
            valid_files = q + [f for f in valid_files if f not in q]
    else:
        threshold = int(np.floor((len(files) - 1) * 0.9))
        train_files, valid_files = files[:threshold], files[threshold:]
    ref_files = [train_files[i] for i in sorted(rng.choice(len(train_files), a.n_ref_files, replace=False))]
    log(f"reference blocks {ref_files}; query block {valid_files[0]}")

    ids_r, R = [], []
    for f in ref_files:
        i, X = read_block(PARQUET_DIR / f)
        ids_r.append(i); R.append(X)
        log(f"loaded {f} {X.shape}")
    ids_r = np.concatenate(ids_r); R = np.concatenate(R)
    ids_v, V = read_block(PARQUET_DIR / valid_files[0])
    qi = np.sort(rng.choice(len(ids_v), a.n_query, replace=False))
    ids_q, Xq = ids_v[qi], V[qi].copy()
    del V
    overlap = set(ids_q) & set(ids_r)
    if overlap:
        raise ValueError(f"{len(overlap)} query IDs also occur in the reference")

    masks, obs_in = [], []
    for q in range(len(Xq)):
        obs = np.flatnonzero(np.isfinite(Xq[q]))
        m = np.zeros(Xq.shape[1], bool)
        m[rng.choice(obs, int(len(obs) * a.mask_ratio), replace=False)] = True
        masks.append(m)
        obs_in.append(np.isfinite(Xq[q]) & ~m)
    Xq_in = np.where(np.stack(obs_in), Xq, 0).astype(np.float32)
    Wq = np.stack(obs_in).astype(np.float32)

    dmap = dataset_map()
    q_ds = [dmap.get(str(i)) for i in ids_q]
    r_ds = [dmap.get(str(i)) for i in ids_r]
    q_union = frozenset().union(*[d for d in q_ds if d])
    keep_b = np.array([not (d and (d & q_union)) for d in r_ds])
    summary = {
        "inputs": {"parquet_dir": str(PARQUET_DIR), "reference_blocks": ref_files,
                   "query_block": valid_files[0], "validation_blocks": valid_files,
                   "file_stats": {f: [os.path.getsize(PARQUET_DIR / f), os.path.getmtime(PARQUET_DIR / f)]
                                  for f in ref_files + [valid_files[0]]},
                   "qc_table": str(QC_TABLE)},
        "params": vars(a),
        "n_reference": int(len(R)), "n_query": int(len(Xq)),
        "query_ids_mapped_to_dataset": int(sum(d is not None for d in q_ds)),
        "reference_ids_mapped_to_dataset": int(sum(d is not None for d in r_ds)),
        "n_query_datasets": len(q_union), "n_reference_regime_b": int(keep_b.sum()),
        "mean_observed_fraction_query": float(np.mean(np.isfinite(Xq))),
        "results": {},
    }
    log(f"query datasets {len(q_union)}; regime-B reference {keep_b.sum()}/{len(R)}")
    ranks = [int(r) for r in a.ranks.split(",")]
    subset_all = list(range(len(Xq)))
    subset_model = list(range(min(a.n_model, len(Xq))))
    preds = {}
    for regime, keep in (("A_random_profiles", np.ones(len(R), bool)), ("B_heldout_datasets", keep_b)):
        Rr = R[keep]
        mu = chunked_nanmean(Rr)
        mu = np.where(np.isfinite(mu), mu, 0.5)
        Rf = np.where(np.isfinite(Rr), Rr, mu[None, :].astype(np.float32)).astype(np.float32)
        preds[f"{regime}:cpg_mean"] = [mu[np.flatnonzero(m)].astype(np.float32) for m in masks]
        log(f"{regime}: kNN")
        preds[f"{regime}:knn{a.k}"] = knn_predict(Xq_in, Wq, Rf, Rr, mu, masks, a.k)
        log(f"{regime}: PPCA fit")
        Vp, lam, s2 = ppca_fit(Rf, mu, max(ranks))
        summary[f"{regime}_ppca_sigma2"] = s2
        log(f"{regime}: PPCA predict ranks {ranks}")
        pp = ppca_predict_all(Xq, obs_in, masks, mu, Vp, lam, s2, ranks)
        for r in ranks:
            preds[f"{regime}:ppca{r}"] = pp[r]
        del Rf, Rr
    if not a.skip_model:
        log("MethylGPT inference")
        mlm, mvc, info = methylgpt_predict(Xq[: len(subset_model)], masks[: len(subset_model)], a.device)
        preds["methylgpt_mlm"] = mlm + [None] * (len(Xq) - len(mlm))
        preds["methylgpt_mvc"] = mvc + [None] * (len(Xq) - len(mvc))
        summary["methylgpt_load"] = {k: info[k] for k in ("loaded_tensors", "parameters", "paths")}
    for name, p in preds.items():
        subsets = {"model_subset": subset_model}
        if not name.startswith("methylgpt"):
            subsets["all_query"] = subset_all
        for sname, sub in subsets.items():
            pp, tt, jj = gather(p, masks, Xq, sub)
            summary["results"].setdefault(sname, {})[name] = metrics(pp, tt, jj)
    for sname, block in summary["results"].items():
        for regime in ("A_random_profiles", "B_heldout_datasets"):
            base = block.get(f"{regime}:cpg_mean", {}).get("mse")
            if base:
                for name, m in block.items():
                    m.setdefault("skill_vs_mean", {})[regime] = 1 - m["mse"] / base
    summary["elapsed_s"] = time.time() - t0
    Path(a.out).write_text(json.dumps(summary, indent=2))
    log(f"wrote {a.out}")
    for sname, block in summary["results"].items():
        print(f"\n== {sname}")
        for name, m in block.items():
            print(f"{name:34s} MSE {m['mse']:.5f}  MAE {m['mae']:.4f}  withinR2med "
                  f"{m['within_cpg_r2_median'] if m['within_cpg_r2_median'] is None else round(m['within_cpg_r2_median'], 3)}")


if __name__ == "__main__":
    main()
