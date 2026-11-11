#!/usr/bin/env python3
"""Fig. 3 cross-array consistency on all 450K and EPIC profiles of set D.

Written on 10-11 Nov 2026 before running. It repeats part E of attribution_stability_fig3_options_2026-11-10.py on more
profiles, with per-profile outputs. Every result is reported. Only public GEO profiles already exported for Fig. 3 appear
in the outputs; no Generation Scotland file is read; nothing is trained.

PROFILES  Every profile of set D (artifacts/figure_tables_2026-11-10/fig3_setD_umap_2026-11-10.csv.gz) whose array
  column is 450K or EPIC: 39,099 expected (11,907 test-split, 27,192 training-split). There is no coverage threshold.
  Native 27K and 'other' profiles are excluded, because they have no CpGs to remove.
COPIES  For each profile, the complete copy (every CpG it was deposited with; values clipped to [0, 1] as the released
  package does) and the 27K copy (every CpG outside the 27K-array mask of platform_vocab_masks_2026-10-01.npz set
  unmeasured), exactly as in part E.
REPRESENTATIONS  Each is computed from the copy itself:
  - raw: beta values with unmeasured CpGs filled with the input layer's training means (core mu);
  - latent: latent scores (1,536);
  - encoder: dimensions 1,537 to 2,048 of the released run-1 embedding.
  Complete copies are taken from the bundle pass of export_figure_tables_2026-11-10.py (bundle_pass_D.npz: same bundle,
  same inputs; check 2). 27K copies are computed here with the released bundle.
  For each representation: centred exact PCA of both copies of all profiles together (78,198 rows), 50 components.
  Matrices above 40,000 x 40,000 (the raw covariance, 49,156 x 49,156) use the float64 subspace iteration of
  export_figure_tables amendment 1. Then one UMAP of both copies (umap-learn, 30 neighbours, min_dist 0.3, Euclidean,
  random_state 0) on the 50 components.
PER-PROFILE OUTPUTS  One row per profile and representation, on the 50 components, cosine distance:
  - rank of the profile's own complete copy among all 39,099 complete copies by distance from its 27K copy
    (1 + the number of other complete copies strictly closer; 1 = nearest);
  - distance between the two copies;
  - distance from the complete copy to its nearest other complete copy;
  - split, dataset, tissue, array, and the number of measured vocabulary CpGs in the complete copy.
SUMMARY METRICS  Per representation, separately for test-split profiles (primary), training-split profiles and all:
  - share of profiles with rank 1;
  - share with rank <= 10;
  - median copy distance divided by median nearest-neighbour distance.
  95% intervals from 2,000 draws resampling the datasets of the subset with replacement (seed 20261110; per-profile
  values held fixed; percentile interval).
CHECK AGAINST PART E  The 10,020 profiles of fig3_option2_cross_array_umap_2026-11-10.csv.gz get the same metrics, with
  ranks and nearest neighbours among their own 10,020 complete copies, on the components of this PCA. They are reported
  beside fig3_option2_cross_array_metrics_2026-11-10.csv. No tolerance applies: the PCA differs.

CHECKS (abort on failure): 1, every profile has array 450K or EPIC and is a GEO profile (GSM in GSE); the profile and
  split counts are reported against 39,099, 11,907 and 27,192; 2, latent and encoder of 64 complete copies recomputed
  with the bundle equal the cache (max abs diff <= 1e-3); 3, the 10,020 part E profiles are all among the profiles.
STAGES  --stage extract (GPU; CUDA_VISIBLE_DEVICES set to one GPU) | umap (CPU) | write | all
OUTPUTS (artifacts/figure_tables_2026-11-10/)
  - fig3_all_cross_array_umap_2026-11-10.csv.gz
  - fig3_all_cross_array_per_profile_2026-11-10.csv.gz
  - fig3_all_cross_array_metrics_2026-11-10.csv
  - an entry in README_options_2026-11-10.md
  Work files go to artifacts/fig3_all_cross_array_2026-11-10_work/ (not exported).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from methylgpt2_lib import ARTIFACTS, CORPUS_DIR, log, require, write_json  # noqa: E402

DATE = "2026-11-10"
FT = ARTIFACTS / f"figure_tables_{DATE}"
FTW = ARTIFACTS / f"figure_tables_{DATE}_work"
WORK = ARTIFACTS / f"fig3_all_cross_array_{DATE}_work"
BUNDLE = ARTIFACTS / "release/methylgpt2-1.1"
RANK = 1536
SEED = 20261110
NBOOT = 2000
REPS = ["raw", "latent", "encoder"]
CHECKS = []


def chk(name, ok, detail=""):
    CHECKS.append({"check": name, "passed": bool(ok), "detail": str(detail)})
    log(f"check {name}: {'pass' if ok else 'FAIL'} {detail}")


def save_checks(stage):
    WORK.mkdir(parents=True, exist_ok=True)
    write_json(WORK / f"checks_{stage}.json", CHECKS)
    bad = [c for c in CHECKS if not c["passed"]]
    if bad:
        sys.exit(f"{stage}: {len(bad)} check(s) failed: {[c['check'] for c in bad]}")


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def profiles():
    t = pd.read_csv(require(FT / f"fig3_setD_umap_{DATE}.csv.gz"),
                    usecols=["corpus_row", "profile_id", "dataset", "split", "tissue", "array", "n_observed_vocab_cpgs"])
    return t[t.array.isin(["450K", "EPIC"])].reset_index(drop=True)


def cosine_metrics(Pc, Pr, dev, chunk=2048):
    """Pc, Pr: complete and 27K copies (n x k). Returns rank of own complete copy, copy distance, nearest-neighbour
    distance among complete copies (cosine)."""
    import torch
    Tc = torch.as_tensor(Pc, dtype=torch.float64, device=dev)
    Tr = torch.as_tensor(Pr, dtype=torch.float64, device=dev)
    Tc, Tr = Tc / Tc.norm(dim=1, keepdim=True), Tr / Tr.norm(dim=1, keepdim=True)
    n = len(Tc)
    rank, dcopy, dnn = np.empty(n, np.int64), np.empty(n), np.empty(n)
    for i in range(0, n, chunk):
        j = torch.arange(i, min(i + chunk, n), device=dev)
        S = Tr[j] @ Tc.T
        own = S[torch.arange(len(j), device=dev), j]
        rank[i:i + chunk] = (1 + (S > own[:, None]).sum(1)).cpu().numpy()
        dcopy[i:i + chunk] = (1 - own).clamp_min(0).cpu().numpy()
        Scc = Tc[j] @ Tc.T
        Scc[torch.arange(len(j), device=dev), j] = -np.inf
        dnn[i:i + chunk] = (1 - Scc.max(1).values).clamp_min(0).cpu().numpy()
    return rank, dcopy, dnn


def stage_extract():
    import torch
    from methylgpt2_lib import Corpus, cuda_preinit
    cuda_preinit()
    torch.backends.cuda.matmul.allow_tf32 = False
    d = torch.device("cuda:0")
    t0 = time.time()
    fx = _load("export_figure_tables", "export_figure_tables_2026-11-10.py")
    P = profiles()
    n = len(P)
    sp = P.split.value_counts().to_dict()
    chk("1 profiles are 450K or EPIC GEO profiles", P.array.isin(["450K", "EPIC"]).all() and fx.is_geo(P.profile_id, P.dataset).all(),
        f"{n} profiles (expected 39,099); split {sp} (expected test 11,907, train 27,192); array {P.array.value_counts().to_dict()}")
    rows = P.corpus_row.to_numpy()
    zb = np.load(require(FTW / "bundle_pass_D.npz"))
    D = zb["rows"]
    pos = np.searchsorted(D, rows)
    assert (D[pos] == rows).all()
    lat_c, enc_c = zb["lat"][pos], zb["emb"][pos][:, RANK:]
    X = Corpus().get(rows)
    np.clip(X, 0, 1, out=X)
    hm27 = np.load(require(ARTIFACTS / "platform_vocab_masks_2026-10-01.npz"))["hm27"]
    sys.path.insert(0, str(HERE / "package/src"))
    from methylgpt2 import MethylGPT
    model = MethylGPT.from_pretrained(str(BUNDLE), device="cuda:0")
    keep = torch.as_tensor(hm27, device=d)

    def bundle(reduced, idx):
        L, E = np.empty((len(idx), RANK), np.float32), np.empty((len(idx), 512), np.float32)
        with torch.no_grad():
            for i in range(0, len(idx), 32):
                x = torch.from_numpy(X[idx[i:i + 32]]).to(d)
                if reduced:
                    x = torch.where(keep, x, torch.full_like(x, float("nan")))
                _, s, e = model._encode(x)
                L[i:i + 32], E[i:i + 32] = s.float().cpu().numpy(), e[:, RANK:].float().cpu().numpy()
        return L, E
    l64, e64 = bundle(False, np.arange(64))
    d2 = max(float(np.abs(l64 - lat_c[:64]).max()), float(np.abs(e64 - enc_c[:64]).max()))
    chk("2 complete-copy latent and encoder equal the cache (64 profiles)", d2 <= 1e-3, f"max abs diff {d2:.3g}")
    t1 = time.time()
    lat_r, enc_r = bundle(True, np.arange(n))
    log(f"27K copies through the bundle: {time.time() - t1:.0f} s")
    mu = model.core.mu.detach().clone()
    del model
    torch.cuda.empty_cache()
    out, info = {"rows": rows}, {}
    for rep in REPS:
        t2 = time.time()
        if rep == "raw":
            T = torch.empty((2 * n, X.shape[1]), dtype=torch.float32, device=d)
            for i in range(0, n, 4096):
                x = torch.from_numpy(X[i:i + 4096]).to(d)
                T[i:i + len(x)] = torch.where(torch.isfinite(x), x, mu)
                T[n + i:n + i + len(x)] = torch.where(keep & torch.isfinite(x), x, mu)
        elif rep == "latent":
            T = torch.from_numpy(np.concatenate([lat_c, lat_r])).to(d)
        else:
            T = torch.from_numpy(np.ascontiguousarray(np.concatenate([enc_c, enc_r]))).to(d)
        S, var, tot, _ = fx.pca(T)
        del T
        torch.cuda.empty_cache()
        out[f"pcs_{rep}"] = S.astype(np.float32)
        rank, dcopy, dnn = cosine_metrics(S[:n], S[n:], d)
        out[f"rank_{rep}"], out[f"dcopy_{rep}"], out[f"dnn_{rep}"] = rank, dcopy, dnn
        info[rep] = {"explained_variance_ratio_50": float(var.sum() / tot), "seconds": round(time.time() - t2, 1)}
        log(f"{rep}: 50 PCs explain {var.sum() / tot:.3f}; rank 1 {np.mean(rank == 1):.4f}; ({time.time() - t2:.0f} s)")
    # check against part E: the 10,020 profiles, ranks among their own complete copies on this PCA
    o = pd.read_csv(require(FT / f"fig3_option2_cross_array_umap_{DATE}.csv.gz"), usecols=["corpus_row", "copy"])
    e_rows = o.loc[o["copy"] == "complete", "corpus_row"].to_numpy()
    inside = np.isin(e_rows, rows)
    chk("3 the part E profiles are among the profiles", inside.all() and len(e_rows) == 10020, f"{inside.sum()} of {len(e_rows)}")
    pe = np.searchsorted(rows, e_rows)
    for rep in REPS:
        S = out[f"pcs_{rep}"].astype(np.float64)
        r_, dc_, dn_ = cosine_metrics(S[:n][pe], S[n:][pe], d)
        out[f"E_rank_{rep}"], out[f"E_dcopy_{rep}"], out[f"E_dnn_{rep}"] = r_, dc_, dn_
    out["E_pos"] = pe
    WORK.mkdir(parents=True, exist_ok=True)
    np.savez(WORK / "extract.npz", **out)
    write_json(WORK / "extract_info.json", {"n": n, "split": sp, "reps": info, "seconds": time.time() - t0})
    save_checks("extract")
    log(f"extract done ({time.time() - t0:.0f} s)")


def _umap(P):
    import umap
    return umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0).fit_transform(np.asarray(P, np.float32)).astype(np.float32)


def stage_umap():
    from concurrent.futures import ProcessPoolExecutor
    z = np.load(require(WORK / "extract.npz"))
    t0 = time.time()
    with ProcessPoolExecutor(3) as ex:
        fut = {rep: ex.submit(_umap, z[f"pcs_{rep}"][:, :50]) for rep in REPS}
        res = {rep: f.result() for rep, f in fut.items()}
    np.savez(WORK / "umap.npz", **res)
    log(f"UMAPs in {time.time() - t0:.0f} s")


def summary(rank, dcopy, dnn, ds, W_by_subset):
    """Point estimates and dataset-resampling intervals for the three metrics."""
    units, codes = np.unique(ds, return_inverse=True)
    rng = np.random.default_rng(SEED)
    members = [np.flatnonzero(codes == g) for g in range(len(units))]
    draws = {"rank1": [], "rank10": [], "ratio": []}
    for _ in range(NBOOT):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(units), len(units))])
        draws["rank1"].append(np.mean(rank[idx] == 1))
        draws["rank10"].append(np.mean(rank[idx] <= 10))
        draws["ratio"].append(np.median(dcopy[idx]) / np.median(dnn[idx]))
    point = {"rank1": float(np.mean(rank == 1)), "rank10": float(np.mean(rank <= 10)), "ratio": float(np.median(dcopy) / np.median(dnn))}
    return {k: (point[k], float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in draws.items()}, \
        {"median_copy_distance": float(np.median(dcopy)), "median_nn_distance": float(np.median(dnn)), "n_datasets": int(len(units))}


METRIC = {"rank1": "share of profiles whose own complete copy is nearest (rank 1)",
          "rank10": "share of profiles whose own complete copy is within rank 10",
          "ratio": "median copy distance / median nearest-neighbour distance among complete copies"}


def stage_write():
    fx = _load("export_figure_tables", "export_figure_tables_2026-11-10.py")
    P = profiles()
    n = len(P)
    z = np.load(require(WORK / "extract.npz"))
    u = np.load(require(WORK / "umap.npz"))
    assert (z["rows"] == P.corpus_row.to_numpy()).all()
    U = pd.DataFrame({"profile_id": np.concatenate([P.profile_id] * 2), "dataset": np.concatenate([P.dataset] * 2),
                      "split": np.concatenate([P.split] * 2), "tissue": np.concatenate([P.tissue] * 2), "array": np.concatenate([P.array] * 2),
                      "copy": np.repeat(["complete", "27K"], n)})
    for rep in REPS:
        U[f"umap1_{rep}"], U[f"umap2_{rep}"] = u[rep][:, 0], u[rep][:, 1]
    chk("1 GEO only (UMAP table)", fx.is_geo(U.profile_id, U.dataset).all())
    per = []
    for rep in REPS:
        per.append(pd.DataFrame({"profile_id": P.profile_id, "dataset": P.dataset, "split": P.split, "tissue": P.tissue, "array": P.array,
                                 "n_observed_vocab_cpgs_complete": P.n_observed_vocab_cpgs, "representation": rep,
                                 "rank_own_complete_copy": z[f"rank_{rep}"], "copy_distance": z[f"dcopy_{rep}"],
                                 "nearest_complete_distance": z[f"dnn_{rep}"]}))
    per = pd.concat(per, ignore_index=True)
    rows = []
    for rep in REPS:
        for sub, m in (("test", (P.split == "test").to_numpy()), ("train", (P.split == "train").to_numpy()), ("all", np.ones(n, bool))):
            est, extra = summary(z[f"rank_{rep}"][m], z[f"dcopy_{rep}"][m], z[f"dnn_{rep}"][m], P.dataset.to_numpy()[m], None)
            for k, (v, lo, hi) in est.items():
                rows.append({"representation": rep, "profiles": sub + (" (primary)" if sub == "test" else ""), "metric": METRIC[k], "value": v,
                             "ci95_low": lo, "ci95_high": hi, "n_profiles": int(m.sum()), **extra, "ranks_among": f"{n:,} complete copies"})
        pe = z["E_pos"]
        est, extra = summary(z[f"E_rank_{rep}"], z[f"E_dcopy_{rep}"], z[f"E_dnn_{rep}"], P.dataset.to_numpy()[pe], None)
        for k, (v, lo, hi) in est.items():
            rows.append({"representation": rep, "profiles": "check: the 10,020 profiles of part E", "metric": METRIC[k], "value": v,
                         "ci95_low": lo, "ci95_high": hi, "n_profiles": int(len(pe)), **extra, "ranks_among": "10,020 complete copies"})
    M = pd.DataFrame(rows)
    files = {}
    for df, name in ((U, f"fig3_all_cross_array_umap_{DATE}.csv.gz"), (per, f"fig3_all_cross_array_per_profile_{DATE}.csv.gz"),
                     (M, f"fig3_all_cross_array_metrics_{DATE}.csv")):
        df.to_csv(FT / name, index=False, float_format="%.6g")
        files[name] = {"rows": int(len(df)), "bytes": (FT / name).stat().st_size}
    old = pd.read_csv(require(FT / f"fig3_option2_cross_array_metrics_{DATE}.csv"))
    info = json.loads((WORK / "extract_info.json").read_text())
    checks = json.loads((WORK / "checks_extract.json").read_text()) + CHECKS
    pd.DataFrame(checks).to_csv(WORK / "checks.csv", index=False)
    # README_options entry (appended)
    L = ["", "## Cross-array consistency on all 450K and EPIC profiles of set D (fig3_all_cross_array_2026-11-10.py)", "",
         f"Design in the script docstring, committed before the run. {n:,} profiles "
         f"({info['split'].get('test', 0):,} test-split, {info['split'].get('train', 0):,} training-split), each as a complete and a 27K copy; one PCA (50 components) "
         "and one UMAP (30 neighbours, min_dist 0.3, seed 0) per representation over both copies of all profiles.", ""]
    desc = {f"fig3_all_cross_array_umap_{DATE}.csv.gz": "both copies of every profile: profile_id, dataset, split, tissue, array, copy (complete or 27K), umap1/umap2 for raw, latent and encoder.",
            f"fig3_all_cross_array_per_profile_{DATE}.csv.gz": "one row per profile and representation: rank_own_complete_copy (1 = the profile's own complete copy is the nearest of all complete copies to its 27K copy), copy_distance, nearest_complete_distance (cosine, 50 components), split, dataset, tissue, array, n_observed_vocab_cpgs_complete.",
            f"fig3_all_cross_array_metrics_{DATE}.csv": "per representation and subset (test, primary; train; all; and the 10,020 profiles of part E as a check): share at rank 1, share within rank 10, median copy distance over median nearest-neighbour distance, with 95% intervals resampling datasets (2,000 draws, seed 20261110)."}
    for k, v in files.items():
        L.append(f"- `{k}`: {v['rows']:,} rows, {v['bytes']:,} bytes; {desc[k]}")
    L.extend(["", "| representation | profiles | metric | value [95% interval] | profiles / datasets |", "|---|---|---|---|---|"])
    for _, r in M.iterrows():
        L.append(f"| {r.representation} | {r.profiles} | {r.metric} | {r.value:.3f} [{r.ci95_low:.3f}, {r.ci95_high:.3f}] | {r.n_profiles:,} / {r.n_datasets} |")
    L.extend(["", "Part E values for comparison (fig3_option2_cross_array_metrics_2026-11-10.csv): " +
              "; ".join(f"{r.representation} {r.metric.split(' among')[0].split(' (')[0]}: {r.value:.3f}" for _, r in old.iterrows()) + ".",
              "Check rows use the components of the larger PCA, so they differ from part E by the change of PCA.",
              "Checks: " + "; ".join(f"{c['check']}: {'pass' if c['passed'] else 'FAIL'} {c['detail']}" for c in checks), ""])
    with open(FT / f"README_options_{DATE}.md", "a") as f:
        f.write("\n".join(L))
    write_json(WORK / "files.json", files)
    save_checks("write")
    log(f"written {files}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["extract", "umap", "write", "all"], required=True)
    a = ap.parse_args()
    for s in (["extract", "umap", "write"] if a.stage == "all" else [a.stage]):
        CHECKS.clear()
        {"extract": stage_extract, "umap": stage_umap, "write": stage_write}[s]()


if __name__ == "__main__":
    main()
