"""Shared code for the MethylGPT retraining (30 Sep 2026).

Corpus access, the probabilistic-PCA (PPCA) core with batched GPU posteriors, masking
helpers and metrics. Model definitions live in methylgpt2_model.py.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
CORPUS_DIR = ARTIFACTS / "corpus_2026-09-30"
N_CPG = 49156


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cuda_preinit(max_tries: int = 300, wait_s: float = 2.0) -> None:
    """Initialise the CUDA driver API before PyTorch does.

    On this shared server cuInit intermittently returns CUDA_ERROR_NOT_INITIALIZED (3)
    while the GPUs are healthy; PyTorch does not retry a failed lazy init, so retry the
    driver call here. Success is sticky for the process.
    """
    import ctypes
    lib = ctypes.CDLL("libcuda.so.1")
    for attempt in range(1, max_tries + 1):
        rc = lib.cuInit(0)
        if rc == 0:
            if attempt > 1:
                log(f"cuInit succeeded on attempt {attempt}")
            return
        time.sleep(wait_s)
    sys.exit(f"cuInit failed {max_tries} times (last code {rc})")


def require(path: Path) -> Path:
    if not Path(path).exists():
        sys.exit(f"Cannot proceed - required data file not found: {path}")
    return Path(path)


class Corpus:
    """Float16 memmap of all pretraining profiles plus the fixed split table."""

    def __init__(self, corpus_dir: Path = CORPUS_DIR):
        self.dir = Path(corpus_dir)
        self.beta = np.load(require(self.dir / "beta_f16.npy"), mmap_mode="r")
        self.profiles = pd.read_parquet(require(self.dir / "profiles.parquet"))
        if len(self.profiles) != self.beta.shape[0] or self.beta.shape[1] != N_CPG:
            raise ValueError("Corpus memmap and profile table disagree")

    def rows(self, split: str) -> np.ndarray:
        return self.profiles.index[self.profiles.split == split].to_numpy()

    def get(self, rows) -> np.ndarray:
        rows = np.sort(np.asarray(rows))
        return np.asarray(self.beta[rows], dtype=np.float32)


class PPCACore(torch.nn.Module):
    """x = mu + V s + e, s_k ~ N(0, lam_k), e ~ N(0, sigma2); V orthonormal over all CpGs.

    The posterior mean of s uses only the observed CpGs of each profile. Because V^T V = I,
    V_O^T V_O = V^T diag(w) V for the 0/1 observation weights w.
    """

    def __init__(self, mu, V, lam, sigma2, rank=None):
        super().__init__()
        rank = rank or V.shape[1]
        self.register_buffer("mu", torch.as_tensor(mu, dtype=torch.float32))
        self.register_buffer("V", torch.as_tensor(V[:, :rank], dtype=torch.float32).contiguous())
        self.register_buffer("lam", torch.as_tensor(lam[:rank], dtype=torch.float32))
        self.sigma2 = float(sigma2)
        self.rank = rank

    @classmethod
    def load(cls, path: Path, rank=None):
        z = np.load(require(path))
        rank = rank or int(z["V"].shape[1])
        trace, lam_all = float(z["trace"]), z["lam"].astype(np.float64)
        sigma2 = (trace - lam_all[:rank].sum()) / (z["V"].shape[0] - rank)
        return cls(z["mu"], z["V"], z["lam"], sigma2, rank)

    @torch.no_grad()
    def posterior(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """x: (B, P) float, any value where w == 0; w: (B, P) {0,1}. Returns s (B, K)."""
        xc = torch.where(w > 0, x - self.mu, torch.zeros((), device=x.device)) * w
        b = (xc @ self.V) / self.sigma2
        prior = torch.diag(1.0 / self.lam)
        s = torch.empty(x.shape[0], self.rank, device=x.device)
        for i in range(x.shape[0]):
            G = (self.V * w[i, :, None]).T @ self.V
            A = G / self.sigma2 + prior
            L = torch.linalg.cholesky(A)
            s[i] = torch.cholesky_solve(b[i, :, None], L)[:, 0]
        return s

    @torch.no_grad()
    def predict(self, s: torch.Tensor, cols=None) -> torch.Tensor:
        if cols is None:
            return self.mu + s @ self.V.T
        return self.mu[cols] + s @ self.V[cols].T


def masked_view(x: torch.Tensor, ratio: torch.Tensor, gen: torch.Generator):
    """Random masks over observed CpGs. ratio: (B,) fraction of observed CpGs to mask."""
    obs = torch.isfinite(x)
    r = torch.rand(x.shape, device=x.device, generator=gen)
    masked = obs & (r < ratio[:, None])
    ctx = obs & ~masked
    return obs, ctx, masked


def mse_metrics(pred: np.ndarray, truth: np.ndarray, cpg: np.ndarray, min_per_cpg: int = 20) -> dict:
    pred = pred.astype(np.float64)
    truth = truth.astype(np.float64)
    res = pred - truth
    out = {"n_targets": int(len(truth)), "mse": float(np.mean(res ** 2)), "mae": float(np.mean(np.abs(res))),
           "mae_clipped": float(np.mean(np.abs(np.clip(pred, 0, 1) - truth)))}
    P = int(cpg.max()) + 1
    n = np.bincount(cpg, minlength=P).astype(np.float64)
    sy = np.bincount(cpg, weights=truth, minlength=P)
    sy2 = np.bincount(cpg, weights=truth ** 2, minlength=P)
    sse = np.bincount(cpg, weights=res ** 2, minlength=P)
    with np.errstate(invalid="ignore", divide="ignore"):
        sst = sy2 - sy ** 2 / n
    keep = (n >= min_per_cpg) & (sst > 1e-12)
    out["within_cpg_r2_pooled"] = float(1 - sse[keep].sum() / sst[keep].sum()) if keep.any() else None
    out["within_cpg_r2_median"] = float(np.median(1 - sse[keep] / sst[keep])) if keep.any() else None
    out["n_cpgs_within"] = int(keep.sum())
    return out


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))


# ---- chronological age utilities (Horvath log-linear transform, adult age 20) ----
ALTUMAGE_DIR = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/finetuning/age_prediction/age_data/altumage_metadata")


def F(age, adult=20.0):
    age = np.asarray(age, float)
    return np.where(age <= adult, np.log(age + 1) - np.log(adult + 1), (age - adult) / (adult + 1))


def Finv(v, adult=20.0):
    v = np.asarray(v, float)
    return np.where(v <= 0, (adult + 1) * np.exp(v) - 1, v * (adult + 1) + adult)


def age_scores(pred, y):
    e = np.abs(np.asarray(pred) - np.asarray(y))
    return {"medae": float(np.median(e)), "mae": float(np.mean(e)),
            "pearson_r": float(np.corrcoef(pred, y)[0, 1]), "n": int(len(y))}


def label_subset(pool_mask, n, seed):
    """Label-efficiency subsets: the first n profiles of a seeded permutation of the pool, so subsets are nested
    across n for one seed; every fifth profile of the subset is held out for model selection (penalty or
    checkpoint). Returns (fitting rows, selection rows)."""
    perm = np.random.default_rng(20261012 + seed).permutation(np.flatnonzero(pool_mask))[:n]
    sel = np.zeros(len(perm), bool)
    sel[::5] = True
    return np.sort(perm[~sel]), np.sort(perm[sel])


def load_altumage_split(name):
    import pyarrow.parquet as pq
    t = pq.read_table(require(ALTUMAGE_DIR / f"altumage_{name}.parquet"), columns=["id", "data", "age"])
    col = t.column("data").combine_chunks()
    X = col.values.to_numpy(zero_copy_only=False).reshape(len(t), -1).astype(np.float32)
    X = np.where(np.isfinite(X), np.clip(X, 0, 1), np.nan).astype(np.float32)
    return np.asarray(t.column("id").to_pylist()), X, np.asarray(t.column("age").to_pylist(), float)
