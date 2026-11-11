#!/usr/bin/env python3
"""Zero-shot comparison of pretrained and randomly initialized MethylGPT (no labels, no training on the task).

Task: reconstruct masked values of held-out profiles (the 2,000 test-split profiles and the 30% masks of
evaluate_heldout_2026-09-30.py) and convert profiles to the vocabulary from their 27K-array CpGs (targets: the
observed CpGs off the 27K array; scenario CpG set of platform_vocab_masks_2026-10-01.npz). No model is trained here.
Arms (same architecture):
  pretrained       MethylGPT after pretraining on the unlabelled corpus (factor module fitted; transformer after
                   masked-value training; artifacts/runs/stopped_cf_dense_2026-09-30/best.pt)
  random           MethylGPT with every pretrained component randomly initialized (random orthonormal factor
                   loadings with the training means and variances, artifacts/ppca_core_random_2026-10-03.npz;
                   transformer at random weights, seed 20261001)
  random_transformer  the fitted factor module with the transformer at random weights (seed 20261001)
Reference: training-set CpG means. Metrics as in the benchmark (pooled MSE, MAE, within-CpG R2).
Output: artifacts/eval/zero_shot_init_2026-10-14.json
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch

from methylgpt2_lib import ARTIFACTS, HERE, Corpus, cuda_preinit, log, mse_metrics, require, write_json
from methylgpt2_model import load_checkpoint

spec = importlib.util.spec_from_file_location("heldout", HERE / "evaluate_heldout_2026-09-30.py")
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)
CKPT = ARTIFACTS / "runs/stopped_cf_dense_2026-09-30/best.pt"
RANDOM_CORE = ARTIFACTS / "ppca_core_random_2026-10-03.npz"


@torch.no_grad()
def main():
    dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:0")
    cuda_preinit()
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time()
    try:
        corpus = Corpus()
        plat = np.load(require(ARTIFACTS / "platform_vocab_masks_2026-10-01.npz"))
        require(CKPT), require(RANDOM_CORE)
    except OSError as e:
        raise SystemExit(f"Cannot proceed - required data file not found: {e}")
    rng = np.random.default_rng(20261001)
    test_rows = np.sort(rng.choice(corpus.rows("test"), 2000, replace=False))
    X = torch.from_numpy(corpus.get(test_rows)).to(dev)
    obs = torch.isfinite(X)
    g = torch.Generator(device=dev).manual_seed(int(1000 * 0.3) + 20261001)
    masked30 = obs & (torch.rand(X.shape, device=dev, generator=g) < 0.3)
    on27 = torch.from_numpy(plat["hm27"]).to(dev)[None, :]
    tasks = {"random_30": (obs & ~masked30, masked30), "hm27_conversion": (obs & on27, obs & ~on27)}
    arms = {"pretrained": dict(core=None, random_init=False), "random": dict(core=str(RANDOM_CORE), random_init=True),
            "random_transformer": dict(core=None, random_init=True)}
    res = {"n_profiles": int(len(X)), "design": __doc__.split("Output:")[0].strip(), "tasks": {}}
    for task, (ctx, masked) in tasks.items():
        keep = masked.sum(1) > 0
        idx = torch.nonzero(keep)[:, 0]
        Xk, ck_, mk = X[idx], ctx[idx], masked[idx]
        truth = np.concatenate([Xk[i][mk[i]].cpu().numpy() for i in range(len(Xk))])
        cpgs = np.concatenate([torch.nonzero(mk[i])[:, 0].cpu().numpy() for i in range(len(Xk))])
        out = {"profiles_with_targets": int(len(idx))}
        for arm, cfg in arms.items():
            torch.manual_seed(20261001)
            model, core, _ = load_checkpoint(CKPT, cfg["core"], dev, random_init=cfg["random_init"])
            model.eval()
            pm, pb = H.predict_model(model, core, Xk, ck_, mk)
            out[arm] = mse_metrics(np.concatenate(pm), truth, cpgs)
            if arm == "pretrained":
                mu = core.mu.float().cpu().numpy()
                out["training CpG means"] = mse_metrics(mu[cpgs], truth, cpgs)
            log(f"{task} {arm}: MSE {out[arm]['mse']:.5f}, within-CpG R2 {out[arm]['within_cpg_r2_pooled']:.3f} ({time.time() - t0:.0f} s)")
            del model, core
            torch.cuda.empty_cache()
        log(f"{task} training CpG means: MSE {out['training CpG means']['mse']:.5f}")
        res["tasks"][task] = out
    write_json(ARTIFACTS / "eval/zero_shot_init_2026-10-14.json", res)


if __name__ == "__main__":
    main()
