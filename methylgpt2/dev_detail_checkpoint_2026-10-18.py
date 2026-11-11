#!/usr/bin/env python3
"""Per-profile development error of a saved pretraining checkpoint (pretraining curve, 18 Oct 2026).

Rebuilds the development set of train_methylgpt2_2026-09-30.py exactly (seed 20260930: 512 development profiles,
30% masks) and saves the per-profile masked-value SSE of the encoder and of the input layer, as evaluate_detail()
does during training. Used for the released 2,000-step checkpoint and the original 4,000-step checkpoint, whose runs
did not save per-profile values. Usage: --ckpt PATH --out NPZ [--device cuda:0]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from methylgpt2_lib import cuda_preinit, Corpus, log, masked_view
from methylgpt2_model import load_checkpoint
import importlib.util

_spec = importlib.util.spec_from_file_location("train_mgpt2", Path(__file__).resolve().parent / "train_methylgpt2_2026-09-30.py")
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--dev-profiles", type=int, default=512)
    a = ap.parse_args()
    if not Path(a.ckpt).is_file():
        raise SystemExit(f"Cannot proceed - required data file not found: {a.ckpt}")
    cuda_preinit()
    torch.backends.cuda.matmul.allow_tf32 = True   # as in training
    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    corpus = Corpus()
    model, core, ck = load_checkpoint(Path(a.ckpt), None, dev)
    rng = np.random.default_rng(a.seed)
    dev_rows = np.sort(rng.choice(corpus.rows("dev"), min(a.dev_profiles, len(corpus.rows("dev"))), replace=False))
    Xd = torch.from_numpy(corpus.get(dev_rows)).to(dev)
    gd = torch.Generator(device=dev).manual_seed(a.seed)
    _, ctx_d, masked_d = masked_view(Xd, torch.full((len(Xd),), 0.3, device=dev), gd)
    pm, pb, pn = T.evaluate_detail(model, core, Xd, ctx_d, masked_d)
    np.savez(a.out, rows=dev_rows, group=corpus.profiles.group.to_numpy()[dev_rows].astype(str), sse_model=pm, sse_core=pb, n=pn,
             step=int(ck["step"]))
    log(f"step {ck['step']}: dev model {pm.sum() / pn.sum():.6f} core {pb.sum() / pn.sum():.6f} -> {a.out}")


if __name__ == "__main__":
    main()
