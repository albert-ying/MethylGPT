#!/usr/bin/env python3
"""No-pretraining control for the factor model: a 'core' with random orthonormal loadings.

Copies the fitted core (ppca_core_meanfilled_2026-09-30.npz) but replaces the 1,536 loading vectors
V with a random orthonormal basis (QR of a Gaussian matrix, seed 20261003). The training mean, the
factor variances and the noise variance are kept, so only the structure learned from the unlabelled
corpus (which CpGs co-vary) is removed. Used with train_aging_heads_2026-09-30.py --random-init --core
to train the full architecture with neither pretrained component.
Output: artifacts/ppca_core_random_2026-10-03.npz
"""
import numpy as np
import torch

from methylgpt_lib import ARTIFACTS


def main():
    z = dict(np.load(ARTIFACTS / "ppca_core_meanfilled_2026-09-30.npz"))
    g = torch.Generator(device="cuda:0").manual_seed(20261003)
    G = torch.randn(z["V"].shape, device="cuda:0", generator=g, dtype=torch.float32)
    Q, _ = torch.linalg.qr(G)
    z["V"] = Q.cpu().numpy().astype(z["V"].dtype)
    np.savez(ARTIFACTS / "ppca_core_random_2026-10-03.npz", **z)
    print("saved", z["V"].shape, "orthonormality error", float(np.abs(z["V"].T @ z["V"] - np.eye(z["V"].shape[1])).max()))


if __name__ == "__main__":
    main()
