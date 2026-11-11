#!/usr/bin/env python3
"""Train MethylGPT-2 (PPCA core + transformer residual) on the training split.

Each step draws profiles uniformly from the training split, masks a per-profile fraction
U(mask_lo, mask_hi) of observed CpGs, computes the core posterior from the remaining CpGs
(fp32), and trains the transformer to predict the masked values (loss on up to
max_targets random masked CpGs per profile). Model selection uses a fixed dev subset
(30% masking); the test split is never loaded here.
"""
from __future__ import annotations

import argparse
import math
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch

from methylgpt2_lib import cuda_preinit, ARTIFACTS, Corpus, PPCACore, log, masked_view, write_json
from methylgpt2_model import MethylGPT2


TABLES = ("id_emb", "load_c", "load_e", "q_emb", "u1", "u2")


def param_groups(model, wd, table_lr_mult):
    """Weight decay on dense matrices; per-CpG tables get a smaller learning rate because
    Adam's per-coordinate steps turn their noisy sparse gradients into a random walk."""
    decay, no_decay, tables = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name in TABLES:
            tables.append(p)
        elif p.ndim >= 2 and name != "latents":
            decay.append(p)
        else:
            no_decay.append(p)
    return [{"params": decay, "weight_decay": wd, "lr_mult": 1.0},
            {"params": no_decay, "weight_decay": 0.0, "lr_mult": 1.0},
            {"params": tables, "weight_decay": 0.0, "lr_mult": table_lr_mult}]


class Prefetcher:
    def __init__(self, corpus, rows, batch, seed, depth=6):
        self.corpus, self.rows, self.batch = corpus, rows, batch
        self.q = queue.Queue(maxsize=depth)
        self.rng = np.random.default_rng(seed)
        self.stop = False
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        while not self.stop:
            idx = np.sort(self.rng.choice(self.rows, self.batch, replace=False))
            self.q.put((idx, torch.from_numpy(self.corpus.get(idx)).pin_memory()))

    def get(self):
        return self.q.get()


def select_targets(masked, k, gen):
    score = torch.rand(masked.shape, device=masked.device, generator=gen)
    score = torch.where(masked, score, torch.full_like(score, -1.0))
    k = min(k, masked.shape[1])
    val, idx = torch.topk(score, k, dim=1)
    return idx, val >= 0


@torch.no_grad()
def evaluate(model, core, X, ctx, masked, batch=8, chunk=4096):
    """Masked-value MSE of the model and of the core over all masked CpGs of the dev set."""
    model.eval()
    sse_m = sse_b = n = 0.0
    for i in range(0, len(X), batch):
        x, c, m = X[i:i + batch], ctx[i:i + batch], masked[i:i + batch]
        s = core.posterior(x, c.float())
        k = int(m.sum(1).max())
        idx, valid = select_targets(m, k, torch.Generator(device=x.device).manual_seed(0))
        with torch.autocast("cuda", enabled=False):
            xs, cc, e = model.inputs(x, c, s)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model.encode(xs, cc, e, c, s)
        for j in range(0, k, chunk):
            sub, val = idx[:, j:j + chunk], valid[:, j:j + chunk]
            pred, base = model.predict_targets(z, s, sub)
            y = torch.where(val, x.gather(1, sub), torch.zeros((), device=x.device))
            zero = torch.zeros((), device=x.device)
            sse_m += float(torch.where(val, (pred - y) ** 2, zero).sum())
            sse_b += float(torch.where(val, (base - y) ** 2, zero).sum())
            n += float(val.sum())
    model.train()
    return sse_m / n, sse_b / n


@torch.no_grad()
def evaluate_detail(model, core, X, ctx, masked, batch=8, chunk=4096):
    """Per-profile masked-value SSE of the model and of the core and the number of masked CpGs (dev set).
    Same computation as evaluate(); used for dataset-resampled intervals (pretraining curve, 18 Oct 2026)."""
    model.eval()
    out_m, out_b, out_n = [], [], []
    for i in range(0, len(X), batch):
        x, c, m = X[i:i + batch], ctx[i:i + batch], masked[i:i + batch]
        s = core.posterior(x, c.float())
        k = int(m.sum(1).max())
        idx, valid = select_targets(m, k, torch.Generator(device=x.device).manual_seed(0))
        with torch.autocast("cuda", enabled=False):
            xs, cc, e = model.inputs(x, c, s)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model.encode(xs, cc, e, c, s)
        sm = torch.zeros(len(x), device=x.device); sb = torch.zeros_like(sm); nn_ = torch.zeros_like(sm)
        for j in range(0, k, chunk):
            sub, val = idx[:, j:j + chunk], valid[:, j:j + chunk]
            pred, base = model.predict_targets(z, s, sub)
            y = torch.where(val, x.gather(1, sub), torch.zeros((), device=x.device))
            zero = torch.zeros((), device=x.device)
            sm += torch.where(val, (pred - y) ** 2, zero).sum(1)
            sb += torch.where(val, (base - y) ** 2, zero).sum(1)
            nn_ += val.sum(1)
        out_m.append(sm.cpu()); out_b.append(sb.cpu()); out_n.append(nn_.cpu())
    model.train()
    return torch.cat(out_m).numpy(), torch.cat(out_b).numpy(), torch.cat(out_n).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", default=str(ARTIFACTS / "ppca_core_meanfilled_2026-09-30.npz"))
    ap.add_argument("--rank", type=int, default=1536)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-targets", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--table-lr-mult", type=float, default=0.3)
    ap.add_argument("--min-lr-frac", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--mask-lo", type=float, default=0.1)
    ap.add_argument("--mask-hi", type=float, default=0.7)
    ap.add_argument("--d-tok", type=int, default=256)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--latents", type=int, default=128)
    ap.add_argument("--score-tokens", type=int, default=16)
    ap.add_argument("--self-layers", type=int, default=8)
    ap.add_argument("--d-read", type=int, default=64)
    ap.add_argument("--encoder", choices=["tokens", "dense"], default="tokens")
    ap.add_argument("--d-dense", type=int, default=2048)
    ap.add_argument("--dense-tokens", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--dev-profiles", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop-at", type=int, default=0,
                    help="stop after this step without changing the learning-rate schedule set by --steps (0 = run to --steps)")
    ap.add_argument("--eval-detail", action="store_true",
                    help="also save per-profile dev SSE at each evaluation (dev_detail_step{N}.npz) for dataset-resampled intervals")
    ap.add_argument("--crossfit-cores", default="",
                    help="comma-separated fold-0 and fold-1 core files (cores fitted on crc32(group) %% 2 halves)")
    a = ap.parse_args()
    cuda_preinit()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # TF32 error in the core posterior (~1e-3 relative) is far below the residuals being learned.
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    corpus = Corpus()
    train_rows, dev_all = corpus.rows("train"), corpus.rows("dev")
    core = PPCACore.load(Path(a.core), rank=a.rank).to(dev)
    model = MethylGPT2(core, d_tok=a.d_tok, d=a.d, heads=a.heads, n_latent=a.latents,
                       n_score_tokens=a.score_tokens, n_self=a.self_layers, d_read=a.d_read,
                       encoder=a.encoder, d_dense=a.d_dense, n_dense_tokens=a.dense_tokens).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    crossfit, group_fold = [], None
    if a.crossfit_cores:
        import zlib
        group_fold = corpus.profiles.group.map(lambda g: zlib.crc32(str(g).encode()) % 2).to_numpy()
        for path in a.crossfit_cores.split(","):
            ck = PPCACore.load(Path(path), rank=a.rank).to(dev)
            Mk = core.V.T @ ck.V                   # (K, K): fold basis -> full basis
            bk = core.V.T @ (ck.mu - core.mu)      # (K,)
            crossfit.append((ck, Mk, bk))
        log(f"cross-fitting with {len(crossfit)} fold cores")
    opt = torch.optim.AdamW(param_groups(model, a.wd, a.table_lr_mult), lr=a.lr, betas=(0.9, 0.95), eps=1e-12)
    step, best = 0, float("inf")
    if a.resume and (out / "last.pt").is_file():
        ck = torch.load(out / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, best = ck["step"], ck["best"]
        log(f"resumed at step {step}, best dev MSE {best:.5f}")
    config = {**vars(a), "n_params": n_params, "n_train": int(len(train_rows)), "core_sigma2": core.sigma2}
    write_json(out / "config.json", config)
    log(f"MethylGPT-2: {n_params / 1e6:.1f}M parameters; train profiles {len(train_rows)}; rank {a.rank}")

    rng = np.random.default_rng(a.seed)
    dev_rows = np.sort(rng.choice(dev_all, min(a.dev_profiles, len(dev_all)), replace=False))
    Xd = torch.from_numpy(corpus.get(dev_rows)).to(dev)
    gd = torch.Generator(device=dev).manual_seed(a.seed)
    _, ctx_d, masked_d = masked_view(Xd, torch.full((len(Xd),), 0.3, device=dev), gd)

    pre = Prefetcher(corpus, train_rows, a.batch, a.seed + step)
    gen = torch.Generator(device=dev).manual_seed(a.seed + 1 + step)
    hist, timing, t_log = [], [], time.time()
    model.train()
    stop = a.stop_at if a.stop_at else a.steps
    while step < stop:
        lr = a.lr * min(1.0, (step + 1) / a.warmup)
        frac = min(1.0, step / a.steps)
        lr *= a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * frac))
        for g in opt.param_groups:
            g["lr"] = lr * g["lr_mult"]
        torch.cuda.synchronize(dev)
        t_start = time.time()
        rows_b, xb = pre.get()
        x = xb.to(dev, non_blocking=True)
        torch.cuda.synchronize(dev)
        t_fetch = time.time()
        ratio = a.mask_lo + (a.mask_hi - a.mask_lo) * torch.rand(len(x), device=dev, generator=gen)
        _, ctx, masked = masked_view(x, ratio, gen)
        if crossfit:
            # Each profile is reconstructed by the core fitted on the other half of the dataset
            # groups; its scores are mapped into the full core's basis for the transformer.
            s = torch.empty(len(x), core.rank, device=dev)
            fit = torch.empty_like(x)
            fold_b = torch.from_numpy(group_fold[rows_b]).to(dev)
            for k, (ck, Mk, bk) in enumerate(crossfit):
                sel = fold_b == 1 - k
                if sel.any():
                    sk = ck.posterior(x[sel], ctx[sel].float())
                    fit[sel] = ck.predict(sk)
                    s[sel] = sk @ Mk.T + bk
        else:
            s, fit = core.posterior(x, ctx.float()), None
        idx, valid = select_targets(masked, a.max_targets, gen)
        torch.cuda.synchronize(dev)
        t_post = time.time()
        pred, base, _ = model(x, ctx, s, idx, fit=fit)
        y = torch.where(valid, x.gather(1, idx), torch.zeros((), device=dev))
        w = valid.float()
        sse = torch.where(valid, (pred - y) ** 2, torch.zeros((), device=dev)).sum()
        with torch.no_grad():
            sse_core = torch.where(valid, (base - y) ** 2, torch.zeros((), device=dev)).sum()
        # Loss relative to the core on the same batch: starts at 1.0, keeps gradients well above
        # Adam's epsilon (raw MSE gradients reach ~1e-12 per encoder weight).
        loss = sse / sse_core.clamp_min(1e-12)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        hist.append((float(sse / w.sum()), float(sse_core / w.sum()), float(gnorm)))
        timing.append((t_fetch - t_start, t_post - t_fetch, time.time() - t_post))
        step += 1
        if step % 100 == 0:
            h = np.array(hist[-100:])
            dt = time.time() - t_log
            t_log = time.time()
            log(f"step {step} loss {h[:, 0].mean():.5f} core {h[:, 1].mean():.5f} "
                f"gain {1 - h[:, 0].mean() / h[:, 1].mean():+.3f} gnorm {h[:, 2].mean():.2f} lr {lr:.2e} "
                f"{100 * a.batch / dt:.0f} prof/s mem {torch.cuda.max_memory_allocated(dev) / 2**30:.1f}G "
                f"t(fetch/post/model) {np.mean([t[0] for t in timing[-100:]]):.3f}/"
                f"{np.mean([t[1] for t in timing[-100:]]):.3f}/{np.mean([t[2] for t in timing[-100:]]):.3f}s")
        if step % a.eval_every == 0 or step == a.steps or step == stop:
            mse_m, mse_b = evaluate(model, core, Xd, ctx_d, masked_d)
            improved = mse_m < best
            best = min(best, mse_m)
            log(f"DEV step {step}: model {mse_m:.5f} core {mse_b:.5f} gain {1 - mse_m / mse_b:+.4f}"
                f"{' (best)' if improved else ''}")
            with open(out / "dev_history.tsv", "a") as fh:
                fh.write(f"{step}\t{mse_m:.6f}\t{mse_b:.6f}\n")
            if a.eval_detail:
                pm, pb, pn = evaluate_detail(model, core, Xd, ctx_d, masked_d)
                np.savez(out / f"dev_detail_step{step}.npz", rows=dev_rows, group=corpus.profiles.group.to_numpy()[dev_rows].astype(str),
                         sse_model=pm, sse_core=pb, n=pn)
            ck = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "best": best, "config": config}
            torch.save(ck, out / "last.pt.tmp")
            (out / "last.pt.tmp").replace(out / "last.pt")
            if improved:
                torch.save({"model": model.state_dict(), "step": step, "dev_mse": mse_m, "config": config},
                           out / "best.pt")
    pre.stop = True
    log("training finished")


if __name__ == "__main__":
    main()
