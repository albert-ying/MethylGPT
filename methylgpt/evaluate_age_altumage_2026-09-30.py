#!/usr/bin/env python3
"""Chronological age on the MethylGPT AltumAge split with MethylGPT-2 (run after pretraining).

AltumAge validation and test profiles were excluded from MethylGPT-2 pretraining; training
profiles were included without labels. Every model is fitted on the 5,461 training
profiles, its settings are chosen by validation MedAE (1,366 profiles), and the 4,626 test
profiles are scored once. Targets use Horvath's log-linear age transform (adult age 20).

Models
  probe_ridge_scores   ridge on the PPCA-core scores (standardized)
  probe_ridge_embed    ridge on the full embedding (scores + pooled latents)
  probe_mlp_embed      MLP on the full embedding, fixed settings, three seeds
  finetune             encoder fine-tuned with an MLP head, masked-input augmentation,
                       epoch chosen on validation, three seeds
Reference numbers on the same split (computed 30 Sep): January MethylGPT fine-tuned
MedAE 2.21; PCA-1024 ridge 2.25; ElasticNet with the Horvath transform 2.31.
"""
from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn

from methylgpt_lib import cuda_preinit, ARTIFACTS, PPCACore, log, require, write_json
from methylgpt_model import MethylGPT2, load_checkpoint

ALT = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/finetuning/age_prediction/age_data/altumage_metadata")


def F(age, adult=20.0):
    age = np.asarray(age, float)
    return np.where(age <= adult, np.log(age + 1) - np.log(adult + 1), (age - adult) / (adult + 1))


def Finv(v, adult=20.0):
    v = np.asarray(v, float)
    return np.where(v <= 0, (adult + 1) * np.exp(v) - 1, v * (adult + 1) + adult)


def scores(pred, y):
    e = np.abs(pred - y)
    return {"medae": float(np.median(e)), "mae": float(np.mean(e)), "pearson_r": float(np.corrcoef(pred, y)[0, 1]),
            "n": int(len(y))}


def load_split(name):
    t = pq.read_table(require(ALT / f"altumage_{name}.parquet"), columns=["id", "data", "age"])
    col = t.column("data").combine_chunks()
    X = col.values.to_numpy(zero_copy_only=False).reshape(len(t), -1).astype(np.float32)
    X = np.where(np.isfinite(X), np.clip(X, 0, 1), np.nan).astype(np.float32)
    return np.asarray(t.column("id").to_pylist()), X, np.asarray(t.column("age").to_pylist(), float)


def load_model(ckpt, core_path, dev):
    return load_checkpoint(ckpt, core_path, dev)


def embed(model, core, X, dev, batch=16, mask_rate=0.0, gen=None):
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch):
            x = torch.from_numpy(X[i:i + batch]).to(dev)
            ctx = torch.isfinite(x)
            if mask_rate > 0:
                ctx &= torch.rand(x.shape, device=dev, generator=gen) >= mask_rate
            s = core.posterior(x, ctx.float())
            with torch.autocast("cuda", enabled=False):
                xs, c, e = model.inputs(x, ctx, s)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = model.encode(xs, c, e, ctx, s)
            out.append(model.embedding(z, s).float().cpu().numpy())
    return np.concatenate(out)


def ridge_select(Ztr, ytr_f, Zva, yva, Zte, yte, lams=np.logspace(-2, 5, 15)):
    m, sd = Ztr.mean(0), Ztr.std(0) + 1e-8
    A, B, C = (Ztr - m) / sd, (Zva - m) / sd, (Zte - m) / sd
    ym = ytr_f.mean()
    G = A.T @ A
    best = None
    for lam in lams:
        w = np.linalg.solve(G + lam * np.eye(G.shape[0]), A.T @ (ytr_f - ym))
        sc = scores(Finv(B @ w + ym), yva)
        if best is None or sc["medae"] < best[1]["medae"]:
            best = (lam, sc, w)
    lam, sc, w = best
    return {"lambda": float(lam), "edge": bool(lam in (lams[0], lams[-1])), "valid": sc,
            "test": scores(Finv(C @ w + ym), yte)}


def mlp_probe(Ztr, ytr_f, Zva, yva, Zte, yte, dev, seeds=(1, 2, 3), epochs=200):
    m, sd = Ztr.mean(0), Ztr.std(0) + 1e-8
    T = [torch.from_numpy(((Z - m) / sd).astype(np.float32)).to(dev) for Z in (Ztr, Zva, Zte)]
    fm, fs = ytr_f.mean(), ytr_f.std()
    t = torch.from_numpy(((ytr_f - fm) / fs).astype(np.float32)).to(dev)
    res = {}
    for seed in seeds:
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(T[0].shape[1], 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, 128), nn.GELU(),
                            nn.Dropout(0.2), nn.Linear(128, 1)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
        best = None
        for ep in range(epochs):
            net.train()
            perm = torch.randperm(len(t), device=dev)
            for i in range(0, len(perm), 128):
                b = perm[i:i + 128]
                loss = ((net(T[0][b]).squeeze(-1) - t[b]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                pv = Finv(net(T[1]).squeeze(-1).cpu().numpy() * fs + fm)
            sc = scores(pv, yva)
            if best is None or sc["medae"] < best[1]["medae"]:
                with torch.no_grad():
                    pt = Finv(net(T[2]).squeeze(-1).cpu().numpy() * fs + fm)
                best = (ep, sc, scores(pt, yte))
        res[seed] = {"epoch": best[0], "valid": best[1], "test": best[2]}
    res["test_medae_mean"] = float(np.mean([v["test"]["medae"] for k, v in res.items() if k in seeds]))
    return res


class AgeHead(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 1))

    def forward(self, e):
        return self.net(e).squeeze(-1)


def finetune(ckpt, core_path, data, dev, seed, epochs=30, batch=32, lr_enc=2e-5, lr_head=1e-3, mask_max=0.3):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data
    model, core, _ = load_model(ckpt, core_path, dev)
    torch.manual_seed(seed)
    head = AgeHead(core.rank + model.latents.shape[1]).to(dev)
    ftr = F(ytr)
    fm, fs = ftr.mean(), ftr.std()
    t = torch.from_numpy(((ftr - fm) / fs).astype(np.float32)).to(dev)
    enc_params = [p for n, p in model.named_parameters() if not n.startswith(("u1", "u2", "q_emb", "out", "dec", "h_norm"))]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": lr_enc, "weight_decay": 0.01},
                             {"params": head.parameters(), "lr": lr_head, "weight_decay": 0.01}])
    gen = torch.Generator(device=dev).manual_seed(seed)
    rng = np.random.default_rng(seed)
    steps_per_epoch = math.ceil(len(Xtr) / batch)
    best = None

    def predict(X):
        model.eval(); head.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 16):
                x = torch.from_numpy(X[i:i + 16]).to(dev)
                ctx = torch.isfinite(x)
                s = core.posterior(x, ctx.float())
                with torch.autocast("cuda", enabled=False):
                    xs, c, e = model.inputs(x, ctx, s)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z = model.encode(xs, c, e, ctx, s)
                    out.append(head(model.embedding(z, s)).float().cpu().numpy())
        return Finv(np.concatenate(out) * fs + fm)

    for ep in range(epochs):
        model.train(); head.train()
        perm = rng.permutation(len(Xtr))
        for i in range(steps_per_epoch):
            b = perm[i * batch:(i + 1) * batch]
            x = torch.from_numpy(Xtr[b]).to(dev)
            ctx = torch.isfinite(x) & (torch.rand(x.shape, device=dev, generator=gen) >= mask_max * torch.rand(
                (len(b), 1), device=dev, generator=gen))
            s = core.posterior(x, ctx.float())
            with torch.autocast("cuda", enabled=False):
                xs, c, e = model.inputs(x, ctx, s)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = model.encode(xs, c, e, ctx, s)
                pred = head(model.embedding(z, s))
            loss = ((pred.float() - t[torch.from_numpy(b).to(dev)]) ** 2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0)
            opt.step()
        sc = scores(predict(Xva), yva)
        if best is None or sc["medae"] < best[1]["medae"]:
            best = (ep, sc, copy.deepcopy(head.state_dict()), {k: v.detach().clone() for k, v in model.state_dict().items()})
        log(f"finetune seed {seed} epoch {ep}: valid MedAE {sc['medae']:.3f}")
    head.load_state_dict(best[2]); model.load_state_dict(best[3])
    return {"epoch": best[0], "valid": best[1], "test": scores(predict(Xte), yte)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--core", default=None, help="defaults to the core recorded in the checkpoint")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-finetune", action="store_true")
    ap.add_argument("--seeds", default="1,2,3")
    a = ap.parse_args()
    cuda_preinit()
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device(a.device)
    (_, Xtr, ytr), (_, Xva, yva), (_, Xte, yte) = load_split("train"), load_split("valid"), load_split("test")
    log(f"AltumAge split: train {len(Xtr)} valid {len(Xva)} test {len(Xte)}")
    model, core, ck = load_model(Path(a.ckpt), a.core, dev)
    Ztr, Zva, Zte = (embed(model, core, X, dev) for X in (Xtr, Xva, Xte))
    K = core.rank
    res = {"checkpoint": a.ckpt, "checkpoint_step": int(ck["step"]), "models": {}}
    ftr = F(ytr)
    res["models"]["probe_ridge_scores"] = ridge_select(Ztr[:, :K], ftr, Zva[:, :K], yva, Zte[:, :K], yte)
    res["models"]["probe_ridge_embed"] = ridge_select(Ztr, ftr, Zva, yva, Zte, yte)
    log(f"ridge probes: scores {res['models']['probe_ridge_scores']['test']['medae']:.3f}, "
        f"embedding {res['models']['probe_ridge_embed']['test']['medae']:.3f}")
    seeds = tuple(int(s) for s in a.seeds.split(","))
    res["models"]["probe_mlp_embed"] = mlp_probe(Ztr, ftr, Zva, yva, Zte, yte, dev, seeds=seeds)
    log(f"MLP probe test MedAE mean {res['models']['probe_mlp_embed']['test_medae_mean']:.3f}")
    write_json(Path(a.out), res)
    if not a.skip_finetune:
        ft = {}
        for seed in seeds:
            ft[seed] = finetune(Path(a.ckpt), a.core, ((Xtr, ytr), (Xva, yva), (Xte, yte)), dev, seed)
            log(f"finetune seed {seed}: valid {ft[seed]['valid']['medae']:.3f} test {ft[seed]['test']['medae']:.3f}")
            res["models"]["finetune"] = ft
            write_json(Path(a.out), res)
        ft["test_medae_mean"] = float(np.mean([ft[s]["test"]["medae"] for s in seeds]))
        res["models"]["finetune"] = ft
    res["elapsed_s"] = time.time() - t0
    write_json(Path(a.out), res)
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
