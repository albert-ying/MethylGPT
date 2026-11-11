#!/usr/bin/env python3
"""Multi-task aging heads on MethylGPT-2 (age, sex, tissue), fine-tuned end to end.

Training profiles: training split only, with metadata age in [0, 120] years, sex M/F or a
tissue label (compiled_metadata.csv). Inputs are randomly masked (0-50% of observed CpGs)
as augmentation. Model selection: AltumAge validation profiles (curated ages; excluded
from pretraining). Final scoring, once, on AltumAge test profiles (curated ages) and on
held-out test-split datasets with metadata ages (cross-dataset generalisation).
Age uses Horvath's log-linear transform (adult age 20), standardised.
"""
from __future__ import annotations

import argparse
import copy
import math
import queue
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from methylgpt2_lib import (cuda_preinit, ARTIFACTS, CORPUS_DIR, Corpus, F, Finv, age_scores as scores, label_subset, log,
                            load_altumage_split as load_split, write_json)
from methylgpt2_model import Heads, embed_batch, load_checkpoint


def clean_labels(prof):
    age = prof.age.where((prof.age >= 0) & (prof.age <= 120))
    sex = prof.sex.map({"M": 1.0, "F": 0.0})
    top = prof[prof.split == "train"].tissue.value_counts().head(20).index.tolist()
    tissue = prof.tissue.map({t: i for i, t in enumerate(top)})
    tissue = tissue.where(prof.tissue.isna(), tissue.fillna(len(top)))  # 20 = other labelled tissue
    return age.to_numpy(float), sex.to_numpy(float), tissue.to_numpy(float), top


@torch.no_grad()
def predict_age(model, core, heads, X, dev, fm, fs, batch=32, emb_fn=embed_batch):
    model.eval(); heads.eval()
    out = []
    for i in range(0, len(X), batch):
        x = torch.from_numpy(X[i:i + batch]).to(dev)
        e = emb_fn(model, core, x, torch.isfinite(x))
        a, _, _ = heads(e)
        out.append(a.float().cpu().numpy())
    model.train(); heads.train()
    return Finv(np.concatenate(out) * fs + fm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--lr-enc", type=float, default=5e-5)
    ap.add_argument("--lr-head", type=float, default=5e-4)
    ap.add_argument("--w-sex", type=float, default=0.2)
    ap.add_argument("--w-tissue", type=float, default=0.2)
    ap.add_argument("--mask-max", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--core-only", action="store_true",
                    help="ablation: heads on the core scores alone (no transformer encoder)")
    ap.add_argument("--random-init", action="store_true",
                    help="control: same architecture and core, transformer not pretrained (initial weights)")
    ap.add_argument("--core", default=None, help="factor-model file replacing the one recorded in --ckpt (e.g. the random-loadings control)")
    ap.add_argument("--n-labels", type=int, default=0,
                    help="label-efficiency experiment: use only this many training profiles with a clean age (their age, sex and "
                         "tissue labels); a fifth of them selects the checkpoint instead of the AltumAge validation set. 0 = all")
    ap.add_argument("--label-seed", type=int, default=0, help="seed of the label subset (nested across --n-labels)")
    ap.add_argument("--freeze-encoder", action="store_true", help="train the output heads only; the encoder keeps its initial weights")
    ap.add_argument("--train-last-layers", type=int, default=0,
                    help="train only the last K self-attention blocks of the encoder (and its output normalization) with the heads; 0 = all")
    a = ap.parse_args()
    cuda_preinit()
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    corpus = Corpus()
    prof = corpus.profiles
    age, sex, tissue, top = clean_labels(prof)
    lab = pd.read_parquet(CORPUS_DIR / "profiles_age_clean_2026-10-01.parquet")
    assert (lab.id.to_numpy() == prof.id.to_numpy()).all()
    age = lab.age_clean.to_numpy(float)          # curated or dataset-checked metadata ages
    age_meta = lab.age_meta.to_numpy(float)       # raw metadata ages (held-out evaluation)
    age_flag = lab.age_group_flag.to_numpy(bool)
    is_train = (prof.split == "train").to_numpy()
    labelled = np.flatnonzero(is_train & (np.isfinite(age) | np.isfinite(sex) | np.isfinite(tissue)))
    val_rows = None
    if a.n_labels:
        labelled, val_rows = label_subset(is_train & np.isfinite(age), a.n_labels, a.label_seed)
    ftr = F(age[labelled]) if a.n_labels else F(age[is_train & np.isfinite(age)])
    fm, fs = float(ftr.mean()), float(ftr.std())
    log(f"labelled training profiles {len(labelled)}; with age {int((is_train & np.isfinite(age)).sum())}")
    model, core, ck = load_checkpoint(Path(a.ckpt), a.core, dev, random_init=a.random_init)
    model.train()
    heads = Heads(core.rank if a.core_only else core.rank + model.latents.shape[1], len(top) + 1).to(dev)
    if a.core_only or a.freeze_encoder:
        for prm in model.parameters():
            prm.requires_grad_(False)
    if a.train_last_layers:
        n_self = len(model.enc_self)
        keep = tuple(f"enc_self.{i}." for i in range(n_self - a.train_last_layers, n_self)) + ("sum_norm.",)
        for n_, prm in model.named_parameters():
            prm.requires_grad_(n_.startswith(keep))
        log(f"training the last {a.train_last_layers} of {n_self} self-attention blocks: "
            f"{sum(p_.numel() for p_ in model.parameters() if p_.requires_grad):,} of {sum(p_.numel() for p_ in model.parameters()):,} encoder parameters")
    emb_fn = (lambda m, c, x, ctx: (c.posterior(x, ctx.float()) / torch.sqrt(c.lam))) if a.core_only else embed_batch
    enc_params = [p for n, p in model.named_parameters() if not n.startswith(("u1", "u2", "q_emb", "out", "dec", "h_norm", "ds_head", "q_load", "sum_proj", "h_proj"))]
    enc_params = [p_ for p_ in enc_params if p_.requires_grad]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": a.lr_enc, "weight_decay": 0.01},
                             {"params": heads.parameters(), "lr": a.lr_head, "weight_decay": 0.01}], eps=1e-10)
    base_lrs = [g["lr"] for g in opt.param_groups]
    if val_rows is None:
        _, Xva, yva = load_split("valid")
    else:
        Xva, yva = corpus.get(val_rows), age[val_rows]
    rng = np.random.default_rng(a.seed)
    q = queue.Queue(maxsize=6)

    def producer():
        r = np.random.default_rng(a.seed + 1)
        while True:
            idx = np.sort(r.choice(labelled, a.batch, replace=False))
            q.put((idx, torch.from_numpy(corpus.get(idx)).pin_memory()))

    threading.Thread(target=producer, daemon=True).start()
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    best, hist = None, []
    for step in range(1, a.steps + 1):
        warm = min(1.0, step / 500)
        cos = 0.5 * (1 + math.cos(math.pi * step / a.steps))
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * warm * (0.05 + 0.95 * cos)
        idx, xb = q.get()
        x = xb.to(dev, non_blocking=True)
        keep = torch.rand(x.shape, device=dev, generator=gen) >= a.mask_max * torch.rand((len(x), 1), device=dev, generator=gen)
        ctx = torch.isfinite(x) & keep
        e = emb_fn(model, core, x, ctx)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pa, ps, pt = heads(e)
        ya = torch.from_numpy(age[idx]).to(dev); ys = torch.from_numpy(sex[idx]).to(dev); yt = torch.from_numpy(tissue[idx]).to(dev)
        ma, ms, mt = torch.isfinite(ya), torch.isfinite(ys), torch.isfinite(yt)
        fa = torch.from_numpy(((F(np.nan_to_num(age[idx], nan=30.0)) - fm) / fs).astype(np.float32)).to(dev)
        loss_a = ((pa.float() - fa) ** 2)[ma].mean() if ma.any() else torch.zeros((), device=dev)
        loss_s = Fn.binary_cross_entropy_with_logits(ps.float()[ms], ys[ms].float()) if ms.any() else torch.zeros((), device=dev)
        loss_t = Fn.cross_entropy(pt.float()[mt], yt[mt].long()) if mt.any() else torch.zeros((), device=dev)
        loss = loss_a + a.w_sex * loss_s + a.w_tissue * loss_t
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(heads.parameters()), 1.0)
        opt.step()
        hist.append((float(loss_a), float(loss_s), float(loss_t)))
        if step % 100 == 0:
            h = np.array(hist[-100:])
            log(f"step {step} age {h[:, 0].mean():.4f} sex {h[:, 1].mean():.4f} tissue {h[:, 2].mean():.4f}")
        if step % a.eval_every == 0 or step == a.steps:
            sc = scores(predict_age(model, core, heads, Xva, dev, fm, fs, emb_fn=emb_fn), yva)
            log(f"VALID step {step}: AltumAge valid MedAE {sc['medae']:.3f} MAE {sc['mae']:.3f} r {sc['pearson_r']:.4f}")
            if best is None or sc["medae"] < best[1]["medae"]:
                best = (step, sc)
                torch.save({"model": model.state_dict(), "heads": heads.state_dict(), "step": step, "valid": sc,
                            "fm": fm, "fs": fs, "tissues": top, "ckpt": a.ckpt, "config": vars(a)}, out / "best.pt")
    # Final one-shot scoring with the best validation checkpoint.
    ck2 = torch.load(out / "best.pt", map_location=dev, weights_only=False)
    model.load_state_dict(ck2["model"]); heads.load_state_dict(ck2["heads"])
    _, Xte, yte = load_split("test")
    pte = predict_age(model, core, heads, Xte, dev, fm, fs, emb_fn=emb_fn)
    res = {"best_step": best[0], "valid": best[1], "altumage_test": scores(pte, yte)}
    test_rows = np.flatnonzero((prof.split == "test").to_numpy() & np.isfinite(age_meta))
    ph = predict_age(model, core, heads, corpus.get(test_rows), dev, fm, fs, emb_fn=emb_fn)
    if a.n_labels:
        # label-efficiency runs: also score the AltumAge test set with CpGs missing (masks of the age scenarios), keep the
        # predictions and drop the checkpoint
        res["n_fit"], res["n_selection"] = int(len(labelled)), int(len(val_rows))
        plat = np.load(ARTIFACTS / "platform_vocab_masks_2026-10-01.npz")
        preds = {"altumage_test|all": pte, "heldout|all": ph}
        for scen in ("random_90", "hm27"):
            Xs = Xte.copy()
            if scen == "random_90":
                Xs[np.random.default_rng(20261001 + 90).random(Xs.shape) < 0.9] = np.nan
            else:
                Xs[:, ~plat[scen]] = np.nan
            preds[f"altumage_test|{scen}"] = predict_age(model, core, heads, Xs, dev, fm, fs, emb_fn=emb_fn)
            res[f"altumage_test_{scen}"] = scores(preds[f"altumage_test|{scen}"], yte)
        np.savez_compressed(out / "predictions.npz", **preds)
        (out / "best.pt").unlink()
    keep = ~age_flag[test_rows]
    res["heldout_datasets_all"] = scores(ph, age_meta[test_rows]) | {"n_groups": int(prof.loc[test_rows, "group"].nunique())}
    res["heldout_datasets_unflagged"] = scores(ph[keep], age_meta[test_rows][keep]) | {"n_groups": int(prof.loc[test_rows[keep], "group"].nunique())}
    res["elapsed_s"] = time.time() - t0
    write_json(out / "results.json", res)
    log(f"AltumAge test {res['altumage_test']}; held-out all {res['heldout_datasets_all']}; unflagged {res['heldout_datasets_unflagged']}")


if __name__ == "__main__":
    main()
