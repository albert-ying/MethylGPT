#!/usr/bin/env python3
"""Multi-task MethylGPT-2: age, sex, tissue and 25 distilled full-array biomarkers.

Continues from the aging-head checkpoint (encoder and heads) and adds a biomarker head whose
targets are ClockBase clock outputs computed from full arrays (DunedinPACE, PhenoAge,
GrimAge v1/v2 and the v2 protein, smoking, HbA1c and CRP surrogates, McCartney scores),
kept only where at most 10% of the clock's CpGs were missing. Targets are robustly
standardised on training profiles (median/IQR) and clipped to +-6. The fixed schedule's
final checkpoint is used (age-selected checkpoint saved too).

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

from methylgpt2_lib import (cuda_preinit, ARTIFACTS, CORPUS_DIR, Corpus, F, Finv, age_scores as scores, log,
                            load_altumage_split as load_split, write_json)
from methylgpt2_model import BioHeads, Heads, embed_batch, load_checkpoint


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
        a = heads(e)[0]
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
    ap.add_argument("--init-aging", default=str(ARTIFACTS / "runs/aging_heads_clean_2026-10-01/best.pt"))
    ap.add_argument("--w-bio", type=float, default=1.0)
    ap.add_argument("--core-only", action="store_true",
                    help="ablation: heads on the core scores alone (no transformer encoder)")
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
    clocks = pd.read_parquet(CORPUS_DIR / "clock_targets_2026-10-01.parquet")
    assert (clocks.id.to_numpy() == prof.id.to_numpy()).all()
    bio_names = [c for c in clocks.columns if c not in ("id", "split")]
    B = clocks[bio_names].to_numpy(float)
    trB = B[is_train]
    med = np.nanmedian(trB, 0)
    iqr = np.nanpercentile(trB, 75, 0) - np.nanpercentile(trB, 25, 0)
    iqr = np.where(iqr > 0, iqr, 1.0)
    Bz = np.clip((B - med) / (iqr / 1.349), -6, 6)
    labelled = np.flatnonzero(is_train & (np.isfinite(age) | np.isfinite(sex) | np.isfinite(tissue) | np.isfinite(Bz).any(1)))
    ftr = F(age[is_train & np.isfinite(age)])
    fm, fs = float(ftr.mean()), float(ftr.std())
    log(f"labelled training profiles {len(labelled)}; with age {int((is_train & np.isfinite(age)).sum())}")
    model, core, ck = load_checkpoint(Path(a.ckpt), None, dev)
    model.train()
    heads = BioHeads(core.rank if a.core_only else core.rank + model.latents.shape[1], len(top) + 1, len(bio_names)).to(dev)
    if a.init_aging:
        ag = torch.load(a.init_aging, map_location=dev, weights_only=False)
        model.load_state_dict(ag["model"])
        missing = heads.load_state_dict(ag["heads"], strict=False)
        log(f"initialised from {a.init_aging}; new parameters: {missing.missing_keys}")
    if a.core_only:
        for prm in model.parameters():
            prm.requires_grad_(False)
    emb_fn = (lambda m, c, x, ctx: (c.posterior(x, ctx.float()) / torch.sqrt(c.lam))) if a.core_only else embed_batch
    enc_params = [p for n, p in model.named_parameters() if not n.startswith(("u1", "u2", "q_emb", "out", "dec", "h_norm", "ds_head", "q_load", "sum_proj", "h_proj"))]
    enc_params = [p_ for p_ in enc_params if p_.requires_grad]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": a.lr_enc, "weight_decay": 0.01},
                             {"params": heads.parameters(), "lr": a.lr_head, "weight_decay": 0.01}], eps=1e-10)
    base_lrs = [g["lr"] for g in opt.param_groups]
    _, Xva, yva = load_split("valid")
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
            pa, ps, pt, pb = heads(e)
        yb = torch.from_numpy(Bz[idx].astype(np.float32)).to(dev); mb = torch.isfinite(yb)
        loss_b = ((pb.float() - torch.nan_to_num(yb)) ** 2)[mb].mean() if mb.any() else torch.zeros((), device=dev)
        ya = torch.from_numpy(age[idx]).to(dev); ys = torch.from_numpy(sex[idx]).to(dev); yt = torch.from_numpy(tissue[idx]).to(dev)
        ma, ms, mt = torch.isfinite(ya), torch.isfinite(ys), torch.isfinite(yt)
        fa = torch.from_numpy(((F(np.nan_to_num(age[idx], nan=30.0)) - fm) / fs).astype(np.float32)).to(dev)
        loss_a = ((pa.float() - fa) ** 2)[ma].mean() if ma.any() else torch.zeros((), device=dev)
        loss_s = Fn.binary_cross_entropy_with_logits(ps.float()[ms], ys[ms].float()) if ms.any() else torch.zeros((), device=dev)
        loss_t = Fn.cross_entropy(pt.float()[mt], yt[mt].long()) if mt.any() else torch.zeros((), device=dev)
        loss = loss_a + a.w_sex * loss_s + a.w_tissue * loss_t + a.w_bio * loss_b
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(heads.parameters()), 1.0)
        opt.step()
        hist.append((float(loss_a), float(loss_s), float(loss_t), float(loss_b)))
        if step % 100 == 0:
            h = np.array(hist[-100:])
            log(f"step {step} age {h[:, 0].mean():.4f} sex {h[:, 1].mean():.4f} tissue {h[:, 2].mean():.4f} bio {h[:, 3].mean():.4f}")
        if step % a.eval_every == 0 or step == a.steps:
            sc = scores(predict_age(model, core, heads, Xva, dev, fm, fs, emb_fn=emb_fn), yva)
            log(f"VALID step {step}: AltumAge valid MedAE {sc['medae']:.3f} MAE {sc['mae']:.3f} r {sc['pearson_r']:.4f}")
            if best is None or sc["medae"] < best[1]["medae"]:
                best = (step, sc)
                torch.save({"model": model.state_dict(), "heads": heads.state_dict(), "step": step, "valid": sc,
                            "fm": fm, "fs": fs, "tissues": top, "ckpt": a.ckpt, "config": vars(a),
                            "bio_names": bio_names, "bio_median": med, "bio_scale": iqr / 1.349}, out / "best.pt")
    torch.save({"model": model.state_dict(), "heads": heads.state_dict(), "step": a.steps, "fm": fm, "fs": fs,
                "tissues": top, "ckpt": a.ckpt, "config": vars(a), "bio_names": bio_names, "bio_median": med,
                "bio_scale": iqr / 1.349}, out / "final.pt")
    # Final one-shot scoring with the best validation checkpoint.
    ck2 = torch.load(out / "final.pt", map_location=dev, weights_only=False)
    model.load_state_dict(ck2["model"]); heads.load_state_dict(ck2["heads"])
    _, Xte, yte = load_split("test")
    res = {"best_age_step": best[0], "valid_at_best": best[1], "scored_checkpoint": "final", "valid": scores(predict_age(model, core, heads, Xva, dev, fm, fs, emb_fn=emb_fn), yva),
           "altumage_test": scores(predict_age(model, core, heads, Xte, dev, fm, fs, emb_fn=emb_fn), yte)}
    test_rows = np.flatnonzero((prof.split == "test").to_numpy() & np.isfinite(age_meta))
    ph = predict_age(model, core, heads, corpus.get(test_rows), dev, fm, fs, emb_fn=emb_fn)
    keep = ~age_flag[test_rows]
    res["heldout_datasets_all"] = scores(ph, age_meta[test_rows]) | {"n_groups": int(prof.loc[test_rows, "group"].nunique())}
    res["heldout_datasets_unflagged"] = scores(ph[keep], age_meta[test_rows][keep]) | {"n_groups": int(prof.loc[test_rows[keep], "group"].nunique())}
    trows = np.flatnonzero((prof.split == "test").to_numpy() & np.isfinite(Bz).any(1))
    preds = []
    model.eval(); heads.eval()
    with torch.no_grad():
        for i in range(0, len(trows), 32):
            x = torch.from_numpy(corpus.get(trows[i:i + 32])).to(dev)
            preds.append(heads(emb_fn(model, core, x, torch.isfinite(x)))[3].float().cpu().numpy())
    P_ = np.concatenate(preds); T_ = Bz[np.sort(trows)]
    res["biomarker_r2_heldout"] = {}
    for j, nm in enumerate(bio_names):
        ok = np.isfinite(T_[:, j])
        if ok.sum() > 50:
            r = np.corrcoef(P_[ok, j], T_[ok, j])[0, 1]
            res["biomarker_r2_heldout"][nm] = {"n": int(ok.sum()), "pearson_r": float(r),
                                               "r2": float(1 - np.mean((P_[ok, j] - T_[ok, j]) ** 2) / np.var(T_[ok, j]))}
    log("held-out biomarker r: " + ", ".join(f"{k} {v['pearson_r']:.3f}" for k, v in res["biomarker_r2_heldout"].items()))
    res["elapsed_s"] = time.time() - t0
    write_json(out / "results.json", res)
    log(f"AltumAge test {res['altumage_test']}; held-out all {res['heldout_datasets_all']}; unflagged {res['heldout_datasets_unflagged']}")


if __name__ == "__main__":
    main()
