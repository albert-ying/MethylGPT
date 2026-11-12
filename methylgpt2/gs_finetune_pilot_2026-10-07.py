#!/usr/bin/env python3
"""Pilot: end-to-end survival fine-tuning of MethylGPT on the Generation Scotland development participants.

Question: does fine-tuning the encoder jointly on the eight incident-disease outcomes give a C-index
considerably above the elastic-net models on frozen features, raw CpGs and GrimAge2 + DunedinPACE?
Development data only (9,866 training-subset participants); the validation and test subsets are not read.

Folds: one assignment for all endpoints (sha1 of the participant identifier mod 5), because joint training
on eight outcomes with endpoint-specific folds would let a held-out participant's other outcomes reach the
shared encoder. All models in this script are fitted and scored on these common folds.

--part ft    For each pilot fold: start from the released run-1 multitask model; add a linear survival head
             (8 outputs) on the 512-unit trunk and a covariate head initialised from Cox(age, sex); train the
             encoder, trunk and heads with the within-batch Breslow partial likelihood averaged over endpoints,
             plus a penalty keeping the 25 distilled outputs at their original values (weight 0.5); 0-50% of
             CpGs removed at random per profile; early stopping on a 15% inner split of the fitting rows.
--part base  Same folds: GPU elastic-net Cox on the frozen embedding (E), distilled outputs (B), factor +
             residual features (R) and raw CpGs (dense fit), and unpenalized Cox for GrimAge2 + DunedinPACE.
--part eval  Pooled within-fold C-index per endpoint and mean, for each model and equal-weight combinations.
Outputs: artifacts/gs_finetune_pilot_2026-10-07/{private_ft_risks,private_base_risks}.parquet, pilot.json.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

from methylgpt2_lib import ARTIFACTS, cuda_preinit, log, require, write_json
from methylgpt2_model import BioHeads, embed_batch, load_checkpoint
from gpu_coxnet_2026_10_01 import select_and_fit

HERE = Path(__file__).resolve().parent


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, HERE / fn)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


stk = _load("stk", "gs_stacked_2026-10-04.py")
tun = _load("tun", "gs_dev_tuning_2026-10-06.py")
SPLIT, CACHE = stk.SPLIT, stk.CACHE
OUT = ARTIFACTS / "gs_finetune_pilot_2026-10-07"
PREDS = ARTIFACTS / "gs_cox/mgpt2_embed_linear_2026-10-01/private_nested_predictions.parquet"
MULTITASK = ARTIFACTS / "runs/multitask_bio_2026-10-01/final.pt"
SCORES = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/revision_server_20260622/server_results/task_a/task_a_gs_clock_scores_fixed_grimage_20260629.csv")


def common_fold(ids):
    return np.array([int(hashlib.sha1(s.encode()).hexdigest(), 16) % 5 for s in ids])


def load_data():
    mp = require(next(CACHE.glob(f"COMMON_SAMPLE_SPLIT_{SPLIT}_train_all_gs_batches_all_probes_*.matrix.npz")))
    ids = pd.read_parquet(mp.with_name(mp.name.replace(".matrix.npz", ".labels.parquet")), columns=["dnam_id"]).dnam_id.astype(str).to_numpy()
    with np.load(mp, allow_pickle=False) as z:
        X = z["matrix"].astype(np.float32)
    preds = pd.read_parquet(require(PREDS)); preds = preds[preds.age_adjustment == "linear"].copy()
    preds["dnam_id"] = preds.dnam_id.astype(str)
    endpoints = sorted(preds.endpoint.unique())
    ev = np.zeros((len(ids), len(endpoints)), bool); du = np.zeros((len(ids), len(endpoints)))
    cov = None
    for j, ep in enumerate(endpoints):
        pe = preds[preds.endpoint == ep].set_index("dnam_id").loc[ids]
        ev[:, j], du[:, j] = pe.event.astype(bool).to_numpy(), pe.duration.to_numpy(float)
        c = pe[["age", "sex"]].to_numpy(float)
        if cov is None:
            cov = c
        elif not np.allclose(cov, c):
            raise ValueError("age or sex differ between endpoint tables")
    return ids, X, endpoints, ev, du, cov


def cox_batch(eta, t, e):
    """Within-batch Breslow partial likelihood for one endpoint (ties broken by order)."""
    order = torch.argsort(t, descending=True)
    s, d = eta[order], e[order]
    return -((s - torch.logcumsumexp(s, 0)) * d).sum() / d.sum().clamp_min(1.0)


def mean_c(ev, du, R, rows):
    return float(np.mean([concordance_index_censored(ev[rows, j], du[rows, j], R[rows, j])[0] for j in range(ev.shape[1])]))


def part_ft(a):
    dev = a.device
    ids, X, endpoints, ev, du, cov = load_data()
    fold = common_fold(ids)
    _, B1 = tun.npz(tun.FEAT1, "train", "mt_bio")
    Xg = torch.from_numpy(X).to(dev)
    evg, dug = torch.from_numpy(ev.astype(np.float32)).to(dev), torch.from_numpy(du.astype(np.float32)).to(dev)
    B1g = torch.from_numpy(B1).to(dev)
    OUT.mkdir(parents=True, exist_ok=True)
    risks = np.full((len(ids), len(endpoints)), np.nan)
    hist = {}
    prev = OUT / f"private_ft_risks{a.tag}.parquet"
    if prev.exists():  # keep folds finished by an earlier run with the same settings tag
        pv = pd.read_parquet(prev); assert (pv.dnam_id.to_numpy() == ids).all()
        risks = pv[[f"FT|{e}" for e in endpoints]].to_numpy(float)
    for k in [int(v) for v in a.folds.split(",")]:
        t0 = time.time()
        torch.manual_seed(20261007 + k)
        fit = np.flatnonzero(fold != k); held = np.flatnonzero(fold == k)
        r = np.random.default_rng(20261007 + k); perm = r.permutation(fit)
        n_es = int(round(0.15 * len(fit))); es, tr = np.sort(perm[:n_es]), np.sort(perm[n_es:])
        cm, cs = cov[tr].mean(0), cov[tr].std(0)
        covz = torch.from_numpy(((cov - cm) / cs).astype(np.float32)).to(dev)
        mt = torch.load(MULTITASK, map_location=dev, weights_only=False)
        model, core, _ = load_checkpoint(Path(mt["ckpt"]), None, dev)
        model.load_state_dict(mt["model"]); model.train()
        heads = BioHeads(core.rank + model.latents.shape[1], len(mt["tissues"]) + 1, len(mt["bio_names"])).to(dev)
        heads.load_state_dict(mt["heads"]); heads.train()
        surv = torch.nn.Linear(512, len(endpoints)).to(dev); torch.nn.init.zeros_(surv.weight); torch.nn.init.zeros_(surv.bias)
        cvh = torch.nn.Linear(2, len(endpoints), bias=False).to(dev)
        with torch.no_grad():
            for j in range(len(endpoints)):
                cx = CoxPHSurvivalAnalysis(ties="breslow", n_iter=200).fit(((cov[tr] - cm) / cs), Surv.from_arrays(ev[tr, j], du[tr, j]))
                cvh.weight[j] = torch.as_tensor(cx.coef_, dtype=torch.float32, device=dev)
        enc = [p for n, p in model.named_parameters() if not n.startswith(("u1", "u2", "q_emb", "out", "dec", "h_norm", "ds_head", "q_load", "sum_proj", "h_proj")) and p.requires_grad]
        if a.lr_enc == 0:  # frozen encoder: train the trunk and heads only
            for p_ in model.parameters():
                p_.requires_grad_(False)
            enc = []
        opt = torch.optim.AdamW(([{"params": enc, "lr": a.lr_enc, "weight_decay": 0.01}] if enc else []) +
                                [{"params": heads.parameters(), "lr": a.lr_trunk, "weight_decay": 0.01},
                                 {"params": list(surv.parameters()) + list(cvh.parameters()), "lr": a.lr_head, "weight_decay": a.wd_head}], eps=1e-10)
        base = [g["lr"] for g in opt.param_groups]
        steps_per_epoch = math.ceil(len(tr) / a.batch); total = steps_per_epoch * a.epochs
        gen = torch.Generator(device=dev).manual_seed(20261007 + k)

        def predict(rows):
            model.eval(); heads.eval()
            out = []
            with torch.no_grad():
                for i in range(0, len(rows), 128):
                    ii = torch.as_tensor(rows[i:i + 128], device=dev)
                    x = Xg[ii]
                    e = embed_batch(model, core, x, torch.isfinite(x))
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        h = heads.trunk(e)
                    out.append((surv(h.float()) + cvh(covz[ii])).float().cpu().numpy())
            model.train(); heads.train()
            return np.concatenate(out)

        best = (-1.0, None, 0); step = 0; hist[k] = []
        for epoch in range(1, a.epochs + 1):
            order = torch.as_tensor(r.permutation(tr), device=dev)
            run = []
            for b in range(0, len(order), a.batch):
                ii = order[b:b + a.batch]
                if len(ii) < 32:
                    continue
                step += 1
                warm = min(1.0, step / 20); cosv = 0.5 * (1 + math.cos(math.pi * step / total))
                for g, lr in zip(opt.param_groups, base):
                    g["lr"] = lr * warm * (0.1 + 0.9 * cosv)
                x = Xg[ii]
                keep = torch.rand(x.shape, device=dev, generator=gen) >= 0.5 * torch.rand((len(x), 1), device=dev, generator=gen)
                e = embed_batch(model, core, x, torch.isfinite(x) & keep)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    h = heads.trunk(e)
                    pb = heads.bio(h)
                eta = surv(h.float()) + cvh(covz[ii])
                lc = torch.stack([cox_batch(eta[:, j], dug[ii, j], evg[ii, j]) for j in range(len(endpoints))]).mean()
                lb = ((pb.float() - B1g[ii]) ** 2).mean()
                loss = lc + a.w_bio * lb
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(heads.parameters()) + list(surv.parameters()), 1.0)
                opt.step(); run.append((float(lc), float(lb)))
            R_es = np.full((len(ids), len(endpoints)), np.nan); R_es[es] = predict(es)
            c_es = mean_c(ev, du, R_es, es)
            hist[k].append({"epoch": epoch, "cox": float(np.mean([v[0] for v in run])), "bio": float(np.mean([v[1] for v in run])), "inner_c": c_es})
            log(f"fold {k} epoch {epoch}: cox {hist[k][-1]['cox']:.4f} bio {hist[k][-1]['bio']:.4f} inner mean C {c_es:.4f} ({time.time() - t0:.0f}s)")
            if c_es > best[0]:
                best = (c_es, {"m": copy.deepcopy(model.state_dict()), "h": copy.deepcopy(heads.state_dict()),
                               "s": copy.deepcopy(surv.state_dict()), "c": copy.deepcopy(cvh.state_dict())}, epoch)
        model.load_state_dict(best[1]["m"]); heads.load_state_dict(best[1]["h"]); surv.load_state_dict(best[1]["s"]); cvh.load_state_dict(best[1]["c"])
        risks[held] = predict(held)
        log(f"fold {k}: best epoch {best[2]} (inner C {best[0]:.4f}); held-out mean C {mean_c(ev, du, risks, held):.4f}")
        del model, heads, surv, cvh, opt, best
        torch.cuda.empty_cache()
        df = pd.DataFrame(risks, columns=[f"FT|{e}" for e in endpoints]); df.insert(0, "fold", fold); df.insert(0, "dnam_id", ids)
        df.to_parquet(OUT / f"private_ft_risks{a.tag}.parquet", index=False)
        write_json(OUT / f"ft_history{a.tag}.json", {"settings": vars(a), "history": hist})


def part_base(a):
    dev = a.device
    ids, X, endpoints, ev, du, cov = load_data()
    fold = common_fold(ids)
    _, E = tun.npz(tun.FEAT1, "train", "mgpt2_embed"); _, B1 = tun.npz(tun.FEAT1, "train", "mt_bio")
    rid, RES = tun.residuals("train", dev); assert (rid == ids).all()
    feats = {"E": E, "B": B1, "R": np.hstack([E, B1, RES]).astype(np.float32), "raw": X}
    sc = pd.read_csv(require(SCORES)); sc["sample_id"] = sc.sample_id.astype(str)
    clk = sc.pivot_table(index="sample_id", columns="model", values="score").loc[ids, ["GrimAgeV2", "DunedinPACE"]].to_numpy(float)
    OUT.mkdir(parents=True, exist_ok=True)
    out = {f"{m}|{e}": np.full(len(ids), np.nan) for m in list(feats) + ["GD"] for e in endpoints}
    prev = OUT / "private_base_risks.parquet"
    if prev.exists():  # keep folds finished by an earlier run
        pv = pd.read_parquet(prev); assert (pv.dnam_id.to_numpy() == ids).all()
        out = {c: pv[c].to_numpy(float).copy() for c in out}
    for k in [int(v) for v in a.folds.split(",")]:
        fit, held = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        cm, cs = stk.fit_scaler(cov[fit].astype(np.float32), dev)
        for j, ep in enumerate(endpoints):
            for m, F in feats.items():
                fm, fs = stk.fit_scaler(F[fit], dev)
                Xf = np.hstack([stk.apply_scaler(cov[fit].astype(np.float32), cm, cs, dev), stk.apply_scaler(F[fit], fm, fs, dev)])
                w, _, _ = select_and_fit(Xf, du[fit, j], ev[fit, j], seed=20260928 + k, dev=dev, iters=200)
                del Xf
                out[f"{m}|{ep}"][held] = np.hstack([stk.apply_scaler(cov[held].astype(np.float32), cm, cs, dev), stk.apply_scaler(F[held], fm, fs, dev)]) @ w
            M = np.column_stack([cov, clk]); mu, sd = M[fit].mean(0), M[fit].std(0)
            cx = CoxPHSurvivalAnalysis(ties="breslow", n_iter=200).fit((M[fit] - mu) / sd, Surv.from_arrays(ev[fit, j], du[fit, j]))
            out[f"GD|{ep}"][held] = ((M[held] - mu) / sd) @ cx.coef_
            log(f"fold {k} {ep} done")
        df = pd.DataFrame(out); df.insert(0, "fold", fold); df.insert(0, "dnam_id", ids)
        df.to_parquet(OUT / "private_base_risks.parquet", index=False)


def part_eval(a):
    ids, X, endpoints, ev, du, cov = load_data()
    del X
    ft = pd.read_parquet(require(OUT / f"private_ft_risks{a.tag}.parquet")); bs = pd.read_parquet(require(OUT / "private_base_risks.parquet"))
    assert (ft.dnam_id.to_numpy() == ids).all() and (bs.dnam_id.to_numpy() == ids).all()
    fold = ft.fold.to_numpy()
    folds = [k for k in range(5) if np.isfinite(ft.loc[fold == k, f"FT|{endpoints[0]}"]).all() and np.isfinite(bs.loc[fold == k, f"E|{endpoints[0]}"]).all()]

    def z(v, rows):
        return (v[rows] - v[rows].mean()) / v[rows].std(ddof=1)

    combos = {"E+B+R": ("E", "B", "R"), "FT+B+R": ("FT", "B", "R"), "FT+R": ("FT", "R"), "FT+B": ("FT", "B")}
    res = {}
    for j, ep in enumerate(endpoints):
        col = {**{m: bs[f"{m}|{ep}"].to_numpy(float) for m in ("E", "B", "R", "raw", "GD")}, "FT": ft[f"FT|{ep}"].to_numpy(float)}
        models = dict(col)
        for cn, parts in combos.items():
            v = np.full(len(ids), np.nan)
            for k in folds:
                rows = np.flatnonzero(fold == k); v[rows] = sum(z(col[p], rows) for p in parts)
            models[cn] = v
        res[ep] = {}
        for m, v in models.items():
            tot = np.zeros(3)
            for k in folds:
                rows = np.flatnonzero(fold == k)
                _, c, d, t, _ = concordance_index_censored(ev[rows, j], du[rows, j], v[rows]); tot += [c, d, t]
            res[ep][m] = float((tot[0] + 0.5 * tot[2]) / tot.sum())
    means = {m: float(np.mean([res[e][m] for e in endpoints])) for m in res[endpoints[0]]}
    write_json(OUT / f"pilot{a.tag}.json", {"folds": folds, "per_endpoint_c": res, "mean_c": means})
    log(f"folds {folds}")
    for m in sorted(means, key=lambda m: -means[m]):
        log(f"{m:8s} mean C {means[m]:.4f}  " + " ".join(f"{e[:6]} {res[e][m]:.3f}" for e in endpoints))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["ft", "base", "eval"], required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--folds", default="0,1")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr-enc", type=float, default=2e-5)
    ap.add_argument("--lr-trunk", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--wd-head", type=float, default=0.05)
    ap.add_argument("--w-bio", type=float, default=0.5)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    cuda_preinit()
    torch.backends.cuda.matmul.allow_tf32 = True
    {"ft": part_ft, "base": part_base, "eval": part_eval}[a.part](a)


if __name__ == "__main__":
    main()
