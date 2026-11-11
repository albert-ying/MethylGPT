#!/usr/bin/env python3
"""Age with missing CpGs: more comparators and tissues (Supplementary Information).

Profiles: the 4,626 AltumAge test profiles (load_altumage_split("test")), 4,016 from datasets that also gave
training-split profiles ("shared") and 610 from datasets that did not ("unshared"; their corpus split is test, dev or
excluded_qc). Datasets are the corpus `group` column.

Scenarios (masks identical to Fig. 4e: degrade() of evaluate_age_revision_2026-10-01.py, imported). Listed in this
order everywhere; array subsets first, random removal as a sensitivity analysis:
  all          every CpG as measured (AltumAge profiles carry their own missing CpGs; these stay missing)
  hm450, epic_v2, hm27   only the vocabulary CpGs of that array are kept (47,826 / 45,668 / 22,231 of 49,156)
  random_p     p = 10, 30, 50, 70, 90: cells with numpy default_rng(20261001 + p).random((4626, 49156)) < p/100 set
               to NaN (one draw per profile and CpG, CpGs in vocabulary order); random_10 is new here
Each scenario's observed-value pattern is hashed (SHA-256 of the packed isfinite matrix) and compared with the hash
written by missing_cpgs_original_model_2026-10-18.py, which applies the same masks to the originally submitted model.

Models (all scored on identical profiles and masks)
  MethylGPT      the released bundle artifacts/release/methylgpt2-1.1 (checksums verified against
                 package/methylgpt2-1.1.checksums.sha256); age = mean of the three training runs. Existing per-profile
                 predictions (eval/age_ensemble_2026-10-04.predictions.npz) are used for all, hm450, epic_v2, hm27 and
                 random_30-90; random_10 is computed with the bundle (package/src methylgpt2 API), and "all" is
                 recomputed with the bundle as a check against the existing file.
  Elastic net    comparator (1): the mean-filled elastic net of Fig. 4e (eval/baseline_corpus_age_clean_rerun_2026-10-01
                 .enet_model.npz; 41,845 corpus training profiles, Horvath transform, training-mean filling). Existing
                 predictions (eval/age_revision_2026-10-01.predictions.npz) for 8 scenarios, random_10 computed here; all
                 9 recomputed as a check.
  MLP            comparator (2): AltumAge-style multilayer perceptron, new here. Architecture of AltumAge (de Lima
                 Camillo, Lapierre & Singh, npj Aging 8, 4, 2022), as implemented in pyaging 0.1.22
                 (models/_base_models.py, AltumAgeNeuralNetwork): batch normalisation of the inputs, five hidden dense
                 layers of 32 units with SELU activation, each preceded by batch normalisation, and a linear output
                 unit after a final batch normalisation (BatchNorm eps 0.001, Keras momentum 0.99 = PyTorch 0.01).
                 The published AltumAge takes 20,318 CpGs; here the input is the 49,156 vocabulary CpGs. Target:
                 Horvath's log-linear age transform (adult age 20), standardised. Inputs: beta values, missing values
                 filled with the training mean of each CpG, then standardised with training mean and s.d. Training
                 profiles: the 5,461 AltumAge training profiles (load_altumage_split("train")). Optimiser Adam, MSE
                 loss, batch 256, L1 and L2 penalties (equal weight lambda) on the dense-layer weights as in a Keras
                 kernel regulariser. Settings chosen on the 1,366 AltumAge validation profiles by validation MedAE:
                 grid learning rate {3e-5, 1e-4, 3e-4, 1e-3, 3e-3} x lambda {0, 1e-5, 1e-4} with seed 1, up to 300 epochs, epoch
                 with the lowest validation MedAE kept (patience 30 epochs); the chosen setting is then trained with
                 seeds 1, 2 and 3. MLP age = mean of the three seeds' predictions (as MethylGPT age is the mean of
                 three runs); single seeds are reported as secondary results. At test time, missing CpGs are filled
                 with the training means.
  MLP, corpus labels   secondary: the same MLP recipe trained on the 41,845 corpus training-split profiles and ages
                 used by the elastic net and by MethylGPT's age head (selection on the same 1,366 validation profiles,
                 up to 60 epochs, patience 10). Reported because the 5,461-profile MLP has fewer age labels than both.
  MethylGPT applications (not comparators): the elastic net, and the 5,461-profile MLP, applied after MethylGPT
                 imputation: missing values replaced by the input-layer reconstruction mu + V s (posterior latent
                 scores from the observed CpGs, ppca_core_meanfilled_2026-09-30, rank 1,536, clipped to [0, 1]);
                 observed values kept (as evaluate_age_enet_factor_imputed_2026-10-02.py; its JSON is the check).

Changes after the first run (18 Oct, 06:47), before any figure: (i) the 5,461-profile MLP had chosen the lowest
learning rate of the grid {1e-4, 3e-4, 1e-3} and the corpus-label MLP the highest, so the grid was widened to 3e-5 and
3e-3 for both; (ii) 136 CpGs are never observed in the 5,461 AltumAge training profiles (nor in the test or validation
profiles); the MLP saw a constant input there, and MethylGPT-imputed values at these CpGs made the MLP-after-imputation
predictions diverge (MedAE > 1,800 years), so the MLP now sets CpGs never observed in its training profiles to the
training mean at every use. Mean-filled MLP inputs are unchanged by (ii). The trained MLPs are saved this time
(artifacts/runs/mlp_altumage_style_2026-10-18/weights.pt).

Pre-specified primary comparison (written before running): MethylGPT minus MLP (5,461 profiles, mean of three seeds)
MedAE on the AltumAge test set with 90% of CpGs removed at random and with only 27K CpGs; 95% interval from 2,000
draws resampling datasets with replacement (paired, paired_ci of evaluate_age_revision_2026-10-01.py, seed 20261002).
Tissue analysis, pre-specified contrast: per tissue (>= 50 profiles; Other and unlabelled dropped), MethylGPT minus
mean-filled elastic net MedAE, with complete profiles and with 27K CpGs, 95% interval resampling datasets within the
tissue (2,000 draws); profile and dataset counts; tissues with fewer than 5 datasets flagged; the same contrast in the
shared and unshared subsets of each tissue (interval only with >= 2 datasets). All other cells are descriptive.
Secondary: MedAE, MAE and Pearson r of every model in every scenario with dataset-resampled MedAE intervals
(cluster_ci, seed 20261001); MethylGPT minus elastic net and minus each MLP in every scenario; single MLP seeds; the
MethylGPT applications; MethylGPT minus MLP per tissue; the originally submitted model (separate block, only if its
complete-profile run reproduced MedAE 2.2097). Added after the first analysis (descriptive): all tissues split into shared
and unshared datasets, MethylGPT minus each comparator; and the count of test profiles whose dataset also gave AltumAge
training profiles (all 4,626: the 5,461-profile MLP saw profiles from every test dataset, including the 7 unshared ones).

Run:
  CUDA_VISIBLE_DEVICES=1 python3 missing_cpgs_comparators_2026-10-18.py --stage predict   (GPU)
  python3 missing_cpgs_comparators_2026-10-18.py --stage analyze                          (CPU; after the
      originally submitted model run, if it is to be included)
Outputs
  artifacts/eval/missing_cpgs_comparators_2026-10-18.predictions.npz   per-profile predictions, every model x scenario
  artifacts/eval/missing_cpgs_comparators_2026-10-18_training.json    MLP grids, selection, checks
  artifacts/eval/missing_cpgs_comparators_2026-10-18.json             all results
  artifacts/eval/missing_cpgs_comparators_2026-10-18_table.csv        Supplementary Table (long format)
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from methylgpt2_lib import ARTIFACTS, CORPUS_DIR, Corpus, F, Finv, PPCACore, age_scores, load_altumage_split, log, require, write_json

HERE = Path(__file__).resolve().parent
EVAL = ARTIFACTS / "eval"
BUNDLE = ARTIFACTS / "release/methylgpt2-1.1"
CHECKSUMS = HERE / "package/methylgpt2-1.1.checksums.sha256"
OUT = EVAL / "missing_cpgs_comparators_2026-10-18"
ORIG = EVAL / "missing_cpgs_original_model_2026-10-18"
WEIGHTS = ARTIFACTS / "runs/mlp_altumage_style_2026-10-18/weights.pt"
SCEN = ["all", "hm450", "epic_v2", "hm27", "random_10", "random_30", "random_50", "random_70", "random_90"]
EXISTING = ["all", "random_30", "random_50", "random_70", "random_90", "hm27", "hm450", "epic_v2"]
LABEL = {"all": "All CpGs", "hm450": "450K CpGs", "epic_v2": "EPIC v2 CpGs", "hm27": "27K CpGs",
         **{f"random_{p}": f"{p}% removed at random" for p in (10, 30, 50, 70, 90)}}
KIND = {s: ("complete profiles" if s == "all" else "array subset" if not s.startswith("random") else "random removal (sensitivity)")
        for s in SCEN}
EN_OLD = "ElasticNet (same labels)"
GRID_LR, GRID_PEN = (3e-5, 1e-4, 3e-4, 1e-3, 3e-3), (0.0, 1e-5, 1e-4)
SEEDS = (1, 2, 3)
PRIMARY = ("random_90", "hm27")

spec = importlib.util.spec_from_file_location("age_revision", HERE / "evaluate_age_revision_2026-10-01.py")
ar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ar)          # degrade, cluster_ci, paired_ci


def pattern_hash(X):
    return hashlib.sha256(np.packbits(np.isfinite(X)).tobytes()).hexdigest()


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------------
# AltumAge-style MLP
# ---------------------------------------------------------------------------------------------------------------------
def make_mlp(n_in, width=32, depth=5):
    import torch.nn as nn
    import torch.nn.functional as Fn

    class AltumAgeMLP(nn.Module):
        """AltumAge network (pyaging 0.1.22 AltumAgeNeuralNetwork) with n_in inputs."""

        def __init__(self):
            super().__init__()
            dims = [n_in] + [width] * depth
            self.bns = nn.ModuleList([nn.BatchNorm1d(d, eps=1e-3, momentum=0.01) for d in dims])
            self.lins = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(depth)] + [nn.Linear(width, 1)])

        def forward(self, x):
            for i in range(depth):
                x = Fn.selu(self.lins[i](self.bns[i](x)))
            return self.lins[-1](self.bns[-1](x)).squeeze(-1)

    return AltumAgeMLP()


class Standardiser:
    """Training-mean filling and standardisation, on the GPU. CpGs never observed in the training profiles carry no
    information for the MLP (their training input is constant); they are set to 0 (the training mean) at every use."""

    def __init__(self, mu, sd, used, dev):
        import torch
        self.mu, self.sd, self.dev = torch.as_tensor(mu, device=dev), torch.as_tensor(sd, device=dev), dev
        self.used = torch.as_tensor(used, device=dev)

    def __call__(self, X):
        import torch
        T = torch.as_tensor(X, device=self.dev, dtype=torch.float32)
        T = torch.where(torch.isfinite(T), T, self.mu)
        return torch.where(self.used, (T - self.mu) / self.sd, torch.zeros((), device=self.dev))


def fit_stats(X_chunks, P, dev):
    """Per-CpG mean over observed values, then s.d. after mean filling (chunks of float32 arrays)."""
    import torch
    s, n = torch.zeros(P, dtype=torch.float64, device=dev), torch.zeros(P, dtype=torch.float64, device=dev)
    for X in X_chunks():
        T = torch.as_tensor(X, device=dev)
        f = torch.isfinite(T)
        s += torch.where(f, T, torch.zeros((), device=dev)).sum(0).double()
        n += f.sum(0).double()
    mu = (s / n.clamp_min(1)).float()
    ss = torch.zeros(P, dtype=torch.float64, device=dev)
    N = 0
    for X in X_chunks():
        T = torch.as_tensor(X, device=dev)
        T = torch.where(torch.isfinite(T), T, mu)
        ss += ((T - mu) ** 2).sum(0).double()
        N += T.shape[0]
    sd = torch.sqrt(ss / (N - 1)).float().clamp_min(1e-4)
    return mu, sd, n > 0


def predict_mlp(net, Z, batch=1024):
    import torch
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, Z.shape[0], batch):
            out.append(net(Z[i:i + batch].float()).float().cpu().numpy())
    return np.concatenate(out)


def train_mlp(Ztr, t, Zva, yva, tm, ts, lr, pen, seed, max_epochs, patience, dev, batch=256):
    """One AltumAge-style MLP; returns the state at the epoch with the lowest validation MedAE and its history."""
    import torch
    torch.manual_seed(seed)
    gen = torch.Generator(device=dev).manual_seed(seed)
    net = make_mlp(Ztr.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    weights = [m.weight for m in net.lins]
    best, hist, n = None, [], Ztr.shape[0]
    for ep in range(max_epochs):
        net.train()
        perm = torch.randperm(n, device=dev, generator=gen)
        tot = 0.0
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            if len(b) < 2:
                continue
            loss = ((net(Ztr[b].float()) - t[b]) ** 2).mean()
            if pen > 0:
                loss = loss + pen * sum(w.abs().sum() + (w * w).sum() for w in weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        pv = Finv(predict_mlp(net, Zva) * ts + tm)
        m = float(np.median(np.abs(pv - yva)))
        hist.append({"epoch": ep, "train_loss": tot / n, "valid_medae": m})
        if best is None or m < best[0]:
            best = (m, ep, copy.deepcopy(net.state_dict()))
        elif ep - best[1] >= patience:
            break
    net.load_state_dict(best[2])
    net.eval()
    return net, {"lr": lr, "penalty": pen, "seed": seed, "best_epoch": best[1], "best_valid_medae": best[0],
                 "epochs_run": len(hist), "history": hist}


def mlp_family(name, Ztr, ytr, Zva, yva, max_epochs, patience, dev):
    """Grid with seed 1, then three seeds with the chosen setting."""
    import torch
    f = F(ytr)
    tm, ts = float(f.mean()), float(f.std())
    t = torch.as_tensor(((f - tm) / ts).astype(np.float32), device=dev)
    grid, nets = [], {}
    for lr in GRID_LR:
        for pen in GRID_PEN:
            t0 = time.time()
            net, h = train_mlp(Ztr, t, Zva, yva, tm, ts, lr, pen, SEEDS[0], max_epochs, patience, dev)
            h["seconds"] = round(time.time() - t0, 1)
            grid.append(h)
            nets[(lr, pen)] = net
            log(f"{name} grid lr {lr:g} lambda {pen:g}: best valid MedAE {h['best_valid_medae']:.3f} at epoch {h['best_epoch']} "
                f"({h['epochs_run']} epochs, {h['seconds']:.0f} s)")
    best = min(grid, key=lambda g: g["best_valid_medae"])
    lr, pen = best["lr"], best["penalty"]
    chosen = {SEEDS[0]: nets[(lr, pen)]}
    seeds = [best]
    for s in SEEDS[1:]:
        net, h = train_mlp(Ztr, t, Zva, yva, tm, ts, lr, pen, s, max_epochs, patience, dev)
        chosen[s] = net
        seeds.append(h)
        log(f"{name} seed {s}: best valid MedAE {h['best_valid_medae']:.3f} at epoch {h['best_epoch']}")
    del nets
    edge = {"lr_at_grid_edge": lr in (GRID_LR[0], GRID_LR[-1]), "penalty_at_grid_edge": pen in (GRID_PEN[0], GRID_PEN[-1])}
    info = {"n_train": int(len(ytr)), "n_valid": int(len(yva)), "target_center": tm, "target_scale": ts,
            "chosen": {"lr": lr, "penalty": pen, **edge}, "max_epochs": max_epochs, "patience": patience,
            "grid_seed1": [{k: v for k, v in g.items() if k != "history"} for g in grid],
            "seeds": [{k: v for k, v in g.items() if k != "history"} for g in seeds],
            "seed_histories": {str(g["seed"]): g["history"] for g in seeds}}
    return chosen, (tm, ts), info


# ---------------------------------------------------------------------------------------------------------------------
# stage 1: predictions
# ---------------------------------------------------------------------------------------------------------------------
def stage_predict(a):
    import torch
    from methylgpt2_lib import cuda_preinit
    cuda_preinit()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device(a.device)
    t0 = time.time()
    info = {"checks": {}}

    # ---- bundle checksums ------------------------------------------------------------------------------------------
    want = dict(reversed(l.split()) for l in require(CHECKSUMS).read_text().splitlines() if l.strip())
    got = {f: sha256_file(BUNDLE / f) for f in want}
    bad = {f for f in want if want[f] != got[f]}
    if bad:
        sys.exit(f"release bundle checksum mismatch: {sorted(bad)}")
    info["checks"]["bundle_checksums_verified"] = sorted(want)
    log("release bundle checksums verified")

    ids, Xte, yte = load_altumage_split("test")
    _, Xtr, ytr = load_altumage_split("train")
    _, Xva, yva = load_altumage_split("valid")
    P = Xte.shape[1]
    log(f"AltumAge train {Xtr.shape}, valid {Xva.shape}, test {Xte.shape}")
    z_ens = np.load(require(EVAL / "age_ensemble_2026-10-04.predictions.npz"))
    z_rev = np.load(require(EVAL / "age_revision_2026-10-01.predictions.npz"))
    assert (z_ens["altumage_ids"].astype(str) == ids).all() and np.array_equal(z_ens["altumage_age"], yte)
    assert (z_rev["altumage_ids"].astype(str) == ids).all()
    prof = pd.read_parquet(CORPUS_DIR / "profiles.parquet", columns=["id", "group", "split", "tissue"])
    pte = prof.set_index("id").reindex(ids)
    grp = pte.group.astype(str).to_numpy()
    train_groups = set(prof.loc[prof.split == "train", "group"].astype(str))
    shared = np.isin(grp, list(train_groups))
    tissue = z_ens["altumage_tissue"].astype(str)
    log(f"shared {shared.sum()}, unshared {(~shared).sum()}, datasets {len(set(grp))}")

    P_ = {}     # predictions: key "model|scenario"
    # ---- MLP on the 5,461 AltumAge training profiles ---------------------------------------------------------------
    mu_a, sd_a, used_a = fit_stats(lambda: [Xtr], P, dev)
    std_a = Standardiser(mu_a, sd_a, used_a, dev)
    info["checks"]["mlp_altumage_train_cpgs_never_observed"] = int((~used_a).sum())
    Ztr, Zva = std_a(Xtr).half(), std_a(Xva).half()
    mlp_a, (tm_a, ts_a), info["mlp_altumage_train"] = mlp_family("MLP (5,461)", Ztr, ytr, Zva, yva, 300, 30, dev)
    del Ztr
    torch.cuda.empty_cache()

    # ---- MLP on the 41,845 corpus training profiles (same labels as the elastic net and MethylGPT) ------------------
    corpus = Corpus()
    lab = pd.read_parquet(CORPUS_DIR / "profiles_age_clean_2026-10-01.parquet")
    assert (lab.id.to_numpy() == corpus.profiles.id.to_numpy()).all()
    age_clean = lab.age_clean.to_numpy(float)
    rows = np.flatnonzero((corpus.profiles.split == "train").to_numpy() & np.isfinite(age_clean))
    log(f"corpus training profiles with age: {len(rows)}")
    chunks = lambda: (corpus.get(rows[i:i + 4096]) for i in range(0, len(rows), 4096))
    mu_c, sd_c, used_c = fit_stats(chunks, P, dev)
    std_c = Standardiser(mu_c, sd_c, used_c, dev)
    info["checks"]["mlp_corpus_labels_cpgs_never_observed"] = int((~used_c).sum())
    Ztr = torch.empty((len(rows), P), dtype=torch.float16, device=dev)
    for k, X in enumerate(chunks()):
        Ztr[k * 4096:k * 4096 + len(X)] = std_c(X).half()
    Zva_c = std_c(Xva).half()
    mlp_c, (tm_c, ts_c), info["mlp_corpus_labels"] = mlp_family("MLP (41,845)", Ztr, age_clean[rows], Zva_c, yva, 60, 10, dev)
    del Ztr, Zva_c, Zva, corpus
    torch.cuda.empty_cache()
    write_json(Path(f"{OUT}_training.json"), info)
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"architecture": "AltumAge (pyaging 0.1.22 AltumAgeNeuralNetwork), 49,156 inputs",
                "altumage_train": {"seeds": {k: v.state_dict() for k, v in mlp_a.items()}, "mu": mu_a.cpu(), "sd": sd_a.cpu(),
                                   "used": used_a.cpu(), "target_center": tm_a, "target_scale": ts_a},
                "corpus_labels": {"seeds": {k: v.state_dict() for k, v in mlp_c.items()}, "mu": mu_c.cpu(), "sd": sd_c.cpu(),
                                  "used": used_c.cpu(), "target_center": tm_c, "target_scale": ts_c}}, WEIGHTS)

    # ---- MethylGPT, released bundle: random_10 and a check of "all" -------------------------------------------------
    sys.path.insert(0, str(HERE / "package/src"))
    from methylgpt2 import MethylGPT
    model = MethylGPT.from_pretrained(str(BUNDLE), device=str(dev))
    assert model.vocab == list(np.load(ARTIFACTS / "platform_vocab_masks_2026-10-01.npz")["vocab"].astype(str))
    for s in ("all", "random_10"):
        X = ar.degrade(Xte, s)
        runs = model.predict_age_runs(pd.DataFrame(X, columns=model.vocab), batch_size=32)
        for r in runs.columns:
            P_[f"MethylGPT {r}|{s}" if s == "random_10" else f"bundle check {r}|{s}"] = runs[r].to_numpy(float)
        P_[f"MethylGPT|{s}" if s == "random_10" else f"bundle check|{s}"] = runs.mean(1).to_numpy(float)
        log(f"bundle {s}: MedAE {age_scores(runs.mean(1).to_numpy(float), yte)['medae']:.3f}")
    d = float(np.max(np.abs(P_["bundle check|all"] - z_ens["ensemble|all"])))
    info["checks"]["bundle_all_vs_existing_ensemble_max_abs_diff_years"] = d
    log(f"bundle vs existing ensemble predictions (all): max abs diff {d:.2e} years")
    del model
    torch.cuda.empty_cache()
    for s in EXISTING:
        P_[f"MethylGPT|{s}"] = z_ens[f"ensemble|{s}"].astype(float)

    # ---- elastic net, MLPs, MethylGPT applications, per scenario --------------------------------------------------
    zen = np.load(require(EVAL / "baseline_corpus_age_clean_rerun_2026-10-01.enet_model.npz"))

    def enet(T):
        return Finv(((T - zen["mu"]) / zen["sd"]) @ zen["w"] * float(zen["ys"]) + float(zen["ym"]))

    core = PPCACore.load(ARTIFACTS / "ppca_core_meanfilled_2026-09-30.npz", rank=1536).to(dev)
    info["checks"]["observed_pattern_sha256"], info["checks"]["observed_fraction"] = {}, {}
    info["checks"]["elastic_net_vs_existing_max_abs_diff"], info["checks"]["enet_imputed_medae_vs_2026-10-02_json"] = {}, {}
    fj = json.loads(require(EVAL / "age_enet_factor_imputed_2026-10-02.json").read_text())["scenarios"]
    for s in SCEN:
        X = ar.degrade(Xte, s)
        info["checks"]["observed_pattern_sha256"][s] = pattern_hash(X)
        info["checks"]["observed_fraction"][s] = float(np.isfinite(X).mean())
        P_[f"Elastic net|{s}"] = enet(np.where(np.isfinite(X), X, zen["mu"]))
        if s in EXISTING:
            info["checks"]["elastic_net_vs_existing_max_abs_diff"][s] = float(np.max(np.abs(P_[f"Elastic net|{s}"] - z_rev[f"{EN_OLD}|{s}"])))
        Za, Zc = std_a(X), std_c(X)
        for k in SEEDS:
            P_[f"MLP seed {k}|{s}"] = Finv(predict_mlp(mlp_a[k], Za) * ts_a + tm_a)
            P_[f"MLP corpus labels seed {k}|{s}"] = Finv(predict_mlp(mlp_c[k], Zc) * ts_c + tm_c)
        P_[f"MLP|{s}"] = np.mean([P_[f"MLP seed {k}|{s}"] for k in SEEDS], axis=0)
        P_[f"MLP corpus labels|{s}"] = np.mean([P_[f"MLP corpus labels seed {k}|{s}"] for k in SEEDS], axis=0)
        del Za, Zc
        filled = np.empty_like(X)
        with torch.no_grad():
            for i in range(0, len(X), 64):
                x = torch.from_numpy(X[i:i + 64]).to(dev)
                obs = torch.isfinite(x)
                rec = core.predict(core.posterior(x, obs.float())).clamp(0, 1)
                filled[i:i + 64] = torch.where(obs, x, rec).cpu().numpy()
        P_[f"Elastic net, MethylGPT imputation|{s}"] = enet(filled)
        Zf = std_a(filled)
        P_[f"MLP, MethylGPT imputation|{s}"] = np.mean([Finv(predict_mlp(mlp_a[k], Zf) * ts_a + tm_a) for k in SEEDS], axis=0)
        del Zf, filled
        torch.cuda.empty_cache()
        if s in fj:
            info["checks"]["enet_imputed_medae_vs_2026-10-02_json"][s] = {
                "here": age_scores(P_[f"Elastic net, MethylGPT imputation|{s}"], yte)["medae"], "json": fj[s]["enet_factor_filled"]["medae"]}
        log(f"{s}: observed {info['checks']['observed_fraction'][s]:.3f}; MedAE MethylGPT {age_scores(P_[f'MethylGPT|{s}'], yte)['medae']:.2f}, "
            f"elastic net {age_scores(P_[f'Elastic net|{s}'], yte)['medae']:.2f}, MLP {age_scores(P_[f'MLP|{s}'], yte)['medae']:.2f}, "
            f"MLP corpus labels {age_scores(P_[f'MLP corpus labels|{s}'], yte)['medae']:.2f}, "
            f"elastic net + imputation {age_scores(P_[f'Elastic net, MethylGPT imputation|{s}'], yte)['medae']:.2f}")
    info["elapsed_s"] = time.time() - t0
    write_json(Path(f"{OUT}_training.json"), info)
    np.savez_compressed(f"{OUT}.predictions.npz", altumage_ids=ids, altumage_age=yte, altumage_tissue=tissue, group=np.asarray(grp, str),
                        shared_training_dataset=shared, **P_)
    log(f"predictions written ({time.time() - t0:.0f} s)")


# ---------------------------------------------------------------------------------------------------------------------
# stage 2: analysis
# ---------------------------------------------------------------------------------------------------------------------
def medae(p, y):
    return float(np.median(np.abs(np.asarray(p) - np.asarray(y))))


def stage_analyze(a):
    t0 = time.time()
    z = np.load(require(Path(f"{OUT}.predictions.npz")), allow_pickle=True)   # own file; group was saved as an object array
    info = json.loads(require(Path(f"{OUT}_training.json")).read_text())
    y, grp, tis, shared = z["altumage_age"], z["group"].astype(str), z["altumage_tissue"].astype(str), z["shared_training_dataset"]
    models = ["MethylGPT", "Elastic net", "MLP", "MLP corpus labels", "Elastic net, MethylGPT imputation", "MLP, MethylGPT imputation"]
    role = {"MethylGPT": "MethylGPT", "Elastic net": "comparator", "MLP": "comparator",
            "MLP corpus labels": "comparator (secondary: corpus age labels)",
            "Elastic net, MethylGPT imputation": "MethylGPT application", "MLP, MethylGPT imputation": "MethylGPT application"}
    res = {"n_profiles": int(len(y)), "n_datasets": int(len(set(grp))), "n_shared": int(shared.sum()), "n_unshared": int((~shared).sum()),
           "n_datasets_shared": int(len(set(grp[shared]))), "n_datasets_unshared": int(len(set(grp[~shared]))),
           "scenario_order": SCEN, "scenario_labels": LABEL, "scenario_kind": KIND, "observed_fraction": info["checks"]["observed_fraction"],
           "model_roles": role, "checks": info["checks"], "mlp_training": {k: {kk: vv for kk, vv in v.items() if kk != "seed_histories"}
                                                                            for k, v in info.items() if k.startswith("mlp")}}
    rows = []

    def row(**kw):
        rows.append(kw)

    # ---- overall per model and scenario --------------------------------------------------------------------------
    ov = {}
    for s in SCEN:
        ov[s] = {}
        for m in models + [f"MLP seed {k}" for k in SEEDS] + [f"MLP corpus labels seed {k}" for k in SEEDS] + \
                ([f"MethylGPT run{k}" for k in SEEDS] if s == "random_10" else []):
            p = z[f"{m}|{s}"]
            sc = {**age_scores(p, y), "medae_ci95_dataset_bootstrap": ar.cluster_ci(p, y, grp)}
            ov[s][m] = sc
            row(part="overall", scenario=s, scenario_label=LABEL[s], scenario_kind=KIND[s], tissue="", subset="all profiles",
                quantity="MedAE (years)", model_or_contrast=m,
                role="comparator, single seed" if " seed " in m else "MethylGPT, single run" if " run" in m else role[m],
                n_profiles=sc["n"], n_datasets=len(set(grp)), estimate=sc["medae"], ci95_lo=sc["medae_ci95_dataset_bootstrap"][0],
                ci95_hi=sc["medae_ci95_dataset_bootstrap"][1], mae=sc["mae"], pearson_r=sc["pearson_r"])
    res["overall"] = ov

    # ---- paired differences --------------------------------------------------------------------------------------
    pd_ = {}
    for s in SCEN:
        pd_[s] = {}
        for c in ("MLP", "Elastic net", "MLP corpus labels"):
            r = ar.paired_ci(z[f"MethylGPT|{s}"], z[f"{c}|{s}"], y, grp)
            pd_[s][f"MethylGPT minus {c}"] = r
            row(part="primary" if (c == "MLP" and s in PRIMARY) else "overall difference", scenario=s, scenario_label=LABEL[s],
                scenario_kind=KIND[s], tissue="", subset="all profiles", quantity="MedAE difference (years)",
                model_or_contrast=f"MethylGPT minus {c}", role="pre-specified primary" if (c == "MLP" and s in PRIMARY) else "secondary",
                n_profiles=len(y), n_datasets=len(set(grp)), estimate=r["difference"], ci95_lo=r["ci95"][0], ci95_hi=r["ci95"][1])
        r = ar.paired_ci(z[f"Elastic net, MethylGPT imputation|{s}"], z[f"Elastic net|{s}"], y, grp)
        pd_[s]["Elastic net with MethylGPT imputation minus mean-filled elastic net"] = r
        row(part="overall difference", scenario=s, scenario_label=LABEL[s], scenario_kind=KIND[s], tissue="", subset="all profiles",
            quantity="MedAE difference (years)", model_or_contrast="Elastic net with MethylGPT imputation minus mean-filled elastic net",
            role="secondary (MethylGPT application)", n_profiles=len(y), n_datasets=len(set(grp)), estimate=r["difference"],
            ci95_lo=r["ci95"][0], ci95_hi=r["ci95"][1])
    res["paired_differences"] = pd_

    # ---- all tissues, split by whether the dataset also gave training profiles (descriptive) -----------------------
    sh_ = {}
    for s in SCEN:
        sh_[s] = {}
        for sub, msub in (("shared training dataset", shared), ("unshared training dataset", ~shared)):
            c = {"n_profiles": int(msub.sum()), "n_datasets": int(len(set(grp[msub]))),
                 **{f"medae_{m}": medae(z[f"{m}|{s}"][msub], y[msub]) for m in ("MethylGPT", "Elastic net", "MLP", "MLP corpus labels")}}
            for comp in ("Elastic net", "MLP", "MLP corpus labels"):
                r = ar.paired_ci(z[f"MethylGPT|{s}"][msub], z[f"{comp}|{s}"][msub], y[msub], grp[msub])
                c[f"MethylGPT minus {comp}"] = r
                row(part="overall by training-dataset sharing", scenario=s, scenario_label=LABEL[s], scenario_kind=KIND[s], tissue="all tissues",
                    subset=sub, quantity="MedAE difference (years)", model_or_contrast=f"MethylGPT minus {comp}", role="descriptive",
                    n_profiles=c["n_profiles"], n_datasets=c["n_datasets"], estimate=r["difference"], ci95_lo=r["ci95"][0], ci95_hi=r["ci95"][1],
                    flag_fewer_than_5_datasets=bool(c["n_datasets"] < 5), medae_methylgpt=c["medae_MethylGPT"],
                    medae_elastic_net=c["medae_Elastic net"], medae_mlp=c["medae_MLP"])
            sh_[s][sub] = c
    res["by_training_dataset_sharing"] = sh_
    import pyarrow.parquet as pq
    from methylgpt2_lib import ALTUMAGE_DIR
    tr_ids = pq.read_table(require(ALTUMAGE_DIR / "altumage_train.parquet"), columns=["id"]).column("id").to_pylist()
    prof = pd.read_parquet(CORPUS_DIR / "profiles.parquet", columns=["id", "group"]).set_index("id")
    tr_groups = set(prof.group.astype(str).reindex(tr_ids).dropna())
    inalt = np.isin(grp, list(tr_groups))
    res["altumage_training_dataset_overlap"] = {
        "note": "shared/unshared refers to the corpus training split (MethylGPT, elastic net, corpus-label MLP); the 5,461-profile "
                "MLP was trained on AltumAge training profiles, which come from the same datasets as the test profiles",
        "altumage_training_datasets": len(tr_groups), "test_profiles_with_altumage_training_profiles_in_dataset": int(inalt.sum()),
        "of_shared": int((inalt & shared).sum()), "of_unshared": int((inalt & ~shared).sum())}
    res["primary"] = {s: {"comparison": "MethylGPT minus MLP (5,461 AltumAge training profiles, mean of three seeds), MedAE",
                          "interval": "95%, 2,000 draws resampling datasets (paired), seed 20261002",
                          **pd_[s]["MethylGPT minus MLP"], "methylgpt_medae": ov[s]["MethylGPT"]["medae"], "mlp_medae": ov[s]["MLP"]["medae"],
                          "n_profiles": int(len(y)), "n_datasets": int(len(set(grp)))} for s in PRIMARY}

    # ---- tissues -------------------------------------------------------------------------------------------------
    keep = [t for t in sorted(set(tis)) if t not in ("Other", "unlabelled") and (tis == t).sum() >= 50]
    tt = {}
    for t in keep:
        mt = tis == t
        tt[t] = {"n_profiles": int(mt.sum()), "n_datasets": int(len(set(grp[mt]))), "fewer_than_5_datasets": bool(len(set(grp[mt])) < 5),
                 "scenarios": {}}
        for s in SCEN:
            cell = {}
            for sub, msub in (("all profiles", mt), ("shared training dataset", mt & shared), ("unshared training dataset", mt & ~shared)):
                n, nd = int(msub.sum()), int(len(set(grp[msub])))
                if n == 0:
                    cell[sub] = {"n_profiles": 0, "n_datasets": 0}
                    continue
                c = {"n_profiles": n, "n_datasets": nd, **{f"medae_{m}": medae(z[f"{m}|{s}"][msub], y[msub]) for m in ("MethylGPT", "Elastic net", "MLP")}}
                for comp in ("Elastic net", "MLP"):
                    d_ = medae(z[f"MethylGPT|{s}"][msub], y[msub]) - medae(z[f"{comp}|{s}"][msub], y[msub])
                    ci = ar.paired_ci(z[f"MethylGPT|{s}"][msub], z[f"{comp}|{s}"][msub], y[msub], grp[msub])["ci95"] if nd >= 2 else [None, None]
                    c[f"MethylGPT minus {comp}"] = {"difference": d_, "ci95": ci}
                    prim = comp == "Elastic net" and s in ("all", "hm27")
                    row(part="tissue", scenario=s, scenario_label=LABEL[s], scenario_kind=KIND[s], tissue=t, subset=sub,
                        quantity="MedAE difference (years)", model_or_contrast=f"MethylGPT minus {comp}",
                        role=("pre-specified tissue contrast" if prim and sub == "all profiles" else "descriptive"),
                        n_profiles=n, n_datasets=nd, estimate=d_, ci95_lo=ci[0], ci95_hi=ci[1],
                        flag_fewer_than_5_datasets=bool(nd < 5), medae_methylgpt=c["medae_MethylGPT"], medae_elastic_net=c["medae_Elastic net"],
                        medae_mlp=c["medae_MLP"])
                cell[sub] = c
            tt[t]["scenarios"][s] = cell
    res["tissues"] = tt
    res["tissue_rule"] = "tissues with >= 50 AltumAge test profiles, Other and unlabelled dropped; intervals resample datasets within the tissue (2,000 draws, seed 20261002); none with a single dataset"

    # ---- originally submitted model (separate block) --------------------------------------------------------------
    oj, on = Path(f"{ORIG}.json"), Path(f"{ORIG}.predictions.npz")
    if oj.exists() and on.exists():
        o, zo = json.loads(oj.read_text()), np.load(on, allow_pickle=True)   # own file; ids were saved as an object array
        assert (zo["altumage_ids"].astype(str) == z["altumage_ids"].astype(str)).all()
        ok = o.get("reproduction_check", {}).get("passed", False)
        mism = {s: (o["observed_pattern_sha256"].get(s) == info["checks"]["observed_pattern_sha256"][s]) for s in o["observed_pattern_sha256"]}
        blk = {"title": "originally submitted model", "limits": o["limits"], "reproduction_check": o.get("reproduction_check"),
               "masks_identical": mism, "missing_value_handling": o["missing_value_handling"], "scenarios": {}}
        if ok and all(mism.values()):
            for s in SCEN:
                if f"original|{s}" not in zo.files:
                    continue
                p = zo[f"original|{s}"]
                sc = {**age_scores(p, y), "medae_ci95_dataset_bootstrap": ar.cluster_ci(p, y, grp)}
                blk["scenarios"][s] = sc
                row(part="originally submitted model", scenario=s, scenario_label=LABEL[s], scenario_kind=KIND[s], tissue="", subset="all profiles",
                    quantity="MedAE (years)", model_or_contrast="originally submitted model", role="originally submitted model (single run; all "
                    "AltumAge profiles in its pretraining corpus; no surviving split manifest)", n_profiles=sc["n"], n_datasets=len(set(grp)),
                    estimate=sc["medae"], ci95_lo=sc["medae_ci95_dataset_bootstrap"][0], ci95_hi=sc["medae_ci95_dataset_bootstrap"][1],
                    mae=sc["mae"], pearson_r=sc["pearson_r"])
        else:
            blk["excluded_reason"] = "reproduction check failed or masks differ"
        res["originally_submitted_model"] = blk
    else:
        res["originally_submitted_model"] = {"status": f"not run: {oj} not found"}

    order = {s: i for i, s in enumerate(SCEN)}
    tab = pd.DataFrame(rows)
    tab["_o"] = tab.scenario.map(order)
    part_o = {"primary": 0, "overall": 1, "overall difference": 2, "overall by training-dataset sharing": 3, "tissue": 4,
              "originally submitted model": 5}
    tab["_p"] = tab.part.map(part_o)
    tab = tab.sort_values(["_p", "_o", "tissue", "subset", "model_or_contrast"], kind="stable").drop(columns=["_o", "_p"])
    tab.to_csv(f"{OUT}_table.csv", index=False, float_format="%.4f")
    res["elapsed_s"] = time.time() - t0
    write_json(Path(f"{OUT}.json"), res)
    for s in PRIMARY:
        r = res["primary"][s]
        log(f"PRIMARY {s}: MethylGPT {r['methylgpt_medae']:.2f} vs MLP {r['mlp_medae']:.2f}; difference {r['difference']:.2f} "
            f"(95% CI {r['ci95'][0]:.2f} to {r['ci95'][1]:.2f})")
    log(f"wrote {OUT}.json and table ({len(tab)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["predict", "analyze"], required=True)
    ap.add_argument("--device", default="cuda:0", help="use with CUDA_VISIBLE_DEVICES=1 (physical GPU 1)")
    a = ap.parse_args()
    stage_predict(a) if a.stage == "predict" else stage_analyze(a)


if __name__ == "__main__":
    main()
