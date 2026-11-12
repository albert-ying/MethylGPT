"""User-facing MethylGPT interface.

Input everywhere is a pandas DataFrame of beta values in [0, 1], one row per sample and one
column per CpG (Illumina probe IDs such as cg00000029). Any subset of CpGs works: CpGs outside
the 49,156-CpG vocabulary are ignored, and vocabulary CpGs that are absent or NaN are treated
as unobserved. Nothing is mean-imputed before the model sees the data.

    from methylgpt2 import MethylGPT
    model = MethylGPT.from_pretrained("path/to/methylgpt2-1.1")
    ages = model.predict_age(betas)          # Series, years (mean of the age-ensemble members)
    runs = model.predict_age_runs(betas)     # one column of predicted age per training run
    clocks = model.predict_clocks(betas)     # 25 distilled full-array clock outputs
    full = model.impute(betas)               # all 49,156 vocabulary CpGs, observed values kept
    emb = model.embed(betas)                 # 2,048-dimensional sample embedding
    fs = model.factor_scores(betas)          # 1,536 factor scores for clustering and visualisation

Age ensemble: bundles from version 1.1 on list several training runs under "age_ensemble" in
config.json. Predicted age is then the mean, in years, of the members' predictions. Sex, tissue,
the clock outputs, embeddings, factor scores and imputation always come from the first run
(weights.pt). A bundle without "age_ensemble" (1.0) has one member and behaves as before.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .model import BioHeads, MethylGPT2, PPCACore


class _AgeMember:
    """One training run used for age: its encoder, heads and age scaling."""

    def __init__(self, name, net, heads, center, scale):
        self.name, self.net, self.heads, self.center, self.scale = name, net, heads, center, scale


def _finv(y, adult_age=20.0):
    """Inverse of Horvath's log-linear age transform."""
    y = np.asarray(y, np.float64)
    return np.where(y < 0, (1 + adult_age) * np.exp(y) - 1, y * (1 + adult_age) + adult_age)


class MethylGPT:
    def __init__(self, bundle_dir, device=None):
        d = Path(bundle_dir)
        cfg = json.loads((d / "config.json").read_text())
        self.config = cfg
        self._bundle_dir, self._stage_nets = d, {}
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.vocab = (d / "vocab.txt").read_text().split()
        self._col = {c: i for i, c in enumerate(self.vocab)}
        z = np.load(d / "core.npz")
        self.core = PPCACore(z["mu"], z["V"], z["lam"], float(z["sigma2"])).to(self.device)
        self.net, self.heads = self._load_run(d / "weights.pt")
        self.clock_names = cfg["clock_names"]
        self.tissues = cfg["tissues"] + ["other"]
        # age-ensemble members; the first run (weights.pt) reuses the modules loaded above
        members = cfg.get("age_ensemble", {}).get("members") or [{"name": "run1", "weights": "weights.pt"}]
        self.age_members = []
        for m in members:
            if m["weights"] == "weights.pt":
                net, heads = self.net, self.heads
            else:
                net, heads = self._load_run(d / m["weights"])
            self.age_members.append(_AgeMember(m["name"], net, heads, float(m.get("age_center", cfg["age_center"])),
                                               float(m.get("age_scale", cfg["age_scale"]))))
        if self.age_members[0].net is not self.net:
            raise RuntimeError("the first age-ensemble member must be weights.pt")

    def _load_run(self, path):
        """Encoder and heads of one training run; all runs share the bundle's factor model core."""
        a, cfg = self.config["architecture"], self.config
        net = MethylGPT2(self.core, d_tok=a["d_tok"], d=a["d"], heads=a["heads"], n_latent=a["latents"],
                         n_score_tokens=a["score_tokens"], n_self=a["self_layers"], d_read=a["d_read"],
                         encoder=a["encoder"], d_dense=a["d_dense"], n_dense_tokens=a["dense_tokens"]).to(self.device)
        state = torch.load(path, map_location=self.device, weights_only=True)
        res = net.load_state_dict(state["encoder"], strict=False)
        if res.unexpected_keys or any(not k.startswith("core.") for k in res.missing_keys):
            raise RuntimeError(f"weights in {path.name} do not match the architecture: {res}")
        heads = BioHeads(self.core.rank + a["d"], len(cfg["tissues"]) + 1, len(cfg["clock_names"])).to(self.device)
        heads.load_state_dict(state["heads"])
        net.eval(); heads.eval()
        return net, heads

    @classmethod
    def from_pretrained(cls, bundle_dir, device=None):
        """Load a bundle directory, including any extra age-ensemble members it lists."""
        return cls(bundle_dir, device)

    def _matrix(self, betas: pd.DataFrame) -> np.ndarray:
        cols = [c for c in betas.columns if c in self._col]
        if not cols:
            raise ValueError("none of the input columns are MethylGPT vocabulary CpGs (expected IDs like cg00000029)")
        X = np.full((len(betas), len(self.vocab)), np.nan, np.float32)
        vals = betas[cols].to_numpy(np.float32)
        if np.nanmax(vals) > 1.5:
            raise ValueError("inputs look like M-values or percentages; convert to beta values in [0, 1]")
        X[:, [self._col[c] for c in cols]] = np.clip(vals, 0, 1)
        cov = np.isfinite(X).mean(1)
        if cov.min() < 0.05:
            warnings.warn(f"some samples observe fewer than 5% of vocabulary CpGs (minimum {cov.min():.1%}); "
                          "predictions for them rest mostly on the corpus prior")
        return X

    @torch.no_grad()
    def _encode(self, x, net=None, s=None):
        net = self.net if net is None else net
        obs = torch.isfinite(x)
        if s is None:
            s = self.core.posterior(x, obs.float())
        amp = self.device.type == "cuda"
        with torch.autocast(self.device.type, enabled=False):
            xs, c, e = net.inputs(x, obs, s)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=amp):
            z = net.encode(xs, c, e, obs, s)
        return z, s, net.embedding(z, s)

    def _batches(self, betas, batch_size):
        X = self._matrix(betas)
        for i in range(0, len(X), batch_size):
            yield i, torch.from_numpy(X[i:i + batch_size]).to(self.device)

    @torch.no_grad()
    def factor_scores(self, betas: pd.DataFrame, batch_size: int = 64) -> pd.DataFrame:
        """Posterior factor scores (1,536 dimensions, natural scale) from the observed CpGs.

        Distances between factor scores are distances between the model's reconstructions of
        the profiles, so this is the representation to use for clustering, visualisation and
        nearest-neighbour searches; embed() is the input to the supervised heads.
        """
        out = []
        for _, x in self._batches(betas, batch_size):
            out.append(self.core.posterior(x, torch.isfinite(x).float()).cpu().numpy())
        S = np.concatenate(out)
        return pd.DataFrame(S, index=betas.index, columns=[f"factor_{k}" for k in range(S.shape[1])])

    def _stage_net(self, stage):
        """Encoder of a named embedding stage listed in the bundle config (loaded on first use)."""
        stages = self.config.get("embedding_stages", {})
        if stage not in stages:
            raise ValueError(f"unknown embedding stage {stage!r}; this bundle provides 'final' and {sorted(stages)}")
        if stage not in self._stage_nets:
            a = self.config["architecture"]
            net = MethylGPT2(self.core, d_tok=a["d_tok"], d=a["d"], heads=a["heads"], n_latent=a["latents"],
                             n_score_tokens=a["score_tokens"], n_self=a["self_layers"], d_read=a["d_read"],
                             encoder=a["encoder"], d_dense=a["d_dense"], n_dense_tokens=a["dense_tokens"]).to(self.device)
            path = self._bundle_dir / stages[stage]["weights"]
            state = torch.load(path, map_location=self.device, weights_only=True)
            res = net.load_state_dict(state["encoder"], strict=False)
            if res.unexpected_keys or any(not k.startswith("core.") for k in res.missing_keys):
                raise RuntimeError(f"weights in {path.name} do not match the architecture: {res}")
            self._stage_nets[stage] = net.eval()
        return self._stage_nets[stage]

    def embed(self, betas: pd.DataFrame, batch_size: int = 32, stage: str = "final") -> pd.DataFrame:
        """2,048-dimensional sample embedding (standardized factor scores and pooled encoder tokens).

        stage="final" (default) uses the released encoder, after the clock-distillation stage; it is the
        input of the age, sex, tissue and clock heads. stage="age_stage" uses the encoder of the same
        training run at the end of the age, sex and tissue stage, before clock distillation: this is the
        embedding on which the disease models of the MethylGPT paper were fitted and evaluated.
        """
        net = None if stage == "final" else self._stage_net(stage)
        out = [self._encode(x, net)[2].float().cpu().numpy() for _, x in self._batches(betas, batch_size)]
        return pd.DataFrame(np.concatenate(out), index=betas.index, columns=[f"emb_{k}" for k in range(out[0].shape[1])])

    def _predict(self, betas, batch_size):
        """First-run outputs (with age set to the ensemble mean) and the per-run ages."""
        rows, runs = [], []
        for _, x in self._batches(betas, batch_size):
            with torch.no_grad():
                _, s, emb = self._encode(x)
                age, sex, tissue, clocks = self.heads(emb)
                ages = []
                for m in self.age_members:
                    a = age if m.net is self.net else m.heads(self._encode(x, m.net, s)[2])[0]
                    ages.append(_finv(a.float().cpu().numpy() * m.scale + m.center))
            ages = np.stack(ages, 1)
            runs.append(pd.DataFrame(ages, columns=[m.name for m in self.age_members]))
            r = pd.DataFrame(clocks.float().cpu().numpy() * np.asarray(self.config["clock_scale"]) +
                             np.asarray(self.config["clock_median"]), columns=self.clock_names)
            r.insert(0, "tissue", [self.tissues[k] for k in tissue.argmax(1).cpu().numpy()])
            r.insert(0, "p_male", torch.sigmoid(sex).float().cpu().numpy())
            r.insert(0, "age", ages.mean(1))
            r.insert(0, "vocab_cpgs_observed", torch.isfinite(x).sum(1).cpu().numpy())
            rows.append(r)
        out, runs = pd.concat(rows, ignore_index=True), pd.concat(runs, ignore_index=True)
        out.index = runs.index = betas.index
        return out, runs

    def predict(self, betas: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
        """Age (years), probability of male sex, most likely tissue and the 25 clock outputs.

        Age is the mean over the bundle's age-ensemble members (see predict_age_runs); all other
        columns come from the first run.
        """
        return self._predict(betas, batch_size)[0]

    def predict_age(self, betas: pd.DataFrame, batch_size: int = 32) -> pd.Series:
        """Predicted age in years: the mean over the age-ensemble members."""
        return self.predict(betas, batch_size)["age"]

    def predict_age_runs(self, betas: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
        """Predicted age in years from each age-ensemble member (one column per training run)."""
        return self._predict(betas, batch_size)[1]

    def predict_clocks(self, betas: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
        return self.predict(betas, batch_size)[self.clock_names]

    @torch.no_grad()
    def reconstruct(self, betas: pd.DataFrame, batch_size: int = 8) -> pd.DataFrame:
        """Factor-model reconstruction mu + V s of every vocabulary CpG, at observed and unobserved CpGs alike.

        s are the posterior factor scores computed from the observed CpGs. Values are not clipped to [0, 1]
        (impute() clips, and returns observed values unchanged).
        """
        X = self._matrix(betas)
        out = np.empty_like(X)
        for i in range(0, len(X), batch_size):
            x = torch.from_numpy(X[i:i + batch_size]).to(self.device)
            out[i:i + batch_size] = self.core.predict(self.core.posterior(x, torch.isfinite(x).float())).float().cpu().numpy()
        return pd.DataFrame(out, index=betas.index, columns=self.vocab)

    @torch.no_grad()
    def residuals(self, betas: pd.DataFrame, batch_size: int = 8) -> pd.DataFrame:
        """Observed beta value minus the factor-model reconstruction (reconstruct()); NaN where a CpG is unobserved.

        These are the residual features of the residual disease model in the MethylGPT paper.
        """
        X = self._matrix(betas)
        out = np.full_like(X, np.nan)
        for i in range(0, len(X), batch_size):
            x = torch.from_numpy(X[i:i + batch_size]).to(self.device)
            obs = torch.isfinite(x)
            r = torch.where(obs, x - self.core.predict(self.core.posterior(x, obs.float())), torch.full_like(x, float("nan")))
            out[i:i + batch_size] = r.float().cpu().numpy()
        return pd.DataFrame(out, index=betas.index, columns=self.vocab)

    def disease_features(self, betas: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
        """Features of the residual disease model of the MethylGPT paper, in the column order of its fitted models.

        Columns: the 2,048-dimensional embedding of the age stage (embed(stage="age_stage")), the 25 distilled clock
        outputs in the standardized units of the clock heads ((value - clock_median) / clock_scale), and the 49,156
        factor-model residuals (residuals(); NaN where unobserved). The paper's Cox models standardize every column
        with the mean and standard deviation of the fitting participants and set unobserved residuals to the fitting
        mean (zero after standardization); age and sex enter as two unpenalized covariates before these columns.
        """
        E = self.embed(betas, batch_size, stage="age_stage")
        B = (self.predict_clocks(betas, batch_size) - np.asarray(self.config["clock_median"])) / np.asarray(self.config["clock_scale"])
        B.columns = [f"clock_{c}" for c in B.columns]
        R = self.residuals(betas, min(batch_size, 8))
        R.columns = [f"residual_{c}" for c in R.columns]
        return pd.concat([E, B.astype(np.float32), R], axis=1)

    @torch.no_grad()
    def impute(self, betas: pd.DataFrame, batch_size: int = 8, chunk: int = 4096, method: str = "factor") -> pd.DataFrame:
        """Return all vocabulary CpGs: observed values unchanged, the rest predicted.

        method="factor" (default) uses the factor model's reconstruction from the posterior factor scores, the
        reconstruction evaluated in the MethylGPT paper. method="transformer" adds the encoder's correction, which
        was trained for masked values before the supervised stages and is not more accurate.
        """
        if method not in ("factor", "transformer"):
            raise ValueError("method must be 'factor' or 'transformer'")
        X = self._matrix(betas)
        out = X.copy()
        if method == "factor":
            with torch.no_grad():
                for i in range(0, len(X), batch_size):
                    x = torch.from_numpy(X[i:i + batch_size]).to(self.device)
                    obs = torch.isfinite(x)
                    rec = self.core.predict(self.core.posterior(x, obs.float())).clamp(0, 1)
                    xi = out[i:i + batch_size]
                    miss = ~obs.cpu().numpy()
                    xi[miss] = rec.float().cpu().numpy()[miss]
            return pd.DataFrame(out, index=betas.index, columns=self.vocab)
        for i in range(0, len(X), batch_size):
            x = torch.from_numpy(X[i:i + batch_size]).to(self.device)
            z, s, _ = self._encode(x)
            for b in range(len(x)):
                miss = torch.nonzero(~torch.isfinite(x[b]))[:, 0]
                preds = []
                for j in range(0, len(miss), chunk):
                    idx = miss[j:j + chunk][None]
                    p, _ = self.net.predict_targets(z[b:b + 1], s[b:b + 1], idx,
                                                    amp_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32)
                    preds.append(p[0])
                if preds:
                    out[i + b, miss.cpu().numpy()] = torch.cat(preds).clamp(0, 1).cpu().numpy()
        return pd.DataFrame(out, index=betas.index, columns=self.vocab)
