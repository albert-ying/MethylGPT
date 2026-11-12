#!/usr/bin/env python3
"""Masked-value pretraining curves of the released MethylGPT encoder (Fig. 1b).

The released encoder was pretrained by train_methylgpt2_2026-09-30.py with a 40,000-step cosine schedule, stopped by
hand after 4,000 steps; the 2,000-step checkpoint (lowest development error) initialized the supervised stages. This
script summarizes three runs that share that schedule and seed:
  original  artifacts/runs/stopped_cf_dense_2026-09-30   development error at 2,000 and 4,000 steps (dev_history.tsv)
  replay    artifacts/runs/pretrain_replay_2026-10-18    steps 0-4,000 again, development error every 250 steps
  extension artifacts/runs/pretrain_extend_2026-10-18    the original run resumed from its 4,000-step checkpoint to
                                                         20,000 steps, development error every 1,000 steps
Development error: masked-value MSE of the encoder and of the input layer on the fixed 512 development profiles
(30% masks), the quantity used for checkpoint selection. Training error: mean masked MSE of training batches (10-70%
masks, cross-fitted input layer) over each 100 steps, parsed from the run logs; it is not comparable with the
development values.

Pre-specified primary quantity: development MSE of the encoder at the end of the extension (20,000 steps) minus that at
2,000 steps, with a 95% interval from resampling development datasets (2,000 draws). Secondary: the same for the input
layer and for the encoder minus the input layer at every evaluation; agreement of the replay with the original run at
2,000 and 4,000 steps; held-out reconstruction MSE (2,000 profiles from 107 test datasets, the masks of Fig. 1) of the
4,000- and 20,000-step checkpoints against the released 2,000-step checkpoint, with 95% intervals from resampling
test datasets, when those evaluations exist.
The original run's history is rounded to six decimals, so its points are recomputed from the saved 2,000- and 4,000-step
checkpoints (dev_detail_checkpoint_2026-10-18.py).
Output: artifacts/eval/pretraining_curve_2026-10-18.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from methylgpt2_lib import ARTIFACTS, Corpus, log

RUNS = {"original": ARTIFACTS / "runs/stopped_cf_dense_2026-09-30", "replay": ARTIFACTS / "runs/pretrain_replay_2026-10-18",
        "extension": ARTIFACTS / "runs/pretrain_extend_2026-10-18"}
LOGS = {"replay": ARTIFACTS / "logs/pretrain_replay_2026-10-18.log", "extension": ARTIFACTS / "logs/pretrain_extend_2026-10-18.log"}
HELDOUT = {2000: ARTIFACTS / "eval/heldout_cf_dense_step2000_2026-09-30.per_profile.csv.gz",
           4000: ARTIFACTS / "eval/heldout_pretrain_step4000_2026-10-18.per_profile.csv.gz",
           20000: ARTIFACTS / "eval/heldout_pretrain_step20000_2026-10-18.per_profile.csv.gz"}
DRAWS, SEED = 2000, 20261018
# per-profile development error of the released checkpoints (dev_detail_checkpoint_2026-10-18.py)
ORIG_DETAIL = {2000: ARTIFACTS / "eval/pretrain_original_dev_detail_step2000_2026-10-18.npz",
               4000: ARTIFACTS / "eval/pretrain_original_dev_detail_step4000_2026-10-18.npz"}


def require(p: Path) -> Path:
    if not p.exists():
        raise SystemExit(f"Cannot proceed - required data file not found: {p}")
    return p


def history(run):
    p = require(RUNS[run] / "dev_history.tsv")
    return pd.read_csv(p, sep="\t", header=None, names=["step", "model", "core"])


def train_curve(run):
    pat = re.compile(r"step (\d+) loss ([0-9.]+) core ([0-9.]+) gain ([+-][0-9.]+)")
    return [{"step": int(m.group(1)), "model": float(m.group(2)), "core": float(m.group(3)), "gain": float(m.group(4))}
            for m in pat.finditer(require(LOGS[run]).read_text())]


def cluster_boot(groups, *arrays, stat, draws=DRAWS, seed=SEED):
    """Percentile interval of stat(arrays restricted to resampled clusters)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in uniq}
    out = []
    for _ in range(draws):
        take = np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        out.append(stat(*(a[take] for a in arrays)))
    return [float(x) for x in np.percentile(out, [2.5, 97.5])]


def dev_points(run):
    files = ([require(ORIG_DETAIL[s]) for s in sorted(ORIG_DETAIL)] if run == "original" else
             sorted(RUNS[run].glob("dev_detail_step*.npz"), key=lambda p: int(p.stem.split("step")[1])))
    pts = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        g, sm, sb, n = z["group"].astype(str), z["sse_model"], z["sse_core"], z["n"]
        step = int(z["step"]) if "step" in z.files else int(f.stem.split("step")[1])
        pts.append({"step": step, "model": float(sm.sum() / n.sum()), "core": float(sb.sum() / n.sum()),
                    "model_ci95": cluster_boot(g, sm, n, stat=lambda a, b: a.sum() / b.sum()),
                    "core_ci95": cluster_boot(g, sb, n, stat=lambda a, b: a.sum() / b.sum()),
                    "model_minus_core": float((sm.sum() - sb.sum()) / n.sum()),
                    "model_minus_core_ci95": cluster_boot(g, sm, sb, n, stat=lambda a, b, c: (a.sum() - b.sum()) / c.sum()),
                    "n_profiles": int(len(n)), "n_datasets": int(len(np.unique(g)))})
    return pts


def detail_path(run, step):
    return ORIG_DETAIL[step] if run == "original" else RUNS[run] / f"dev_detail_step{step}.npz"


def step_difference(run_a, step_a, run_b, step_b):
    za = np.load(require(detail_path(run_a, step_a)), allow_pickle=True)
    zb = np.load(require(detail_path(run_b, step_b)), allow_pickle=True)
    if not np.array_equal(za["rows"], zb["rows"]):
        raise SystemExit("development rows differ between runs")
    g, n = za["group"].astype(str), za["n"]
    res = {}
    for k in ("model", "core"):
        a, b = za[f"sse_{k}"], zb[f"sse_{k}"]
        res[k] = {"difference": float((a.sum() - b.sum()) / n.sum()),
                  "ci95": cluster_boot(g, a, b, n, stat=lambda x, y, c: (x.sum() - y.sum()) / c.sum())}
    return res


def heldout_comparison():
    if not all(p.exists() for p in HELDOUT.values()):
        return {"status": "not run", "missing": [str(p) for p in HELDOUT.values() if not p.exists()]}
    corpus = Corpus()
    rng = np.random.default_rng(20261001)
    test_rows = np.sort(rng.choice(corpus.rows("test"), 2000, replace=False))
    grp = dict(zip(corpus.profiles.id.to_numpy()[test_rows], corpus.profiles.group.to_numpy()[test_rows].astype(str)))
    tabs = {s: pd.read_csv(p) for s, p in HELDOUT.items()}
    out = {}
    for rate in sorted(tabs[2000].rate.unique()):
        base = tabs[2000][tabs[2000].rate == rate].set_index("id")
        blk = {"2000": {"mse": float(base.sse_methylgpt2.sum() / base.n_masked.sum()),
                        "input_layer_mse": float(base.sse_ppca_core.sum() / base.n_masked.sum())}}
        for s in (4000, 20000):
            t = tabs[s][tabs[s].rate == rate].set_index("id").loc[base.index]
            if not np.array_equal(t.n_masked.to_numpy(), base.n_masked.to_numpy()):
                raise SystemExit(f"masks differ between step {s} and step 2000 at rate {rate}")
            g = np.array([grp[i] for i in base.index])
            a, b, n = t.sse_methylgpt2.to_numpy(), base.sse_methylgpt2.to_numpy(), base.n_masked.to_numpy()
            blk[str(s)] = {"mse": float(a.sum() / n.sum()), "minus_2000": float((a.sum() - b.sum()) / n.sum()),
                           "minus_2000_ci95": cluster_boot(g, a, b, n, stat=lambda x, y, c: (x.sum() - y.sum()) / c.sum())}
        out[str(rate)] = blk
    out["n_profiles"], out["n_datasets"] = 2000, int(len(set(grp.values())))
    return out


def main():
    res = {"schedule": "cosine over 40,000 steps after 1,000 warm-up steps; batch 32; seed 20260930; same for all three runs",
           "original_dev_history": history("original").to_dict(orient="records")}
    res["original_dev"] = dev_points("original")   # per-profile values recomputed from the saved checkpoints
    for run in ("replay", "extension"):
        res[f"{run}_dev"] = dev_points(run)
        res[f"{run}_train"] = train_curve(run)
    orig = history("original").set_index("step")
    rep = {p["step"]: p for p in res["replay_dev"]}
    res["replay_vs_original"] = {str(s): {"original_model": float(orig.loc[s, "model"]), "replay_model": rep[s]["model"] if s in rep else None,
                                          "original_core": float(orig.loc[s, "core"]), "replay_core": rep[s]["core"] if s in rep else None}
                                 for s in (2000, 4000)}
    ext_steps = [p["step"] for p in res["extension_dev"]]
    if ext_steps:
        last = max(ext_steps)
        res["primary"] = {"definition": f"development MSE at step {last} (extension of the original run) minus the released 2,000-step checkpoint",
                          **step_difference("extension", last, "original", 2000)}
        res["original_4000_minus_2000"] = step_difference("original", 4000, "original", 2000)
        res["replay_minus_original_2000"] = step_difference("replay", 2000, "original", 2000)
        best = min(res["extension_dev"], key=lambda p: p["model"])
        res["extension_best"] = {"step": best["step"], "model": best["model"], "model_ci95": best["model_ci95"]}
    res["heldout"] = heldout_comparison()
    out = ARTIFACTS / "eval/pretraining_curve_2026-10-18.json"
    out.write_text(json.dumps(res, indent=1))
    log(f"wrote {out}; primary {res.get('primary')}")


if __name__ == "__main__":
    main()
