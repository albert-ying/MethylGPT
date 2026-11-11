#!/usr/bin/env python3
"""Paired bootstrap of nested-development C-index differences in Generation Scotland.

MethylGPT-2 out-of-fold risks (locked protocol copy) are compared, on the same subjects and
outer folds, with the age+sex baseline of the same run, the v12 raw-CpG Cox run and the v12
corrected January embedding run. Subjects are resampled within outer folds (1,000 draws,
seed 20260928); the combined C-index pools within-fold comparable pairs. BH adjustment
across all endpoint x comparator contrasts.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored

from methylgpt2_lib import ARTIFACTS, log, write_json

V12 = Path("/Entropy/kying/tmp/methylgpt_publication_2026-09-28/results")


def load_preds(paths):
    frames = [pd.read_parquet(Path(p) / "private_nested_predictions.parquet") for p in paths]
    d = pd.concat(frames)
    return d[d.age_adjustment == "linear"]


def counts(ev, du, r):
    _, c, dd, tr, _ = concordance_index_censored(ev, du, r)
    return np.array([c, dd, tr], float)


def cidx(cnt):
    return (cnt[0] + 0.5 * cnt[2]) / cnt.sum()


def bh(p):
    p = np.asarray(p); n = len(p); order = np.argsort(p)
    q = np.empty(n); prev = 1.0
    for rank, i in reversed(list(enumerate(order, 1))):
        prev = min(prev, p[i] * n / rank); q[i] = prev
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ARTIFACTS / "gs_cox/mgpt2_embed_linear_2026-10-01"))
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--out", default=str(ARTIFACTS / "gs_cox/paired_bootstrap_mgpt2_embed_2026-10-01.json"))
    ap.add_argument("--extra", action="append", default=[],
                    help="name=directory of an additional comparator prediction file (repeatable)")
    ap.add_argument("--skip-v12", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    me = load_preds([a.run])
    comps = {"age_sex": None,
             "raw_cpg_v12": load_preds([V12 / f"nested_raw_cpg_{p}_2026-09-28" for p in
                                        ("recovered_part1", "linear_part2", "recovered_part3", "linear_part4")]),
             "january_corrected_v12": load_preds([V12 / "nested_observed_frozen_linear_2026-09-30"])}
    if a.skip_v12:
        comps = {"age_sex": None}
    for spec in a.extra:
        name, d = spec.split("=", 1)
        comps[name] = load_preds([d])
    rows = []
    rng = np.random.default_rng(20260928)
    for ep in sorted(me.endpoint.unique()):
        m = me[me.endpoint == ep].set_index("dnam_id")
        for name, other in comps.items():
            if other is None:
                r_other = m.baseline_risk
            else:
                o = other[other.endpoint == ep].set_index("dnam_id")
                if set(o.index) != set(m.index) or not (o.loc[m.index, "outer_fold"].to_numpy() == m.outer_fold.to_numpy()).all():
                    raise ValueError(f"Subject or fold mismatch for {ep} vs {name}")
                r_other = o.loc[m.index, "risk"]
            ev, du = m.event.astype(bool).to_numpy(), m.duration.to_numpy(float)
            r1, r0 = m.risk.to_numpy(float), r_other.to_numpy(float)
            folds = [np.flatnonzero(m.outer_fold.to_numpy() == f) for f in sorted(m.outer_fold.unique())]
            c1 = cidx(sum(counts(ev[f], du[f], r1[f]) for f in folds))
            c0 = cidx(sum(counts(ev[f], du[f], r0[f]) for f in folds))
            diffs = []
            for _ in range(a.draws):
                t1, t0_ = np.zeros(3), np.zeros(3)
                for f in folds:
                    b = rng.choice(f, len(f), replace=True)
                    t1 += counts(ev[b], du[b], r1[b]); t0_ += counts(ev[b], du[b], r0[b])
                diffs.append(cidx(t1) - cidx(t0_))
            diffs = np.array(diffs)
            p = float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())))
            rows.append({"endpoint": ep, "comparator": name, "c_mgpt2": c1, "c_comparator": c0, "difference": c1 - c0,
                         "ci95_low": float(np.percentile(diffs, 2.5)), "ci95_high": float(np.percentile(diffs, 97.5)), "p": p})
            log(f"{ep} vs {name}: {c1:.4f} vs {c0:.4f} diff {c1 - c0:+.4f} [{rows[-1]['ci95_low']:+.4f}, {rows[-1]['ci95_high']:+.4f}] p {p:.3f}")
    df = pd.DataFrame(rows)
    df["q_bh"] = bh(df.p.to_numpy())
    df.to_csv(Path(a.out).with_suffix(".csv"), index=False)
    write_json(Path(a.out), {"run": a.run, "draws": a.draws, "contrasts": df.to_dict("records"), "elapsed_s": time.time() - t0})
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
