#!/usr/bin/env python3
"""Does the package reproduce the residual disease model?

For the Generation Scotland development participants, the package call disease_features() builds
[age-stage embedding (2,048), 25 distilled clock outputs in standardized units, 49,156 factor-model residuals].
The stored fold models of the residual model (gs_factor_residual_cox_2026-10-01.py; means, scales and
coefficients fitted on four outer folds) are applied to the held-out fold, with age and sex as in the analysis,
and the resulting risks are compared with the stored held-out risks of the manuscript. TF32 matrix products are
enabled, as in the analysis. Aggregate output only:
artifacts/eval/disease_features_reproduction_2026-10-09.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sksurv.metrics import concordance_index_censored

from methylgpt2_lib import ARTIFACTS, cuda_preinit, log, require, write_json

HERE = Path(__file__).resolve().parent
CACHE = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/task_a_private_training_20260629/_frame_cache")
SPLIT = "full18859_common_task_a_incident_positive_duration_20260629"
D = ARTIFACTS / "gs_cox/factor_residual_2026-10-01"


def main():
    cuda_preinit(); torch.backends.cuda.matmul.allow_tf32 = True
    sys.path.insert(0, str(HERE / "package/src"))
    from methylgpt2 import MethylGPT, __version__
    model = MethylGPT.from_pretrained(ARTIFACTS / "release/methylgpt2-1.1", device="cuda:0")
    mp = require(next(CACHE.glob(f"COMMON_SAMPLE_SPLIT_{SPLIT}_train_all_gs_batches_all_probes_*.matrix.npz")))
    ids = pd.read_parquet(mp.with_name(mp.name.replace(".matrix.npz", ".labels.parquet")), columns=["dnam_id"]).dnam_id.astype(str).to_numpy()
    with np.load(mp, allow_pickle=False) as z:
        X = z["matrix"].astype(np.float32)
    pos = {s: i for i, s in enumerate(ids)}
    preds = pd.read_parquet(require(D / "factor_residual/private_nested_predictions.parquet"))
    preds = preds[preds.age_adjustment == "linear"].copy(); preds["dnam_id"] = preds.dnam_id.astype(str)
    res = {"package_version": __version__, "bundle": "methylgpt2-1.1", "feature_columns": None, "endpoints": {}}
    for ep in ("Endocrine_and_Metabolic_Diseases", "date_of_death"):
        pe = preds[preds.endpoint == ep].reset_index(drop=True)
        out = {}
        for f in sorted(pe.outer_fold.unique()):
            h = pe[pe.outer_fold == f]
            F = model.disease_features(pd.DataFrame(X[[pos[s] for s in h.dnam_id]], columns=model.vocab))
            res["feature_columns"] = int(F.shape[1])
            m = dict(np.load(require(D / f"models/factor_residual_{ep}_outer{f}.npz"), allow_pickle=False))
            Fv = F.to_numpy(np.float64)
            zf = (np.where(np.isfinite(Fv), Fv, m["mean"]) - m["mean"]) / m["scale"]
            c = (h[["age", "sex"]].to_numpy(float) - m["cov_mean"]) / m["cov_scale"]
            risk = np.hstack([c, zf]) @ m["coefficient"].astype(np.float64)
            st = h.risk.to_numpy(float)
            ev, du = h.event.astype(bool).to_numpy(), h.duration.to_numpy(float)
            out[str(int(f))] = {"heldout_participants": int(len(h)), "max_abs_risk_difference": float(np.abs(risk - st).max()),
                                "risk_sd": float(st.std()), "pearson_r": float(np.corrcoef(risk, st)[0, 1]),
                                "c_index_package_features": float(concordance_index_censored(ev, du, risk)[0]),
                                "c_index_stored": float(concordance_index_censored(ev, du, st)[0])}
            log(f"{ep} fold {f}: {out[str(int(f))]}")
            del F, Fv, zf
        res["endpoints"][ep] = out
    write_json(ARTIFACTS / "eval/disease_features_reproduction_2026-10-09.json", res)


if __name__ == "__main__":
    main()
