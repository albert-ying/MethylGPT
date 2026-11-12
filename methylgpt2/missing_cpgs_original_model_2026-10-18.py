#!/usr/bin/env python3
"""Originally submitted MethylGPT on the AltumAge test set with CpGs removed (Supplementary Information).

Model: the surviving fine-tuned age checkpoint of the originally submitted model (pretraining run Jan07-04-50, 6 layers,
256 dimensions; ResNet head on all CpG token embeddings; epoch 211):
  /Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/finetuning/age_prediction/age_finetune_ckpt/
  test-Finetune-Fold_0-...-Jan07-04-50-allemb-resnet-mlmwt0-bsz16_epoch=211-...-test_medae=2.2097-....ckpt
Inference code: regenerate_age_eval_results.py of revision_analyses/54_finetuning_recovery_audit (June 2026), which
regenerated its complete-profile AltumAge predictions (test MedAE 2.2097, n = 4,626). Its functions are imported
unchanged (load_model_args, setup_import_path, run_split): MinMax age scaler fitted on the AltumAge training ages,
batch size 16, float16 autocast; a missing CpG enters the model as the tokenizer's pad value (-2), which is how this
code treats every CpG that is NaN in the input.

Scenarios and masks: identical to the methylgpt2 analyses (degrade() of evaluate_age_revision_2026-10-01.py):
  all; random_p (p = 10, 30, 50, 70, 90): cells with numpy default_rng(20261001 + p).random((4626, 49156)) < p/100
  are set to NaN, over the test matrix in vocabulary order; hm27, hm450, epic_v2: only the vocabulary CpGs of that
  array are kept (artifacts/platform_vocab_masks_2026-10-01.npz). The SHA-256 of each scenario's observed-value pattern
  is written out and compared with the one computed by missing_cpgs_comparators_2026-10-18.py.

Gate (pre-specified): the complete-profile run must reproduce the saved predictions (test MedAE 2.2097 within 0.0005
years); otherwise the script stops before any masked scenario. Primary metric: test MedAE per scenario (years).
Limits stated wherever these results appear: single run; all AltumAge profiles were in its pretraining corpus; no
surviving split manifest. Results go only into the panel and table titled "originally submitted model".

Run (methygpt conda env: Python 3.9.21, torch 2.1.0+cu121, the environment of the June regeneration):
  CUDA_VISIBLE_DEVICES=1 /home/kying/anaconda3/envs/methygpt/bin/python missing_cpgs_original_model_2026-10-18.py
Outputs: artifacts/eval/missing_cpgs_original_model_2026-10-18.json and .predictions.npz
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
AUDIT = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/revision/revision_analyses/54_finetuning_recovery_audit")
REGEN = AUDIT / "code/regenerate_age_eval_results.py"
SAVED = AUDIT / "results/regenerated_age_eval_results/altumage_test_jan07_checkpoint_predictions.csv"
SAVED_AUDIT = AUDIT / "results/regenerated_age_eval_results/altumage_jan07_checkpoint_regeneration_audit.json"
PLAT_FILE = ARTIFACTS / "platform_vocab_masks_2026-10-01.npz"
SCEN = ["all", "hm450", "epic_v2", "hm27", "random_10", "random_30", "random_50", "random_70", "random_90"]
EXPECTED_MEDAE = 2.2097


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def require(p):
    if not Path(p).exists():
        sys.exit(f"Cannot proceed - required data file not found: {p}")
    return Path(p)


def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def drop_mask(scen, shape, plat):
    """Cells set to NaN by degrade() of evaluate_age_revision_2026-10-01.py (same expressions)."""
    if scen == "all":
        return np.zeros(shape, bool)
    if scen.startswith("random_"):
        p = int(scen.split("_")[1]) / 100
        return np.random.default_rng(20261001 + int(p * 100)).random(shape) < p
    return np.broadcast_to(~plat[scen], shape)


def medae(p, y):
    return float(np.median(np.abs(np.asarray(p, float) - np.asarray(y, float))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="smoke test on the first N profiles (no outputs written)")
    ap.add_argument("--scenarios", default=",".join(SCEN))
    ap.add_argument("--out", default=str(ARTIFACTS / "eval/missing_cpgs_original_model_2026-10-18.json"))
    a = ap.parse_args()
    t0 = time.time()
    spec = importlib.util.spec_from_file_location("regen", require(REGEN))
    regen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(regen)
    ckpt = require(regen.CHECKPOINT)
    audit = json.loads(require(SAVED_AUDIT).read_text())
    sha = sha256_file(ckpt)
    if sha != audit["checkpoint"]["sha256"]:
        sys.exit(f"checkpoint checksum {sha} differs from the June regeneration audit {audit['checkpoint']['sha256']}")
    log(f"checkpoint checksum matches the June regeneration audit ({sha[:12]})")

    model_args = regen.load_model_args()
    regen.setup_import_path()
    from finetuning_age_datasets import CollatableVocab
    from finetuning_age_models import FintuneAltumAgeModel_EMB_MLM
    from sklearn import preprocessing

    dev = torch.device("cuda")
    torch.cuda.set_device(0)          # CUDA_VISIBLE_DEVICES=1 maps physical GPU 1 to device 0
    train_age = pd.read_parquet(require(model_args["train_file"]), columns=["age"])["age"].to_numpy()
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(train_age.reshape(-1, 1))
    test_df = pd.read_parquet(require(model_args["test_file"]))
    if a.limit:
        test_df = test_df.iloc[:a.limit].reset_index(drop=True)
    vocab = CollatableVocab(model_args)
    model = FintuneAltumAgeModel_EMB_MLM(model_args=model_args, vocab=vocab, scaler=scaler)
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"], strict=True)
    model.to(dev).eval()
    plat = np.load(require(PLAT_FILE))
    plat = {k: plat[k].astype(bool) for k in ("hm27", "hm450", "epic_v2")}

    X0 = np.stack(test_df["data"].to_numpy())
    y = test_df["age"].to_numpy(float)
    ids = test_df["id"].astype(str).to_numpy()
    log(f"test matrix {X0.shape} {X0.dtype}; natively missing {np.isnan(X0).mean():.4f}; non-finite non-NaN "
        f"{int((~np.isfinite(X0) & ~np.isnan(X0)).sum())}")
    full_shape = (4626, X0.shape[1])
    res = {"model": "originally submitted MethylGPT, fine-tuned age checkpoint (Jan07-04-50 encoder, ResNet head, epoch 211)",
           "limits": "single run; all AltumAge profiles were in its pretraining corpus; no surviving split manifest",
           "checkpoint": str(ckpt), "checkpoint_sha256": sha, "inference_code": str(REGEN),
           "missing_value_handling": "removed CpGs set to NaN; the tokenizer maps NaN to the pad value (-2), as for natively missing CpGs",
           "runtime": {"python": sys.version, "torch": torch.__version__, "platform": platform.platform(),
                       "gpu": torch.cuda.get_device_name(0), "batch_size": 16},
           "n": int(len(y)), "scenarios": {}, "observed_pattern_sha256": {}}
    preds = {}
    for s in a.scenarios.split(","):
        drop = drop_mask(s, full_shape, plat)[:len(X0)]
        X = X0.copy()
        X[drop] = np.nan
        res["observed_pattern_sha256"][s] = hashlib.sha256(np.packbits(np.isfinite(X)).tobytes()).hexdigest()
        df = test_df[["id", "age"]].copy()
        df["data"] = list(X)
        t1 = time.time()
        fr = regen.run_split(split="test", df=df, model=model, vocab=vocab, scaler=scaler, batch_size=16, device=dev, limit=None)
        assert (fr["sample_id"].astype(str).to_numpy() == ids).all()
        preds[s] = fr["pred_age"].to_numpy(float)
        res["scenarios"][s] = {"observed_fraction": float(np.isfinite(X).mean()), "medae": medae(preds[s], y),
                               "mae": float(np.mean(np.abs(preds[s] - y))), "pearson_r": float(np.corrcoef(preds[s], y)[0, 1]),
                               "n": int(len(y)), "seconds": round(time.time() - t1, 1)}
        log(f"{s}: observed {res['scenarios'][s]['observed_fraction']:.3f}, MedAE {res['scenarios'][s]['medae']:.4f} "
            f"({res['scenarios'][s]['seconds']:.0f} s)")
        if s == "all":
            saved = pd.read_csv(require(SAVED)).iloc[:len(y)]
            assert (saved["sample_id"].astype(str).to_numpy() == ids).all()
            d = float(np.max(np.abs(saved["pred_age"].to_numpy(float) - preds[s])))
            res["reproduction_check"] = {"saved_predictions": str(SAVED), "max_abs_diff_years_vs_saved": d,
                                         "medae": res["scenarios"][s]["medae"], "expected_medae": EXPECTED_MEDAE,
                                         "passed": bool(a.limit or abs(res["scenarios"][s]["medae"] - EXPECTED_MEDAE) < 5e-4)}
            log(f"reproduction check: max |pred - saved| {d:.2e} years; MedAE {res['scenarios'][s]['medae']:.4f}")
            if not res["reproduction_check"]["passed"]:
                sys.exit(f"complete-profile MedAE {res['scenarios'][s]['medae']:.4f} does not reproduce {EXPECTED_MEDAE}; stopping")
    res["elapsed_s"] = time.time() - t0
    if a.limit:
        log(f"smoke test done ({a.limit} profiles); nothing written")
        return
    out = Path(a.out)
    out.write_text(json.dumps(res, indent=1))
    np.savez_compressed(out.with_suffix(".predictions.npz"), altumage_ids=ids, altumage_age=y,
                        **{f"original|{s}": v for s, v in preds.items()})
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
