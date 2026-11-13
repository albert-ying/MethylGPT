#!/usr/bin/env python3
"""Tables on the originally submitted MethylGPT model for the Supplementary Information (no model is rerun).

These tables show how the original model was built and why its age
results differ from the revision. This script collects, from surviving files only, three tables:
  (a) architecture: the original submission's text, the surviving checkpoint that the revision uses as
      "originally submitted model" (Jan07-04-50, read from its args.json and state dict), and MethylGPT
      (read from the pretraining config.json of the released model);
  (b) the 8-configuration output-head ablation of the original model, as reported in the previous revision
      draft (age_prediction_architecture_ablation.csv; no header in the file, columns as named below);
  (c) age results of the original model: values as reported in the original submission, the logged training
      histories recovered by the 30 Sep 2026 provenance audit, the MedAE recomputed from the surviving
      checkpoint's saved predictions, and MethylGPT on the same AltumAge test set.
Every value is read from the files listed in SOURCES; values quoted from the submission text are marked as such.
Outputs: artifacts/eval/original_model_tables_2026-10-18.json and copies of the two source CSVs in artifacts/eval/.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
A = HERE / "artifacts"
PROJ = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT")
PRE = PROJ / "dev_pretraining_test-dataset_CpGs_type3-preprocessing_False-Jan07-04-50"
SOURCES = {
    "original_args": PRE / "args.json",
    "original_checkpoint": PRE / "best_model_epoch6.pt",
    "original_run_log": PRE / "run.log",
    "original_test_log": PRE / "test.log",
    "head_ablation": PROJ / "revision_round3_20260703/submission_package_draft_2026-05-26/Supplementary_Information/draft_tables/age_prediction_architecture_ablation.csv",
    "age_reconciliation": Path("/Entropy/kying/tmp/methylgpt_publication_2026-09-28/results/figure_age_source_reconciliation_2026-09-30/figure_age_source_reconciliation_2026-09-30.csv"),
    "age_provenance": Path("/Entropy/kying/tmp/methylgpt_publication_2026-09-28/results/legacy_age_figure_provenance_2026-09-28.json"),
    "methylgpt_config": A / "runs/stopped_cf_dense_2026-09-30/config.json",
    "methylgpt_age": A / "eval/age_ensemble_2026-10-04.json",
}
ABLATION_COLUMNS = ["head", "input", "encoder", "validation MedAE", "validation MAE", "validation r", "test MedAE", "test MAE", "test r"]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    for k, p in SOURCES.items():
        if not p.is_file():
            raise SystemExit(f"Cannot proceed - required data file not found: {p}")
    args = json.loads(SOURCES["original_args"].read_text())
    sd = torch.load(SOURCES["original_checkpoint"], map_location="cpu", weights_only=True)
    n_params = int(sum(v.numel() for v in sd.values()))
    layers = sorted({int(k.split(".")[2]) for k in sd if k.startswith("transformer_encoder.layers.")})
    emb = tuple(sd["encoder.embedding.weight"].shape)
    cfg = json.loads(SOURCES["methylgpt_config"].read_text())
    age = json.loads(SOURCES["methylgpt_age"].read_text())
    mg_test = age["a_altumage_test"]["ensemble"]["medae"]
    runs = [age["a_altumage_test"][k]["medae"] for k in ("run 1", "seed 2", "seed 3") if k in age["a_altumage_test"]]

    arch = [
        {"attribute": "Input", "original submission (text)": "Each CpG a token: CpG ID embedding plus value embedding; [CLS] token prepended",
         "checkpoint used as originally submitted model": f"{args['n_hvg']:,} CpG tokens plus 3 special tokens (embedding table {emb[0]:,} x {emb[1]})",
         "MethylGPT": "Probabilistic input layer over 49,156 CpGs (latent scores from the CpGs present) and three linear maps of all CpGs to 2,048 units"},
        {"attribute": "Transformer blocks", "original submission (text)": "12 (Results); 6 (Methods)",
         "checkpoint used as originally submitted model": f"{args['nlayers']} (state dict layers {layers[0]}-{layers[-1]})",
         "MethylGPT": f"{cfg['self_layers']} self-attention blocks over {cfg['dense_tokens']} + {cfg['score_tokens']} + {cfg['latents']} tokens"},
        {"attribute": "Attention heads", "original submission (text)": "4", "checkpoint used as originally submitted model": str(args["nhead"]), "MethylGPT": str(cfg["heads"])},
        {"attribute": "Model width", "original submission (text)": "64 (larger variants of 7M and 15M parameters mentioned)",
         "checkpoint used as originally submitted model": str(args["layer_size"]), "MethylGPT": f"{cfg['d']} (sample embedding 2,048)"},
        {"attribute": "Parameters", "original submission (text)": "not stated for the main model; larger variants 7M and 15M",
         "checkpoint used as originally submitted model": f"{n_params:,}", "MethylGPT": f"{cfg['n_params']:,} (encoder)"},
        {"attribute": "Pretraining objective", "original submission (text)": "Masked values of 30% of CpGs plus profile reconstruction from the [CLS] embedding",
         "checkpoint used as originally submitted model": f"mask ratio {args['mask_ratio']}; GEPC (profile reconstruction) {args['GEPC']}",
         "MethylGPT": f"Masked values, {int(100 * cfg['mask_lo'])}-{int(100 * cfg['mask_hi'])}% of observed CpGs per profile, loss relative to the input layer"},
        {"attribute": "Training", "original submission (text)": "AdamW, learning rate 0.001, 10 epochs, batch size 16, learning rate decayed 10% per epoch",
         "checkpoint used as originally submitted model": f"learning rate {args['lr']}, batch size {args['batch_size']}, {args['epochs']} epochs planned; best checkpoint at epoch 6 (run.log)",
         "MethylGPT": f"AdamW, learning rate {cfg['lr']}, batch size {cfg['batch']}, cosine schedule over {cfg['steps']:,} steps; stopped at 4,000, checkpoint at 2,000 selected on development datasets"},
        {"attribute": "Data split", "original submission (text)": "Random split of samples (train and test sets)",
         "checkpoint used as originally submitted model": "All AltumAge profiles were in the pretraining corpus",
         "MethylGPT": "Split by dataset before any fitting (145,700 training profiles, 1,941 datasets)"},
        {"attribute": "Genomic position or sequence", "original submission (text)": "Not used; CpGs identified by ID",
         "checkpoint used as originally submitted model": "Not used", "MethylGPT": "Not used; each CpG has a learned vector"},
        {"attribute": "Age head", "original submission (text)": "ResNet1D on token embeddings, fine-tuned",
         "checkpoint used as originally submitted model": "ResNet on all token embeddings, fine-tuned (age checkpoint epoch 211)",
         "MethylGPT": "Shared two-layer network (1,024 and 512 units) on the 2,048-dimensional embedding, trained with the encoder"},
    ]

    abl = pd.read_csv(SOURCES["head_ablation"], header=None, names=ABLATION_COLUMNS)
    if len(abl) != 8:
        raise SystemExit(f"expected 8 ablation rows, found {len(abl)}")

    rec = pd.read_csv(SOURCES["age_reconciliation"])
    prov = json.loads(SOURCES["age_provenance"].read_text())
    age_rows = [
        {"source": "Original submission, Results (as reported)", "model": "MethylGPT as submitted (64-dimensional)", "validation MedAE": 4.45, "test MedAE": None,
         "note": "Values quoted from the submission text; comparators included PhenoAge and GrimAge"},
        {"source": "Original submission, Results (as reported)", "model": "Larger variant", "validation MedAE": 2.73, "test MedAE": 2.81,
         "note": "Values quoted from the submission text"},
    ]
    # run widths as stated by the provenance audit (safe_text_reconciliation.can_state_now)
    RUN = {"Sep26-10-27": "64-dimensional run, September 2024", "Nov29-12-01": "128-dimensional run, November 2024",
           "Jan07-04-50": "256-dimensional checkpoint, January 2025"}
    for _, r in rec.iterrows():
        age_rows.append({"source": ("training log: " + r["source_stream"]) if isinstance(r["source_stream"], str) else "recomputed from the checkpoint's saved predictions",
                         "model": f"{RUN.get(r['run'], r['run'])} (epoch {int(r['epoch'])})", f"{r['split'].lower()} MedAE": float(r["medae_years"]),
                         "n": None if pd.isna(r["sample_n"]) else int(r["sample_n"]), "note": r["source_type"]})
    age_rows.append({"source": "This revision (AltumAge test, curated ages)", "model": "MethylGPT (mean of three training runs)", "test MedAE": mg_test,
                     "n": age["a_altumage_test"]["ensemble"].get("n", 4626),
                     "note": f"single runs {min(runs):.2f}-{max(runs):.2f}; AltumAge validation and test profiles excluded from pretraining"})
    out = {
        "architecture": arch,
        "head_ablation_original_model": abl.to_dict(orient="records"),
        "age_results": age_rows,
        "provenance_decision": prov.get("decision"),
        "original_checkpoint": {"path": str(SOURCES["original_checkpoint"]), "sha256": sha(SOURCES["original_checkpoint"]), "n_parameters": n_params,
                                "args": {k: args[k] for k in ("layer_size", "nlayers", "nhead", "n_hvg", "mask_ratio", "GEPC", "lr", "batch_size", "epochs", "dropout")},
                                "run_log": SOURCES["original_run_log"].read_text().strip(), "test_log": SOURCES["original_test_log"].read_text().strip()},
        "sources": {k: {"path": str(p), "sha256": sha(p)} for k, p in SOURCES.items() if k != "original_checkpoint"},
    }
    (A / "eval/original_model_tables_2026-10-18.json").write_text(json.dumps(out, indent=1, default=float))
    shutil.copy2(SOURCES["head_ablation"], A / "eval/original_model_head_ablation_2026-05-26.csv")
    shutil.copy2(SOURCES["age_reconciliation"], A / "eval/original_model_age_reconciliation_2026-09-30.csv")
    print(f"original checkpoint: {n_params:,} parameters, {len(layers)} layers, width {args['layer_size']}; ablation rows {len(abl)}; age rows {len(age_rows)}")


if __name__ == "__main__":
    main()
