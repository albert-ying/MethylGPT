#!/usr/bin/env python3
"""Training-selected joint Cox models on empirical Generation Scotland inputs.

Copy of the locked compare_joint_cox_2026-09-28.py (publication checkout, sha256 recorded in
inputs) for MethylGPT-2 features. The only change is that the "embedding" representation
reads <embedding-dir>/COMMON_SAMPLE_SPLIT_<split>_<split>_<embedding-tag>.embedding.npz.

The original test split is never loaded. The original validation split is
development evidence, because it has informed previous MethylGPT experiments.
Raw CpGs and frozen MethylGPT embeddings share the same subject split, objective,
unpenalized age/sex covariates and selection procedure. Each run records its grid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import numpy as np
import pandas as pd
import scipy
import sklearn
import sksurv
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from threadpoolctl import threadpool_limits

PROJECT = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT")
CACHE = PROJECT / "task_a_private_training_20260629/_frame_cache"
LABELS = PROJECT / "revision_server_20260622/server_results/task_a/results/regenerated_survival_split_label_tables"
SPLIT = "full18859_common_task_a_incident_positive_duration_20260629"
ENDPOINTS = [
    "Autoimmune_Diseases", "Cancers", "Cardiovascular_Diseases", "date_of_death",
    "Endocrine_and_Metabolic_Diseases", "Kidney_Diseases",
    "Neurological_Diseases", "Respiratory_Diseases",
]
SEED = 20260928
EMBEDDING_DIR = str(CACHE / "pooled_embeddings")
EMBEDDING_TAG = "cls_mean_all_probes_*"


def write_json(path: Path, value: object) -> None:
    try:
        with path.open("x") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
    except Exception as exc:
        raise RuntimeError(f"Cannot write analysis record {path}: {exc}") from exc


def fingerprint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Cannot proceed - required data file not found: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        stat = path.stat()
    except Exception as exc:
        raise RuntimeError(f"Cannot fingerprint empirical input {path}: {exc}") from exc
    return {"path": str(path), "bytes": stat.st_size,
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "sha256": digest.hexdigest()}


def save_inner_path(path: Path, **arrays) -> None:
    """Publish a complete, losslessly compressed archive without overwriting."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=".inner_path_", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)  # Atomic publication; an existing target fails.
    except Exception as exc:
        raise RuntimeError(f"Cannot save inner Cox path {path}: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def peak_process_memory_gib() -> float:
    units = 1024**3 if sys.platform == "darwin" else 1024**2
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / units


def unique_file(parent: Path, pattern: str) -> Path:
    matches = sorted(parent.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Cannot proceed - required data file not found: {parent / pattern}")
    if len(matches) != 1:
        raise RuntimeError(f"Ambiguous empirical input {parent / pattern}: {len(matches)} matches")
    return matches[0]


def load_features(representation: str, split: str, inputs: dict) -> tuple[np.ndarray, np.ndarray]:
    if split not in ("train", "val"):
        raise ValueError("Only train and validation features may be loaded by this experiment")
    if representation == "embedding":
        path = unique_file(Path(EMBEDDING_DIR), f"COMMON_SAMPLE_SPLIT_{SPLIT}_{split}_{EMBEDDING_TAG}.embedding.npz")
        with np.load(path, allow_pickle=False) as payload:
            ids = payload["dnam_id"].astype(str)
            x = payload["embedding"]
        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite frozen embeddings: {path}")
    else:
        path = unique_file(CACHE, f"COMMON_SAMPLE_SPLIT_{SPLIT}_{split}_all_gs_batches_all_probes_*.matrix.npz")
        label_path = path.with_name(path.name.replace(".matrix.npz", ".labels.parquet"))
        with np.load(path, allow_pickle=False) as payload:
            x = payload["matrix"]
        ids = pd.read_parquet(label_path, columns=["dnam_id"])["dnam_id"].astype(str).to_numpy()
        inputs[f"{split}_feature_row_order"] = fingerprint(label_path)
        finite = x[np.isfinite(x)]
        if finite.size == 0 or finite.min() < 0 or finite.max() > 1:
            raise ValueError(f"Raw methylation cache is not on the beta scale: {path}")
        del finite
        if representation.startswith("cpg_projection"):
            # A prespecified CpG-specific readout of the pretrained gene encoder.
            # No outcome or fitted sample statistic enters this projection.
            import torch
            pretrain = PROJECT / "dev_pretraining_test-dataset_CpGs_type3-preprocessing_False-Jan07-04-50"
            checkpoint_path = pretrain / "best_model_epoch6.pt"
            vocab_path = pretrain / "vocab.json"
            probes_path = PROJECT / "processed_dataset/probe_ids_type3.csv"
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            vocab = json.loads(vocab_path.read_text())
            probes = pd.read_csv(probes_path)["illumina_probe_id"].astype(str).tolist()
            indices = np.asarray([vocab[probe] for probe in probes], dtype=int)
            if len(probes) != x.shape[1] or len(set(probes)) != len(probes):
                raise ValueError("CpG projection vocabulary does not match the matrix columns")
            weights = state["encoder.embedding.weight"][indices]
            weights = torch.nn.functional.layer_norm(weights, (weights.shape[1],),
                       state["encoder.enc_norm.weight"], state["encoder.enc_norm.bias"], eps=1e-5).numpy()
            if representation == "cpg_projection_permuted":
                weights = weights[np.random.default_rng(SEED).permutation(len(weights))]
            # Missing structural CpGs contribute zero; this is a fixed mask,
            # not an estimate using validation observations or outcome labels.
            finite_mask = np.isfinite(x)
            observed_per_row = finite_mask.sum(axis=1)
            if (observed_per_row == 0).any():
                raise ValueError("CpG projection has samples with no observed CpGs")
            x = np.where(finite_mask, x, 0.0) @ weights
            x = (x / np.sqrt(observed_per_row[:, None])).astype(np.float32)
            inputs["projection_checkpoint"] = fingerprint(checkpoint_path)
            inputs["projection_vocabulary"] = fingerprint(vocab_path)
            inputs["projection_probe_order"] = fingerprint(probes_path)
            logging.info("Projected beta values through %s pretrained CpG coordinates", x.shape[1])
    inputs[f"{split}_features"] = fingerprint(path)
    if x.shape[0] != len(ids) or len(np.unique(ids)) != len(ids):
        raise ValueError("Feature row count or uniqueness check failed")
    logging.info("Loaded %s %s: %s", representation, split, x.shape)
    return ids, x


def load_labels(endpoint: str, split: str, ids: np.ndarray, inputs: dict) -> pd.DataFrame:
    if split not in ("train", "val"):
        raise ValueError("Test labels are excluded from this experiment")
    path = LABELS / f"survial_{endpoint}_{SPLIT}_split_labels.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Cannot proceed - required data file not found: {path}")
    frame = pd.read_parquet(path, filters=[("split", "==", split)])
    frame["dnam_id"] = frame["dnam_id"].astype(str)
    if frame["dnam_id"].duplicated().any() or set(frame["dnam_id"]) != set(ids):
        raise ValueError(f"Label-feature identity mismatch: {endpoint} {split}")
    frame = frame.set_index("dnam_id").loc[ids]
    mapped_sex = frame["sex"].map({"F": 1.0, "M": 0.0, "Female": 1.0, "Male": 0.0,
                                   "female": 1.0, "male": 0.0})
    frame["sex"] = mapped_sex.where(mapped_sex.notna(), pd.to_numeric(frame["sex"], errors="coerce"))
    for column in ["age", "sex", "duration", "event"]:
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite {column}: {endpoint} {split}")
    if (frame["duration"] <= 0).any() or not frame["event"].isin([0, 1]).all():
        raise ValueError(f"Invalid survival labels: {endpoint} {split}")
    if not frame["sex"].isin([0, 1]).all():
        raise ValueError(f"Unexpected sex coding: {endpoint} {split}")
    if f"labels_{endpoint}" not in inputs:
        inputs[f"labels_{endpoint}"] = fingerprint(path)
    return frame


class TrainTransform:
    """Fit imputation and population standardization only on fitting rows."""

    def fit(self, raw: np.ndarray, covariates: np.ndarray) -> "TrainTransform":
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
            self.mean = np.nanmean(raw, axis=0, dtype=np.float64)
            self.scale = np.nanstd(raw, axis=0, dtype=np.float64)
        self.mean = np.nan_to_num(self.mean, nan=0.0)
        self.scale = np.where(np.isfinite(self.scale) & (self.scale > 1e-8), self.scale, 1.0)
        self.cov_mean = covariates.mean(axis=0)
        self.cov_scale = covariates.std(axis=0)
        self.cov_scale[self.cov_scale < 1e-8] = 1.0
        return self

    def transform(self, raw: np.ndarray, covariates: np.ndarray) -> np.ndarray:
        n_covariates = len(self.cov_mean)
        out = np.empty((len(raw), raw.shape[1] + n_covariates), dtype=np.float64, order="F")
        out[:, :n_covariates] = (covariates - self.cov_mean) / self.cov_scale
        for start in range(0, raw.shape[1], 1024):
            stop = min(start + 1024, raw.shape[1])
            values = raw[:, start:stop].astype(np.float64)
            values = np.where(np.isfinite(values), values, self.mean[start:stop])
            out[:, start + n_covariates:stop + n_covariates] = (values - self.mean[start:stop]) / self.scale[start:stop]
        if not np.isfinite(out).all():
            raise ValueError("Preprocessing left non-finite features")
        return out


def target(frame: pd.DataFrame) -> np.ndarray:
    return Surv.from_arrays(frame["event"].astype(bool).to_numpy(), frame["duration"].to_numpy(dtype=float))


def cindex(frame: pd.DataFrame, risk: np.ndarray) -> float:
    if not np.isfinite(risk).all():
        raise ValueError("Non-finite risk predictions")
    return float(concordance_index_censored(frame["event"].astype(bool), frame["duration"], risk)[0])


def fit_path(x: np.ndarray, y: np.ndarray, args: argparse.Namespace, n_unpenalized: int = 2) -> CoxnetSurvivalAnalysis:
    started = time.perf_counter()
    logging.info("Starting Cox path: rows=%d columns=%d alphas=%d peak_process_memory=%.3f GiB",
                 len(x), x.shape[1], args.n_alphas, peak_process_memory_gib())
    penalty = np.ones(x.shape[1])
    penalty[:n_unpenalized] = 0.0
    model = CoxnetSurvivalAnalysis(n_alphas=args.n_alphas, alpha_min_ratio=args.alpha_min_ratio,
                                 l1_ratio=args.l1_ratio, penalty_factor=penalty,
                                 normalize=False, tol=1e-7, max_iter=args.max_iterations)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        model.fit(x, y)
    for warning in captured:
        logging.warning("Cox fit warning: %s", warning.message)
        if issubclass(warning.category, ConvergenceWarning):
            raise RuntimeError(f"Unconverged Cox path: {warning.message}")
    if len(model.alphas_) != args.n_alphas:
        raise RuntimeError(f"Cox path returned {len(model.alphas_)} of {args.n_alphas} requested alpha values")
    if not np.isfinite(model.coef_).all():
        raise ValueError("Non-finite Cox coefficients")
    logging.info("Completed Cox path: %.3f seconds; peak_process_memory=%.3f GiB",
                 time.perf_counter() - started, peak_process_memory_gib())
    return model


def covariate_basis(fit_cov: np.ndarray, eval_cov: np.ndarray, adjustment: str):
    """Four-knot restricted cubic age spline plus additive sex.

    Quantiles and norm=2 follow Hmisc rcspline.eval: the outer 5th/95th
    percentiles and equally spaced inner quantiles. Knots use fitting ages.
    """
    if adjustment == "linear":
        return fit_cov, eval_cov, None
    knots = np.quantile(fit_cov[:, 0], [0.05, 0.35, 0.65, 0.95])
    if (np.diff(knots) <= 0).any():
        raise ValueError("Restricted cubic age spline requires four distinct training quantile knots")
    def expand(cov):
        age = cov[:, 0]
        high, previous = knots[-1], knots[-2]
        columns = [age]
        for knot in knots[:-2]:
            term = (np.maximum(age - knot, 0.0) ** 3
                    - (high - knot) / (high - previous) * np.maximum(age - previous, 0.0) ** 3
                    + (previous - knot) / (high - previous) * np.maximum(age - high, 0.0) ** 3)
            columns.append(term / (high - knots[0]) ** 2)
        return np.column_stack((*columns, cov[:, 1]))
    return expand(fit_cov), expand(eval_cov), knots


def fit_baseline(x: np.ndarray, y: np.ndarray, alpha: float = 0.0) -> CoxPHSurvivalAnalysis:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        for category in (RuntimeWarning, ConvergenceWarning, scipy.linalg.LinAlgWarning):
            warnings.filterwarnings("error", category=category)
        model = CoxPHSurvivalAnalysis(alpha=alpha, ties="breslow", n_iter=200).fit(x, y)
    for warning in captured:
        logging.warning("Baseline Cox fit warning: %s", warning.message)
        if issubclass(warning.category, (ConvergenceWarning, scipy.linalg.LinAlgWarning)):
            raise RuntimeError(f"Unreliable baseline Cox fit: {warning.message}")
    if not np.isfinite(model.coef_).all():
        raise ValueError("Non-finite baseline Cox coefficients")
    return model


def concordance_counts(frame: pd.DataFrame, risk: np.ndarray) -> dict:
    score, concordant, discordant, tied_risk, tied_time = concordance_index_censored(
        frame["event"].astype(bool), frame["duration"], risk)
    return {"c_index": float(score), "concordant": int(concordant),
            "discordant": int(discordant), "tied_risk": int(tied_risk), "tied_time": int(tied_time)}


def nested_development(args, ids, raw, train_labels, cov, inputs) -> None:
    """Nested development evaluation confined to the original training subjects.

    Each outer fold has a separate risk scale. Only within-fold comparable pairs
    contribute to the combined C-index; cross-fold risk comparisons are excluded.
    """
    adjustments = ["linear", "spline"] if args.age_adjustment == "both" else [args.age_adjustment]
    write_json(args.output / "inputs_before_nested_fit.json", inputs)
    save_inner = getattr(args, "save_inner_artifacts", False)
    input_manifest = fingerprint(args.output / "inputs_before_nested_fit.json")
    producer = fingerprint(Path(__file__))
    protocol_manifest = fingerprint(args.output / "protocol_before_fit.json")
    protocol = json.loads((args.output / "protocol_before_fit.json").read_text())
    if protocol["source"]["sha256"] != producer["sha256"]:
        raise RuntimeError("Runner source changed between protocol creation and nested fitting")
    if save_inner and args.representation == "raw_cpg" and raw.shape != (9866, 49156):
        raise ValueError("This raw-CpG benchmark requires the documented 9,866 training subjects and 49,156 CpGs")
    score_rows, selection_rows, cv_rows, predictions = [], [], [], []
    resumed = set()
    if args.resume_from is not None:
        previous_protocol = json.loads((args.resume_from / "protocol_before_fit.json").read_text())
        previous_adjustments = (["linear", "spline"] if previous_protocol["age_adjustment_if_nested"] == "both"
                                else [previous_protocol["age_adjustment_if_nested"]])
        if (previous_protocol["representation"] != args.representation
                or previous_protocol["folds"] != args.folds
                or previous_protocol["outer_folds_if_nested"] != args.outer_folds
                or not set(adjustments).issubset(previous_adjustments)
                or previous_protocol["alpha_path"]["count"] != args.n_alphas
                or previous_protocol["alpha_path"]["min_to_max_ratio"] != args.alpha_min_ratio
                or previous_protocol["alpha_path"]["l1_ratio"] != args.l1_ratio
                or not previous_protocol["spline_if_used"].startswith("Restricted cubic age spline")):
            raise ValueError("Resume source has a different statistical protocol")
        old_inputs = json.loads((args.resume_from / "inputs_before_nested_fit.json").read_text())
        if inputs != old_inputs:
            raise ValueError("Empirical inputs differ from the resume source")
        old_scores = pd.read_csv(args.resume_from / "outer_fold_results.csv")
        old_cv = pd.read_csv(args.resume_from / "inner_cv_results.csv")
        copied_sources = []
        for endpoint, frame in train_labels.items():
            endpoint_rows = old_scores[(old_scores.endpoint == endpoint) & old_scores.age_adjustment.isin(adjustments)]
            expected = {(adjustment, fold) for adjustment in adjustments for fold in range(1, args.outer_folds + 1)}
            observed = set(zip(endpoint_rows.age_adjustment, endpoint_rows.outer_fold))
            if observed != expected or len(endpoint_rows) != len(expected):
                continue
            endpoint_cv = old_cv[(old_cv.endpoint == endpoint) & old_cv.age_adjustment.isin(adjustments)]
            if len(endpoint_cv) != len(expected) * args.folds * args.n_alphas:
                raise ValueError(f"Incomplete inner selection history in resume endpoint {endpoint}")
            endpoint_predictions, endpoint_selections = [], []
            for adjustment, fold in sorted(expected):
                tag = f"{endpoint}_{adjustment}_outer{fold}"
                pred_path = args.resume_from / f"{tag}_private_predictions.parquet"
                selection_path = args.resume_from / f"{tag}_selection_before_evaluation.json"
                model_path = args.resume_from / f"{tag}_model.npz"
                pred = pd.read_parquet(pred_path)
                matching = frame.loc[pred.dnam_id]
                for column in ["duration", "event", "age", "sex"]:
                    np.testing.assert_allclose(pred[column], matching[column], atol=0, rtol=0)
                endpoint_predictions.append(pred)
                endpoint_selections.append(json.loads(selection_path.read_text()))
                for source in [pred_path, selection_path, model_path]:
                    if (args.output / source.name).exists():
                        raise FileExistsError(args.output / source.name)
                    shutil.copy2(source, args.output / source.name)
                    copied_sources.append(fingerprint(source))
            combined = pd.concat(endpoint_predictions)
            for adjustment in adjustments:
                part = combined[combined.age_adjustment == adjustment]
                if len(part) != len(ids) or set(part.dnam_id) != set(ids) or part.dnam_id.duplicated().any():
                    raise ValueError("Resume predictions do not cover each training subject once")
            predictions.extend(endpoint_predictions)
            selection_rows.extend(endpoint_selections)
            score_rows.extend(endpoint_rows.to_dict("records"))
            cv_rows.extend(endpoint_cv.to_dict("records"))
            resumed.add(endpoint)
            logging.info("Resumed fully evaluated endpoint %s from %s", endpoint, args.resume_from)
        write_json(args.output / "resumed_source_manifest.json", {
            "source_protocol": fingerprint(args.resume_from / "protocol_before_fit.json"),
            "source_outer_results": fingerprint(args.resume_from / "outer_fold_results.csv"),
            "source_inner_results": fingerprint(args.resume_from / "inner_cv_results.csv"),
            "completed_endpoints_reused": sorted(resumed), "copied_files": copied_sources,
            "numerical_change": "Iteration budget may increase and baseline warning checks may be stricter; convergence tolerance and statistical protocol for each retained adjustment are unchanged. Only complete requested endpoint/age-adjustment combinations are reused. Omitted adjustment branches remain in the source outputs and are not included as completed evidence.",
        })
    with threadpool_limits(limits=getattr(args, "threads", 8)):
        for endpoint, frame in train_labels.items():
            if endpoint in resumed:
                continue
            y = target(frame)
            outer = StratifiedKFold(n_splits=args.outer_folds, shuffle=True, random_state=SEED + 1000)
            for outer_fold, (train_idx, held_idx) in enumerate(outer.split(ids, frame["event"]), 1):
                if np.intersect1d(ids[train_idx], ids[held_idx]).size:
                    raise ValueError("Outer training and evaluation subjects overlap")
                for adjustment in adjustments:
                    tag = f"{endpoint}_{adjustment}_outer{outer_fold}"
                    logging.info("Nested %s %s", args.representation, tag)
                    inner = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED + outer_fold)
                    fold_scores = []
                    for inner_fold, (local_fit, local_eval) in enumerate(inner.split(train_idx, frame.iloc[train_idx]["event"]), 1):
                        fit_idx, eval_idx = train_idx[local_fit], train_idx[local_eval]
                        fit_cov, eval_cov, inner_knots = covariate_basis(cov[fit_idx], cov[eval_idx], adjustment)
                        n_cov = fit_cov.shape[1]
                        transform = TrainTransform().fit(raw[fit_idx], fit_cov)
                        fit_x = transform.transform(raw[fit_idx], fit_cov)
                        eval_x = transform.transform(raw[eval_idx], eval_cov)
                        fitted = fit_path(fit_x, y[fit_idx], args, n_cov)
                        if save_inner:
                            save_inner_path(args.output / f"{tag}_inner{inner_fold}_path.npz",
                                endpoint=np.array(endpoint), age_adjustment=np.array(adjustment),
                                outer_fold=np.array(outer_fold), inner_fold=np.array(inner_fold),
                                fit_ids=ids[fit_idx].astype(str), evaluation_ids=ids[eval_idx].astype(str),
                                coefficient=fitted.coef_, offset=fitted.offset_, alphas=fitted.alphas_,
                                alpha_fractions=fitted.alphas_ / fitted.alphas_[0], n_covariates=np.array(n_cov),
                                mean=transform.mean, scale=transform.scale,
                                cov_mean=transform.cov_mean, cov_scale=transform.cov_scale,
                                spline_knots=np.array([]) if inner_knots is None else inner_knots,
                                inputs_manifest_sha256=np.array(input_manifest["sha256"]),
                                source_sha256=np.array(producer["sha256"]),
                                created_utc=np.array(datetime.now(timezone.utc).isoformat()))
                        scores = []
                        for index, alpha in enumerate(fitted.alphas_):
                            score = cindex(frame.iloc[eval_idx], fitted.predict(eval_x, alpha=alpha))
                            scores.append(score)
                            cv_rows.append({"endpoint": endpoint, "age_adjustment": adjustment,
                                "outer_fold": outer_fold, "inner_fold": inner_fold, "alpha_index": index,
                                "alpha": float(alpha), "alpha_fraction": float(alpha / fitted.alphas_[0]),
                                "c_index": score, "n_fit": len(fit_idx), "n_eval": len(eval_idx),
                                "eval_events": int(frame.iloc[eval_idx]["event"].sum())})
                        fold_scores.append(scores)
                    means = np.asarray(fold_scores).mean(axis=0)
                    selected = int(np.argmax(means))
                    selection = {"endpoint": endpoint, "age_adjustment": adjustment, "outer_fold": outer_fold,
                        "selected_alpha_index": selected, "inner_mean_c_index": float(means[selected]),
                        "alpha_fraction": float(np.geomspace(1, args.alpha_min_ratio, args.n_alphas)[selected]),
                        "strongest_penalty_selected": selected == 0,
                        "weakest_penalty_selected": selected == args.n_alphas - 1,
                        "outer_evaluation_used_for_selection": False,
                        "created_utc": datetime.now(timezone.utc).isoformat()}
                    write_json(args.output / f"{tag}_selection_before_evaluation.json", selection)
                    selection_rows.append(selection)
                    fit_cov, held_cov, spline = covariate_basis(cov[train_idx], cov[held_idx], adjustment)
                    n_cov = fit_cov.shape[1]
                    transform = TrainTransform().fit(raw[train_idx], fit_cov)
                    fit_x = transform.transform(raw[train_idx], fit_cov)
                    held_x = transform.transform(raw[held_idx], held_cov)
                    fitted = fit_path(fit_x, y[train_idx], args, n_cov)
                    alpha = float(fitted.alphas_[selected])
                    baseline = fit_baseline(fit_x[:, :n_cov], y[train_idx])
                    risk = fitted.predict(held_x, alpha=alpha)
                    base_risk = baseline.predict(held_x[:, :n_cov])
                    counts = concordance_counts(frame.iloc[held_idx], risk)
                    base_counts = concordance_counts(frame.iloc[held_idx], base_risk)
                    row = {"endpoint": endpoint, "representation": args.representation,
                        "age_adjustment": adjustment, "outer_fold": outer_fold,
                        "n_fit": len(train_idx), "n_eval": len(held_idx),
                        "eval_events": int(frame.iloc[held_idx]["event"].sum()), "selected_alpha": alpha,
                        "selected_alpha_index": selected, "nonzero_methylation_coefficients": int(np.count_nonzero(fitted.coef_[n_cov:, selected])),
                        **counts, **{f"baseline_{key}": value for key, value in base_counts.items()}}
                    score_rows.append(row)
                    pred = frame.iloc[held_idx][["duration", "event", "age", "sex"]].copy()
                    pred["endpoint"], pred["age_adjustment"], pred["outer_fold"] = endpoint, adjustment, outer_fold
                    pred["risk"], pred["baseline_risk"] = risk, base_risk
                    pred = pred.reset_index()
                    pred.to_parquet(args.output / f"{tag}_private_predictions.parquet", index=False)
                    predictions.append(pred)
                    np.savez_compressed(args.output / f"{tag}_model.npz",
                        coefficient=fitted.coef_[:, selected], offset=fitted.offset_[selected],
                        baseline_coefficient=baseline.coef_, alpha=alpha,
                        mean=transform.mean, scale=transform.scale,
                        cov_mean=transform.cov_mean, cov_scale=transform.cov_scale,
                        spline_knots=np.array([]) if spline is None else spline,
                        fit_ids=ids[train_idx].astype(str) if save_inner else ids[train_idx])
                    logging.info("Nested %s C %.6f baseline %.6f delta %+.6f", tag,
                                 counts["c_index"], base_counts["c_index"], counts["c_index"] - base_counts["c_index"])
                    logging.info("Completed outer fit %s; peak_process_memory=%.3f GiB", tag, peak_process_memory_gib())
                    pd.DataFrame(score_rows).to_csv(args.output / "outer_fold_results.csv", index=False)
                    pd.DataFrame(cv_rows).to_csv(args.output / "inner_cv_results.csv", index=False)
    all_predictions = pd.concat(predictions, ignore_index=True)
    if all_predictions.duplicated(["endpoint", "age_adjustment", "dnam_id"]).any():
        raise ValueError("A subject was evaluated more than once in an endpoint/adjustment")
    all_predictions.to_parquet(args.output / "private_nested_predictions.parquet", index=False)
    pd.DataFrame(selection_rows).to_csv(args.output / "selected_penalties.csv", index=False)
    result_rows = []
    for (endpoint, adjustment), group in pd.DataFrame(score_rows).groupby(["endpoint", "age_adjustment"], sort=False):
        denominator = (group["concordant"] + group["discordant"] + group["tied_risk"]).sum()
        base_denominator = (group["baseline_concordant"] + group["baseline_discordant"] + group["baseline_tied_risk"]).sum()
        if denominator != base_denominator:
            raise ValueError("Paired models have different comparable-pair denominators")
        combined = (group["concordant"].sum() + 0.5 * group["tied_risk"].sum()) / denominator
        base_combined = (group["baseline_concordant"].sum() + 0.5 * group["baseline_tied_risk"].sum()) / denominator
        result_rows.append({"endpoint": endpoint, "representation": args.representation,
            "age_adjustment": adjustment, "within_fold_pairs_c_index": combined,
            "baseline_within_fold_pairs_c_index": base_combined,
            "increment_over_baseline": combined - base_combined,
            "mean_fold_c_index": group["c_index"].mean(), "n_evaluated": int(group["n_eval"].sum()),
            "events_evaluated": int(group["eval_events"].sum()), "comparable_pairs": int(denominator),
            "evaluation_scope": "nested_development_original_train_only"})
    pd.DataFrame(result_rows).to_csv(args.output / "nested_development_results.csv", index=False)
    verified_artifacts = {}
    if save_inner:
        expected_outer = {(endpoint, adjustment, outer) for endpoint in train_labels for adjustment in adjustments
                          for outer in range(1, args.outer_folds + 1)}
        observed_outer = [(r["endpoint"], r["age_adjustment"], r["outer_fold"]) for r in score_rows]
        selected_outer = [(r["endpoint"], r["age_adjustment"], r["outer_fold"]) for r in selection_rows]
        expected_cv = {(*key, inner, alpha) for key in expected_outer for inner in range(1, args.folds + 1)
                       for alpha in range(args.n_alphas)}
        observed_cv = [(r["endpoint"], r["age_adjustment"], r["outer_fold"], r["inner_fold"], r["alpha_index"])
                       for r in cv_rows]
        if (set(observed_outer) != expected_outer or len(observed_outer) != len(expected_outer)
                or set(selected_outer) != expected_outer or len(selected_outer) != len(expected_outer)
                or set(observed_cv) != expected_cv or len(observed_cv) != len(expected_cv)):
            raise ValueError("Nested shard has incomplete or duplicated fit/selection/CV coverage")
        expected_groups = {(endpoint, adjustment) for endpoint in train_labels for adjustment in adjustments}
        if set(all_predictions.groupby(["endpoint", "age_adjustment"]).groups) != expected_groups:
            raise ValueError("Nested shard prediction groups differ from its endpoint protocol")
        for _, group in all_predictions.groupby(["endpoint", "age_adjustment"]):
            if len(group) != len(ids) or set(group.dnam_id) != set(ids):
                raise ValueError("Nested shard does not evaluate every subject exactly once per endpoint")
        names = ["protocol_before_fit.json", "inputs_before_nested_fit.json", "private_nested_predictions.parquet",
                 "outer_fold_results.csv", "inner_cv_results.csv", "selected_penalties.csv", "nested_development_results.csv"]
        for endpoint, adjustment, outer in sorted(expected_outer):
            tag = f"{endpoint}_{adjustment}_outer{outer}"
            names += [f"{tag}_{suffix}" for suffix in ("selection_before_evaluation.json", "model.npz", "private_predictions.parquet")]
            for inner in range(1, args.folds + 1):
                filename = f"{tag}_inner{inner}_path.npz"
                with np.load(args.output / filename, allow_pickle=False) as saved:
                    if (saved["coefficient"].shape != (raw.shape[1] + len(saved["cov_mean"]), args.n_alphas)
                            or saved["inputs_manifest_sha256"].item() != input_manifest["sha256"]
                            or saved["source_sha256"].item() != producer["sha256"]
                            or saved["endpoint"].item() != endpoint or saved["age_adjustment"].item() != adjustment
                            or saved["outer_fold"].item() != outer or saved["inner_fold"].item() != inner):
                        raise ValueError(f"Inner-path artifact identity differs: {filename}")
                names.append(filename)
        if fingerprint(Path(__file__))["sha256"] != producer["sha256"]:
            raise RuntimeError("Runner source changed during nested fitting")
        for manifest in (input_manifest, protocol_manifest):
            if fingerprint(Path(manifest["path"]))["sha256"] != manifest["sha256"]:
                raise RuntimeError(f"Run manifest changed during nested fitting: {manifest['path']}")
        for entry in inputs.values():
            if fingerprint(Path(entry["path"]))["sha256"] != entry["sha256"]:
                raise RuntimeError(f"Empirical source changed during nested fitting: {entry['path']}")
        verified_artifacts = {"coverage": {"endpoints": list(train_labels), "subjects_per_endpoint": len(ids),
                              "outer_fits": len(expected_outer), "inner_paths": len(expected_outer) * args.folds,
                              "inner_cv_scores": len(expected_cv), "prediction_rows": len(all_predictions)},
                              "source": producer, "source_and_input_hashes_unchanged": True,
                              "protocol": protocol_manifest, "inputs_manifest": input_manifest,
                              "protocol_and_input_manifest_hashes_unchanged": True,
                              "artifacts": [fingerprint(args.output / name) for name in names]}
    write_json(args.output / "completion.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs, "results": result_rows, "validation_loaded": False, "test_loaded": False,
        "simulated_data": False, "independent_confirmation": False, "verified_artifacts": verified_artifacts})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation", choices=["embedding", "raw_cpg", "cpg_projection", "cpg_projection_permuted"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", action="append", choices=ENDPOINTS)
    parser.add_argument("--n-alphas", type=int, default=12)
    parser.add_argument("--alpha-min-ratio", type=float, default=0.05)
    parser.add_argument("--l1-ratio", type=float, default=0.5)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=100000)
    parser.add_argument("--nested-development", action="store_true")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--age-adjustment", choices=["linear", "spline", "both"], default="both")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--save-inner-artifacts", action="store_true")
    parser.add_argument("--embedding-dir", default=None)
    parser.add_argument("--embedding-tag", default=None)
    args = parser.parse_args()
    global EMBEDDING_DIR, EMBEDDING_TAG
    if args.embedding_dir:
        EMBEDDING_DIR = args.embedding_dir
    if args.embedding_tag:
        EMBEDDING_TAG = args.embedding_tag
    if args.threads < 1:
        raise ValueError("Numerical thread limit must be positive")
    if args.endpoint is not None and len(args.endpoint) != len(set(args.endpoint)):
        raise ValueError("Endpoint subsets must not contain duplicates")
    if args.save_inner_artifacts and (not args.nested_development or args.resume_from is not None):
        raise ValueError("Inner artifacts require a new nested run; resume is unsupported for this artifact mode")
    if args.resume_from is not None and not args.nested_development:
        raise ValueError("Resume is supported only for nested development")
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.output / "execution.log")])
    endpoints = args.endpoint or ENDPOINTS
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "empirical": True,
        "purpose": "Publication-oriented model development; no independent confirmation claim",
        "representation": args.representation, "endpoints": endpoints, "seed": SEED,
        "projection_if_used": "Beta-weighted sum of LayerNorm-transformed pretrained CpG-token embeddings, divided by sqrt(observed CpGs); missing CpGs contribute zero. Fixed outcome-independent transformation, distinct from contextual sample embeddings.",
        "permuted_projection_control": "Same pretrained token vectors with CpG-row assignment permuted once with seed 20260928; randomized model control, no simulated observations.",
        "selection": "Stratified training-only CV; maximum mean C-index, ties favor stronger penalty",
        "alpha_path": {"count": args.n_alphas, "min_to_max_ratio": args.alpha_min_ratio,
                       "fold_specific_alpha_max": True, "l1_ratio": args.l1_ratio},
        "folds": args.folds, "age_sex_penalized": False,
        "coxnet_max_iterations": args.max_iterations, "coxnet_tolerance": 1e-7,
        "numerical_threads": args.threads, "inner_artifacts_saved": args.save_inner_artifacts,
        "resume_from": None if args.resume_from is None else str(args.resume_from.resolve()),
        "sex_encoding": "Female/F=1, Male/M=0, matching the source training driver",
        "preprocessing": "Fitting-fold means and observed-value population SD before mean imputation; all-missing features become zero",
        "validation": "Not loaded" if args.nested_development else "Previously used validation split; development evidence only",
        "nested_development": args.nested_development,
        "outer_folds_if_nested": args.outer_folds,
        "age_adjustment_if_nested": args.age_adjustment,
        "spline_if_used": "Restricted cubic age spline with four knots at fitting-fold quantiles .05/.35/.65/.95; three age terms, norm=2 normalization, linear tails; additive sex; all four covariates unpenalized",
        "spline_reference": "https://search.r-project.org/CRAN/refmans/Hmisc/html/rcspline.eval.html",
        "nested_numerical_amendment": "Initial 5-df B-spline runs aborted at cardiovascular inner fits in all three representations. Coxnet failed even at stronger penalties and with QR orthogonalization. Independent baseline Newton fits were finite. A 3-df restricted cubic age basis passed all three inner fits in the failing outer fold. No held-out performance was used to choose this numerical repair; original incomplete runs are preserved.",
        "nested_aggregation": "Concordant/comparable counts summed within outer folds only; no cross-fold risk comparisons",
        "nested_limitations": "Development after model-family exploration; random subject folds, no family identifiers in source labels; external confirmation still needed",
        "test_features_loaded": False, "test_labels_loaded": False,
        "all_endpoints_reported": set(endpoints) == set(ENDPOINTS), "simulated_data": False,
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__,
                     "sklearn": sklearn.__version__, "sksurv": sksurv.__version__},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "command": sys.argv, "source": fingerprint(Path(__file__)),
    }
    write_json(args.output / "protocol_before_fit.json", protocol)
    inputs: dict = {}
    ids, raw = load_features(args.representation, "train", inputs)
    train_labels = {endpoint: load_labels(endpoint, "train", ids, inputs) for endpoint in endpoints}
    cov = train_labels[endpoints[0]][["age", "sex"]].to_numpy(dtype=float)
    for frame in train_labels.values():
        np.testing.assert_array_equal(cov, frame[["age", "sex"]].to_numpy(dtype=float))
    if args.nested_development:
        nested_development(args, ids, raw, train_labels, cov, inputs)
        return
    selection_rows, cv_rows, models, baselines = [], [], {}, {}
    with threadpool_limits(limits=args.threads):
        for endpoint in endpoints:
            frame = train_labels[endpoint]
            y = target(frame)
            scores = []
            folds = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
            for fold, (fit_idx, eval_idx) in enumerate(folds.split(ids, frame["event"]), 1):
                logging.info("Fitting %s fold %s/%s", endpoint, fold, args.folds)
                transform = TrainTransform().fit(raw[fit_idx], cov[fit_idx])
                fit_x = transform.transform(raw[fit_idx], cov[fit_idx])
                eval_x = transform.transform(raw[eval_idx], cov[eval_idx])
                fitted = fit_path(fit_x, y[fit_idx], args)
                base = fit_baseline(fit_x[:, :2], y[fit_idx])
                base_c = cindex(frame.iloc[eval_idx], base.predict(eval_x[:, :2]))
                fold_scores = []
                for index, alpha in enumerate(fitted.alphas_):
                    score = cindex(frame.iloc[eval_idx], fitted.predict(eval_x, alpha=alpha))
                    fold_scores.append(score)
                    cv_rows.append({"endpoint": endpoint, "fold": fold, "alpha_index": index,
                                    "alpha": float(alpha), "alpha_fraction": float(alpha / fitted.alphas_[0]),
                                    "c_index": score, "age_sex_c_index": base_c,
                                    "n_fit": len(fit_idx), "n_eval": len(eval_idx),
                                    "eval_events": int(frame.iloc[eval_idx]["event"].sum())})
                scores.append(fold_scores)
                del fit_x, eval_x, transform, fitted, base
            means = np.asarray(scores).mean(axis=0)
            selected = int(np.argmax(means))
            selection_rows.append({"endpoint": endpoint, "selected_alpha_index": selected,
                                   "alpha_fraction": float(np.geomspace(1, args.alpha_min_ratio, args.n_alphas)[selected]),
                                   "mean_training_cv_c_index": float(means[selected])})
            logging.info("Selected %s alpha index %s; training CV C-index %.6f", endpoint, selected, means[selected])
        pd.DataFrame(cv_rows).to_csv(args.output / "all_training_cv_results.csv", index=False)
        write_json(args.output / "selection_lock_before_validation.json", {
            "created_utc": datetime.now(timezone.utc).isoformat(), "selected": selection_rows,
            "validation_labels_loaded": False, "validation_features_loaded": False,
            "test_labels_loaded": False, "test_features_loaded": False,
            "source_inputs": inputs,
        })
        transform = TrainTransform().fit(raw, cov)
        fit_x = transform.transform(raw, cov)
        for row in selection_rows:
            endpoint = row["endpoint"]
            fitted = fit_path(fit_x, target(train_labels[endpoint]), args)
            idx = row["selected_alpha_index"]
            models[endpoint] = (fitted, float(fitted.alphas_[idx]))
            baselines[endpoint] = fit_baseline(fit_x[:, :2], target(train_labels[endpoint]))
        validation_ids, validation_raw = load_features(args.representation, "val", inputs)
        if set(ids) & set(validation_ids):
            raise ValueError("Training and validation IDs overlap")
        validation_frames = {endpoint: load_labels(endpoint, "val", validation_ids, inputs) for endpoint in endpoints}
        val_cov = validation_frames[endpoints[0]][["age", "sex"]].to_numpy(dtype=float)
        for frame in validation_frames.values():
            np.testing.assert_array_equal(val_cov, frame[["age", "sex"]].to_numpy(dtype=float))
        val_x = transform.transform(validation_raw, val_cov)
        result_rows, predictions = [], []
        for row in selection_rows:
            endpoint = row["endpoint"]
            fitted, alpha = models[endpoint]
            frame = validation_frames[endpoint]
            risk = fitted.predict(val_x, alpha=alpha)
            baseline = baselines[endpoint].predict(val_x[:, :2])
            score, base_score = cindex(frame, risk), cindex(frame, baseline)
            result_rows.append({**row, "representation": args.representation, "n_train": len(ids),
                                "train_events": int(train_labels[endpoint]["event"].sum()),
                                "n_validation": len(frame), "validation_events": int(frame["event"].sum()),
                                "validation_c_index": score, "age_sex_validation_c_index": base_score,
                                "increment_over_age_sex": score - base_score,
                                "selected_alpha": alpha,
                                "nonzero_methylation_coefficients": int(np.count_nonzero(fitted.coef_[2:, row["selected_alpha_index"]])),
                                "evaluation_scope": "previously_used_validation_development_only"})
            pred = frame[["duration", "event", "age", "sex"]].copy()
            pred["endpoint"], pred["risk"], pred["age_sex_risk"] = endpoint, risk, baseline
            predictions.append(pred.reset_index())
            logging.info("Development validation %s: joint %.6f, age+sex %.6f, increment %+.6f", endpoint, score, base_score, score - base_score)
        pd.DataFrame(result_rows).to_csv(args.output / "development_validation_results.csv", index=False)
        pd.concat(predictions, ignore_index=True).to_parquet(args.output / "private_validation_predictions.parquet", index=False)
        np.savez_compressed(args.output / "train_fitted_preprocessing.npz", mean=transform.mean, scale=transform.scale,
                            cov_mean=transform.cov_mean, cov_scale=transform.cov_scale)
        np.savez_compressed(args.output / "selected_coefficients.npz",
                            **{endpoint: model.coef_[:, selection_rows[i]["selected_alpha_index"]]
                               for i, (endpoint, (model, _)) in enumerate(models.items())})
        write_json(args.output / "completion.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
                   "inputs": inputs, "selection_lock": fingerprint(args.output / "selection_lock_before_validation.json"),
                   "test_features_loaded": False, "test_labels_loaded": False, "simulated_data": False,
                   "independent_confirmation": False, "rows": result_rows})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Joint Cox development experiment failed; incomplete outputs must not support claims")
        raise
