"""Finetuning evaluation metrics for MethylGPT downstream tasks.

Provides metric computation functions for regression (age prediction),
binary classification (disease prediction), and multi-label classification.
"""

import logging

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def elasticnet_metric(labels, preds):
    """Compute regression metrics for ElasticNet-style predictions.

    Args:
        labels: Ground truth values.
        preds: Predicted values.

    Returns:
        Dictionary with r2, rmse, mae, pearson_r, spearman_r, and medae.
    """
    R2 = r2_score(labels, preds)
    rmse = np.sqrt(mean_squared_error(labels, preds))
    mae = mean_absolute_error(labels, preds)
    medae = median_absolute_error(labels, preds)

    try:
        pearson_r = pearsonr(labels, preds)[0]
    except Exception:
        pearson_r = -1e-9
    try:
        sp_cor = spearmanr(labels, preds)[0]
    except Exception:
        sp_cor = -1e-9

    return {
        "r2": R2,
        "rmse": rmse,
        "mae": mae,
        "p_r": pearson_r,
        "s_r": sp_cor,
        "medae": medae,
    }


def regression_metric(validation_step_outputs):
    """Compute regression metrics from validation step outputs.

    Args:
        validation_step_outputs: List of dicts with 'pred_age' and 'label' tensors.

    Returns:
        Dictionary with r2, rmse, mae, pearson_r, spearman_r, and medae.
    """
    preds_item = []
    labels_item = []
    for batch in validation_step_outputs:
        preds_item.append(batch["pred_age"])
        labels_item.append(batch["label"])

    preds = torch.cat(preds_item).detach().to(torch.float).cpu().numpy()
    labels = torch.cat(labels_item).detach().to(torch.float).cpu().numpy()

    R2 = r2_score(labels, preds)
    rmse = np.sqrt(mean_squared_error(labels, preds))
    mae = mean_absolute_error(labels, preds)
    medae = median_absolute_error(labels, preds)

    try:
        pearson_r = pearsonr(labels, preds)[0]
    except Exception:
        pearson_r = -1e-9
    try:
        sp_cor = spearmanr(labels, preds)[0]
    except Exception:
        sp_cor = -1e-9

    return {
        "r2": R2,
        "rmse": rmse,
        "mae": mae,
        "p_r": pearson_r,
        "s_r": sp_cor,
        "medae": medae,
    }


class EvalMultiLabel:
    """Multi-label classification evaluator.

    Args:
        y_trues: Ground truth labels (list of lists or array).
        y_preds: Predicted labels (list of lists or array).
    """

    def __init__(self, y_trues, y_preds):
        self.y_trues = y_trues
        self.y_preds = y_preds

        def single_sample(y_true, y_pred):
            y_true = list(set(y_true))
            y_pred = list(set(y_pred))
            y_ = list(set(y_true) | set(y_pred))
            K = len(y_)
            tp1 = fn1 = fp1 = tn1 = 0
            for item in y_:
                if item in y_true and item in y_pred:
                    tp1 += 1 / K
                elif item in y_true and item not in y_pred:
                    fn1 += 1 / K
                elif item not in y_true and item in y_pred:
                    fp1 += 1 / K
                elif item not in y_true and item not in y_pred:
                    tn1 += 1 / K
            return tp1, fn1, fp1, tn1

        def some_samples(y_trues, y_preds):
            if len(y_trues) != len(y_preds):
                logger.warning("Different length between y_trues and y_preds!")
                return 0, 0, 0, 0
            tp = fn = fp = tn = 0
            for y_true, y_pred in zip(y_trues, y_preds):
                tpi, fni, fpi, tni = single_sample(y_true, y_pred)
                tp += tpi
                fn += fni
                fp += fpi
                tn += tni
            return tp, fn, fp, tn

        self.tp, self.fn, self.fp, self.tn = some_samples(self.y_trues, self.y_preds)
        try:
            self.recall = self.tp / (self.tp + self.fn)
        except ZeroDivisionError:
            self.recall = 0
        try:
            self.precision = self.tp / (self.tp + self.fp)
        except ZeroDivisionError:
            self.precision = 0
        try:
            self.f1 = 2 * self.recall * self.precision / (self.precision + self.recall)
        except ZeroDivisionError:
            self.f1 = 0


def _compute_classification_metrics(labels, preds, average):
    """Helper to compute classification metrics for a given averaging strategy.

    Args:
        labels: Ground truth labels.
        preds: Predicted labels.
        average: Averaging strategy ('micro', 'macro', or 'weighted').

    Returns:
        Dictionary with accuracy, precision, recall, f1, and auc.
    """
    try:
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, average=average)
        rec = recall_score(labels, preds, average=average)
        f1 = f1_score(labels, preds, average=average)
        auc_val = roc_auc_score(labels, preds, average=average)
    except Exception:
        acc = prec = rec = f1 = auc_val = 0
    return {
        f"acc_{average}": acc,
        f"pr_{average}": prec,
        f"rec_{average}": rec,
        f"f1_{average}": f1,
        f"auc_{average}": auc_val,
    }


def disease_age_metric(validation_step_outputs):
    """Compute combined age regression + disease classification metrics.

    Args:
        validation_step_outputs: List of dicts with pred_age, label_age,
            pred_disease, and label_disease tensors.

    Returns:
        Dictionary with regression and classification metrics.
    """
    preds_age_item = []
    labels_age_item = []
    preds_disease_item = []
    labels_disease_item = []
    threshold = 0.5

    for batch in validation_step_outputs:
        preds_age_item.append(batch["pred_age"])
        labels_age_item.append(batch["label_age"])
        preds_disease_item.append(batch["pred_disease"])
        labels_disease_item.append(batch["label_disease"])

    preds_age = torch.cat(preds_age_item).detach().to(torch.float).cpu().numpy()
    labels_age = torch.cat(labels_age_item).detach().to(torch.float).cpu().numpy()
    preds_disease = torch.cat(preds_disease_item).detach().to(torch.float).cpu().numpy()
    preds_disease = np.where(preds_disease >= threshold, 1, 0)
    labels_disease = torch.cat(labels_disease_item).detach().to(torch.float).cpu().numpy()

    # Age metrics
    R2 = r2_score(labels_age, preds_age)
    rmse = np.sqrt(mean_squared_error(labels_age, preds_age))
    mae = mean_absolute_error(labels_age, preds_age)
    medae = median_absolute_error(labels_age, preds_age)

    try:
        pearson_r = pearsonr(labels_age, preds_age)[0]
    except Exception:
        pearson_r = -1e-9
    try:
        sp_cor = spearmanr(labels_age, preds_age)[0]
    except Exception:
        sp_cor = -1e-9

    EML = EvalMultiLabel(labels_disease, preds_disease)

    result = {
        "r2": R2,
        "rmse": rmse,
        "mae": mae,
        "p_r": pearson_r,
        "s_r": sp_cor,
        "medae": medae,
        "rec_eml": EML.recall,
        "pr_eml": EML.precision,
        "f1_eml": EML.f1,
    }

    for avg in ("micro", "macro", "weighted"):
        result.update(_compute_classification_metrics(labels_disease, preds_disease, avg))

    return result


def disease_metric(validation_step_outputs):
    """Compute disease classification metrics.

    Args:
        validation_step_outputs: List of dicts with pred_disease and
            label_disease tensors.

    Returns:
        Dictionary with classification metrics across micro/macro/weighted averaging.
    """
    preds_disease_item = []
    labels_disease_item = []
    threshold = 0.5

    for batch in validation_step_outputs:
        preds_disease_item.append(batch["pred_disease"])
        labels_disease_item.append(batch["label_disease"])

    preds_disease = torch.cat(preds_disease_item).detach().to(torch.float).cpu().numpy()
    preds_disease = np.where(preds_disease >= threshold, 1, 0)
    labels_disease = torch.cat(labels_disease_item).detach().to(torch.float).cpu().numpy()

    EML = EvalMultiLabel(labels_disease, preds_disease)

    result = {
        "rec_eml": EML.recall,
        "pr_eml": EML.precision,
        "f1_eml": EML.f1,
    }

    for avg in ("micro", "macro", "weighted"):
        result.update(_compute_classification_metrics(labels_disease, preds_disease, avg))

    return result


def disease_new_metric(validation_step_outputs, step=0.001):
    """Compute disease prediction metrics with threshold sweep.

    Args:
        validation_step_outputs: List of dicts with pred_disease and
            label_disease tensors.
        step: Threshold sweep step size.

    Returns:
        Dictionary with AUC, ACC, and optimal threshold.
    """
    T = np.arange(0, 1 + step, step)[None, :]

    preds_disease_item = []
    labels_disease_item = []

    for batch in validation_step_outputs:
        preds_disease_item.append(batch["pred_disease"])
        labels_disease_item.append(batch["label_disease"])

    targets = torch.cat(labels_disease_item).detach().to(torch.float).cpu().numpy()
    predictions = torch.cat(preds_disease_item).detach().to(torch.float).cpu().numpy()

    targets = targets.reshape(-1, 1)
    predictions = predictions.reshape(-1, 1)

    outputs_T = np.greater_equal(predictions, T)
    tp = np.sum(np.logical_and(outputs_T, targets), axis=0)
    tn = np.sum(np.logical_and(~outputs_T, ~targets), axis=0)
    fp = np.sum(np.logical_and(outputs_T, ~targets), axis=0)
    fn = np.sum(np.logical_and(~outputs_T, targets), axis=0)

    sens = tp / (tp + fn).astype(float)
    spec = tn / (tn + fp).astype(float)
    ACC = (tp + tn) / (tp + tn + fp + fn).astype(float)
    AUC = np.trapz(y=sens, x=spec)

    threshold = T[:, np.nanargmax(ACC)][0]
    ACC_best = np.nanmax(ACC)

    return {"AUC": AUC, "ACC": ACC_best, "Thresh": threshold}


def disease_category_metric(validation_step_outputs, model_args, step=0.001, data_type="valid"):
    """Compute disease category classification metrics with threshold sweep.

    Args:
        validation_step_outputs: List of dicts with pred_disease and
            label_disease tensors.
        model_args: Model configuration dictionary.
        step: Threshold sweep step size.
        data_type: Data split type ('valid' or 'test').

    Returns:
        Dictionary with AUC, ACC, and optimal threshold.
    """
    T = np.arange(0, 1 + step, step)[None, :]

    preds_disease_item = []
    labels_disease_item = []

    for batch in validation_step_outputs:
        preds_disease_item.append(batch["pred_disease"])
        labels_disease_item.append(batch["label_disease"])

    targets = torch.cat(labels_disease_item).detach().to(torch.float).cpu().numpy()
    predictions = torch.cat(preds_disease_item).detach().to(torch.float).cpu().numpy()

    targets = targets.reshape(-1, 1)
    predictions = predictions.reshape(-1, 1)

    outputs_T = np.greater_equal(predictions, T)
    tp = np.sum(np.logical_and(outputs_T, targets), axis=0)
    tn = np.sum(np.logical_and(~outputs_T, ~targets), axis=0)
    fp = np.sum(np.logical_and(outputs_T, ~targets), axis=0)
    fn = np.sum(np.logical_and(~outputs_T, targets), axis=0)

    sens = tp / (tp + fn).astype(float)
    spec = tn / (tn + fp).astype(float)
    ACC = (tp + tn) / (tp + tn + fp + fn).astype(float)

    fpr, tpr, thresholds = roc_curve(targets, predictions)
    roc_auc = auc(fpr, tpr)

    threshold = T[:, np.nanargmax(ACC)][0]
    ACC_best = np.nanmax(ACC)

    return {"AUC": roc_auc, "ACC": ACC_best, "Thresh": threshold}
