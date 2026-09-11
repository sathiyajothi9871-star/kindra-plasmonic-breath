"""Metrics, calibration and bootstrap confidence intervals."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, roc_auc_score,
                             average_precision_score, confusion_matrix)


def expected_calibration_error(prob, y, n_bins=15):
    conf = prob.max(1)
    pred = prob.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def brier_score(prob, y, n_classes):
    oh = np.eye(n_classes)[y]
    return float(((prob - oh) ** 2).sum(1).mean())


def negative_log_likelihood(prob, y):
    p = np.clip(prob[np.arange(len(y)), y], 1e-12, 1.0)
    return float(-np.log(p).mean())


def all_metrics(prob, y, n_classes=4):
    pred = prob.argmax(1)
    out = dict(
        accuracy=float(accuracy_score(y, pred)),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        macro_f1=float(f1_score(y, pred, average="macro")),
        mcc=float(matthews_corrcoef(y, pred)),
        ece=expected_calibration_error(prob, y),
        brier=brier_score(prob, y, n_classes),
        nll=negative_log_likelihood(prob, y))
    try:
        out["macro_auroc"] = float(roc_auc_score(y, prob, multi_class="ovr",
                                                 average="macro"))
    except ValueError:
        out["macro_auroc"] = float("nan")
    oh = np.eye(n_classes)[y]
    out["macro_auprc"] = float(average_precision_score(oh, prob, average="macro"))
    return out


def bootstrap_ci(prob, y, metric="balanced_accuracy", n_boot=2000, seed=0,
                 n_classes=4):
    """Percentile bootstrap over the test sessions."""
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < n_classes:
            continue
        vals.append(all_metrics(prob[idx], y[idx], n_classes)[metric])
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def confusion(prob, y, n_classes=4):
    return confusion_matrix(y, prob.argmax(1), labels=list(range(n_classes)))


def risk_coverage(prob, y, uncertainty):
    """Selective-prediction curve: error against retained fraction."""
    order = np.argsort(uncertainty)
    correct = (prob.argmax(1) == y)[order]
    cov = np.arange(1, len(y) + 1) / len(y)
    risk = 1.0 - np.cumsum(correct) / np.arange(1, len(y) + 1)
    return cov, risk
