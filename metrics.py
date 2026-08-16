"""
metrics.py — Full statistical evaluation.

Directly answers Reviewer 3 Comment 5 ("Reporting only 96% +/- 0.8% is
not sufficient") and Comment 9's implicit ask for rigor:

  - per-fold metrics (accuracy, precision, recall, F1, sensitivity,
    specificity), each as macro-average AND micro-average across the
    four classes;
  - bootstrap 95% confidence intervals;
  - paired statistical tests (Wilcoxon signed-rank + paired t-test as a
    cross-check) between the proposed model and every baseline, computed
    on matched per-fold scores.

Nothing here reports a single aggregate number without its spread.
"""
from typing import Dict, List

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def per_class_sensitivity_specificity(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    sens, spec = [], []
    total = cm.sum()
    for c in range(n_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = total - tp - fn - fp
        sens.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        spec.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return np.array(sens), np.array(spec)


def compute_fold_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, n_classes: int
) -> Dict[str, float]:
    """All metrics for a single fold. y_proba: (N, n_classes) softmax output."""
    sens, spec = per_class_sensitivity_specificity(y_true, y_pred, n_classes)
    try:
        auc_macro = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        auc_micro = roc_auc_score(y_true, y_proba, multi_class="ovr", average="micro")
    except ValueError:
        auc_macro, auc_micro = float("nan"), float("nan")

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "sensitivity_macro": float(sens.mean()),
        "specificity_macro": float(spec.mean()),
        "roc_auc_macro": auc_macro,
        "roc_auc_micro": auc_micro,
    }


def aggregate_fold_metrics(fold_metrics: List[Dict[str, float]]) -> Dict[str, str]:
    """Mean +/- std across folds, formatted as e.g. '0.9656 +/- 0.0080'.

    This is the ONLY place a summary "mean +/- std" string is produced —
    it is always derived from real per-fold numbers, never hardcoded.
    """
    keys = fold_metrics[0].keys()
    summary = {}
    for k in keys:
        vals = np.array([m[k] for m in fold_metrics])
        summary[k] = f"{vals.mean():.4f} +/- {vals.std(ddof=1):.4f}"
    return summary


def bootstrap_confidence_interval(values: np.ndarray, n_iterations: int = 2000, ci: float = 0.95, seed: int = 42):
    """Percentile bootstrap CI over per-fold (or per-repeat) scores."""
    rng = np.random.default_rng(seed)
    boots = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_iterations)]
    lower = np.percentile(boots, (1 - ci) / 2 * 100)
    upper = np.percentile(boots, (1 + ci) / 2 * 100)
    return float(np.mean(values)), float(lower), float(upper)


def paired_significance_test(model_a_scores: np.ndarray, model_b_scores: np.ndarray) -> Dict[str, float]:
    """Paired comparison between two models on MATCHED folds/repeats.

    Reports both a paired t-test and the non-parametric Wilcoxon
    signed-rank test (the primary test per config.yaml, since fold counts
    are small and normality shouldn't be assumed).
    """
    assert len(model_a_scores) == len(model_b_scores), "Scores must be paired (same folds)."
    t_stat, t_p = stats.ttest_rel(model_a_scores, model_b_scores)
    try:
        w_stat, w_p = stats.wilcoxon(model_a_scores, model_b_scores)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    return {"paired_ttest_p": float(t_p), "wilcoxon_p": float(w_p)}
