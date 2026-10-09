"""
evaluation.py
--------------
Shared evaluation utilities used by both train_xgboost.py and
train_baselines.py, so every model in the comparison is scored identically.

Metrics computed:
    Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, MCC, Confusion Matrix
    VD-S (Vulnerability-Detection Score, PrimeVul-style):
        the false-negative rate on the positive (vulnerable) class,
        MEASURED AT the highest decision threshold that keeps the
        false-positive rate on the negative (benign) class at or below a
        target ceiling (default 0.5%). This mirrors the PrimeVul paper's
        emphasis on realistic, low-false-alarm operating points instead of
        Accuracy, which is dominated by the huge benign majority class.

Figures (all saved at 300 DPI to outputs/figures/):
    confusion matrix, ROC curve, Precision-Recall curve, model-comparison bar
    chart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
)

from utils import FIGURES_DIR, METRICS_DIR, get_logger, save_json

logger = get_logger("evaluation", log_file="evaluation.log")

FIGURE_DPI = 300


# --------------------------------------------------------------------------- #
# VD-S
# --------------------------------------------------------------------------- #

def compute_vds(
    y_true: np.ndarray, y_proba: np.ndarray, target_fpr_ceiling: float = 0.005
) -> dict[str, float]:
    """Vulnerability Detection Score (false-negative rate of vulnerable
    functions), evaluated at the loosest threshold whose false-positive rate
    on benign functions does not exceed `target_fpr_ceiling`.

    Returns the achieved FPR (which may be below the ceiling if no threshold
    hits it exactly), the resulting VD-S, and the threshold used.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)

    valid_idx = np.where(fpr <= target_fpr_ceiling)[0]
    if len(valid_idx) == 0:
        # Even the strictest threshold exceeds the ceiling; use the strictest.
        best_idx = int(np.argmin(fpr))
    else:
        # Among thresholds satisfying the FPR ceiling, pick the one with the
        # highest TPR (i.e. lowest FNR / lowest VD-S).
        best_idx = valid_idx[np.argmax(tpr[valid_idx])]

    achieved_fpr = float(fpr[best_idx])
    achieved_tpr = float(tpr[best_idx])
    threshold_at_ceiling = float(thresholds[best_idx])
    vds = 1.0 - achieved_tpr  # fraction of vulnerable functions MISSED

    return {
        "target_fpr_ceiling": target_fpr_ceiling,
        "achieved_fpr": achieved_fpr,
        "achieved_tpr": achieved_tpr,
        "threshold_at_ceiling": threshold_at_ceiling,
        "vds": vds,
    }


# --------------------------------------------------------------------------- #
# Full metric bundle
# --------------------------------------------------------------------------- #

def compute_all_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    model_name: str,
) -> dict[str, Any]:

    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    false_negative_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0.0
    )

    metrics = {
        "model_name": model_name,
        "threshold_used": float(threshold),

        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),

        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),

        "f1_score": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),

        "f2_score": float(
            fbeta_score(
                y_true,
                y_pred,
                beta=2.0,
                zero_division=0,
            )
        ),

        "specificity": float(specificity),

        "false_positive_rate": float(
            false_positive_rate
        ),

        "false_negative_rate": float(
            false_negative_rate
        ),

        "roc_auc": float(
            roc_auc_score(y_true, y_proba)
        )
        if len(set(y_true)) > 1
        else float("nan"),

        "pr_auc": float(
            average_precision_score(
                y_true,
                y_proba,
            )
        )
        if len(set(y_true)) > 1
        else float("nan"),

        "mcc": float(
            matthews_corrcoef(
                y_true,
                y_pred,
            )
        ),

        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },

        "vds_at_0.5pct_fpr": compute_vds(
            y_true,
            y_proba,
            target_fpr_ceiling=0.005,
        ),

        "vds_at_1pct_fpr": compute_vds(
            y_true,
            y_proba,
            target_fpr_ceiling=0.01,
        ),

        "support": {
            "total": int(len(y_true)),
            "positive": int(y_true.sum()),
            "negative": int(
                (y_true == 0).sum()
            ),
        },
    }

    return metrics


def select_best_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Select the decision threshold on the VALIDATION set that maximizes F1.
    Called ONLY on validation predictions — never on test — to prevent data
    leakage (project Rule #2)."""
    return select_best_threshold_fbeta(y_true, y_proba, beta=1.0)


def select_best_threshold_fbeta(y_true: np.ndarray, y_proba: np.ndarray, beta: float = 1.0) -> float:
    """Select the VALIDATION-set threshold maximizing the F-beta score.

    beta=1.0 -> F1 (precision/recall weighted equally; used as the default,
                model-selection-consistent operating point).
    beta=2.0 -> F2 (recall weighted 4x more than precision; a more
                security-appropriate operating point, since in vulnerability
                detection a missed vulnerability (false negative) is usually
                far costlier than a false alarm). Reported as a SECOND,
                explicitly-labeled operating point alongside the F1 one —
                never used to silently replace it.

    As with select_best_threshold, this must only ever be called on
    validation-set (y_true, y_proba), never on test.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    b2 = beta ** 2
    denom = np.maximum(b2 * precisions + recalls, 1e-12)
    f_beta = (1 + b2) * precisions * recalls / denom
    f_beta = f_beta[:-1]  # align with `thresholds` (len n-1)
    if len(f_beta) == 0:
        return 0.5
    best_idx = int(np.argmax(f_beta))
    return float(thresholds[best_idx])


# --------------------------------------------------------------------------- #
# Per-example prediction persistence (needed for McNemar's test, section 18)
# --------------------------------------------------------------------------- #

PREDICTIONS_DIR = METRICS_DIR / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


def save_test_predictions(model_name: str, y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> Path:
    """Persist one model's raw TEST predictions so a later stage (McNemar's
    test, further bootstrap analyses, error inspection) never has to retrain
    the model to reproduce them."""
    out_path = PREDICTIONS_DIR / f"{model_name}_test_predictions.npz"
    np.savez(out_path, y_true=y_true, y_proba=y_proba, threshold=np.array([threshold]))
    logger.info("Saved test predictions for '%s' -> %s", model_name, out_path)
    return out_path


def load_test_predictions(model_name: str) -> tuple[np.ndarray, np.ndarray, float]:
    path = PREDICTIONS_DIR / f"{model_name}_test_predictions.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"No saved test predictions for '{model_name}' at {path}. "
            f"Re-run the stage that trains/evaluates this model."
        )
    data = np.load(path)
    return data["y_true"], data["y_proba"], float(data["threshold"][0])


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

def plot_confusion_matrix(y_true, y_pred, model_name: str, out_name: str) -> Path:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Benign", "Vulnerable"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Benign", "Vulnerable"])
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix — {model_name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path = FIGURES_DIR / out_name
    fig.savefig(out_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return out_path


def plot_roc_curve(y_true, y_proba, model_name: str, out_name: str) -> Path:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve"); ax.legend(loc="lower right")
    fig.tight_layout()
    out_path = FIGURES_DIR / out_name
    fig.savefig(out_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return out_path


def plot_pr_curve(y_true, y_proba, model_name: str, out_name: str) -> Path:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(recall, precision, label=f"{model_name} (AP={ap:.4f})")
    base_rate = y_true.mean()
    ax.axhline(base_rate, linestyle="--", color="gray", label=f"Base rate ({base_rate:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve"); ax.legend(loc="upper right")
    fig.tight_layout()
    out_path = FIGURES_DIR / out_name
    fig.savefig(out_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return out_path


def plot_model_comparison(all_model_metrics: list[dict[str, Any]], out_name: str = "model_comparison.png") -> Path:
    names = [m["model_name"] for m in all_model_metrics]
    metric_keys = ["precision", "recall", "f1_score", "roc_auc", "pr_auc", "mcc"]
    x = np.arange(len(names))
    width = 0.13

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.6), 6))
    for i, key in enumerate(metric_keys):
        values = [m[key] for m in all_model_metrics]
        ax.bar(x + i * width, values, width, label=key)
    ax.set_xticks(x + width * (len(metric_keys) - 1) / 2)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Score"); ax.set_title("Model Comparison")
    ax.legend(loc="lower right", ncol=3, fontsize=8)
    fig.tight_layout()
    out_path = FIGURES_DIR / out_name
    fig.savefig(out_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return out_path


def full_evaluation_pipeline(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    model_name: str,
    save_prefix: str,
) -> dict[str, Any]:
    """Compute metrics + save all standard plots for one model on the TEST set."""
    metrics = compute_all_metrics(y_true, y_proba, threshold, model_name)
    y_pred = (y_proba >= threshold).astype(int)

    plot_confusion_matrix(y_true, y_pred, model_name, f"{save_prefix}_confusion_matrix.png")
    plot_roc_curve(y_true, y_proba, model_name, f"{save_prefix}_roc_curve.png")
    plot_pr_curve(y_true, y_proba, model_name, f"{save_prefix}_pr_curve.png")

    save_json(metrics, METRICS_DIR / f"{save_prefix}_test_metrics.json")
    logger.info("[%s] Test metrics: %s", model_name, {k: v for k, v in metrics.items() if k not in ("confusion_matrix",)})
    return metrics
