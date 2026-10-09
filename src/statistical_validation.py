"""
statistical_validation.py
--------------------------
STAGE 7 (project notes section 18) — Statistical Validation.

Two tools, both computed on the TEST set only, after the model and threshold
are already frozen (no leakage risk since nothing here influences training):

  1. McNemar's test: pairwise significance test between two models' TEST
     predictions on the SAME examples. Answers "is Model A significantly
     better than Model B, or could the difference be chance?"

  2. Bootstrap confidence intervals: resample the test set with replacement
     many times, recompute each metric each time, and report the 95% CI.
     This tells us how much a metric like F1 or ROC-AUC could plausibly
     vary just from test-set sampling noise.

Run (after train_xgboost.py and train_baselines.py have produced predictions):
    python src/statistical_validation.py --model-a Proposed_CodeBERT_XGBoost --model-b RandomForest
"""

from __future__ import annotations

import argparse
from typing import Callable

import numpy as np
from sklearn.metrics import (
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
)
from statsmodels.stats.contingency_tables import mcnemar

from utils import METRICS_DIR, get_logger, load_json, save_json, set_global_seed

logger = get_logger("statistical_validation", log_file="stage7_statistical_validation.log")


def run_mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> dict:
    """McNemar's test on paired binary predictions from two models on the
    SAME test set. Uses the exact-binomial version for small discordant
    counts (statsmodels default `exact=True` is auto-selected there)."""
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # 2x2 contingency table of (A correct/incorrect) x (B correct/incorrect)
    both_correct = int(np.sum(correct_a & correct_b))
    a_correct_b_wrong = int(np.sum(correct_a & ~correct_b))
    a_wrong_b_correct = int(np.sum(~correct_a & correct_b))
    both_wrong = int(np.sum(~correct_a & ~correct_b))

    table = [[both_correct, a_correct_b_wrong], [a_wrong_b_correct, both_wrong]]
    result = mcnemar(table, exact=(a_correct_b_wrong + a_wrong_b_correct < 25))

    return {
        "contingency_table": {
            "both_correct": both_correct,
            "model_a_correct_only": a_correct_b_wrong,
            "model_b_correct_only": a_wrong_b_correct,
            "both_wrong": both_wrong,
        },
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "significant_at_0.05": bool(result.pvalue < 0.05),
    }


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Stratified bootstrap confidence intervals for the final TEST metrics.

    Positive and negative examples are resampled separately so that each
    bootstrap replicate preserves the original class counts. This is more
    appropriate for the severely imbalanced PrimeVul test set.
    """

    rng = np.random.RandomState(seed)

    positive_idx = np.where(y_true == 1)[0]
    negative_idx = np.where(y_true == 0)[0]

    f1_samples = []
    f2_samples = []
    precision_samples = []
    recall_samples = []
    roc_auc_samples = []
    pr_auc_samples = []
    mcc_samples = []

    for _ in range(n_bootstrap):

        sampled_positive = rng.choice(
            positive_idx,
            size=len(positive_idx),
            replace=True,
        )

        sampled_negative = rng.choice(
            negative_idx,
            size=len(negative_idx),
            replace=True,
        )

        idx = np.concatenate(
            [sampled_positive, sampled_negative]
        )

        rng.shuffle(idx)

        yt = y_true[idx]
        yp = y_proba[idx]

        y_pred = (
            yp >= threshold
        ).astype(int)

        precision_samples.append(
            precision_score(
                yt,
                y_pred,
                zero_division=0,
            )
        )

        recall_samples.append(
            recall_score(
                yt,
                y_pred,
                zero_division=0,
            )
        )

        f1_samples.append(
            f1_score(
                yt,
                y_pred,
                zero_division=0,
            )
        )

        f2_samples.append(
            fbeta_score(
                yt,
                y_pred,
                beta=2.0,
                zero_division=0,
            )
        )

        roc_auc_samples.append(
            roc_auc_score(
                yt,
                yp,
            )
        )

        pr_auc_samples.append(
            average_precision_score(
                yt,
                yp,
            )
        )

        mcc_samples.append(
            matthews_corrcoef(
                yt,
                y_pred,
            )
        )

    alpha = (1 - ci) / 2

    def _ci_bounds(
        samples: list[float],
    ) -> dict:

        arr = np.asarray(
            samples,
            dtype=float,
        )

        return {
            "mean": float(
                arr.mean()
            ),
            "lower": float(
                np.percentile(
                    arr,
                    100 * alpha,
                )
            ),
            "upper": float(
                np.percentile(
                    arr,
                    100 * (1 - alpha),
                )
            ),
        }

    return {
        "bootstrap_method": (
            "stratified_by_true_class"
        ),
        "n_bootstrap": int(
            n_bootstrap
        ),
        "confidence_level": float(
            ci
        ),
        "positive_samples_per_resample": int(
            len(positive_idx)
        ),
        "negative_samples_per_resample": int(
            len(negative_idx)
        ),

        "precision": _ci_bounds(
            precision_samples
        ),

        "recall": _ci_bounds(
            recall_samples
        ),

        "f1_score": _ci_bounds(
            f1_samples
        ),

        "f2_score": _ci_bounds(
            f2_samples
        ),

        "roc_auc": _ci_bounds(
            roc_auc_samples
        ),

        "pr_auc": _ci_bounds(
            pr_auc_samples
        ),

        "mcc": _ci_bounds(
            mcc_samples
        ),
    }


def run_statistical_validation(model_a: str, model_b: str) -> dict:
    """Load each model's saved TEST predictions (written by train_xgboost.py /
    train_baselines.py via evaluation.save_test_predictions) and run
    McNemar's test between them on the SAME test examples."""
    from evaluation import load_test_predictions

    y_true_a, proba_a, threshold_a = load_test_predictions(model_a)
    y_true_b, proba_b, threshold_b = load_test_predictions(model_b)

    if not np.array_equal(y_true_a, y_true_b):
        raise ValueError(
            f"'{model_a}' and '{model_b}' were evaluated on different test "
            f"label arrays — they must share the same test set for McNemar's "
            f"test to be valid."
        )

    y_pred_a = (proba_a >= threshold_a).astype(int)
    y_pred_b = (proba_b >= threshold_b).astype(int)

    result = run_mcnemar_test(y_true_a, y_pred_a, y_pred_b)
    result["model_a"] = model_a
    result["model_b"] = model_b
    logger.info(
        "McNemar's test [%s vs %s]: statistic=%.4f, p=%.4g, significant@0.05=%s",
        model_a, model_b, result["statistic"], result["p_value"], result["significant_at_0.05"],
    )
    save_json(result, METRICS_DIR / f"stage7_mcnemar_{model_a}_vs_{model_b}.json")
    return result


if __name__ == "__main__":
    import xgboost as xgb
    from utils import MODELS_DIR, load_embeddings

    parser = argparse.ArgumentParser(description="Stage 7: Statistical validation")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument(
        "--model-a", default="Proposed_CodeBERT_XGBoost",
        help="First model name for McNemar's test (must have saved test predictions)",
    )
    parser.add_argument(
        "--model-b", default=None,
        help="Second model name for McNemar's test, e.g. RandomForest, LogisticRegression, "
        "LinearSVM, A_no_weighting, D_smote_balanced. If omitted, McNemar's test is skipped.",
    )
    args = parser.parse_args()

    set_global_seed()
    X_test, y_test = load_embeddings("test")

    config = load_json(METRICS_DIR / "proposed_model_config.json")
    threshold = config.get("best_threshold_f1", config.get("best_threshold"))

    model = xgb.XGBClassifier()
    model.load_model(str(MODELS_DIR / "xgboost_proposed_model.json"))
    proba_test = model.predict_proba(X_test)[:, 1]

    ci_result = bootstrap_confidence_intervals(
        y_test, proba_test, threshold, n_bootstrap=args.n_bootstrap
    )
    save_json(ci_result, METRICS_DIR / "stage7_bootstrap_ci_proposed_model.json")
    logger.info("Bootstrap 95%% CIs: %s", ci_result)

    if args.model_b:
        run_statistical_validation(args.model_a, args.model_b)
    else:
        logger.info(
            "No --model-b given; skipping McNemar's test. Example: "
            "python src/statistical_validation.py --model-a Proposed_CodeBERT_XGBoost --model-b RandomForest"
        )
