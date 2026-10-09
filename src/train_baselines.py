"""
train_baselines.py
-------------------
STAGE 5 — Baseline models for comparison against the proposed
Imbalance-Aware CodeBERT-XGBoost framework.

All baselines consume the SAME frozen CodeBERT embeddings (fair comparison —
only the classifier head changes):
    - Logistic Regression (class_weight='balanced')
    - Linear SVM (class_weight='balanced', probability via Platt scaling)
    - Random Forest (class_weight='balanced')
    - XGBoost without imbalance handling (duplicated here for a single
      consolidated comparison table/figure; already computed as
      "A_no_weighting" in train_xgboost.py)

Every model's threshold is selected on VALIDATION only (never test), using
the same F1-maximizing rule as the proposed model, for a fair comparison.

Run:
    python src/train_baselines.py
"""

from __future__ import annotations

import argparse
from typing import Any

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from evaluation import (
    compute_all_metrics,
    plot_model_comparison,
    select_best_threshold,
    save_test_predictions,
)
from utils import (
    METRICS_DIR,
    get_logger,
    load_embeddings,
    load_json,
    save_json,
    set_global_seed,
    RANDOM_SEED_DEFAULT,
)

logger = get_logger("train_baselines", log_file="stage5_train_baselines.log")


def _fit_and_score(name: str, model, X_train, y_train, X_valid, y_valid, X_test, y_test) -> dict[str, Any]:
    logger.info("Training baseline: %s", name)
    model.fit(X_train, y_train)

    proba_valid = model.predict_proba(X_valid)[:, 1]
    proba_test = model.predict_proba(X_test)[:, 1]

    threshold = select_best_threshold(y_valid, proba_valid)  # validation only
    metrics = compute_all_metrics(y_test, proba_test, threshold, model_name=name)
    save_test_predictions(name, y_test, proba_test, threshold=threshold)
    logger.info("[%s] test F1=%.4f | ROC-AUC=%.4f | threshold=%.4f",
                name, metrics["f1_score"], metrics["roc_auc"], threshold)
    return metrics


def run_baselines() -> list[dict[str, Any]]:
    set_global_seed()

    X_train, y_train = load_embeddings("train")
    X_valid, y_valid = load_embeddings("valid")
    X_test, y_test = load_embeddings("test")

    models = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=RANDOM_SEED_DEFAULT
        ),
        # NOTE: SVC(kernel="linear", probability=True) was replaced with
        # LinearSVC + CalibratedClassifierCV. SVC's libsvm backend scales
        # roughly quadratically-to-cubically with sample count and its
        # internal probability=True adds a further 5-fold Platt-scaling pass
        # on top — on PrimeVul's ~230k training functions this made Stage 5
        # run for hours without finishing. LinearSVC uses liblinear's
        # optimized linear-kernel solver (near-linear in sample count), and
        # CalibratedClassifierCV(cv=3) reproduces the same sigmoid/Platt
        # probability calibration with fewer folds and parallelized via
        # n_jobs=-1, which is both correct and dramatically faster.
        "LinearSVM": CalibratedClassifierCV(
            estimator=LinearSVC(
                class_weight="balanced",
                random_state=RANDOM_SEED_DEFAULT,
                max_iter=5000,
            ),
            method="sigmoid",
            cv=3,
            n_jobs=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, class_weight="balanced", random_state=RANDOM_SEED_DEFAULT,
            n_jobs=-1,
        ),
    }

    all_results = []
    for name, model in models.items():
        metrics = _fit_and_score(name, model, X_train, y_train, X_valid, y_valid, X_test, y_test)
        all_results.append(metrics)

    # Pull in the two XGBoost variants already computed by train_xgboost.py,
    # so the comparison figure/table is complete without retraining them.
    ablation_path = METRICS_DIR / "stage4_ablation_and_final_results.json"
    if ablation_path.exists():
        ablation = load_json(ablation_path)["ablation"]
        for key in ("A_no_weighting", "C_weighted_plus_threshold"):
            if key in ablation:
                all_results.append(ablation[key])
    else:
        logger.warning(
            "stage4_ablation_and_final_results.json not found — run train_xgboost.py "
            "first for a complete comparison table."
        )

    save_json(all_results, METRICS_DIR / "stage5_baseline_comparison.json")
    plot_model_comparison(all_results, out_name="model_comparison.png")
    logger.info("Stage 5 complete. Comparison table + figure saved.")
    return all_results


if __name__ == "__main__":
    argparse.ArgumentParser(description="Stage 5: Train baseline models").parse_args()
    run_baselines()
