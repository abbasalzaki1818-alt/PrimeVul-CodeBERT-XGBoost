"""
train_xgboost.py
-----------------
STAGE 4 — Imbalance-aware XGBoost training + Stage 6 (threshold selection)
+ Stage 5's ablation study.

Hard rules enforced here (project constitution, section 23):
    1. Test set is NEVER used for training.
    2. Test set is NEVER used for hyperparameter or threshold selection.
    3. scale_pos_weight is computed from the TRAIN split only.
    4. Threshold is selected on VALIDATION only, then frozen and applied to
       TEST exactly once.

Ablation study (Stage 17 in the notes):
    A. CodeBERT + plain XGBoost (no imbalance handling)
    B. + Class weighting (scale_pos_weight)
    C. + Class weighting + threshold optimization
    D. Full framework (same as C here; paired-eval / explainability are
       separate stages) — reported for completeness.

Run:
    python src/train_xgboost.py
"""

from __future__ import annotations

import argparse
import itertools
from typing import Any

import numpy as np
import xgboost as xgb

from evaluation import (
    full_evaluation_pipeline,
    compute_all_metrics,
    select_best_threshold,
    select_best_threshold_fbeta,
    save_test_predictions,
)
from utils import (
    MODELS_DIR,
    METRICS_DIR,
    get_logger,
    load_embeddings,
    save_json,
    set_global_seed,
    RANDOM_SEED_DEFAULT,
)

logger = get_logger("train_xgboost", log_file="stage4_train_xgboost.log")

BASE_XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    eval_metric="aucpr",
    tree_method="hist",
    random_state=RANDOM_SEED_DEFAULT,
)


def compute_scale_pos_weight(y_train: np.ndarray) -> float:
    """scale_pos_weight = (#benign) / (#vulnerable), computed on TRAIN only."""
    n_benign = int((y_train == 0).sum())
    n_vulnerable = int((y_train == 1).sum())
    if n_vulnerable == 0:
        raise ValueError("No vulnerable samples in training set; cannot compute scale_pos_weight.")
    weight = n_benign / n_vulnerable
    logger.info(
        "Train class counts -> benign=%d, vulnerable=%d -> scale_pos_weight=%.4f",
        n_benign, n_vulnerable, weight,
    )
    return weight


def tune_hyperparameters(
    X_train, y_train, X_valid, y_valid, scale_pos_weight: float,
    param_grid: dict | None = None,
) -> dict:
    """STAGE 4a — Hyperparameter tuning, evaluated ONLY on the validation
    set (never test). Uses PR-AUC as the selection metric rather than
    Accuracy/ROC-AUC, since PR-AUC is far more sensitive to performance on
    the rare positive (vulnerable) class under severe imbalance — exactly
    the failure mode this project cares about (see project notes, section 3).

    A small, deliberately compact grid is used (rather than an exhaustive
    GridSearchCV) because each configuration requires a full XGBoost fit on
    ~180k rows x 768 dims; an exhaustive grid would take hours. This keeps
    tuning tractable while still being a real, validation-driven search
    instead of hard-coded defaults.
    """
    from sklearn.metrics import average_precision_score

    if param_grid is None:
        param_grid = {
            "max_depth": [4, 6, 8],
            "learning_rate": [0.03, 0.05, 0.1],
            "min_child_weight": [1, 5],
        }

    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    logger.info("Hyperparameter search: %d configurations over %s", len(combos), keys)

    best_score = -1.0
    best_params: dict = {}
    search_log = []

    for combo in combos:
        overrides = dict(zip(keys, combo))
        params = dict(BASE_XGB_PARAMS)
        params.update(overrides)
        params["scale_pos_weight"] = scale_pos_weight
        params["n_estimators"] = 300  # smaller budget during search; final model retrains at 400+ with early stopping

        model = xgb.XGBClassifier(**params, early_stopping_rounds=20)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        proba_valid = model.predict_proba(X_valid)[:, 1]
        pr_auc = average_precision_score(y_valid, proba_valid)

        search_log.append({**overrides, "valid_pr_auc": float(pr_auc)})
        logger.info("  params=%s -> valid PR-AUC=%.4f", overrides, pr_auc)

        if pr_auc > best_score:
            best_score = pr_auc
            best_params = overrides

    logger.info("Best hyperparameters (by validation PR-AUC=%.4f): %s", best_score, best_params)
    save_json(
        {"search_log": search_log, "best_params": best_params, "best_valid_pr_auc": float(best_score)},
        METRICS_DIR / "stage4a_hyperparameter_search.json",
    )
    return best_params


def train_one_xgboost(
    X_train, y_train, X_valid, y_valid, use_class_weight: bool, params_override: dict | None = None
) -> xgb.XGBClassifier:
    params = dict(BASE_XGB_PARAMS)
    if params_override:
        params.update(params_override)
    if use_class_weight:
        params["scale_pos_weight"] = compute_scale_pos_weight(y_train)

    model = xgb.XGBClassifier(**params, early_stopping_rounds=30)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )
    return model

def tune_scale_pos_weight(
    X_train,
    y_train,
    X_valid,
    y_valid,
    tuned_params: dict,
) -> tuple[float, list[dict]]:
    """
    Select scale_pos_weight using VALIDATION PR-AUC only.

    The TEST set is never used for weight selection.
    """

    from sklearn.metrics import average_precision_score

    full_ratio = compute_scale_pos_weight(y_train)

    candidates = sorted({
        1.0,
        float(np.sqrt(full_ratio)),
        float(full_ratio / 4.0),
        float(full_ratio / 2.0),
        float(full_ratio * 0.75),
        float(full_ratio),
    })

    logger.info(
        "=== Stage 4b: scale_pos_weight tuning "
        "(validation PR-AUC) ==="
    )

    logger.info(
        "Weight candidates: %s",
        [round(x, 4) for x in candidates],
    )

    best_weight = 1.0
    best_pr_auc = -1.0

    search_results = []

    for weight in candidates:

        params = dict(tuned_params)
        params["scale_pos_weight"] = float(weight)

        model = train_one_xgboost(
            X_train,
            y_train,
            X_valid,
            y_valid,
            use_class_weight=False,
            params_override=params,
        )

        proba_valid = model.predict_proba(
            X_valid
        )[:, 1]

        pr_auc = average_precision_score(
            y_valid,
            proba_valid,
        )

        result = {
            "scale_pos_weight": float(weight),
            "valid_pr_auc": float(pr_auc),
        }

        search_results.append(result)

        logger.info(
            "scale_pos_weight=%.4f "
            "-> validation PR-AUC=%.6f",
            weight,
            pr_auc,
        )

        if pr_auc > best_pr_auc:
            best_pr_auc = float(pr_auc)
            best_weight = float(weight)

    logger.info(
        "Best validation-selected "
        "scale_pos_weight=%.4f "
        "(PR-AUC=%.6f)",
        best_weight,
        best_pr_auc,
    )

    save_json(
        {
            "selection_metric": "validation_pr_auc",
            "candidates": search_results,
            "best_scale_pos_weight": best_weight,
            "best_valid_pr_auc": best_pr_auc,
        },
        METRICS_DIR
        / "stage4b_scale_pos_weight_search.json",
    )

    return best_weight, search_results


def run_ablation_and_final_training(
    use_hyperparameter_tuning: bool = True,
    run_smote_experiment: bool = True,
) -> dict[str, Any]:

    set_global_seed()

    X_train, y_train = load_embeddings("train")
    X_valid, y_valid = load_embeddings("valid")
    X_test, y_test = load_embeddings("test")

    logger.info(
        "Loaded embeddings: train=%s valid=%s test=%s",
        X_train.shape,
        X_valid.shape,
        X_test.shape,
    )

    results: dict[str, Any] = {
        "ablation": {},
        "final": {},
    }

    # -------------------------------------------------------
    # Full class-ratio weight (reference value only)
    # -------------------------------------------------------

    full_scale_pos_weight = (
        compute_scale_pos_weight(y_train)
    )

    # -------------------------------------------------------
    # Stage 4a — Hyperparameter tuning
    # VALIDATION ONLY
    # -------------------------------------------------------

    tuned_params: dict = {}

    if use_hyperparameter_tuning:

        logger.info(
            "=== Stage 4a: Hyperparameter tuning "
            "(validation PR-AUC) ==="
        )

        tuned_params = tune_hyperparameters(
            X_train,
            y_train,
            X_valid,
            y_valid,
            1.0,
        )

    # -------------------------------------------------------
    # Stage 4b — Select the best class weight
    # VALIDATION ONLY
    # -------------------------------------------------------

    best_weight, weight_search = (
        tune_scale_pos_weight(
            X_train,
            y_train,
            X_valid,
            y_valid,
            tuned_params,
        )
    )

    # =======================================================
    # EXPERIMENT A
    # Unweighted XGBoost + default threshold 0.5
    # =======================================================

    logger.info(
        "=== Experiment A: "
        "Unweighted XGBoost, threshold=0.5 ==="
    )

    model_a = train_one_xgboost(
        X_train,
        y_train,
        X_valid,
        y_valid,
        use_class_weight=False,
        params_override=tuned_params,
    )

    proba_a_valid = model_a.predict_proba(
        X_valid
    )[:, 1]

    proba_a_test = model_a.predict_proba(
        X_test
    )[:, 1]

    metrics_a = compute_all_metrics(
        y_test,
        proba_a_test,
        threshold=0.5,
        model_name="A_no_weighting",
    )

    results["ablation"][
        "A_no_weighting"
    ] = metrics_a

    save_test_predictions(
        "A_no_weighting",
        y_test,
        proba_a_test,
        threshold=0.5,
    )

    # =======================================================
    # EXPERIMENT A2
    # Same unweighted model
    # + F1 threshold selected on VALIDATION
    # =======================================================

    logger.info(
        "=== Experiment A2: "
        "Unweighted XGBoost "
        "+ validation F1 threshold ==="
    )

    threshold_a_f1 = select_best_threshold(
        y_valid,
        proba_a_valid,
    )

    metrics_a2 = compute_all_metrics(
        y_test,
        proba_a_test,
        threshold=threshold_a_f1,
        model_name=(
            "A2_unweighted_optimized_threshold"
        ),
    )

    results["ablation"][
        "A2_unweighted_optimized_threshold"
    ] = metrics_a2

    save_test_predictions(
        "A2_unweighted_optimized_threshold",
        y_test,
        proba_a_test,
        threshold=threshold_a_f1,
    )

    logger.info(
        "[A2] threshold=%.4f | "
        "F1=%.4f | PR-AUC=%.4f | "
        "Recall=%.4f",
        threshold_a_f1,
        metrics_a2["f1_score"],
        metrics_a2["pr_auc"],
        metrics_a2["recall"],
    )

    # =======================================================
    # EXPERIMENT B
    # Full class-ratio weighting
    # threshold = 0.5
    # =======================================================

    logger.info(
        "=== Experiment B: "
        "Full-ratio class weighting, "
        "threshold=0.5 ==="
    )

    full_weight_params = dict(
        tuned_params
    )

    full_weight_params[
        "scale_pos_weight"
    ] = full_scale_pos_weight

    model_b = train_one_xgboost(
        X_train,
        y_train,
        X_valid,
        y_valid,
        use_class_weight=False,
        params_override=full_weight_params,
    )

    proba_b_valid = model_b.predict_proba(
        X_valid
    )[:, 1]

    proba_b_test = model_b.predict_proba(
        X_test
    )[:, 1]

    metrics_b = compute_all_metrics(
        y_test,
        proba_b_test,
        threshold=0.5,
        model_name="B_class_weighted",
    )

    results["ablation"][
        "B_class_weighted"
    ] = metrics_b

    save_test_predictions(
        "B_class_weighted",
        y_test,
        proba_b_test,
        threshold=0.5,
    )

    # =======================================================
    # EXPERIMENT B2
    # Full-ratio weighting
    # + VALIDATION F1 threshold
    # =======================================================

    logger.info(
        "=== Experiment B2: "
        "Full-ratio weighting "
        "+ validation F1 threshold ==="
    )

    threshold_b_f1 = select_best_threshold(
        y_valid,
        proba_b_valid,
    )

    metrics_b2 = compute_all_metrics(
        y_test,
        proba_b_test,
        threshold=threshold_b_f1,
        model_name=(
            "B2_full_weight_optimized_threshold"
        ),
    )

    results["ablation"][
        "B2_full_weight_optimized_threshold"
    ] = metrics_b2

    save_test_predictions(
        "B2_full_weight_optimized_threshold",
        y_test,
        proba_b_test,
        threshold=threshold_b_f1,
    )

    # =======================================================
    # PROPOSED MODEL
    # Validation-selected class weight
    # =======================================================

    logger.info(
        "=== Experiment C: Proposed model "
        "with validation-selected class weight ==="
    )

    proposed_params = dict(
        tuned_params
    )

    proposed_params[
        "scale_pos_weight"
    ] = best_weight

    model_proposed = train_one_xgboost(
        X_train,
        y_train,
        X_valid,
        y_valid,
        use_class_weight=False,
        params_override=proposed_params,
    )

    proba_proposed_valid = (
        model_proposed.predict_proba(
            X_valid
        )[:, 1]
    )

    proba_proposed_test = (
        model_proposed.predict_proba(
            X_test
        )[:, 1]
    )

    # -------------------------------------------------------
    # F1 operating point
    # -------------------------------------------------------

    best_threshold_f1 = (
        select_best_threshold(
            y_valid,
            proba_proposed_valid,
        )
    )

    logger.info(
        "Selected proposed-model "
        "F1 threshold on validation: %.4f",
        best_threshold_f1,
    )

    final_metrics = (
        full_evaluation_pipeline(
            y_test,
            proba_proposed_test,
            best_threshold_f1,
            model_name=(
                "Proposed_CodeBERT_XGBoost"
            ),
            save_prefix="proposed_model",
        )
    )

    final_metrics[
        "selected_threshold_source"
    ] = "validation_set_only"

    final_metrics[
        "selected_scale_pos_weight"
    ] = float(best_weight)

    results["ablation"][
        "C_weighted_plus_threshold"
    ] = final_metrics

    results["final"] = final_metrics

    save_test_predictions(
        "Proposed_CodeBERT_XGBoost",
        y_test,
        proba_proposed_test,
        threshold=best_threshold_f1,
    )

    # -------------------------------------------------------
    # F2 operating point
    # -------------------------------------------------------

    logger.info(
        "=== Experiment C2: "
        "Proposed model + F2 threshold ==="
    )

    best_threshold_f2 = (
        select_best_threshold_fbeta(
            y_valid,
            proba_proposed_valid,
            beta=2.0,
        )
    )

    metrics_c2 = compute_all_metrics(
        y_test,
        proba_proposed_test,
        best_threshold_f2,
        model_name=(
            "Proposed_CodeBERT_XGBoost_F2"
        ),
    )

    results["ablation"][
        "C2_recall_priority_threshold"
    ] = metrics_c2

    logger.info(
        "F2 threshold=%.4f | "
        "Recall=%.4f | Precision=%.4f | "
        "F1=%.4f | F2=%.4f",
        best_threshold_f2,
        metrics_c2["recall"],
        metrics_c2["precision"],
        metrics_c2["f1_score"],
        metrics_c2["f2_score"],
    )

    # =======================================================
    # EXPERIMENT D
    # SMOTE ablation
    # =======================================================

    if run_smote_experiment:

        logger.info(
            "=== Experiment D: "
            "BorderlineSMOTE-balanced XGBoost ==="
        )

        try:
            from imblearn.over_sampling import (
                BorderlineSMOTE,
            )

            smote = BorderlineSMOTE(
                random_state=RANDOM_SEED_DEFAULT
            )

            X_train_sm, y_train_sm = (
                smote.fit_resample(
                    X_train,
                    y_train,
                )
            )

            if len(y_train_sm) == len(y_train):

                logger.warning(
                    "BorderlineSMOTE generated no "
                    "synthetic samples; "
                    "falling back to SMOTE."
                )

                from imblearn.over_sampling import (
                    SMOTE,
                )

                smote = SMOTE(
                    random_state=(
                        RANDOM_SEED_DEFAULT
                    )
                )

                X_train_sm, y_train_sm = (
                    smote.fit_resample(
                        X_train,
                        y_train,
                    )
                )

            logger.info(
                "SMOTE train set: %d -> %d "
                "(benign=%d, vulnerable=%d)",
                len(y_train),
                len(y_train_sm),
                int(
                    (y_train_sm == 0).sum()
                ),
                int(
                    (y_train_sm == 1).sum()
                ),
            )

            model_d = train_one_xgboost(
                X_train_sm,
                y_train_sm,
                X_valid,
                y_valid,
                use_class_weight=False,
                params_override=tuned_params,
            )

            proba_d_valid = (
                model_d.predict_proba(
                    X_valid
                )[:, 1]
            )

            proba_d_test = (
                model_d.predict_proba(
                    X_test
                )[:, 1]
            )

            threshold_d = (
                select_best_threshold(
                    y_valid,
                    proba_d_valid,
                )
            )

            metrics_d = compute_all_metrics(
                y_test,
                proba_d_test,
                threshold_d,
                model_name="D_smote_balanced",
            )

            results["ablation"][
                "D_smote_balanced"
            ] = metrics_d

            save_test_predictions(
                "D_smote_balanced",
                y_test,
                proba_d_test,
                threshold=threshold_d,
            )

            logger.info(
                "[D] F1=%.4f | "
                "PR-AUC=%.4f | "
                "ROC-AUC=%.4f",
                metrics_d["f1_score"],
                metrics_d["pr_auc"],
                metrics_d["roc_auc"],
            )

        except Exception:
            logger.exception(
                "Experiment D failed."
            )
            raise

    # =======================================================
    # Save results
    # =======================================================

    save_json(
        results,
        METRICS_DIR
        / "stage4_ablation_and_final_results.json",
    )

    model_proposed.save_model(
        str(
            MODELS_DIR
            / "xgboost_proposed_model.json"
        )
    )

    save_json(
        {
            "best_threshold_f1": (
                best_threshold_f1
            ),
            "best_threshold_f2": (
                best_threshold_f2
            ),

            "scale_pos_weight_full_ratio": (
                full_scale_pos_weight
            ),

            "selected_scale_pos_weight": (
                best_weight
            ),

            "scale_pos_weight_search": (
                weight_search
            ),

            "tuned_params": tuned_params,

            "weight_selection_metric": (
                "validation_pr_auc"
            ),

            "threshold_selection_source": (
                "validation_set_only"
            ),
        },
        METRICS_DIR
        / "proposed_model_config.json",
    )

    logger.info(
        "Stage 4 complete. "
        "Proposed model uses "
        "scale_pos_weight=%.4f",
        best_weight,
    )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4: Train imbalance-aware XGBoost")
    parser.add_argument("--no-tuning", action="store_true", help="Skip hyperparameter tuning (use BASE_XGB_PARAMS)")
    parser.add_argument("--no-smote", action="store_true", help="Skip the SMOTE ablation experiment")
    args = parser.parse_args()
    run_ablation_and_final_training(
        use_hyperparameter_tuning=not args.no_tuning,
        run_smote_experiment=not args.no_smote,
    )
