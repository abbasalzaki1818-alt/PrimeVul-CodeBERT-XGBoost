"""
main.py
-------
Full pipeline orchestrator for the Explainable Hybrid CodeBERT-XGBoost
vulnerability detection framework (PrimeVul).

Stages:
    1. inspect
    2. preprocess
    3. embed
    4. train_xgboost
    5. train_baselines
    6. explain
    7. statistical_validation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from utils import get_logger  # noqa: E402

logger = get_logger("main")


def stage_inspect(args) -> None:
    from data_inspection import run_inspection
    run_inspection(include_paired=args.paired)


def stage_preprocess(args) -> None:
    from preprocessing import run_preprocessing
    run_preprocessing(
        code_field=args.code_field,
        label_field=args.label_field,
        positive_label=args.positive_label,
    )


def stage_embed(args) -> None:
    from embedding_extractor import run_embedding_extraction
    run_embedding_extraction(
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )


def stage_train_xgboost(args) -> None:
    from train_xgboost import run_ablation_and_final_training

    run_ablation_and_final_training(
        use_hyperparameter_tuning=not args.no_tuning,
        run_smote_experiment=not args.no_smote,
    )


def stage_train_baselines(args) -> None:
    from train_baselines import run_baselines
    run_baselines()


def stage_explain(args) -> None:
    from explainability import run_explainability
    run_explainability(num_local_examples=args.num_local_examples)


def stage_statistical_validation(args) -> None:
    import xgboost as xgb

    from evaluation import load_test_predictions
    from statistical_validation import (
        bootstrap_confidence_intervals,
        run_statistical_validation,
    )
    from utils import (
        METRICS_DIR,
        MODELS_DIR,
        load_embeddings,
        load_json,
        save_json,
        set_global_seed,
    )

    set_global_seed()

    X_test, y_test = load_embeddings("test")

    config = load_json(METRICS_DIR / "proposed_model_config.json")
    threshold = config.get(
        "best_threshold_f1",
        config.get("best_threshold"),
    )

    model = xgb.XGBClassifier()
    model.load_model(str(MODELS_DIR / "xgboost_proposed_model.json"))

    proba_test = model.predict_proba(X_test)[:, 1]

    ci_result = bootstrap_confidence_intervals(
        y_test,
        proba_test,
        threshold,
        n_bootstrap=args.n_bootstrap,
    )

    save_json(
        ci_result,
        METRICS_DIR / "stage7_bootstrap_ci_proposed_model.json",
    )

    logger.info("Bootstrap 95%% CIs: %s", ci_result)

    if args.model_b:
        run_statistical_validation(
            "Proposed_CodeBERT_XGBoost",
            args.model_b,
        )
    else:
        logger.info(
            "No --model-b supplied; McNemar test skipped."
        )


STAGE_FUNCS = {
    "inspect": stage_inspect,
    "preprocess": stage_preprocess,
    "embed": stage_embed,
    "train_xgboost": stage_train_xgboost,
    "train_baselines": stage_train_baselines,
    "explain": stage_explain,
    "statistical_validation": stage_statistical_validation,
}

STAGE_ORDER = [
    "inspect",
    "preprocess",
    "embed",
    "train_xgboost",
    "train_baselines",
    "explain",
    "statistical_validation",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PrimeVul CodeBERT-XGBoost full pipeline"
    )

    parser.add_argument(
        "--stage",
        choices=["all", *STAGE_ORDER],
        default="all",
        help="Stage to run; default=all",
    )

    # Stage 1
    parser.add_argument(
        "--paired",
        action="store_true",
        help="[inspect] inspect *_paired.jsonl",
    )

    # Stage 2
    parser.add_argument(
        "--code-field",
        default=None,
        help="[preprocess] override code field",
    )
    parser.add_argument(
        "--label-field",
        default=None,
        help="[preprocess] override label field",
    )
    parser.add_argument(
        "--positive-label",
        default=None,
        help="[preprocess] positive label",
    )

    # Stage 3
    parser.add_argument(
        "--model-name",
        default="microsoft/codebert-base",
        help="[embed] HF model name",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="[embed] batch size",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="[embed] max token length",
    )

    # Stage 4
    parser.add_argument(
        "--no-tuning",
        action="store_true",
        help="[train_xgboost] skip hyperparameter tuning",
    )
    parser.add_argument(
        "--no-smote",
        action="store_true",
        help="[train_xgboost] skip SMOTE ablation",
    )

    # Stage 6
    parser.add_argument(
        "--num-local-examples",
        type=int,
        default=5,
        help="[explain] number of local examples",
    )

    # Stage 7
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=2000,
        help="[statistical_validation] bootstrap repetitions",
    )
    parser.add_argument(
        "--model-b",
        default="RandomForest",
        help="[statistical_validation] baseline for McNemar",
    )

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    stages_to_run = (
        STAGE_ORDER
        if args.stage == "all"
        else [args.stage]
    )

    for stage_name in stages_to_run:
        logger.info("=" * 70)
        logger.info("RUNNING STAGE: %s", stage_name)
        logger.info("=" * 70)

        try:
            STAGE_FUNCS[stage_name](args)

        except FileNotFoundError as exc:
            logger.error(
                "Stage '%s' failed — missing prerequisite: %s",
                stage_name,
                exc,
            )
            sys.exit(1)

        except Exception:
            logger.exception(
                "Stage '%s' raised an unexpected error.",
                stage_name,
            )
            sys.exit(1)

    logger.info("Pipeline finished successfully.")


if __name__ == "__main__":
    main()