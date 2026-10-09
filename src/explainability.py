"""
explainability.py
-------------------
STAGE 6 — Two-level Explainability, as specified in the project notes:

  1. GLOBAL explanation (XGBoost + SHAP):
     Standard SHAP TreeExplainer over the 768-dim CodeBERT embedding space.
     This tells us which embedding DIMENSIONS matter most overall — useful
     for the paper's global-importance figure, even though embedding
     dimensions are not directly interpretable to a human programmer.

  2. LOCAL, SOURCE-CODE-LEVEL explanation (the more valuable one):
     For a handful of selected test examples, we explain the decision in
     terms of actual CODE TOKENS rather than opaque embedding dimensions.
     Method: leave-one-token-out occlusion.
         - Tokenize the function with the CodeBERT tokenizer.
         - For each token, mask it out (replace with the tokenizer's
           <mask>/<unk> token), recompute the mean-pooled CodeBERT
           embedding, and re-score it with the trained XGBoost model.
         - The importance of a token = drop in predicted vulnerability
           probability caused by removing it. Large positive drops mean
           "this token was pushing the prediction toward Vulnerable."
     This is model-agnostic, requires no gradient access, and produces
     directly human-readable output such as:
         strcpy       -> +0.41 (pushed toward Vulnerable)
         buffer       -> +0.12
         sizeof       -> -0.08 (pushed toward Benign)

Run:
    python src/explainability.py --num-local-examples 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from utils import (
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    DatasetSchema,
    get_logger,
    load_embeddings,
    save_json,
    set_global_seed,
)

logger = get_logger("explainability", log_file="stage6_explainability.log")

CLEANED_DIR = OUTPUTS_DIR / "cleaned"


# --------------------------------------------------------------------------- #
# 1. Global SHAP explanation
# --------------------------------------------------------------------------- #

def run_global_shap(model, X_background: np.ndarray, X_explain: np.ndarray, top_k: int = 20) -> dict:
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logger.info("Computing global SHAP values over %d embedding dimensions...", X_explain.shape[1])
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_explain)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:top_k]

    fig, ax = plt.subplots(figsize=(7, max(4, top_k * 0.3)))
    ax.barh(
        [f"emb_dim_{i}" for i in top_indices][::-1],
        mean_abs_shap[top_indices][::-1],
    )
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Global Feature Importance (Top {top_k} CodeBERT Embedding Dimensions)")
    fig.tight_layout()
    out_path = FIGURES_DIR / "global_shap_importance.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    result = {
        "top_k_dimensions": [int(i) for i in top_indices],
        "top_k_mean_abs_shap": [float(mean_abs_shap[i]) for i in top_indices],
        "figure_path": str(out_path),
    }
    save_json(result, METRICS_DIR / "stage6_global_shap.json")
    logger.info("Global SHAP explanation saved -> %s", out_path)
    return result


# --------------------------------------------------------------------------- #
# 2. Local, token-level explanation via leave-one-token-out occlusion
# --------------------------------------------------------------------------- #

def _mean_pool_embedding(model, tokenizer, code: str, device: str, max_length: int = 512) -> np.ndarray:
    import torch

    encoded = tokenizer(code, max_length=max_length, truncation=True, padding=False, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    last_hidden = outputs.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return (summed / counts).squeeze(0).cpu().numpy()


def _group_tokens_into_words(tokens: list[str]) -> list[list[int]]:
    """Group CodeBERT/RoBERTa BPE subword-token indices into whole-word
    groups, using the standard convention that a token starting with 'Ġ'
    (or being a newline 'Ċ') marks the start of a new word/segment.

    This directly targets the Stage-6 weakness: occluding a single subword
    (e.g. the "_" inside "str_cpy", or a lone "(" or "Ġ") barely perturbs
    the mean-pooled embedding and mostly reflects tokenization artifacts
    rather than anything a programmer would recognize as an explanation.
    Occluding whole words ("strcpy", "buffer", "sizeof") is both more
    faithful to how CodeBERT represents the token sequence's meaning and
    directly readable in the paper's explainability figures.
    """
    groups: list[list[int]] = []
    current: list[int] = []
    for i, tok in enumerate(tokens):
        starts_new_word = tok.startswith("Ġ") or tok.startswith("Ċ") or i == 0
        if starts_new_word and current:
            groups.append(current)
            current = []
        current.append(i)
    if current:
        groups.append(current)
    return groups


def _is_meaningful_word(tokens: list[str], indices: list[int]) -> bool:
    """Filter out whitespace-only / pure-punctuation word groups from the
    reported top-explanations (they are still occluded for completeness of
    the perturbation analysis, just not surfaced as 'important tokens',
    since they are not informative to a human reader)."""
    joined = "".join(tokens[i] for i in indices)
    stripped = joined.replace("Ġ", "").replace("Ċ", "").strip()
    if not stripped:
        return False
    # Drop groups that are pure punctuation (e.g. ";", "(", "->", "{}")
    return any(ch.isalnum() for ch in stripped)


# --------------------------------------------------------------------------- #
# 2. Local, word-level explanation via leave-one-word-out occlusion
# --------------------------------------------------------------------------- #

def explain_local_example(
    code: str,
    codebert_tokenizer,
    codebert_model,
    xgb_model,
    device: str,
    max_length: int = 512,
) -> dict:
    """Leave-one-WORD-out occlusion importance for a single function.

    Improvement over the original subword-level occlusion: tokens are first
    grouped into whole words (see _group_tokens_into_words), and an entire
    word is masked at once. Reported top_tokens excludes pure
    punctuation/whitespace groups so the explanation is directly readable
    (e.g. "strcpy", "buffer") rather than dominated by "Ġ", ";", "(".
    """
    tokens = codebert_tokenizer.tokenize(code, truncation=True, max_length=max_length)
    if len(tokens) == 0:
        return {"words": [], "importances": []}

    word_groups = _group_tokens_into_words(tokens)

    base_embedding = _mean_pool_embedding(codebert_model, codebert_tokenizer, code, device, max_length)
    base_proba = float(xgb_model.predict_proba(base_embedding.reshape(1, -1))[0, 1])

    mask_token = codebert_tokenizer.mask_token or codebert_tokenizer.unk_token

    # Cap the number of WORDS we occlude individually to keep this tractable
    # on CPU for long functions; sample uniformly if needed.
    max_words_to_test = 80
    group_indices_to_test = (
        list(range(len(word_groups)))
        if len(word_groups) <= max_words_to_test
        else sorted(np.random.choice(len(word_groups), max_words_to_test, replace=False).tolist())
    )

    importances = []
    for g_idx in group_indices_to_test:
        token_indices = word_groups[g_idx]
        occluded_tokens = tokens.copy()
        for t_idx in token_indices:
            occluded_tokens[t_idx] = mask_token
        occluded_code = codebert_tokenizer.convert_tokens_to_string(occluded_tokens)

        occluded_embedding = _mean_pool_embedding(
            codebert_model, codebert_tokenizer, occluded_code, device, max_length
        )
        occluded_proba = float(xgb_model.predict_proba(occluded_embedding.reshape(1, -1))[0, 1])

        importance = base_proba - occluded_proba  # drop in P(vulnerable) when word removed
        word_text = codebert_tokenizer.convert_tokens_to_string(
            [tokens[i] for i in token_indices]
        ).strip()

        importances.append({
            "word": word_text,
            "token_positions": token_indices,
            "importance": importance,
            "is_meaningful": _is_meaningful_word(tokens, token_indices),
        })

    importances.sort(key=lambda x: abs(x["importance"]), reverse=True)
    meaningful_only = [imp for imp in importances if imp["is_meaningful"]]

    return {
        "base_probability_vulnerable": base_proba,
        "num_tokens": len(tokens),
        "num_words": len(word_groups),
        "num_words_tested": len(group_indices_to_test),
        "top_words": meaningful_only[:15],
        "top_words_including_punctuation": importances[:15],
    }


def run_local_explanations(num_examples: int = 5) -> list[dict]:
    import torch
    from transformers import AutoModel, AutoTokenizer
    import xgboost as xgb

    from evaluation import load_test_predictions

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/codebert-base"
    )

    codebert_model = AutoModel.from_pretrained(
        "microsoft/codebert-base"
    ).to(device).eval()

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(
        str(
            MODELS_DIR
            / "xgboost_proposed_model.json"
        )
    )

    # -------------------------------------------------------
    # Load cleaned TEST functions
    # -------------------------------------------------------

    test_path = CLEANED_DIR / "test_clean.jsonl"

    records = []

    with test_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            records.append(
                json.loads(line)
            )

    # -------------------------------------------------------
    # Load FINAL proposed-model predictions
    # -------------------------------------------------------

    y_true, y_proba, threshold = (
        load_test_predictions(
            "Proposed_CodeBERT_XGBoost"
        )
    )

    y_pred = (
        y_proba >= threshold
    ).astype(int)

    if len(records) != len(y_true):
        raise ValueError(
            "Number of TEST records does not match "
            "the saved proposed-model predictions."
        )

    # -------------------------------------------------------
    # Identify TP / FN / FP / TN cases
    # -------------------------------------------------------

    tp_indices = np.where(
        (y_true == 1) & (y_pred == 1)
    )[0]

    fn_indices = np.where(
        (y_true == 1) & (y_pred == 0)
    )[0]

    fp_indices = np.where(
        (y_true == 0) & (y_pred == 1)
    )[0]

    tn_indices = np.where(
        (y_true == 0) & (y_pred == 0)
    )[0]

    logger.info(
        "Available prediction outcomes: "
        "TP=%d FN=%d FP=%d TN=%d",
        len(tp_indices),
        len(fn_indices),
        len(fp_indices),
        len(tn_indices),
    )

    # -------------------------------------------------------
    # Preferred paper examples:
    # 2 TP + 1 FN + 1 FP + 1 TN
    # -------------------------------------------------------

    selection_plan = [
        ("TP", tp_indices, 2),
        ("FN", fn_indices, 1),
        ("FP", fp_indices, 1),
        ("TN", tn_indices, 1),
    ]

    selected_examples = []
    used_indices = set()

    for outcome, indices, count in selection_plan:

        for idx in indices[:count]:

            idx = int(idx)

            selected_examples.append(
                (idx, outcome)
            )

            used_indices.add(idx)

    # -------------------------------------------------------
    # If user requests more than five examples,
    # fill remaining positions deterministically.
    # -------------------------------------------------------

    if len(selected_examples) < num_examples:

        extra_groups = [
            ("TP", tp_indices),
            ("FN", fn_indices),
            ("FP", fp_indices),
            ("TN", tn_indices),
        ]

        for outcome, indices in extra_groups:

            for idx in indices:

                idx = int(idx)

                if idx in used_indices:
                    continue

                selected_examples.append(
                    (idx, outcome)
                )

                used_indices.add(idx)

                if (
                    len(selected_examples)
                    >= num_examples
                ):
                    break

            if (
                len(selected_examples)
                >= num_examples
            ):
                break

    selected_examples = (
        selected_examples[:num_examples]
    )

    # -------------------------------------------------------
    # Generate local explanations
    # -------------------------------------------------------

    all_explanations = []

    for i, (
        record_index,
        outcome,
    ) in enumerate(selected_examples):

        record = records[record_index]

        logger.info(
            "Explaining local example %d/%d "
            "(outcome=%s, true=%d, predicted=%d, "
            "probability=%.4f)...",
            i + 1,
            len(selected_examples),
            outcome,
            int(y_true[record_index]),
            int(y_pred[record_index]),
            float(y_proba[record_index]),
        )

        explanation = explain_local_example(
            record["code"],
            tokenizer,
            codebert_model,
            xgb_model,
            device,
        )

        explanation["test_index"] = int(
            record_index
        )

        explanation[
            "prediction_outcome"
        ] = outcome

        explanation["true_label"] = int(
            y_true[record_index]
        )

        explanation["predicted_label"] = int(
            y_pred[record_index]
        )

        explanation[
            "predicted_probability"
        ] = float(
            y_proba[record_index]
        )

        explanation[
            "threshold_used"
        ] = float(threshold)

        explanation[
            "code_preview"
        ] = record["code"][:200]

        all_explanations.append(
            explanation
        )

    save_json(
        all_explanations,
        METRICS_DIR
        / "stage6_local_explanations.json",
    )

    logger.info(
        "Local explanations saved for %d "
        "prediction-aware examples.",
        len(all_explanations),
    )

    return all_explanations


# --------------------------------------------------------------------------- #

def run_explainability(num_local_examples: int = 5) -> None:
    set_global_seed()
    import xgboost as xgb

    X_train, _ = load_embeddings("train")
    X_test, _ = load_embeddings("test")

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(MODELS_DIR / "xgboost_proposed_model.json"))

    # Background/explain sample sizes capped for tractability on large datasets.
    rng = np.random.RandomState(42)
    bg_idx = rng.choice(len(X_train), size=min(200, len(X_train)), replace=False)
    ex_idx = rng.choice(len(X_test), size=min(200, len(X_test)), replace=False)

    run_global_shap(xgb_model, X_train[bg_idx], X_test[ex_idx])
    run_local_explanations(num_examples=num_local_examples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 6: Explainability (global SHAP + local token-level)")
    parser.add_argument("--num-local-examples", type=int, default=5)
    args = parser.parse_args()
    run_explainability(args.num_local_examples)
