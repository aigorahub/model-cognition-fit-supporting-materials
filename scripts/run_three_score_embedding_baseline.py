#!/usr/bin/env python3
"""Evaluate three-score EmbeddingGemma baselines for the cooked-ham study.

The target is deliberately narrow: for each consumer-product pair, compute
three actual-vs-ideal similarity scores, one for visual, texture, and flavor,
then predict liking from those three numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "analysis" / "raw" / "dataset.xlsx"
OUT = ROOT / "analysis" / "embedding_baselines"
CACHE = OUT / "cache"

MODALITIES = ["visual", "texture", "flavor"]
ACTUAL_COLS = {
    "visual": "DescriptionVisual",
    "texture": "DescriptionTexture",
    "flavor": "DescriptionFlavor",
}
IDEAL_COLS = {
    "visual": "IdealVisual",
    "texture": "IdealTexture",
    "flavor": "IdealFlavor",
}
MODALITY_QUESTIONS = {
    "visual": "How similar is the product's visual appearance to this consumer's ideal ham?",
    "texture": "How similar is the product's texture to this consumer's ideal ham?",
    "flavor": "How similar is the product's flavor to this consumer's ideal ham?",
}


@dataclass
class ScoreSet:
    name: str
    scores: np.ndarray
    description: str


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else re.sub(r"\s+", " ", text)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 1e-12).astype(np.float32)


def load_data(limit: int | None) -> pd.DataFrame:
    sensory = pd.read_excel(DATASET, sheet_name="product sensory properties")
    ideal = pd.read_excel(DATASET, sheet_name="consumer questionnaire (home)")
    ideal = ideal.drop_duplicates(subset=["Consumer"])
    keep = ["Consumer", "Product", "Liking", *ACTUAL_COLS.values()]
    merged = sensory[keep].merge(ideal[["Consumer", *IDEAL_COLS.values()]], on="Consumer", how="inner")
    merged = merged.rename(columns={"Liking": "liking"})
    for col in [*ACTUAL_COLS.values(), *IDEAL_COLS.values()]:
        merged[col] = merged[col].map(clean_text)
    if limit:
        merged = merged.head(limit).copy()
    return merged.reset_index(drop=True)


def cache_key(texts: list[str], prompt_name: str | None, model_id: str) -> str:
    digest = hashlib.sha256()
    digest.update(model_id.encode())
    digest.update(str(prompt_name).encode())
    for text in texts:
        digest.update(b"\0")
        digest.update(text.encode("utf-8"))
    return digest.hexdigest()[:16]


def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    prompt_name: str | None,
    model_id: str,
    batch_size: int,
) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    key = cache_key(texts, prompt_name, model_id)
    path = CACHE / f"{key}.npy"
    meta_path = CACHE / f"{key}.json"
    if path.exists():
        return normalize_rows(np.load(path))
    kwargs = {"normalize_embeddings": True, "batch_size": batch_size, "show_progress_bar": True}
    if prompt_name:
        kwargs["prompt_name"] = prompt_name
    embeddings = normalize_rows(model.encode(texts, **kwargs))
    np.save(path, embeddings)
    meta_path.write_text(
        json.dumps(
            {
                "model_id": model_id,
                "prompt_name": prompt_name,
                "n_texts": len(texts),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return embeddings


def encode_unique(
    model: SentenceTransformer,
    texts: list[str],
    prompt_name: str | None,
    model_id: str,
    batch_size: int,
) -> np.ndarray:
    unique = list(dict.fromkeys(texts))
    embeddings = encode_texts(model, unique, prompt_name, model_id, batch_size)
    index = {text: i for i, text in enumerate(unique)}
    return embeddings[[index[text] for text in texts]]


def cosine_scores(actual: np.ndarray, ideal: np.ndarray) -> np.ndarray:
    return np.sum(normalize_rows(actual) * normalize_rows(ideal), axis=1)


def all_modality_text(df: pd.DataFrame, cols: dict[str, str]) -> list[str]:
    return (
        df[[cols["visual"], cols["texture"], cols["flavor"]]]
        .fillna("")
        .agg(" ".join, axis=1)
        .map(clean_text)
        .tolist()
    )


def tagged_text(modality: str, text: str) -> str:
    label = {"visual": "Visual", "texture": "Texture", "flavor": "Flavor"}[modality]
    return f"{label} description: {text if text else '(empty)'}"


def qa_text(modality: str, text: str) -> str:
    return f"Question: {MODALITY_QUESTIONS[modality]}\nAnswer: {text if text else '(empty)'}"


def build_score_sets(
    df: pd.DataFrame,
    model: SentenceTransformer,
    model_id: str,
    batch_size: int,
) -> list[ScoreSet]:
    whole_actual = all_modality_text(df, ACTUAL_COLS)
    whole_ideal = all_modality_text(df, IDEAL_COLS)
    whole_actual_emb = encode_unique(model, whole_actual, "STS", model_id, batch_size)
    whole_ideal_emb = encode_unique(model, whole_ideal, "STS", model_id, batch_size)
    whole_score = cosine_scores(whole_actual_emb, whole_ideal_emb)[:, None]

    raw_scores = []
    tagged_scores = []
    qa_scores = []
    retrieval_scores = []
    for modality in MODALITIES:
        actual_text = df[ACTUAL_COLS[modality]].tolist()
        ideal_text = df[IDEAL_COLS[modality]].tolist()

        actual_raw = encode_unique(model, actual_text, "STS", model_id, batch_size)
        ideal_raw = encode_unique(model, ideal_text, "STS", model_id, batch_size)
        raw_scores.append(cosine_scores(actual_raw, ideal_raw))

        actual_tagged = encode_unique(model, [tagged_text(modality, text) for text in actual_text], "STS", model_id, batch_size)
        ideal_tagged = encode_unique(model, [tagged_text(modality, text) for text in ideal_text], "STS", model_id, batch_size)
        tagged_scores.append(cosine_scores(actual_tagged, ideal_tagged))

        actual_qa = encode_unique(model, [qa_text(modality, text) for text in actual_text], "STS", model_id, batch_size)
        ideal_qa = encode_unique(model, [qa_text(modality, text) for text in ideal_text], "STS", model_id, batch_size)
        qa_scores.append(cosine_scores(actual_qa, ideal_qa))

        actual_retrieval = encode_unique(model, [tagged_text(modality, text) for text in actual_text], "Retrieval-document", model_id, batch_size)
        ideal_retrieval = encode_unique(model, [tagged_text(modality, text) for text in ideal_text], "Retrieval-document", model_id, batch_size)
        retrieval_scores.append(cosine_scores(actual_retrieval, ideal_retrieval))

    raw_matrix = np.vstack(raw_scores).T
    tagged_matrix = np.vstack(tagged_scores).T
    qa_matrix = np.vstack(qa_scores).T
    retrieval_matrix = np.vstack(retrieval_scores).T

    return [
        ScoreSet("whole_text_one_score", whole_score, "One cosine score after concatenating visual, texture, and flavor."),
        ScoreSet("three_modality_scores_raw_sts", raw_matrix, "Three STS cosine scores from the full comments in each modality."),
        ScoreSet("three_modality_scores_tagged_sts", tagged_matrix, "Three STS cosine scores with each comment labeled by modality."),
        ScoreSet("three_modality_scores_qa_sts", qa_matrix, "Three STS cosine scores using Q+A framing to focus the modality."),
        ScoreSet("three_modality_scores_retrieval_doc", retrieval_matrix, "Three cosine scores using Retrieval-document embeddings."),
    ]


def regressors() -> dict[str, object]:
    return {
        "dummy": DummyRegressor(strategy="mean"),
        "hist_gbr": HistGradientBoostingRegressor(max_iter=250, learning_rate=0.04, l2_regularization=0.05, random_state=42),
        "gbr": GradientBoostingRegressor(n_estimators=250, learning_rate=0.035, max_depth=2, random_state=42),
        "extra_trees": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=8, random_state=42, n_jobs=-1),
        "svr_rbf": make_pipeline(StandardScaler(), SVR(C=2.0, epsilon=0.25, gamma="scale")),
    }


def evaluate(
    score_sets: list[ScoreSet],
    y: np.ndarray,
    groups: np.ndarray,
    splits: int,
) -> pd.DataFrame:
    rows = []
    for score_set in score_sets:
        split_iter = GroupShuffleSplit(n_splits=splits, test_size=0.2, random_state=42).split(score_set.scores, y, groups)
        for split_id, (train_idx, test_idx) in enumerate(split_iter):
            x_train = np.nan_to_num(score_set.scores[train_idx].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
            x_test = np.nan_to_num(score_set.scores[test_idx].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
            for model_name, estimator in regressors().items():
                if score_set.name != "whole_text_one_score" and model_name == "dummy":
                    continue
                if score_set.name == "whole_text_one_score" and model_name == "dummy":
                    feature_name = "dummy_mean"
                else:
                    feature_name = score_set.name
                est = clone(estimator)
                est.fit(x_train, y[train_idx])
                pred = np.asarray(est.predict(x_test), dtype=np.float64)
                if not np.isfinite(pred).all():
                    pred = np.nan_to_num(pred, nan=float(np.mean(y[train_idx])), posinf=10.0, neginf=0.0)
                pred = np.clip(pred, 0.0, 10.0)
                rows.append(
                    {
                        "feature_set": feature_name,
                        "model": model_name,
                        "split": split_id,
                        "r2": r2_score(y[test_idx], pred),
                        "mae": mean_absolute_error(y[test_idx], pred),
                        "rmse": mean_squared_error(y[test_idx], pred) ** 0.5,
                        "n_train": len(train_idx),
                        "n_test": len(test_idx),
                        "n_scores": score_set.scores.shape[1],
                        "description": "Mean-only baseline." if feature_name == "dummy_mean" else score_set.description,
                    }
                )
    return pd.DataFrame(rows)


def write_score_table(score_sets: list[ScoreSet], df: pd.DataFrame, suffix: str) -> None:
    rows = []
    for score_set in score_sets:
        if score_set.scores.shape[1] != 3:
            continue
        for i, modality in enumerate(MODALITIES):
            rows.append(
                {
                    "feature_set": score_set.name,
                    "modality": modality,
                    "mean_similarity": float(np.mean(score_set.scores[:, i])),
                    "sd_similarity": float(np.std(score_set.scores[:, i], ddof=1)),
                    "correlation_with_liking": float(np.corrcoef(score_set.scores[:, i], df["liking"].to_numpy())[0, 1]),
                }
            )
    pd.DataFrame(rows).to_csv(OUT / f"three_score_similarity_descriptives_{suffix}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--splits", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--model-id", default="google/embeddinggemma-300m")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    df = load_data(args.limit)
    y = df["liking"].to_numpy(dtype=np.float64)
    groups = df["Consumer"].to_numpy()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Rows: {len(df)}; consumers: {df['Consumer'].nunique()}; device: {device}")
    print("Target: predict liking from three visual, texture, and flavor similarity scores.")

    model = SentenceTransformer(args.model_id).to(device=device)
    score_sets = build_score_sets(df, model, args.model_id, args.batch_size)
    results = evaluate(score_sets, y, groups, args.splits)

    suffix = "smoke" if args.limit else "full"
    results_path = OUT / f"three_score_embedding_results_{suffix}.csv"
    summary_path = OUT / f"three_score_embedding_summary_{suffix}.csv"
    metadata_path = OUT / f"three_score_embedding_metadata_{suffix}.json"
    results.to_csv(results_path, index=False)
    summary = (
        results.groupby(["feature_set", "model"], as_index=False)
        .agg(
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            r2_max=("r2", "max"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            n_scores=("n_scores", "first"),
            splits=("split", "nunique"),
            description=("description", "first"),
        )
        .sort_values(["r2_mean", "mae_mean"], ascending=[False, True])
    )
    summary.to_csv(summary_path, index=False)
    write_score_table(score_sets, df, suffix)
    metadata_path.write_text(
        json.dumps(
            {
                "model_id": args.model_id,
                "n_rows": len(df),
                "n_consumers": int(df["Consumer"].nunique()),
                "splits": args.splits,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source_dataset": str(DATASET.relative_to(ROOT)),
                "target": "Predict 0-10 liking from three actual-vs-ideal modality similarity scores.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("\nTop results:")
    print(summary.head(20).to_string(index=False))
    print(f"\nWrote {results_path}\nWrote {summary_path}\nWrote {metadata_path}")


if __name__ == "__main__":
    main()
