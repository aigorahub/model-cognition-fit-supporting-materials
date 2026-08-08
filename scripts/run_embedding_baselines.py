#!/usr/bin/env python3
"""Run sklearn embedding baselines for the cooked-ham IFC study.

This script is intentionally independent of TabPFN. It tests whether better
EmbeddingGemma feature construction can improve the original raw cosine
baseline before the slower TabPFN refresh.
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
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
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
    "visual": "Décrire l'apparence visuelle du jambon: couleur, gras, humidité, homogénéité, tranche.",
    "texture": "Décrire la texture en bouche du jambon: tendre, moelleux, sec, caoutchouteux, fibreux.",
    "flavor": "Décrire le goût du jambon: salé, fumé, goût de viande, arômes, arrière-goût.",
}


@dataclass
class FeatureSet:
    name: str
    low: np.ndarray
    high: np.ndarray | None
    description: str


def sanitize_matrix(values: np.ndarray) -> np.ndarray:
    """Return a finite float64 matrix for sklearn."""
    matrix = np.asarray(values, dtype=np.float64)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(matrix, -1e6, 1e6)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    matrix = sanitize_matrix(values)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 1e-12).astype(np.float32)


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else re.sub(r"\s+", " ", text)


def qa(question: str, answer: str) -> str:
    answer = answer if answer else "(empty)"
    return f"Question: {question}\nAnswer: {answer}"


def split_chunks(text: str) -> list[str]:
    text = clean_text(text)
    if not text:
        return ["(empty)"]
    chunks = {text}
    for part in re.split(r"[.!?;]|,| mais | cependant | toutefois | aussi | et | puis ", text, flags=re.I):
        part = part.strip()
        if part:
            chunks.add(part)
    words = text.split()
    for size in (8, 12, 16):
        if len(words) >= size:
            step = max(1, size // 2)
            for start in range(0, len(words) - size + 1, step):
                chunks.add(" ".join(words[start : start + size]))
    return sorted(chunks, key=lambda s: (len(s.split()), len(s)), reverse=True)


def load_data(limit: int | None = None) -> pd.DataFrame:
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
    embeddings = model.encode(texts, **kwargs)
    embeddings = normalize_rows(embeddings)
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
    emb_unique = encode_texts(model, unique, prompt_name, model_id, batch_size)
    index = {text: i for i, text in enumerate(unique)}
    return emb_unique[[index[text] for text in texts]]


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(sanitize_matrix(a) * sanitize_matrix(b), axis=1)


def modality_texts(df: pd.DataFrame, prefix: str, modality: str) -> list[str]:
    cols = ACTUAL_COLS if prefix == "actual" else IDEAL_COLS
    return df[cols[modality]].tolist()


def all_texts(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = ACTUAL_COLS if prefix == "actual" else IDEAL_COLS
    return (
        df[[cols["visual"], cols["texture"], cols["flavor"]]]
        .fillna("")
        .agg(" ".join, axis=1)
        .map(clean_text)
        .tolist()
    )


def pooled_chunk_embeddings(
    model: SentenceTransformer,
    texts: list[str],
    modality: str,
    model_id: str,
    batch_size: int,
    top_k: int = 3,
) -> np.ndarray:
    query = encode_texts(
        model,
        [MODALITY_QUESTIONS[modality]],
        "Retrieval-query",
        model_id,
        batch_size,
    )[0]
    all_chunks: list[str] = []
    spans: list[tuple[int, int]] = []
    for text in texts:
        chunks = split_chunks(text)
        start = len(all_chunks)
        all_chunks.extend(chunks)
        spans.append((start, len(all_chunks)))
    chunk_embs = encode_texts(model, all_chunks, "Retrieval-document", model_id, batch_size)
    pooled = []
    max_scores = []
    for start, end in spans:
        embs = sanitize_matrix(chunk_embs[start:end])
        scores = embs @ sanitize_matrix(query[None, :]).ravel()
        scores = np.nan_to_num(scores, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
        take = np.argsort(scores)[-min(top_k, len(scores)) :]
        weights = np.maximum(scores[take], 0)
        if weights.sum() <= 1e-12:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights = weights / weights.sum()
        vec = (embs[take] * weights[:, None]).sum(axis=0)
        norm = np.linalg.norm(vec)
        pooled.append(vec / norm if norm > 1e-12 else vec)
        max_scores.append(float(scores.max()))
    return np.asarray(pooled, dtype=np.float32), np.asarray(max_scores, dtype=np.float32)


def make_feature_set(name: str, low: np.ndarray, high: np.ndarray | None, description: str) -> FeatureSet:
    return FeatureSet(
        name,
        sanitize_matrix(low),
        sanitize_matrix(high) if high is not None else None,
        description,
    )


def build_feature_sets(
    df: pd.DataFrame,
    model: SentenceTransformer,
    model_id: str,
    batch_size: int,
    include_chunked: bool,
) -> list[FeatureSet]:
    actual_all = all_texts(df, "actual")
    ideal_all = all_texts(df, "ideal")
    actual_all_emb = encode_unique(model, actual_all, "STS", model_id, batch_size)
    ideal_all_emb = encode_unique(model, ideal_all, "STS", model_id, batch_size)
    full_cos = cosine(actual_all_emb, ideal_all_emb)[:, None]
    full_diff = ideal_all_emb - actual_all_emb

    actual_mod = {}
    ideal_mod = {}
    actual_mod_qa = {}
    ideal_mod_qa = {}
    mod_cos = []
    mod_cos_qa = []
    low_parts = [full_cos]
    length_parts = []
    for modality in MODALITIES:
        a_text = modality_texts(df, "actual", modality)
        i_text = modality_texts(df, "ideal", modality)
        actual_mod[modality] = encode_unique(model, a_text, "STS", model_id, batch_size)
        ideal_mod[modality] = encode_unique(model, i_text, "STS", model_id, batch_size)
        mod_cos.append(cosine(actual_mod[modality], ideal_mod[modality]))
        a_qa = [qa(MODALITY_QUESTIONS[modality], text) for text in a_text]
        i_qa = [qa(MODALITY_QUESTIONS[modality], text) for text in i_text]
        actual_mod_qa[modality] = encode_unique(model, a_qa, "STS", model_id, batch_size)
        ideal_mod_qa[modality] = encode_unique(model, i_qa, "STS", model_id, batch_size)
        mod_cos_qa.append(cosine(actual_mod_qa[modality], ideal_mod_qa[modality]))
        length_parts.append(np.log1p([len(x.split()) for x in a_text]))
        length_parts.append(np.log1p([len(x.split()) for x in i_text]))
    mod_cos_matrix = np.vstack(mod_cos).T
    mod_cos_qa_matrix = np.vstack(mod_cos_qa).T
    length_matrix = np.vstack(length_parts).T
    low_mod = np.hstack([full_cos, mod_cos_matrix, length_matrix])
    low_mod_qa = np.hstack([full_cos, mod_cos_matrix, mod_cos_qa_matrix, length_matrix])
    mod_diff = np.hstack([ideal_mod[m] - actual_mod[m] for m in MODALITIES])
    mod_absdiff = np.hstack([np.abs(ideal_mod[m] - actual_mod[m]) for m in MODALITIES])
    mod_diff_qa = np.hstack([ideal_mod_qa[m] - actual_mod_qa[m] for m in MODALITIES])

    features = [
        make_feature_set("dummy_mean", np.zeros((len(df), 1), dtype=np.float32), None, "Mean-only baseline."),
        make_feature_set("whole_text_cosine", full_cos, None, "STS embedding cosine between all actual and all ideal comments."),
        make_feature_set("whole_text_cosine_plus_diff", full_cos, full_diff, "Whole-text cosine plus PCA of ideal-minus-actual vector."),
        make_feature_set("modality_cosines", low_mod, None, "Whole cosine, three modality cosines, and text-length controls."),
        make_feature_set("modality_cosines_plus_diff", low_mod, mod_diff, "Modality cosines plus PCA of modality-specific vector differences."),
        make_feature_set("modality_cosines_plus_absdiff", low_mod, mod_absdiff, "Modality cosines plus PCA of absolute modality differences."),
        make_feature_set("qa_modality_cosines_plus_diff", low_mod_qa, mod_diff_qa, "Q+A modality embeddings plus PCA of modality-specific differences."),
    ]

    if include_chunked:
        chunk_actual = {}
        chunk_ideal = {}
        chunk_scores = []
        chunk_cos = []
        for modality in MODALITIES:
            a_text = modality_texts(df, "actual", modality)
            i_text = modality_texts(df, "ideal", modality)
            chunk_actual[modality], a_score = pooled_chunk_embeddings(model, a_text, modality, model_id, batch_size)
            chunk_ideal[modality], i_score = pooled_chunk_embeddings(model, i_text, modality, model_id, batch_size)
            chunk_scores.extend([a_score, i_score])
            chunk_cos.append(cosine(chunk_actual[modality], chunk_ideal[modality]))
        chunk_cos_matrix = np.vstack(chunk_cos).T
        chunk_scores_matrix = np.vstack(chunk_scores).T
        chunk_low = np.hstack([full_cos, chunk_cos_matrix, chunk_scores_matrix, length_matrix])
        chunk_diff = np.hstack([chunk_ideal[m] - chunk_actual[m] for m in MODALITIES])
        features.append(
            make_feature_set(
                "retrieval_chunk_pooling_plus_diff",
                chunk_low,
                chunk_diff,
                "Question-conditioned top-k chunk pooling plus PCA of modality differences.",
            )
        )
    return features


def regressors() -> dict[str, object]:
    return {
        "dummy": DummyRegressor(strategy="mean"),
        "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 4, 9), cv=5)),
        "hist_gbr": HistGradientBoostingRegressor(max_iter=250, learning_rate=0.04, l2_regularization=0.05, random_state=42),
        "gbr": GradientBoostingRegressor(n_estimators=350, learning_rate=0.03, max_depth=2, random_state=42),
        "extra_trees": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1),
        "svr_rbf": make_pipeline(StandardScaler(), SVR(C=3.0, epsilon=0.2, gamma="scale")),
    }


def evaluate_feature_set(
    fs: FeatureSet,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    pca_components: int,
) -> list[dict[str, object]]:
    rows = []
    split_iter = GroupShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42).split(fs.low, y, groups)
    for split_id, (train_idx, test_idx) in enumerate(split_iter):
        x_train_low = sanitize_matrix(fs.low[train_idx])
        x_test_low = sanitize_matrix(fs.low[test_idx])
        if fs.high is not None:
            n_comp = min(pca_components, fs.high.shape[1], len(train_idx) - 1)
            high_train = sanitize_matrix(fs.high[train_idx])
            high_test = sanitize_matrix(fs.high[test_idx])
            pca = PCA(n_components=n_comp, random_state=42, svd_solver="full")
            train_high = pca.fit_transform(high_train)
            test_high = pca.transform(high_test)
            x_train = sanitize_matrix(np.hstack([x_train_low, train_high]))
            x_test = sanitize_matrix(np.hstack([x_test_low, test_high]))
        else:
            x_train = x_train_low
            x_test = x_test_low
        for model_name, estimator in regressors().items():
            if fs.name == "dummy_mean" and model_name != "dummy":
                continue
            if fs.name != "dummy_mean" and model_name == "dummy":
                continue
            est = clone(estimator)
            est.fit(x_train, y[train_idx])
            pred = np.asarray(est.predict(x_test), dtype=np.float64)
            if not np.isfinite(pred).all():
                pred = np.nan_to_num(pred, nan=float(np.mean(y[train_idx])), posinf=10.0, neginf=0.0)
            pred = np.clip(pred, 0.0, 10.0)
            rows.append(
                {
                    "feature_set": fs.name,
                    "model": model_name,
                    "split": split_id,
                    "r2": r2_score(y[test_idx], pred),
                    "mae": mean_absolute_error(y[test_idx], pred),
                    "rmse": mean_squared_error(y[test_idx], pred) ** 0.5,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "pca_components": pca_components if fs.high is not None else 0,
                    "description": fs.description,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke tests.")
    parser.add_argument("--splits", type=int, default=20)
    parser.add_argument("--pca-components", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model-id", default="google/embeddinggemma-300m")
    parser.add_argument("--skip-chunked", action="store_true", help="Skip retrieval chunk-pooling features.")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    df = load_data(args.limit)
    y = df["liking"].to_numpy(dtype=np.float32)
    groups = df["Consumer"].to_numpy()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Rows: {len(df)}; consumers: {df['Consumer'].nunique()}; device: {device}")
    model = SentenceTransformer(args.model_id).to(device=device)
    features = build_feature_sets(
        df,
        model,
        args.model_id,
        args.batch_size,
        include_chunked=not args.skip_chunked,
    )
    all_rows = []
    for fs in features:
        print(f"\nEvaluating {fs.name}: {fs.description}")
        all_rows.extend(evaluate_feature_set(fs, y, groups, args.splits, args.pca_components))
    results = pd.DataFrame(all_rows)
    suffix = "smoke" if args.limit else "full"
    result_path = OUT / f"embedding_baseline_results_{suffix}.csv"
    summary_path = OUT / f"embedding_baseline_summary_{suffix}.csv"
    metadata_path = OUT / f"embedding_baseline_metadata_{suffix}.json"
    results.to_csv(result_path, index=False)
    summary = (
        results.groupby(["feature_set", "model"], as_index=False)
        .agg(
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            r2_max=("r2", "max"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            splits=("split", "nunique"),
            description=("description", "first"),
        )
        .sort_values(["r2_mean", "mae_mean"], ascending=[False, True])
    )
    summary.to_csv(summary_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "model_id": args.model_id,
                "n_rows": len(df),
                "n_consumers": int(df["Consumer"].nunique()),
                "splits": args.splits,
                "pca_components": args.pca_components,
                "skip_chunked": args.skip_chunked,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source_dataset": str(DATASET.relative_to(ROOT)),
                "guide_source": "EMBEDDING_GUIDE.md",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("\nTop results:")
    print(summary.head(15).to_string(index=False))
    print(f"\nWrote {result_path}\nWrote {summary_path}\nWrote {metadata_path}")


if __name__ == "__main__":
    main()
