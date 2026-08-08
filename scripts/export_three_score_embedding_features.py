#!/usr/bin/env python3
"""Export three-score EmbeddingGemma feature CSVs for remote ML runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]

from run_three_score_embedding_baseline import (
    MODALITIES,
    OUT,
    SentenceTransformer,
    build_score_sets,
    load_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="google/embeddinggemma-300m")
    parser.add_argument("--batch-size", default=96, type=int)
    parser.add_argument("--test-size", default=0.2, type=float)
    parser.add_argument("--random-state", default=42, type=int)
    parser.add_argument(
        "--output-dir",
        default=OUT / "tabpfn_inputs",
        type=Path,
        help="Directory for generated TabPFN input CSVs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(limit=None)
    split = pd.Series("train", index=df.index, dtype="object")
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
    _train_idx, test_idx = next(splitter.split(df, df["liking"], groups=df["Consumer"]))
    split.iloc[test_idx] = "holdout"

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(args.model_id).to(device=device)
    score_sets = build_score_sets(df, model, args.model_id, args.batch_size)

    manifest = {
        "model_id": args.model_id,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "rows": int(len(df)),
        "consumers": int(df["Consumer"].nunique()),
        "holdout_rows": int((split == "holdout").sum()),
        "holdout_consumers": int(df.loc[split == "holdout", "Consumer"].nunique()),
        "feature_columns": ["visual_similarity", "texture_similarity", "flavor_similarity"],
        "files": [],
    }
    for score_set in score_sets:
        if score_set.scores.shape[1] != 3:
            continue
        out = pd.DataFrame(
            {
                "consumer": df["Consumer"],
                "product": df["Product"],
                "liking": df["liking"],
                "split": split,
            }
        )
        for index, modality in enumerate(MODALITIES):
            out[f"{modality}_similarity"] = score_set.scores[:, index]
        path = args.output_dir / f"{score_set.name}_seed{args.random_state}.csv"
        out.to_csv(path, index=False)
        try:
            rel_path = str(path.resolve().relative_to(ROOT))
        except ValueError:
            rel_path = str(path)
        manifest["files"].append(
            {
                "feature_set": score_set.name,
                "path": rel_path,
                "description": score_set.description,
            }
        )

    manifest_path = args.output_dir / f"manifest_seed{args.random_state}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
