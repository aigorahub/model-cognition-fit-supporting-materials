#!/usr/bin/env python3
"""Run local TabPFN v2 on the three-score embedding feature CSVs."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["visual_similarity", "texture_similarity", "flavor_similarity"]


def default_device() -> str:
    try:
        import torch

        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tabpfn-python",
        type=Path,
        default=None,
        help="Python with TabPFN installed (or set TABPFN_PYTHON). Required.",
    )
    parser.add_argument(
        "--runner",
        type=Path,
        default=None,
        help="Path to tabpfn-launcher scripts/run_tabpfn_csv.py (or set TABPFN_RUNNER). Required.",
    )
    parser.add_argument("--input-dir", default=ROOT / "analysis" / "embedding_baselines" / "tabpfn_inputs", type=Path)
    parser.add_argument("--outputs-dir", default=ROOT / "analysis" / "embedding_baselines" / "tabpfn_local_outputs", type=Path)
    parser.add_argument("--summary-csv", default=ROOT / "analysis" / "embedding_baselines" / "three_score_embedding_tabpfn_summary.csv", type=Path)
    parser.add_argument("--device", default=default_device(), choices=["mps", "cpu", "auto"])
    parser.add_argument("--prediction-batch-size", default=64, type=int)
    parser.add_argument("--drop-artifacts", action="store_true", help="Remove per-row prediction artifact folders after each run.")
    return parser.parse_args()


def variants(input_dir: Path) -> dict[str, Path]:
    return {
        "qa_sts": input_dir / "three_modality_scores_qa_sts_seed42.csv",
        "tagged_sts": input_dir / "three_modality_scores_tagged_sts_seed42.csv",
        "raw_sts": input_dir / "three_modality_scores_raw_sts_seed42.csv",
        "retrieval_doc": input_dir / "three_modality_scores_retrieval_doc_seed42.csv",
    }


def run_variant(args: argparse.Namespace, name: str, csv_path: Path) -> dict[str, object]:
    artifact_dir = args.outputs_dir / f"{name}_seed42_artifacts"
    output_json = args.outputs_dir / f"{name}_seed42.json"
    shutil.rmtree(artifact_dir, ignore_errors=True)
    output_json.unlink(missing_ok=True)
    command = [
        str(args.tabpfn_python),
        str(args.runner),
        "--csv",
        str(csv_path),
        "--target",
        "liking",
        "--model",
        "v2",
        "--task",
        "regression",
        "--features-json",
        json.dumps(FEATURES, separators=(",", ":")),
        "--split-column",
        "split",
        "--train-split-value",
        "train",
        "--test-split-value",
        "holdout",
        "--device",
        args.device,
        "--prediction-batch-size",
        str(args.prediction_batch_size),
        "--artifacts-dir",
        str(artifact_dir),
        "--output-json",
        str(output_json),
        "--save-predictions",
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    result = json.loads(output_json.read_text(encoding="utf-8"))
    if args.drop_artifacts:
        shutil.rmtree(artifact_dir, ignore_errors=True)
    return {
        "engine": "tabpfn_local",
        "validation": "single explicit GroupShuffleSplit seed42 by consumer",
        "feature_set": name,
        "model": f"TabPFN {result['model']}",
        "r2_mean": result["r2"],
        "r2_std": "",
        "mae_mean": result["mae"],
        "rmse_mean": result["rmse"],
        "pearsonr": "",
        "train_rows": result["train_rows"],
        "test_rows": result["test_rows"],
        "device": result["device"],
        "elapsed_seconds": result["elapsed_seconds"],
        "tabpfn_package": result["tabpfn_package"],
        "torch_package": result["torch_package"],
    }


def main() -> None:
    import os

    args = parse_args()
    if args.tabpfn_python is None:
        env = os.environ.get("TABPFN_PYTHON")
        args.tabpfn_python = Path(env) if env else None
    if args.runner is None:
        env = os.environ.get("TABPFN_RUNNER")
        args.runner = Path(env) if env else None
    if args.tabpfn_python is None or not Path(args.tabpfn_python).exists():
        raise SystemExit(
            "Pass --tabpfn-python PATH (or set TABPFN_PYTHON) to a Python with TabPFN installed."
        )
    if args.runner is None or not Path(args.runner).exists():
        raise SystemExit(
            "Pass --runner PATH (or set TABPFN_RUNNER) to tabpfn-launcher scripts/run_tabpfn_csv.py."
        )
    if args.device == "auto":
        args.device = default_device()
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, csv_path in variants(args.input_dir).items():
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        print(f"Running {name}...", flush=True)
        rows.append(run_variant(args, name, csv_path))
    summary = pd.DataFrame(rows).sort_values("r2_mean", ascending=False)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_csv, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
