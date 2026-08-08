#!/usr/bin/env python3
"""Run topic-level LLM scoring for the cooked-ham liking analysis.

The script has two stages:
1. Score each home-use consumer/product row with Gemini Flash Lite.
2. Prepare a TabPFN-ready CSV and optionally run TabPFN on Mini 1 over SSH.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import queue
import random
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd


TOPICS = [
    "overall_visual_match",
    "color_pinkness_match",
    "fat_lean_appearance_match",
    "surface_moisture_shine_match",
    "slice_structure_homogeneity_match",
    "overall_texture_match",
    "tenderness_softness_match",
    "firmness_rubberiness_match",
    "dryness_juiciness_match",
    "fibrous_stringy_pieces_match",
    "thickness_chew_match",
    "overall_flavor_match",
    "saltiness_match",
    "ham_taste_match",
    "aromatic_smoky_spiced_match",
    "bland_insipid_intensity_match",
    "offnote_aftertaste_match",
]

SCALE_LABELS = {
    1: "extremely different",
    2: "very different",
    3: "somewhat different",
    4: "somewhat similar",
    5: "very similar",
    6: "extremely similar",
}

DEFAULT_DATA_XLSX = Path("analysis/raw/dataset.xlsx")
DEFAULT_OUTPUT_DIR = Path("analysis/topic_level")
DEFAULT_KEYPOOL_ENV = None  # set GEMINI_API_KEY or pass --keypool-env
REMOTE_LAUNCHER_ROOT = None  # optional local launcher; not required for archived results


def parse_keypool_env(text: str) -> list[str]:
    env: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = strip_shell_quotes(value.strip())
        env[key] = value

    ordered_names = ["GEMINI_API_KEY", "GEMINI_API_KEYS"]
    numbered = sorted(
        [name for name in env if re.fullmatch(r"GEMINI_API_KEY_\d+", name)],
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    ordered_names.extend(numbered)

    keys: list[str] = []
    seen: set[str] = set()
    for name in ordered_names:
        if name not in env:
            continue
        values = split_key_values(env[name]) if name == "GEMINI_API_KEYS" else [env[name]]
        for value in values:
            value = value.strip()
            if value and value not in seen:
                seen.add(value)
                keys.append(value)
    return keys


def strip_shell_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def split_key_values(value: str) -> list[str]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [item for item in re.split(r"[\s,;]+", value) if item]


def load_api_keys(keypool_env: Path | None) -> list[str]:
    parts: list[str] = []
    if keypool_env and keypool_env.exists():
        parts.append(keypool_env.read_text())
    env_lines = []
    for name, value in os.environ.items():
        if name == "GEMINI_API_KEY" or name == "GEMINI_API_KEYS" or re.fullmatch(
            r"GEMINI_API_KEY_\d+", name
        ):
            env_lines.append(f"{name}={value}")
    if env_lines:
        parts.append("\n".join(env_lines))
    keys: list[str] = []
    seen: set[str] = set()
    for part in parts:
        for key in parse_keypool_env(part):
            if key not in seen:
                keys.append(key)
                seen.add(key)
    return keys


def load_ham_home_use_data(path: Path) -> pd.DataFrame:
    evaluations = pd.read_excel(path, sheet_name="product sensory properties")
    ideals = pd.read_excel(path, sheet_name="consumer questionnaire (home)")

    evaluations = evaluations.dropna(subset=["Liking"]).copy()
    evaluations = evaluations[evaluations["Consumer"].astype(str).str.startswith("H")].copy()
    evaluations = evaluations.reset_index(drop=True)
    evaluations.insert(0, "row_id", evaluations.index)

    merged = evaluations.merge(
        ideals[["Consumer", "IdealVisual", "IdealTexture", "IdealFlavor"]],
        on="Consumer",
        how="left",
    )
    text_columns = [
        "DescriptionVisual",
        "DescriptionTexture",
        "DescriptionFlavor",
        "IdealVisual",
        "IdealTexture",
        "IdealFlavor",
    ]
    for column in text_columns:
        merged[column] = merged[column].fillna("").astype(str)
    return merged


def build_prompt(row: pd.Series, scale_points: int) -> str:
    scale_text = "\n".join(
        f"{value} = {label}" for value, label in SCALE_LABELS.items() if value <= scale_points
    )
    json_shape = ",\n  ".join(f'"{topic}": 1-{scale_points} or null' for topic in TOPICS)
    return f"""
You are a sensory and consumer scientist analyzing French consumer comments about cooked ham.

Task: Compare the ACTUAL ham experience with the consumer's IDEAL ham experience.
Score alignment for each modality/topic pair.

Scale:
{scale_text}

Use null only when neither the actual nor ideal text gives enough evidence for that topic.
Focus only on sensory descriptions of the ham itself. Ignore price, brand, health claims, and purchase intent.
The comments are in French.

IDEAL EXPERIENCE:
Visual: {row.get("IdealVisual", "") or "(empty)"}
Texture: {row.get("IdealTexture", "") or "(empty)"}
Flavor: {row.get("IdealFlavor", "") or "(empty)"}

ACTUAL EXPERIENCE:
Visual: {row.get("DescriptionVisual", "") or "(empty)"}
Texture: {row.get("DescriptionTexture", "") or "(empty)"}
Flavor: {row.get("DescriptionFlavor", "") or "(empty)"}

Return JSON only, exactly this object shape:
{{
  {json_shape}
}}
""".strip()


def parse_topic_response(text: str, scale_points: int) -> dict[str, int | None]:
    if not text:
        raise ValueError("empty response")
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("response JSON is not an object")

    parsed: dict[str, int | None] = {}
    for topic in TOPICS:
        value = data.get(topic)
        if value is None or value == "":
            parsed[topic] = None
            continue
        try:
            int_value = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{topic} is not an integer or null: {value!r}") from exc
        if not 1 <= int_value <= scale_points:
            raise ValueError(f"{topic} out of range: {int_value}")
        parsed[topic] = int_value
    return parsed


def extract_response_text(response_json: dict[str, Any]) -> str:
    candidates = response_json.get("candidates") or []
    if not candidates:
        raise ValueError("Gemini response has no candidates")
    parts = candidates[0].get("content", {}).get("parts", [])
    for part in parts:
        if part.get("thought"):
            continue
        if "text" in part:
            return str(part["text"])
    raise ValueError("Gemini response has no text part")


def post_gemini_json(
    *,
    model: str,
    api_key: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> tuple[int, str]:
    params = urllib.parse.urlencode({"key": api_key})
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?{params}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")


def score_one_row(
    row: pd.Series,
    *,
    key_queue: queue.Queue[str],
    model: str,
    scale_points: int,
    temperature: float,
    timeout_seconds: float,
    attempts: int,
) -> dict[str, Any]:
    api_key = key_queue.get()
    try:
        payload = {
            "contents": [{"parts": [{"text": build_prompt(row, scale_points)}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": temperature,
            },
        }
        last_error = ""
        for attempt in range(attempts):
            status, body = post_gemini_json(
                model=model,
                api_key=api_key,
                payload=payload,
                timeout_seconds=timeout_seconds,
            )
            if status == 429:
                last_error = "rate_limited"
                time.sleep(2**attempt)
                continue
            if status != 200:
                last_error = f"http_{status}: {body[:300]}"
                time.sleep(1)
                continue
            try:
                response_text = extract_response_text(json.loads(body))
                scores = parse_topic_response(response_text, scale_points)
                return row_result(row, scores, "")
            except Exception as exc:  # noqa: BLE001 - stored for row-level audit.
                last_error = f"parse_error: {exc}"
                time.sleep(1)
        return row_result(row, {topic: None for topic in TOPICS}, last_error)
    except Exception as exc:  # noqa: BLE001 - stored for row-level audit.
        return row_result(row, {topic: None for topic in TOPICS}, str(exc))
    finally:
        key_queue.put(api_key)


def row_result(row: pd.Series, scores: dict[str, int | None], parse_error: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "row_id": int(row["row_id"]),
        "Consumer": row["Consumer"],
        "Product": row["Product"],
        "Liking": row["Liking"],
        "parse_error": parse_error,
    }
    result.update(scores)
    return result


def load_existing_scores(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def scored_row_ids(existing: pd.DataFrame, retry_errors: bool) -> set[int]:
    if existing.empty or "row_id" not in existing.columns:
        return set()
    if retry_errors and "parse_error" in existing.columns:
        successful = existing["parse_error"].fillna("").astype(str).eq("")
        return set(existing.loc[successful, "row_id"].astype(int).tolist())
    return set(existing["row_id"].astype(int).tolist())


def combine_scores(existing: pd.DataFrame, new_rows: list[dict[str, Any]]) -> pd.DataFrame:
    frames = []
    if not existing.empty:
        frames.append(existing)
    if new_rows:
        frames.append(pd.DataFrame(new_rows))
    if not frames:
        return pd.DataFrame(columns=["row_id", "Consumer", "Product", "Liking", "parse_error", *TOPICS])
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["row_id"], keep="last")
    return combined.sort_values("row_id").reset_index(drop=True)


def run_scoring(args: argparse.Namespace, source: pd.DataFrame, scores_path: Path) -> pd.DataFrame:
    existing = load_existing_scores(scores_path) if args.resume else pd.DataFrame()
    skip_ids = scored_row_ids(existing, retry_errors=args.retry_errors)

    work = source[~source["row_id"].isin(skip_ids)].copy()
    if args.limit is not None:
        work = work.head(args.limit)
    if work.empty:
        print("No rows to score.")
        return existing

    api_keys = load_api_keys(args.keypool_env)
    if not api_keys:
        raise RuntimeError("No Gemini API keys found. Use --keypool-env or GEMINI_API_KEY.")

    key_queue: queue.Queue[str] = queue.Queue()
    for key in api_keys:
        key_queue.put(key)

    max_workers = max(1, min(args.max_workers, len(api_keys), len(work)))
    print(f"Scoring {len(work)} rows with {max_workers} workers and {len(api_keys)} API keys.")

    new_rows: list[dict[str, Any]] = []
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                score_one_row,
                row,
                key_queue=key_queue,
                model=args.model,
                scale_points=args.scale_points,
                temperature=args.temperature,
                timeout_seconds=args.request_timeout,
                attempts=args.attempts,
            )
            for _, row in work.iterrows()
        ]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            new_rows.append(future.result())
            if index % args.progress_every == 0 or index == len(futures):
                elapsed = time.time() - started
                print(f"  {index}/{len(futures)} scored in {elapsed:.1f}s")
                combined = combine_scores(existing, new_rows)
                scores_path.parent.mkdir(parents=True, exist_ok=True)
                combined.to_csv(scores_path, index=False)

    combined = combine_scores(existing, new_rows)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(scores_path, index=False)
    return combined


def prepare_modeling_frame(
    scores: pd.DataFrame,
    *,
    scale_points: int,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, list[str]]:
    if scores.empty:
        raise ValueError("No scores available for modeling")
    active = scores.copy()
    active["parse_error"] = active.get("parse_error", "").fillna("").astype(str)
    active = active[active["parse_error"].eq("")].copy()
    active = active.dropna(subset=["Liking"]).reset_index(drop=True)

    neutral = (scale_points + 1) / 2
    feature_columns: list[str] = []
    for topic in TOPICS:
        active[topic] = pd.to_numeric(active.get(topic), errors="coerce")
        missing_col = f"{topic}_missing"
        active[missing_col] = active[topic].isna().astype(int)
        active[topic] = active[topic].fillna(neutral)
        feature_columns.extend([topic, missing_col])

    active = add_consumer_group_split(active, test_size=test_size, random_state=random_state)
    ordered = ["row_id", "Consumer", "Product", "split", "Liking", *feature_columns]
    return active[ordered].copy(), feature_columns


def add_consumer_group_split(
    frame: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
) -> pd.DataFrame:
    groups = sorted(frame["Consumer"].astype(str).unique())
    rng = random.Random(random_state)
    rng.shuffle(groups)
    test_count = max(1, round(len(groups) * test_size))
    holdout = set(groups[:test_count])
    frame = frame.copy()
    frame["split"] = frame["Consumer"].astype(str).map(lambda value: "holdout" if value in holdout else "train")
    return frame


def run_tabpfn_on_mini(
    *,
    modeling_csv: Path,
    feature_columns: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    run_id = args.tabpfn_run_id or f"topic_level_{int(time.time())}"
    remote_csv_name = f"{run_id}.csv"
    remote_csv = f"{REMOTE_LAUNCHER_ROOT}/data/{remote_csv_name}"
    local_summary_dir = args.output_dir / "tabpfn_remote" / run_id
    local_summary_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        ["scp", str(modeling_csv), f"{args.ssh_host}:{remote_csv}"],
        check=True,
        text=True,
    )

    feature_types = {feature: "continuous" for feature in feature_columns}
    remote_command = " ".join(
        [
            "cd",
            shlex.quote(REMOTE_LAUNCHER_ROOT),
            "&&",
            "TABPFN_ALLOW_CPU_LARGE_DATASET=1" if args.tabpfn_device == "cpu" else "",
            "python3",
            "scripts/tabpfn_cluster.py",
            "--node",
            shlex.quote(args.tabpfn_node),
            "--csv",
            shlex.quote(f"data/{remote_csv_name}"),
            "--target",
            "Liking",
            "--task",
            "regression",
            "--split-column",
            "split",
            "--train-split-value",
            "train",
            "--test-split-value",
            "holdout",
            "--random-state",
            str(args.random_state),
            "--test-size",
            str(args.test_size),
            "--features-json",
            shlex.quote(json.dumps(feature_columns, separators=(",", ":"))),
            "--feature-types-json",
            shlex.quote(json.dumps(feature_types, separators=(",", ":"))),
            "--run-id",
            shlex.quote(run_id),
            "--prediction-batch-size",
            str(args.tabpfn_prediction_batch_size),
            "--json",
        ]
    )
    remote_command = " ".join(part for part in remote_command.split(" ") if part)
    if args.tabpfn_device:
        remote_command += f" --device {shlex.quote(args.tabpfn_device)}"
    if args.no_permutation_importance:
        remote_command += " --no-permutation-importance"

    completed = subprocess.run(
        ["ssh", args.ssh_host, remote_command],
        check=False,
        text=True,
        capture_output=True,
    )
    (local_summary_dir / "tabpfn_stdout.txt").write_text(completed.stdout)
    (local_summary_dir / "tabpfn_stderr.txt").write_text(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr[-2000:] or completed.stdout[-2000:])

    summary = extract_json_object(completed.stdout)
    (local_summary_dir / "tabpfn_summary.json").write_text(json.dumps(summary, indent=2))

    if not args.skip_tabpfn_download:
        remote_pattern = f"{args.ssh_host}:{REMOTE_LAUNCHER_ROOT}/outputs/{run_id}*"
        subprocess.run(["scp", "-r", remote_pattern, str(local_summary_dir)], check=False, text=True)
    return summary


def extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in command output")
    return json.loads(text[start:])


def write_metadata(
    *,
    path: Path,
    args: argparse.Namespace,
    source_rows: int,
    scores: pd.DataFrame,
    modeling: pd.DataFrame,
    features: list[str],
    tabpfn_summary: dict[str, Any] | None,
) -> None:
    errors = 0
    if not scores.empty and "parse_error" in scores.columns:
        errors = int(scores["parse_error"].fillna("").astype(str).ne("").sum())
    metadata = {
        "source_rows": source_rows,
        "scored_rows": int(len(scores)),
        "parse_errors": errors,
        "modeling_rows": int(len(modeling)),
        "train_rows": int((modeling["split"] == "train").sum()) if not modeling.empty else 0,
        "holdout_rows": int((modeling["split"] == "holdout").sum()) if not modeling.empty else 0,
        "feature_count": len(features),
        "features": features,
        "gemini_model": args.model,
        "scale_points": args.scale_points,
        "temperature": args.temperature,
        "tabpfn_summary": tabpfn_summary,
    }
    path.write_text(json.dumps(metadata, indent=2))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-xlsx", default=DEFAULT_DATA_XLSX, type=Path)
    parser.add_argument("--keypool-env", default=DEFAULT_KEYPOOL_ENV, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--model", default="gemini-flash-lite-latest")
    parser.add_argument("--scale-points", default=6, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--max-workers", default=32, type=int)
    parser.add_argument("--request-timeout", default=90.0, type=float)
    parser.add_argument("--attempts", default=3, type=int)
    parser.add_argument("--progress-every", default=100, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Score only the first N unscored rows.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing score CSV.")
    parser.add_argument("--retry-errors", action="store_true", help="When resuming, retry rows with parse errors.")
    parser.add_argument("--skip-scoring", action="store_true", help="Use existing score CSV only.")
    parser.add_argument("--test-size", default=0.2, type=float, help="Consumer holdout fraction.")
    parser.add_argument("--random-state", default=42, type=int)
    parser.add_argument("--run-tabpfn", action="store_true", help="Run TabPFN on Mini 1 through SSH.")
    parser.add_argument("--ssh-host", default="")
    parser.add_argument("--tabpfn-node", default="")
    parser.add_argument("--tabpfn-run-id", default=None)
    parser.add_argument("--tabpfn-device", default=None, choices=["auto", "mps", "cpu"])
    parser.add_argument("--tabpfn-prediction-batch-size", default=16, type=int)
    parser.add_argument("--skip-tabpfn-download", action="store_true")
    parser.add_argument("--no-permutation-importance", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = args.output_dir / "topic_level_flash_lite_scores.csv"
    modeling_path = args.output_dir / "topic_level_tabpfn_input.csv"
    metadata_path = args.output_dir / "topic_level_analysis_metadata.json"

    source = load_ham_home_use_data(args.data_xlsx)
    if args.skip_scoring:
        scores = load_existing_scores(scores_path)
        if scores.empty:
            raise FileNotFoundError(f"No existing scores found: {scores_path}")
    else:
        scores = run_scoring(args, source, scores_path)

    modeling, features = prepare_modeling_frame(
        scores,
        scale_points=args.scale_points,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    modeling.to_csv(modeling_path, index=False)
    print(f"Wrote scores: {scores_path}")
    print(f"Wrote TabPFN input: {modeling_path}")

    tabpfn_summary = None
    if args.run_tabpfn:
        tabpfn_summary = run_tabpfn_on_mini(
            modeling_csv=modeling_path,
            feature_columns=features,
            args=args,
        )
        print(json.dumps(tabpfn_summary.get("results", tabpfn_summary), indent=2)[:2000])

    write_metadata(
        path=metadata_path,
        args=args,
        source_rows=len(source),
        scores=scores,
        modeling=modeling,
        features=features,
        tabpfn_summary=tabpfn_summary,
    )
    print(f"Wrote metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
