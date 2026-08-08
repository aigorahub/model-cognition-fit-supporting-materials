#!/usr/bin/env python3
"""Topic-level scoring for reasoning (thinking) Gemini configs.

Mirrors scripts/run_topic_level_analysis.py (same prompt, scale, parser, key
pool) but adds a thinkingConfig to the request so we can score the same 17
topic-alignment variables with Gemini 3 Flash at a given thinking level. Used for
the cross-model JAR validation: does a reasoning model recover the held-out JAR
structure better or worse than the non-thinking Flash Lite model?

Writes a per-config scores CSV with the same schema as the Flash Lite run.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import queue
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_topic_level_analysis import (  # noqa: E402
    TOPICS,
    build_prompt,
    combine_scores,
    extract_response_text,
    load_api_keys,
    load_existing_scores,
    load_ham_home_use_data,
    parse_topic_response,
    post_gemini_json,
    row_result,
    scored_row_ids,
)

DEFAULT_KEYPOOL_ENV = None  # set GEMINI_API_KEY or pass --keypool-env
DEFAULT_DATA_XLSX = Path("analysis/raw/dataset.xlsx")
DEFAULT_OUTPUT_DIR = Path("analysis/topic_level")


def score_one_row(
    row: pd.Series,
    *,
    key_queue: "queue.Queue[str]",
    model: str,
    thinking_level: str,
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
                "thinkingConfig": {"thinkingLevel": thinking_level},
            },
        }
        last_error = ""
        for attempt in range(attempts):
            status, body = post_gemini_json(
                model=model, api_key=api_key, payload=payload, timeout_seconds=timeout_seconds
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
                import json

                response_text = extract_response_text(json.loads(body))
                scores = parse_topic_response(response_text, scale_points)
                return row_result(row, scores, "")
            except Exception as exc:  # noqa: BLE001
                last_error = f"parse_error: {exc}"
                time.sleep(1)
        return row_result(row, {topic: None for topic in TOPICS}, last_error)
    except Exception as exc:  # noqa: BLE001
        return row_result(row, {topic: None for topic in TOPICS}, str(exc))
    finally:
        key_queue.put(api_key)


def run(args: argparse.Namespace) -> int:
    source = load_ham_home_use_data(args.data_xlsx)
    scores_path = args.output_dir / args.output_csv
    existing = load_existing_scores(scores_path) if args.resume else pd.DataFrame()
    skip_ids = scored_row_ids(existing, retry_errors=args.retry_errors)
    work = source[~source["row_id"].isin(skip_ids)].copy()
    if args.limit is not None:
        work = work.head(args.limit)
    if work.empty:
        print("No rows to score.")
        return 0

    keys = load_api_keys(args.keypool_env)
    if not keys:
        raise RuntimeError("No Gemini API keys found.")
    key_queue: "queue.Queue[str]" = queue.Queue()
    for k in keys:
        key_queue.put(k)

    max_workers = max(1, min(args.max_workers, len(keys), len(work)))
    print(f"Scoring {len(work)} rows | model={args.model} thinking={args.thinking_level} "
          f"| {max_workers} workers, {len(keys)} keys")

    new_rows: list[dict[str, Any]] = []
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(
                score_one_row, row,
                key_queue=key_queue, model=args.model, thinking_level=args.thinking_level,
                scale_points=args.scale_points, temperature=args.temperature,
                timeout_seconds=args.request_timeout, attempts=args.attempts,
            )
            for _, row in work.iterrows()
        ]
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            new_rows.append(fut.result())
            if i % args.progress_every == 0 or i == len(futures):
                elapsed = time.time() - started
                print(f"  {i}/{len(futures)} in {elapsed:.1f}s")
                combined = combine_scores(existing, new_rows)
                scores_path.parent.mkdir(parents=True, exist_ok=True)
                combined.to_csv(scores_path, index=False)

    combined = combine_scores(existing, new_rows)
    combined.to_csv(scores_path, index=False)
    errors = int(combined["parse_error"].fillna("").astype(str).ne("").sum())
    total_time = time.time() - started
    print(f"Done. rows={len(combined)} parse_errors={errors} wall={total_time:.1f}s -> {scores_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gemini-3-flash-preview")
    p.add_argument("--thinking-level", required=True, choices=["minimal", "low", "high"])
    p.add_argument("--output-csv", required=True, help="Filename within --output-dir.")
    p.add_argument("--data-xlsx", default=DEFAULT_DATA_XLSX, type=Path)
    p.add_argument("--keypool-env", default=DEFAULT_KEYPOOL_ENV, type=Path)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    p.add_argument("--scale-points", default=6, type=int)
    p.add_argument("--temperature", default=0.7, type=float)
    p.add_argument("--max-workers", default=32, type=int)
    p.add_argument("--request-timeout", default=120.0, type=float)
    p.add_argument("--attempts", default=3, type=int)
    p.add_argument("--progress-every", default=200, type=int)
    p.add_argument("--limit", default=None, type=int)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--retry-errors", action="store_true")
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
