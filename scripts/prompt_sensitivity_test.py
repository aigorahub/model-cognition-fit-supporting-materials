#!/usr/bin/env python3
"""Prompt-sensitivity robustness check for the cross-model JAR comparison.

The full comparison gives every model the SAME prompt (the fair controlled
design). This script tests whether the model ranking on JAR recovery is an
artifact of that particular prompt. On a fixed sample it scores three model
configs under two prompts: the baseline prompt and a more explicit,
reasoning-friendlier "rubric" prompt that a thinking model might benefit from.
If the non-thinking model recovers JAR at least as well under both prompts, the
conclusion is robust to prompting.

Runs at low concurrency so it can coexist with the full background run on the
shared key pool.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import queue
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_topic_level_analysis import (  # noqa: E402
    SCALE_LABELS,
    TOPICS,
    build_prompt,
    extract_response_text,
    load_api_keys,
    load_ham_home_use_data,
    parse_topic_response,
    post_gemini_json,
    row_result,
)

DEFAULT_KEYPOOL_ENV = None  # set GEMINI_API_KEY or pass --keypool-env
DEFAULT_DATA_XLSX = Path("analysis/raw/dataset.xlsx")
OUT_DIR = Path("analysis/topic_level")

PAIRS = [
    ("saltiness_match", "JARSalt"),
    ("fat_lean_appearance_match", "JARFat"),
    ("color_pinkness_match", "JARColor"),
    ("tenderness_softness_match", "JARTender"),
]

MODELS = [
    ("flash-lite", "gemini-flash-lite-latest", None),
    ("g3-minimal", "gemini-3-flash-preview", "minimal"),
    ("g3-low", "gemini-3-flash-preview", "low"),
]


def build_prompt_rubric(row: pd.Series, scale_points: int) -> str:
    scale_text = "\n".join(
        f"{v} = {label}" for v, label in SCALE_LABELS.items() if v <= scale_points
    )
    json_shape = ",\n  ".join(f'"{t}": 1-{scale_points} or null' for t in TOPICS)
    return f"""
You are a sensory and consumer scientist analyzing French consumer comments about cooked ham.

Task: For each sensory topic below, judge how closely the ACTUAL ham matched this consumer's IDEAL ham.

Scoring guidance ({scale_points}-point scale):
{scale_text}
- Judge each topic on its own. A ham can match the ideal on some topics and miss on others, so do not let one strong impression set every score.
- A high score means the actual experience is close to the ideal for that topic. A low score means it is far from the ideal, whether the attribute is too much OR too little.
- Weigh only the sensory content of the comments. Ignore price, brand, health claims, and purchase intent.
- Use null only when neither the actual nor the ideal text gives any evidence for that topic.

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


PROMPTS: dict[str, Callable[[pd.Series, int], str]] = {
    "baseline": build_prompt,
    "rubric": build_prompt_rubric,
}


def score_row(row, *, key_queue, model, thinking_level, prompt_fn, scale_points, temperature, attempts, timeout):
    api_key = key_queue.get()
    try:
        gen: dict[str, Any] = {"responseMimeType": "application/json", "temperature": temperature}
        if thinking_level:
            gen["thinkingConfig"] = {"thinkingLevel": thinking_level}
        payload = {"contents": [{"parts": [{"text": prompt_fn(row, scale_points)}]}], "generationConfig": gen}
        for attempt in range(attempts):
            status, body = post_gemini_json(model=model, api_key=api_key, payload=payload, timeout_seconds=timeout)
            if status == 429:
                time.sleep(2**attempt)
                continue
            if status != 200:
                time.sleep(1)
                continue
            try:
                scores = parse_topic_response(extract_response_text(json.loads(body)), scale_points)
                return row_result(row, scores, "")
            except Exception as exc:  # noqa: BLE001
                last = f"parse_error: {exc}"
                time.sleep(1)
        return row_result(row, {t: None for t in TOPICS}, "failed")
    finally:
        key_queue.put(api_key)


def score_sample(sample, *, model, thinking_level, prompt_fn, keys, workers, scale_points, temperature):
    kq: "queue.Queue[str]" = queue.Queue()
    for k in keys:
        kq.put(k)
    out = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(score_row, row, key_queue=kq, model=model, thinking_level=thinking_level,
                      prompt_fn=prompt_fn, scale_points=scale_points, temperature=temperature,
                      attempts=3, timeout=120.0)
            for _, row in sample.iterrows()
        ]
        for f in concurrent.futures.as_completed(futs):
            out.append(f.result())
    return pd.DataFrame(out)


def recovery(scored: pd.DataFrame, jar: pd.DataFrame) -> dict:
    valid = scored[scored["parse_error"].fillna("").astype(str).eq("")].copy()
    for t, _ in PAIRS:
        valid[t] = pd.to_numeric(valid.get(t), errors="coerce")
    m = valid.merge(jar, on=["Consumer", "Product"], how="inner")
    per = {}
    rhos = []
    for t, j in PAIRS:
        sub = m[m[t].notna() & m[j].notna()]
        rho = spearmanr(sub[t], sub[j].abs())[0] if len(sub) > 10 else np.nan
        per[j] = {"n": int(len(sub)), "rho_abs": float(rho)}
        if np.isfinite(rho):
            rhos.append(abs(rho))
    return {"mean_recovery": float(np.mean(rhos)) if rhos else float("nan"),
            "merged": int(len(m)), "per_attr": per}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=250)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--scale-points", type=int, default=6)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    keys = load_api_keys(DEFAULT_KEYPOOL_ENV)
    src = load_ham_home_use_data(DEFAULT_DATA_XLSX)
    sample = src.sample(n=min(args.n, len(src)), random_state=args.seed).reset_index(drop=True)
    jar = pd.read_excel(DEFAULT_DATA_XLSX, sheet_name="product sensory properties")[
        ["Consumer", "Product", "JARColor", "JARFat", "JARSalt", "JARTender"]
    ]

    results: dict[str, dict] = {}
    for mlabel, model, thinking in MODELS:
        results[mlabel] = {}
        for plabel, pfn in PROMPTS.items():
            t0 = time.time()
            scored = score_sample(sample, model=model, thinking_level=thinking, prompt_fn=pfn,
                                   keys=keys, workers=args.workers, scale_points=args.scale_points,
                                   temperature=args.temperature)
            rec = recovery(scored, jar)
            rec["seconds"] = round(time.time() - t0, 1)
            rec["fail_rate"] = float(scored["parse_error"].fillna("").astype(str).ne("").mean())
            results[mlabel][plabel] = rec
            print(f"{mlabel:>11} | {plabel:>8} | recovery={rec['mean_recovery']:.3f} "
                  f"merged={rec['merged']} fail={rec['fail_rate']:.2f} {rec['seconds']}s")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "prompt_sensitivity_summary.json").write_text(
        json.dumps({"n": int(len(sample)), "results": results}, indent=2)
    )
    print("\nRanking by mean JAR recovery (higher = better):")
    for plabel in PROMPTS:
        ranked = sorted(MODELS, key=lambda m: results[m[0]][plabel]["mean_recovery"], reverse=True)
        order = " > ".join(f"{m[0]}({results[m[0]][plabel]['mean_recovery']:.3f})" for m in ranked)
        print(f"  {plabel:>8}: {order}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
