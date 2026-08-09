#!/usr/bin/env python3
"""Cross-model JAR recovery: does a reasoning model recover the held-out JAR
structure better than the non-thinking model?

For each model's topic-level scores, we join to the held-out JAR ratings and
measure, per attribute, the Spearman correlation between LLM alignment and the
absolute JAR deviation |JAR| (more negative = better recovery). We also report
attribute specificity (matched vs mismatched correlation) and data completeness
(share of rows the model left null / failed to parse). The comparison mirrors the
paper's controlled six-point, temperature 0.7 setting across the same three model
configurations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BG = "#FFFFFF"
INK = "#111111"
MUTED = "#444444"
TERRACOTTA = "#222222"
SANDSTONE = "#DDDDDD"
SAGE = "#555555"

# LLM topic  ->  (JAR column, short label)
PAIRS = [
    ("saltiness_match", "JARSalt", "Saltiness"),
    ("fat_lean_appearance_match", "JARFat", "Fat / lean"),
    ("color_pinkness_match", "JARColor", "Color"),
    ("tenderness_softness_match", "JARTender", "Tenderness"),
]

CONFIGS = [
    ("Direct-response", "topic_level_flash_lite_scores.csv", "#333333"),
    ("Minimal thinking", "topic_level_g3flash_minimal_scores.csv", "#777777"),
    ("Low thinking", "topic_level_g3flash_low_scores.csv", "#DDDDDD"),
]


def load_jar(workbook: Path) -> pd.DataFrame:
    sp = pd.read_excel(workbook, sheet_name="product sensory properties")
    return sp[["Consumer", "Product", "JARColor", "JARFat", "JARSalt", "JARTender"]]


def evaluate(scores_path: Path, jar: pd.DataFrame) -> dict:
    scores = pd.read_csv(scores_path)
    n_total = len(scores)
    parse_err = int(scores.get("parse_error", "").fillna("").astype(str).ne("").sum())
    valid = scores[scores.get("parse_error", "").fillna("").astype(str).eq("")].copy()
    for topic, _, _ in PAIRS:
        valid[topic] = pd.to_numeric(valid.get(topic), errors="coerce")
    merged = valid.merge(jar, on=["Consumer", "Product"], how="inner")

    per_attr = {}
    topics = [p[0] for p in PAIRS]
    matrix, diag, off = [], [], []
    for ti, (topic, jcol, label) in enumerate(PAIRS):
        row = []
        for ji, (_, jcol2, _) in enumerate(PAIRS):
            sub = merged[merged[topic].notna() & merged[jcol2].notna()]
            rho, _ = spearmanr(sub[topic], sub[jcol2].abs()) if len(sub) > 10 else (np.nan, np.nan)
            row.append(float(rho))
            (diag if ti == ji else off).append(float(rho))
        matrix.append(row)
        own = merged[merged[topic].notna() & merged[jcol].notna()]
        rho_abs, p_abs = spearmanr(own[topic], own[jcol].abs())
        # null rate for this topic among merged rows
        null_rate = float(merged[topic].isna().mean())
        per_attr[label] = {
            "n": int(len(own)),
            "rho_abs": float(rho_abs),
            "p_abs": float(p_abs),
            "null_rate": null_rate,
        }
    return {
        "rows": n_total,
        "parse_errors": parse_err,
        "merged_rows": int(len(merged)),
        "per_attr": per_attr,
        "mean_diagonal": float(np.mean(diag)),
        "mean_off_diagonal": float(np.mean(off)),
        "specificity_gap": float(np.mean(off) - np.mean(diag)),
        "mean_null_rate": float(np.mean([v["null_rate"] for v in per_attr.values()])),
        "matrix": matrix,
    }


def render(results: dict, out_path: Path) -> None:
    labels = [p[2] for p in PAIRS]
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.5, 4.6), facecolor=BG, gridspec_kw={"width_ratios": [1.7, 1.05]}
    )
    x = np.arange(len(labels))
    n_models = sum(1 for n, _, _ in CONFIGS if n in results)
    width = 0.78 / max(n_models, 1)
    mi = 0
    for name, _, color in CONFIGS:
        if name not in results:
            continue
        vals = [abs(results[name]["per_attr"][lab]["rho_abs"]) for lab in labels]
        offset = (mi - (n_models - 1) / 2) * width
        bars = ax1.bar(
            x + offset,
            vals,
            width,
            label=name,
            color=color,
            edgecolor=INK,
            linewidth=0.7,
            zorder=2,
        )
        if name == "Minimal thinking":
            for b in bars:
                b.set_hatch("///")
        mi += 1
    ax1.set_facecolor(BG)
    ax1.set_xticks(x, labels, fontsize=10)
    ax1.set_ylabel(r"Absolute Spearman $|\rho|$ (alignment vs $|JAR|$)", color=INK, fontsize=9)
    ax1.set_ylim(0, 0.85)
    ax1.legend(frameon=False, fontsize=8, loc="upper right")
    ax1.tick_params(colors=MUTED)
    ax1.grid(axis="y", color=SANDSTONE, linestyle=":", linewidth=0.8, zorder=0)
    ax1.set_axisbelow(True)
    for sp in ax1.spines.values():
        sp.set_color(SANDSTONE)
    ax1.text(-0.08, 1.05, "A", transform=ax1.transAxes, fontsize=11, fontweight="bold", va="bottom")

    names = [n for n, _, _ in CONFIGS if n in results]
    colors = [c for n, _, c in CONFIGS if n in results]
    recov = [abs(results[n]["mean_diagonal"]) for n in names]
    gap = [results[n]["specificity_gap"] for n in names]
    xx = np.arange(len(names))
    b1 = ax2.bar(xx - 0.18, recov, 0.34, label="Mean recovery", color=colors, edgecolor=INK, linewidth=0.7, zorder=2)
    b2 = ax2.bar(xx + 0.18, gap, 0.34, label="Specificity gap", color=colors, alpha=0.45, edgecolor=INK, linewidth=0.7, zorder=2)
    ax2.bar_label(b1, labels=[f"{v:.2f}" for v in recov], padding=2, fontsize=7.5)
    ax2.bar_label(b2, labels=[f"{v:.2f}" for v in gap], padding=2, fontsize=7.5)
    ax2.set_facecolor(BG)
    short = [n.replace(" ", "\n") if n != "Direct-response" else "Direct-\nresponse" for n in names]
    ax2.set_xticks(xx, short, fontsize=7.5)
    ax2.set_ylim(0, max(max(recov), max(gap)) * 1.25)
    ax2.legend(frameon=False, fontsize=8)
    ax2.tick_params(colors=MUTED)
    ax2.grid(axis="y", color=SANDSTONE, linestyle=":", linewidth=0.8, zorder=0)
    ax2.set_axisbelow(True)
    for sp in ax2.spines.values():
        sp.set_color(SANDSTONE)
    ax2.text(-0.08, 1.05, "B", transform=ax2.transAxes, fontsize=11, fontweight="bold", va="bottom")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, facecolor=BG, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workbook", type=Path,
                   default=Path("analysis/raw/dataset.xlsx"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("analysis/topic_level"))
    args = p.parse_args()
    jar = load_jar(args.workbook)

    results = {}
    for name, fname, _ in CONFIGS:
        path = args.output_dir / fname
        if not path.exists():
            print(f"skip {name}: {path} not found")
            continue
        results[name] = evaluate(path, jar)

    render(results, args.output_dir / "cross_model_jar.png")
    (args.output_dir / "cross_model_jar_summary.json").write_text(json.dumps(results, indent=2))

    labels = [p[2] for p in PAIRS]
    print(f"\n{'model':<28}{'merged':>8}{'parseE':>8}{'nullR':>8}{'recov':>8}{'specGap':>9}   per-attr |rho|")
    for name, _, _ in CONFIGS:
        if name not in results:
            continue
        r = results[name]
        pa = "  ".join(f"{lab.split()[0]}={abs(r['per_attr'][lab]['rho_abs']):.2f}" for lab in labels)
        print(f"{name:<28}{r['merged_rows']:>8}{r['parse_errors']:>8}{r['mean_null_rate']:>8.2f}"
              f"{abs(r['mean_diagonal']):>8.3f}{r['specificity_gap']:>9.3f}   {pa}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
