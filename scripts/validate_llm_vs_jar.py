#!/usr/bin/env python3
"""Independent validation: do LLM free-text alignment scores recover the human JAR structure?

The original cooked-ham study (Visalli et al. 2024 open data; Mahieu et al. 2022)
collected structured Just-About-Right ratings (JARColor, JARFat, JARSalt,
JARTender), coded -2 (really not enough) ... 0 (just about right) ... +2 (really
too much), alongside the free-text comments.

Our LLM pipeline produced actual-versus-ideal alignment scores for the matching
attributes (color/pinkness, fat/lean, saltiness, tenderness) from the free text
ALONE. The JAR ratings were never shown to the model. If the LLM alignment score
peaks at JAR = 0 and falls off toward both extremes, the model recovered the
human structured judgment from open text. This is an external, non-circular
check, unlike the liking-to-ideal geometry.
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

# LLM alignment topic  ->  (human JAR column, attribute label, "too much" word)
PAIRS = [
    ("saltiness_match", "JARSalt", "Saltiness", "salty"),
    ("fat_lean_appearance_match", "JARFat", "Fat / lean", "fatty"),
    ("color_pinkness_match", "JARColor", "Color", "colored"),
    ("tenderness_softness_match", "JARTender", "Tenderness", "tender"),
]

JAR_TICKS = {-2: "--", -1: "-", 0: "JAR", 1: "+", 2: "++"}


def load_merged(workbook: Path, scores_path: Path) -> pd.DataFrame:
    sp = pd.read_excel(workbook, sheet_name="product sensory properties")
    llm = pd.read_csv(scores_path)
    parse_error = llm.get("parse_error")
    if parse_error is not None:
        llm = llm[parse_error.fillna("").astype(str).eq("")].copy()
    keep = ["Consumer", "Product", "JARColor", "JARFat", "JARSalt", "JARTender"]
    merged = llm.merge(sp[keep], on=["Consumer", "Product"], how="inner")
    return merged


def summarize(merged: pd.DataFrame) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for topic, jar, label, word in PAIRS:
        sub = merged[merged[topic].notna() & merged[jar].notna()].copy()
        sub["absjar"] = sub[jar].abs()
        rho_abs, p_abs = spearmanr(sub[topic], sub["absjar"])
        rho_signed, p_signed = spearmanr(sub[topic], sub[jar])
        grp = sub.groupby(jar)[topic]
        levels = sorted(sub[jar].unique())
        means = grp.mean()
        sems = grp.sem()
        counts = grp.size()
        out[topic] = {
            "jar": jar,
            "label": label,
            "word": word,
            "n": int(len(sub)),
            "rho_abs": float(rho_abs),
            "p_abs": float(p_abs),
            "rho_signed": float(rho_signed),
            "p_signed": float(p_signed),
            "levels": [int(x) for x in levels],
            "means": [float(means[x]) for x in levels],
            "sems": [float(sems[x]) if np.isfinite(sems[x]) else 0.0 for x in levels],
            "counts": [int(counts[x]) for x in levels],
            "peak_at_jar": float(means.get(0, np.nan)),
        }
    return out


def specificity(merged: pd.DataFrame) -> dict:
    """4x4 matrix of rho(LLM attribute alignment, |JAR|). Diagonal should dominate."""
    topics = [p[0] for p in PAIRS]
    jars = [p[1] for p in PAIRS]
    labels = [p[2] for p in PAIRS]
    matrix = []
    diag, off = [], []
    for ti, topic in enumerate(topics):
        row = []
        for ji, jar in enumerate(jars):
            sub = merged[merged[topic].notna() & merged[jar].notna()]
            rho, _ = spearmanr(sub[topic], sub[jar].abs())
            row.append(float(rho))
            (diag if ti == ji else off).append(float(rho))
        matrix.append(row)
    return {
        "row_labels": labels,
        "col_labels": labels,
        "matrix": matrix,
        "mean_diagonal": float(np.mean(diag)),
        "mean_off_diagonal": float(np.mean(off)),
        "specificity_gap": float(np.mean(off) - np.mean(diag)),
        "own_jar_strongest_for_all": all(
            min(range(len(jars)), key=lambda j: matrix[i][j]) == i for i in range(len(topics))
        ),
    }


def render(summary: dict[str, dict], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2), facecolor=BG)
    for ax, (topic, jar, label, word) in zip(axes.ravel(), PAIRS):
        s = summary[topic]
        ax.set_facecolor(BG)
        x = np.array(s["levels"], dtype=float)
        y = np.array(s["means"])
        err = 1.96 * np.array(s["sems"])
        ax.grid(True, zorder=0, color=SANDSTONE, linestyle=":", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.axvline(0, color=MUTED, linewidth=1.0, linestyle="--", zorder=1)
        ax.plot(x, y, color=TERRACOTTA, linewidth=2.0, zorder=3)
        ax.errorbar(
            x,
            y,
            yerr=err,
            fmt="o",
            color=TERRACOTTA,
            ecolor=MUTED,
            elinewidth=1.1,
            capsize=3.5,
            markersize=7,
            markeredgecolor=INK,
            markeredgewidth=0.6,
            zorder=4,
        )
        for xi, yi, ni in zip(x, y, s["counts"]):
            # push n= above error bar
            ax.annotate(
                f"n={ni}",
                (xi, yi),
                textcoords="offset points",
                xytext=(0, 14),
                ha="center",
                fontsize=6.5,
                color=MUTED,
                zorder=5,
            )
        ax.set_xticks(list(JAR_TICKS))
        ax.set_xticklabels([JAR_TICKS[t] for t in JAR_TICKS], fontsize=9, color=INK)
        ax.set_ylim(1, 6.35)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylabel("LLM alignment score (1-6)", color=INK, fontsize=8.5)
        ax.set_xlabel(f"Human JAR: {word}", color=INK, fontsize=8.5)
        ax.tick_params(colors=MUTED, labelsize=8.5)
        for spine in ax.spines.values():
            spine.set_color(SANDSTONE)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, facecolor=BG, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path,
                        default=Path("analysis/raw/dataset.xlsx"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("analysis/topic_level"))
    parser.add_argument("--scores-path", type=Path, default=None)
    args = parser.parse_args()
    scores_path = args.scores_path or args.output_dir / "topic_level_flash_lite_scores.csv"

    merged = load_merged(args.workbook, scores_path)
    summary = summarize(merged)
    spec = specificity(merged)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    render(summary, args.output_dir / "llm_vs_jar_validation.png")
    merged.to_csv(args.output_dir / "llm_jar_merged.csv", index=False)
    (args.output_dir / "llm_vs_jar_summary.json").write_text(
        json.dumps({"per_attribute": summary, "specificity": spec}, indent=2)
    )
    pd.DataFrame(spec["matrix"], index=spec["row_labels"], columns=spec["col_labels"]).to_csv(
        args.output_dir / "llm_vs_jar_specificity.csv"
    )

    print(f"merged rows: {len(merged)}")
    for topic, s in summary.items():
        peak = s["peak_at_jar"]
        ends = [m for lvl, m in zip(s["levels"], s["means"]) if lvl in (-2, 2)]
        print(f"{s['label']:12s} n={s['n']:4d}  peak@JAR={peak:.2f}  "
              f"ends~{np.mean(ends):.2f}  rho(|JAR|)={s['rho_abs']:+.3f}  rho(signed)={s['rho_signed']:+.3f}")
    print(f"specificity: diag={spec['mean_diagonal']:.3f}  off={spec['mean_off_diagonal']:.3f}  "
          f"gap={spec['specificity_gap']:.3f}  own_strongest_all={spec['own_jar_strongest_for_all']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
