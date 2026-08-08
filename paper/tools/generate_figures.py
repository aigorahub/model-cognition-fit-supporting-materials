#!/usr/bin/env python3
"""Generate greyscale-safe manuscript figures (FQAP, Gemini 3.6 design guidance)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

try:
    from adjustText import adjust_text
except ImportError:  # pragma: no cover
    adjust_text = None

ROOT = Path(__file__).resolve().parents[2]
if (ROOT / "analysis" / "grid" / "feature_importance_results.csv").exists():
    GRID_DATA = ROOT / "analysis" / "grid"
    TOPIC_DATA = ROOT / "analysis" / "topic_level"
elif (ROOT / "data" / "grid" / "feature_importance_results.csv").exists():
    GRID_DATA = ROOT / "data" / "grid"
    TOPIC_DATA = ROOT / "data" / "topic_level"
else:
    GRID_DATA = ROOT / "analysis" / "grid"
    TOPIC_DATA = ROOT / "analysis" / "topic_level"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#111111"
MUTED = "#555555"
GREY_DARK = "#333333"
GREY_MID = "#777777"
GREY_LIGHT = "#DDDDDD"
FILL_LIGHT = "#F5F5F5"
GRID_C = "#E5E5E5"

MODEL_ORDER = [
    "Gemini 2.5 Flash Lite",
    "Gemini 3 Flash low",
    "Gemini 3 Flash minimal",
]
MODEL_SHORT = {
    "Gemini 2.5 Flash Lite": "Direct\nresponse",
    "Gemini 3 Flash low": "Low\nthinking",
    "Gemini 3 Flash minimal": "Minimal\nthinking",
}
MODEL_STYLE = {
    "Gemini 2.5 Flash Lite": {
        "marker": "o",
        "face": GREY_DARK,
        "edge": INK,
        "fill": GREY_DARK,
        "label": "Direct-response",
    },
    "Gemini 3 Flash low": {
        "marker": "s",
        "face": GREY_MID,
        "edge": INK,
        "fill": GREY_MID,
        "label": "Low thinking",
    },
    "Gemini 3 Flash minimal": {
        "marker": "^",
        "face": "white",
        "edge": INK,
        "fill": GREY_LIGHT,
        "label": "Minimal thinking",
    },
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)

GRID_RESULTS = GRID_DATA / "grid_comparison_results.csv"


def repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def load_grid_results() -> pd.DataFrame:
    df = pd.read_csv(GRID_RESULTS)
    required = {
        "config_id",
        "model",
        "scale",
        "temperature",
        "r2",
        "mae",
        "llm_time",
        "errors",
        "n_valid",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{GRID_RESULTS} is missing columns: {sorted(missing)}")
    if "error_rate" not in df.columns:
        df["error_rate"] = df["errors"] / df["n_valid"]
    return df


def save(fig: plt.Figure, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=INK,
    )


def draw_workflow() -> None:
    """Compact pipeline banner (also embedded in methods figure)."""
    fig, ax = plt.subplots(figsize=(7.4, 1.55))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 2.2)

    boxes = [
        (0.25, 0.45, 1.9, 1.15, "Actual and\nideal comments"),
        (2.55, 0.45, 1.9, 1.15, "LLM alignment\nscoring"),
        (4.85, 0.45, 1.7, 1.15, "Visual\nTexture\nFlavor"),
        (6.95, 0.45, 1.8, 1.15, "TabPFN\nregression"),
        (9.15, 0.45, 2.0, 1.15, "Held-out\nliking (0-10)"),
    ]
    for x, y, w, h, label in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.04",
                facecolor=FILL_LIGHT,
                edgecolor=INK,
                linewidth=0.9,
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=8.0,
            linespacing=1.15,
            color=INK,
        )
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + boxes[i][2]
        x2 = boxes[i + 1][0]
        y = boxes[i][1] + boxes[i][3] / 2
        ax.add_patch(
            FancyArrowPatch(
                (x1 + 0.06, y),
                (x2 - 0.06, y),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=0.9,
                color=MUTED,
            )
        )
    plt.close(fig)  # methods figure is build_scoring_method_figure.py -> workflow_scoring.png


def draw_model_grid() -> None:
    df = load_grid_results()
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.35), sharex=True)
    metrics = [
        (axes[0], "r2", r"$R^2$ (held-out liking)"),
        (axes[1], "mae", r"MAE on liking (0-10)"),
    ]
    for ax, metric, ylabel in metrics:
        for i, model in enumerate(MODEL_ORDER):
            vals = df.loc[df["model"] == model, metric].to_numpy()
            # spread points; use discrete jitter so means stay readable
            jitter = np.linspace(-0.12, 0.12, len(vals))
            rng.shuffle(jitter)
            xs = i + jitter
            st = MODEL_STYLE[model]
            ax.scatter(
                xs,
                vals,
                s=40,
                marker=st["marker"],
                facecolors=st["face"],
                edgecolors=st["edge"],
                linewidths=0.7,
                zorder=3,
                alpha=0.95,
            )
        ax.set_xticks(range(len(MODEL_ORDER)))
        ax.set_xticklabels([MODEL_SHORT[m] for m in MODEL_ORDER], fontsize=7.5)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=GRID_C, linestyle=":", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        panel_label(ax, "A" if metric == "r2" else "B")

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker=MODEL_STYLE[m]["marker"],
            color="w",
            markerfacecolor=MODEL_STYLE[m]["face"],
            markeredgecolor=MODEL_STYLE[m]["edge"],
            markersize=7,
            label=MODEL_STYLE[m]["label"],
        )
        for m in MODEL_ORDER

    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        fontsize=7.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, "grid_performance")


def draw_operations() -> None:
    df = load_grid_results().set_index("config_id")
    configs = [
        "Flash Lite_6pt_t07",
        "G3 Flash (low)_6pt_t07",
        "G3 Flash (minimal)_6pt_t07",
    ]
    labels = ["Direct\nresponse", "Low\nthinking", "Minimal\nthinking"]
    fills = [GREY_DARK, GREY_MID, GREY_LIGHT]
    seconds = [float(df.loc[c, "llm_time"]) for c in configs]
    errors = [100.0 * float(df.loc[c, "error_rate"]) for c in configs]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    panels = [
        (axes[0], seconds, "Wall-clock time (s)", "{:.0f} s"),
        (axes[1], errors, "Rows with scoring errors (%)", "{:.1f}%"),
    ]
    for ax, vals, ylabel, fmt in panels:
        bars = ax.bar(
            labels,
            vals,
            color=fills,
            edgecolor=INK,
            linewidth=0.7,
            width=0.62,
            zorder=2,
        )
        # light hatch only on middle bar for greyscale separation
        bars[1].set_hatch("///")
        bars[1].set_edgecolor(INK)
        ax.bar_label(bars, labels=[fmt.format(v) for v in vals], padding=3, fontsize=8, color=INK)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(vals) * 1.18)
        ax.grid(axis="y", color=GRID_C, linestyle=":", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=7.5)
    panel_label(axes[0], "A")
    panel_label(axes[1], "B")
    fig.tight_layout()
    save(fig, "operational_comparison")


def draw_modality_importance() -> None:
    imp = pd.read_csv(GRID_DATA / "feature_importance_results.csv")
    ours = imp.groupby("feature", as_index=False)["importance"].mean()
    ours["feature"] = ours["feature"].str.capitalize()
    ours_map = dict(zip(ours["feature"], ours["importance"]))
    mahieu = {"Flavor": 0.261, "Texture": 0.201, "Visual": 0.139}
    labels = ["Flavor", "Texture", "Visual"]
    ours_raw = np.array([ours_map[x] for x in labels])
    mahieu_raw = np.array([mahieu[x] for x in labels])
    ours_vals = 100 * ours_raw / ours_raw.sum()
    mahieu_vals = 100 * mahieu_raw / mahieu_raw.sum()

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(labels))
    width = 0.36
    b1 = ax.bar(
        x - width / 2,
        mahieu_vals,
        width,
        label="Mahieu et al. (literature baseline)",
        color="white",
        edgecolor=INK,
        linewidth=0.8,
        hatch="////",
        zorder=2,
    )
    b2 = ax.bar(
        x + width / 2,
        ours_vals,
        width,
        label="LLM feature importance (this study)",
        color=GREY_MID,
        edgecolor=INK,
        linewidth=0.8,
        zorder=2,
    )
    ax.bar_label(b1, labels=[f"{v:.0f}%" for v in mahieu_vals], padding=2, fontsize=7.5)
    ax.bar_label(b2, labels=[f"{v:.0f}%" for v in ours_vals], padding=2, fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Within-method share of modality signal (%)")
    ax.set_ylim(0, max(ours_vals.max(), mahieu_vals.max()) * 1.2)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(axis="y", color=GRID_C, linestyle=":", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "modality_hierarchy")


def draw_topic_rank() -> None:
    path = TOPIC_DATA / "topic_level_contrast_points.csv"
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    style = {
        "Flavor": {"marker": "o", "face": GREY_DARK, "edge": INK},
        "Texture": {"marker": "s", "face": GREY_MID, "edge": INK},
        "Visual": {"marker": "^", "face": "white", "edge": INK},
    }
    label_map = {
        "Off-notes and aftertaste": "Off-notes",
        "Blandness and intensity": "Blandness",
        "Fibrous pieces": "Fibrous",
        "Overall texture match": "Overall texture",
        "Firmness and rubberiness": "Firmness",
        "Overall flavor match": "Overall flavor",
        "Dryness and juiciness": "Dryness",
        "Smoky and spiced notes": "Smoky/spiced",
        "Ham taste": "Ham taste",
        "Moisture and shine": "Moisture",
        "Slice structure": "Slice structure",
        "Color and pinkness": "Color",
        "Saltiness": "Saltiness",
        "Tenderness and softness": "Tenderness",
        "Overall visual match": "Overall visual",
        "Fat and lean appearance": "Fat/lean",
        "Thickness and chew": "Thickness",
    }

    texts = []
    for _, row in df.iterrows():
        st = style.get(row["modality"], {"marker": "o", "face": "white", "edge": INK})
        ax.scatter(
            row["mahieu_rank"],
            row["our_rank"],
            s=48,
            marker=st["marker"],
            facecolors=st["face"],
            edgecolors=st["edge"],
            linewidths=0.9,
            zorder=3,
        )
        lab = label_map.get(row["label"], row["label"])
        texts.append(
            ax.text(
                row["mahieu_rank"],
                row["our_rank"],
                lab,
                fontsize=6.4,
                color=INK,
                zorder=4,
            )
        )

    lim = [0.0, 18.5]
    ax.plot(lim, lim, color=MUTED, linewidth=0.9, linestyle=":", zorder=1)
    ax.set_xlim(18.5, 0.0)
    ax.set_ylim(18.5, 0.0)
    ax.set_xticks(list(range(1, 18, 2)))
    ax.set_yticks(list(range(1, 18, 2)))
    ax.set_xlabel("Mahieu et al. topic rank\n(1 = strongest published driver)")
    ax.set_ylabel("LLM topic rank\n(1 = strongest LLM feature importance)")
    ax.grid(color=GRID_C, linestyle=":", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker=style[m]["marker"],
            color="w",
            markerfacecolor=style[m]["face"],
            markeredgecolor=style[m]["edge"],
            markersize=8,
            label=m,
        )
        for m in ("Flavor", "Texture", "Visual")
    ]
    ax.legend(handles=legend_elements, frameon=True, loc="lower right", fontsize=8)

    if adjust_text is not None:
        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="#888888", lw=0.5),
            expand_points=(1.3, 1.4),
            force_text=(0.4, 0.6),
            force_points=(0.2, 0.3),
            only_move={"points": "y", "text": "xy"},
        )
    else:
        # fallback offset if adjustText missing
        for t in texts:
            x, y = t.get_position()
            t.set_position((x, y - 0.45))

    fig.tight_layout()
    save(fig, "topic_rank_comparison")


def write_summary() -> None:
    summary = {
        "figures": [
            "workflow_scoring",
            "grid_performance",
            "operational_comparison",
            "modality_hierarchy",
            "topic_rank_comparison",
            "llm_scoring_example",
            "jar_validation",
            "cross_model_jar",
        ],
        "source_data": [
            repo_path(GRID_DATA / "bootstrap_family_win_probabilities.csv"),
            repo_path(GRID_DATA / "bootstrap_pairwise_probabilities.csv"),
            repo_path(GRID_DATA / "bootstrap_top_config_results.csv"),
            repo_path(GRID_DATA / "downstream_model_comparison_results.csv"),
            repo_path(GRID_RESULTS),
            repo_path(GRID_DATA / "feature_importance_results.csv"),
            repo_path(TOPIC_DATA / "topic_level_contrast_points.csv"),
        ],
        "design": "Gemini 3.6 Flash guidance: greyscale fills, marker legends, adjustText labels, bar callouts",
    }
    (OUT / "figure_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    # draw_workflow()  # unused; methods figure built separately
    draw_model_grid()
    draw_operations()
    draw_modality_importance()
    draw_topic_rank()
    write_summary()
    print(f"Wrote core figures to {OUT}")


if __name__ == "__main__":
    main()
