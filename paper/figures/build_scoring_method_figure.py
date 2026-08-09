"""Build methods figure: pipeline banner + actual/ideal scoring example."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INK = "#111111"
MUTED = "#555555"
ACCENT = "#222222"
SURFACE = "#F0F0F0"
CANVAS = "#FFFFFF"
LINE = "#888888"


def add_box(ax, x, y, w, h, title, kicker, rows, fill, edge):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=0.9,
        edgecolor=edge,
        facecolor=fill,
    )
    ax.add_patch(box)
    ax.text(x + 0.018, y + h - 0.035, kicker, ha="left", va="top", fontsize=5.8, color=MUTED)
    ax.text(
        x + 0.018,
        y + h - 0.095,
        title,
        ha="left",
        va="top",
        fontsize=9.2,
        color=INK,
        fontstyle="italic",
    )
    row_y = y + h - 0.185
    for label, body in rows:
        ax.text(x + 0.02, row_y, label, ha="left", va="top", fontsize=5.9, color=ACCENT)
        ax.text(
            x + 0.105,
            row_y,
            "\n".join(textwrap.wrap(body, 34)),
            ha="left",
            va="top",
            fontsize=5.7,
            color=INK,
            fontstyle="italic",
            linespacing=1.1,
        )
        row_y -= 0.105


def draw_pipeline_banner(ax, y0=0.88, height=0.10):
    """Compact five-step pipeline banner (short labels, low height)."""
    labels = [
        "Actual + ideal",
        "LLM scoring",
        "Three scores",
        "TabPFN",
        "Held-out liking",
    ]
    n = len(labels)
    left, right = 0.04, 0.96
    gap = 0.022
    total_w = right - left
    box_w = (total_w - gap * (n - 1)) / n
    y = y0
    h = height
    xs = []
    for i, lab in enumerate(labels):
        x = left + i * (box_w + gap)
        xs.append(x)
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                box_w,
                h,
                boxstyle="round,pad=0.006,rounding_size=0.012",
                facecolor=SURFACE,
                edgecolor=INK,
                linewidth=0.75,
            )
        )
        ax.text(
            x + box_w / 2,
            y + h / 2,
            lab,
            ha="center",
            va="center",
            fontsize=6.8,
            color=INK,
        )
    for i in range(n - 1):
        x1 = xs[i] + box_w
        x2 = xs[i + 1]
        mid_y = y + h / 2
        ax.add_patch(
            FancyArrowPatch(
                (x1 + 0.003, mid_y),
                (x2 - 0.003, mid_y),
                arrowstyle="-|>",
                mutation_scale=7,
                linewidth=0.75,
                color=MUTED,
            )
        )


def main():
    fig, ax = plt.subplots(figsize=(7.4, 5.5), dpi=300)
    ax.set_facecolor(CANVAS)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_pipeline_banner(ax)

    ax.text(
        0.04,
        0.82,
        "Example evaluation (liking withheld from the model; used only in downstream TabPFN).",
        ha="left",
        va="top",
        fontsize=7.0,
        color=MUTED,
    )

    actual_rows = [
        ("Visual", "Couleur moins rose que les jambons habituels. Belle tranche."),
        ("Texture", "Le jambon se défait mais il a une bonne texture."),
        ("Flavor", "Très bon, pas trop salé."),
    ]
    ideal_rows = [
        ("Visual", "Belle tranche légèrement rose persillée."),
        ("Texture", "Tranche un peu épaisse qui ne se défait pas en bouche."),
        ("Flavor", "Léger goût de fumé, pas trop salé."),
    ]

    # slightly narrower cards so titles stay inside borders
    add_box(
        ax,
        0.04,
        0.20,
        0.40,
        0.56,
        "Actual product response",
        "Consumer H041 · product J09 · liking 8.90 hidden",
        actual_rows,
        SURFACE,
        LINE,
    )
    add_box(
        ax,
        0.56,
        0.20,
        0.40,
        0.56,
        "Same consumer ideal response",
        "Ideal Free-Comment questionnaire",
        ideal_rows,
        CANVAS,
        LINE,
    )

    ax.annotate(
        "",
        xy=(0.545, 0.48),
        xytext=(0.455, 0.48),
        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0),
    )
    ax.text(0.50, 0.51, "same\nprompt", ha="center", va="bottom", fontsize=5.6, color=MUTED, linespacing=0.95)

    strip = FancyBboxPatch(
        (0.03, 0.02),
        0.94,
        0.16,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        linewidth=0.6,
        edgecolor=LINE,
        facecolor=SURFACE,
    )
    ax.add_patch(strip)

    ax.text(0.05, 0.145, "Fixed rubric (1-6)", ha="left", va="top", fontsize=6.8, color=ACCENT, weight="bold")
    ax.text(
        0.05,
        0.105,
        "1 = extremely different\n6 = extremely similar",
        ha="left",
        va="top",
        fontsize=6.0,
        color=MUTED,
    )

    ax.add_patch(Rectangle((0.32, 0.04), 0.0015, 0.11, facecolor=LINE, edgecolor=LINE))
    ax.text(0.35, 0.145, "Validated JSON", ha="left", va="top", fontsize=6.8, color=ACCENT, weight="bold")
    scores = [("visual", 4), ("texture", 5), ("flavor", 6)]
    for i, (label, value) in enumerate(scores):
        x = 0.38 + i * 0.09
        ax.text(x, 0.095, label, ha="center", va="center", fontsize=6.0, color=MUTED)
        ax.text(
            x,
            0.055,
            str(value),
            ha="center",
            va="center",
            fontsize=7.2,
            color=CANVAS,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.02", facecolor=ACCENT, edgecolor=ACCENT),
        )

    ax.add_patch(Rectangle((0.66, 0.04), 0.0015, 0.11, facecolor=LINE, edgecolor=LINE))
    ax.text(0.69, 0.145, "Downstream only", ha="left", va="top", fontsize=6.8, color=ACCENT, weight="bold")
    ax.text(
        0.69,
        0.095,
        "Three scores enter TabPFN;\nliking stays held out",
        ha="left",
        va="top",
        fontsize=6.0,
        color=MUTED,
    )

    # Manuscript uses workflow_scoring.png only (merged pipeline + example).
    fig.savefig(OUT_DIR / "workflow_scoring.png", dpi=300, bbox_inches="tight", facecolor=CANVAS, pad_inches=0.06)
    print(f"wrote paper/figures/workflow_scoring.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
