#!/usr/bin/env python3
"""Per-modality preference maps from LLM topic-alignment scores.

Reproduces the *form* of Mahieu et al. (2022) Fig. 3: one biplot per sensory
modality (visual, texture, flavor) showing the 30 cooked hams positioned in a
sensory space, the ideal product projected as a supplementary point, and mean
liking projected as a supplementary vector.

This is an analog, not a literal rerun of the original MR-CA. The original ran
MR-CA on descriptor citation proportions. Here the space is built from LLM
actual-versus-ideal topic-alignment scores, so the "ideal" is the point of
perfect alignment (every topic at the scale maximum) rather than a projected
ideal-product descriptor profile. The map answers the same question: where do
the products sit, where is the ideal, and which way does liking point.
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

BG = "#F7F6F2"
INK = "#1A2721"
MUTED = "#4A5D53"
TERRACOTTA = "#C05A45"
SANDSTONE = "#EAE6DB"
SAGE = "#6F8A78"

SCALE_MAX = 6  # six-point alignment scale used for the topic-level run

MODALITY_TOPICS: dict[str, list[str]] = {
    "Visual": [
        "overall_visual_match",
        "color_pinkness_match",
        "fat_lean_appearance_match",
        "surface_moisture_shine_match",
        "slice_structure_homogeneity_match",
    ],
    "Texture": [
        "overall_texture_match",
        "tenderness_softness_match",
        "firmness_rubberiness_match",
        "dryness_juiciness_match",
        "fibrous_stringy_pieces_match",
        "thickness_chew_match",
    ],
    "Flavor": [
        "overall_flavor_match",
        "saltiness_match",
        "ham_taste_match",
        "aromatic_smoky_spiced_match",
        "bland_insipid_intensity_match",
        "offnote_aftertaste_match",
    ],
}

TOPIC_LABELS = {
    "overall_visual_match": "Overall visual",
    "color_pinkness_match": "Color/pinkness",
    "fat_lean_appearance_match": "Fat/lean",
    "surface_moisture_shine_match": "Moisture/shine",
    "slice_structure_homogeneity_match": "Slice structure",
    "overall_texture_match": "Overall texture",
    "tenderness_softness_match": "Tenderness",
    "firmness_rubberiness_match": "Firmness/rubbery",
    "dryness_juiciness_match": "Dryness/juiciness",
    "fibrous_stringy_pieces_match": "Fibrous pieces",
    "thickness_chew_match": "Thickness/chew",
    "overall_flavor_match": "Overall flavor",
    "saltiness_match": "Saltiness",
    "ham_taste_match": "Ham taste",
    "aromatic_smoky_spiced_match": "Smoky/spiced",
    "bland_insipid_intensity_match": "Blandness",
    "offnote_aftertaste_match": "Off-notes",
}

# Mahieu et al. (2022) reported the average absolute weighted correlation of
# mean liking with the sensory axes for each modality. Used here only as a
# reference point for the modality hierarchy comparison.
MAHIEU_LIKING_AXIS_STRENGTH = {"Visual": 0.139, "Texture": 0.201, "Flavor": 0.261}


def load_valid_scores(scores_path: Path) -> pd.DataFrame:
    scores = pd.read_csv(scores_path)
    parse_error = scores.get("parse_error")
    if parse_error is not None:
        mask = parse_error.fillna("").astype(str).eq("")
        scores = scores[mask].copy()
    for topic in TOPIC_LABELS:
        if topic in scores.columns:
            scores[topic] = pd.to_numeric(scores[topic], errors="coerce")
    scores["Liking"] = pd.to_numeric(scores["Liking"], errors="coerce")
    return scores


def product_means(scores: pd.DataFrame, topics: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    grouped = scores.groupby("Product")
    matrix = grouped[topics].mean(numeric_only=True)
    matrix = matrix.dropna(axis=1, how="all")
    matrix = matrix.fillna(matrix.mean())
    liking = grouped["Liking"].mean().reindex(matrix.index).astype(float)
    return matrix, liking


def covariance_pca(matrix: pd.DataFrame) -> dict[str, object]:
    """Center-only PCA via SVD. Topics share one 1-6 scale, so no rescaling."""
    values = matrix.to_numpy(dtype=float)
    center = values.mean(axis=0)
    centered = values - center
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    axes = vt.T  # p x k principal axes
    product_coords = centered @ axes  # n x k principal coordinates
    var_share = (s**2) / (s**2).sum() if (s**2).sum() else s**2
    return {
        "center": center,
        "axes": axes,
        "product_coords": product_coords,
        "singular_values": s,
        "var_share": var_share,
        "index": list(matrix.index),
        "topics": list(matrix.columns),
    }


def project_point(point: np.ndarray, pca: dict[str, object]) -> np.ndarray:
    centered = point - pca["center"]
    return centered @ pca["axes"][:, :2]


def liking_vector(product_coords: np.ndarray, liking: np.ndarray) -> np.ndarray:
    """Correlation of liking with each of the first two axes."""
    out = []
    for dim in range(2):
        axis = product_coords[:, dim]
        if axis.std() < 1e-9 or np.asarray(liking).std() < 1e-9:
            out.append(0.0)
        else:
            out.append(float(np.corrcoef(axis, liking)[0, 1]))
    return np.array(out)


def orient(pca: dict[str, object], like_vec: np.ndarray, ideal2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flip axis signs so liking points up and to the right (readability only)."""
    coords = pca["product_coords"].copy()
    axes = pca["axes"].copy()
    flips = np.array([1.0, 1.0])
    for dim in range(2):
        if like_vec[dim] < 0:
            flips[dim] = -1.0
    coords[:, :2] *= flips
    axes[:, :2] *= flips
    pca["product_coords"] = coords
    pca["axes"] = axes
    return like_vec * flips, ideal2 * flips


def build_modality_map(
    scores: pd.DataFrame, modality: str, topics: list[str]
) -> dict[str, object]:
    present = [t for t in topics if t in scores.columns]
    matrix, liking = product_means(scores, present)
    pca = covariance_pca(matrix)
    coords = pca["product_coords"]
    like_vec = liking_vector(coords, liking.to_numpy())

    ideal_point = np.full(len(pca["topics"]), float(SCALE_MAX))
    ideal2 = project_point(ideal_point, pca)

    like_vec, ideal2 = orient(pca, like_vec, ideal2)
    coords = pca["product_coords"]

    # cosine between liking direction and ideal direction (1.0 = liking points
    # straight at the ideal, reproducing the original's central finding).
    ideal_dir = ideal2 / (np.linalg.norm(ideal2) + 1e-12)
    like_dir = like_vec / (np.linalg.norm(like_vec) + 1e-12)
    cos_like_ideal = float(np.dot(ideal_dir, like_dir))

    return {
        "modality": modality,
        "matrix": matrix,
        "liking": liking,
        "pca": pca,
        "coords": coords,
        "like_vec": like_vec,
        "ideal2": ideal2,
        "liking_axis_strength": float(np.mean(np.abs(like_vec))),
        "cos_like_ideal": cos_like_ideal,
        "var_share": pca["var_share"],
    }


def draw_map(ax: plt.Axes, m: dict[str, object]) -> None:
    coords = m["coords"]
    pca = m["pca"]
    liking = m["liking"].to_numpy()
    index = pca["index"]
    var = m["var_share"]

    # Frame to the product cloud only. The ideal (perfect alignment) sits far
    # outside it by construction, so it is clamped to the frame edge below and
    # drawn in its true direction, like the original Fig. 3.
    span = max(np.abs(coords[:, :2]).max(), 1e-6)

    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=liking,
        cmap="YlOrRd",
        edgecolor=INK,
        linewidth=0.5,
        s=70,
        alpha=0.92,
        zorder=3,
    )
    for label, x, y in zip(index, coords[:, 0], coords[:, 1]):
        ax.text(x, y, str(label), fontsize=6, color=INK, ha="center", va="center", zorder=4)

    # topic loading vectors
    axes = pca["axes"][:, :2]
    load_scale = 0.62 * span / (np.abs(axes).max() + 1e-9)
    for j, topic in enumerate(pca["topics"]):
        vx, vy = axes[j, 0] * load_scale, axes[j, 1] * load_scale
        ax.plot([0, vx], [0, vy], color=TERRACOTTA, linewidth=1.0, alpha=0.55, zorder=2)
        ax.text(
            vx * 1.08,
            vy * 1.08,
            TOPIC_LABELS.get(topic, topic),
            fontsize=7,
            color=TERRACOTTA,
            ha="center",
            va="center",
            zorder=5,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": BG, "edgecolor": "none", "alpha": 0.78},
        )

    # ideal point (perfect alignment), placed at its projected location
    ix, iy = m["ideal2"]
    # keep the ideal marker inside the frame if the raw projection is far out
    norm = np.hypot(ix, iy)
    if norm > 1.15 * span:
        ix, iy = ix / norm * 1.12 * span, iy / norm * 1.12 * span
    ax.scatter([ix], [iy], marker="*", s=320, color=SAGE, edgecolor=INK, linewidth=0.8, zorder=6)
    ax.text(ix, iy + 0.07 * span, "Ideal", fontsize=9, color=INK, ha="center", va="bottom", zorder=6)

    # liking supplementary vector (label offset perpendicular to avoid the
    # near-collinear ideal marker)
    lx, ly = m["like_vec"] * 0.92 * span
    ax.annotate(
        "",
        xy=(lx, ly),
        xytext=(0, 0),
        arrowprops={"arrowstyle": "-|>", "color": INK, "linewidth": 2.0},
        zorder=7,
    )
    lnorm = np.hypot(lx, ly) + 1e-12
    perp = np.array([-ly, lx]) / lnorm
    mid = np.array([lx, ly]) * 0.55 + perp * 0.13 * span
    ax.text(mid[0], mid[1], "Liking", fontsize=10, color=INK, fontweight="bold", ha="center", va="center", zorder=7)

    ax.axhline(0, color=MUTED, linewidth=0.7, alpha=0.3)
    ax.axvline(0, color=MUTED, linewidth=0.7, alpha=0.3)
    lim = 1.25 * span
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Dim 1 ({var[0] * 100:.0f}%)", color=INK, fontsize=9)
    ax.set_ylabel(f"Dim 2 ({var[1] * 100:.0f}%)", color=INK, fontsize=9)
    ax.set_title(
        f"{m['modality']}   (liking-to-ideal cos = {m['cos_like_ideal']:.2f})",
        color=INK,
        fontsize=13,
        loc="left",
        pad=8,
    )
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(SANDSTONE)
    return sc


def render(output_dir: Path, scores_path: Path) -> dict[str, object]:
    scores = load_valid_scores(scores_path)
    maps = {
        modality: build_modality_map(scores, modality, topics)
        for modality, topics in MODALITY_TOPICS.items()
    }

    # individual panels
    output_dir.mkdir(parents=True, exist_ok=True)
    for modality, m in maps.items():
        fig, ax = plt.subplots(figsize=(6.6, 6.2), facecolor=BG)
        ax.set_facecolor(BG)
        sc = draw_map(ax, m)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("Product mean liking", color=INK, fontsize=8)
        cbar.ax.tick_params(colors=MUTED, labelsize=7)
        fig.tight_layout()
        out = output_dir / f"modality_map_{modality.lower()}.png"
        fig.savefig(out, dpi=190, facecolor=BG, bbox_inches="tight")
        plt.close(fig)

    # combined three-panel figure (matches the original Fig. 3 layout)
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.2), facecolor=BG)
    last = None
    for ax, modality in zip(axes, MODALITY_TOPICS):
        ax.set_facecolor(BG)
        last = draw_map(ax, maps[modality])
    cbar = fig.colorbar(last, ax=axes, fraction=0.018, pad=0.02)
    cbar.set_label("Product mean liking", color=INK, fontsize=9)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    fig.suptitle(
        "LLM topic-alignment preference maps by modality (analog of Mahieu et al. 2022, Fig. 3)",
        color=INK,
        fontsize=15,
        x=0.02,
        ha="left",
        y=1.02,
    )
    combined = output_dir / "modality_maps_combined.png"
    fig.savefig(combined, dpi=190, facecolor=BG, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "scores_path": str(scores_path),
        "products": int(next(iter(maps.values()))["coords"].shape[0]),
        "scale_max": SCALE_MAX,
        "modalities": {
            modality: {
                "liking_axis_strength": m["liking_axis_strength"],
                "cos_like_ideal": m["cos_like_ideal"],
                "var_share_dim1": float(m["var_share"][0]),
                "var_share_dim2": float(m["var_share"][1]),
                "mahieu_liking_axis_strength": MAHIEU_LIKING_AXIS_STRENGTH.get(modality),
            }
            for modality, m in maps.items()
        },
    }
    ours_order = " > ".join(
        sorted(maps, key=lambda k: maps[k]["liking_axis_strength"], reverse=True)
    )
    mahieu_order = " > ".join(
        sorted(MAHIEU_LIKING_AXIS_STRENGTH, key=MAHIEU_LIKING_AXIS_STRENGTH.get, reverse=True)
    )
    summary["our_modality_order"] = ours_order
    summary["mahieu_modality_order"] = mahieu_order
    (output_dir / "modality_maps_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=Path("analysis/topic_level"),
        type=Path,
    )
    parser.add_argument(
        "--scores-path",
        default=None,
        type=Path,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scores_path = args.scores_path or args.output_dir / "topic_level_flash_lite_scores.csv"
    summary = render(args.output_dir, scores_path)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
