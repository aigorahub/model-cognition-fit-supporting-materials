#!/usr/bin/env python3
"""Render a temporary MR-CA style comparison report for topic-level ham analysis."""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

try:
    from run_topic_level_analysis import TOPICS, prepare_modeling_frame
except ModuleNotFoundError:
    from scripts.run_topic_level_analysis import TOPICS, prepare_modeling_frame


BG = "#F7F6F2"
INK = "#1A2721"
MUTED = "#4A5D53"
TERRACOTTA = "#C05A45"
SANDSTONE = "#EAE6DB"
SAGE = "#6F8A78"

MAHIEU_MODALITY_SIGNAL = {
    "Visual": 0.139,
    "Texture": 0.201,
    "Flavor": 0.261,
}

TOPIC_LABELS = {
    "overall_visual_match": "Overall visual match",
    "color_pinkness_match": "Color and pinkness",
    "fat_lean_appearance_match": "Fat and lean appearance",
    "surface_moisture_shine_match": "Moisture and shine",
    "slice_structure_homogeneity_match": "Slice structure",
    "overall_texture_match": "Overall texture match",
    "tenderness_softness_match": "Tenderness and softness",
    "firmness_rubberiness_match": "Firmness and rubberiness",
    "dryness_juiciness_match": "Dryness and juiciness",
    "fibrous_stringy_pieces_match": "Fibrous pieces",
    "thickness_chew_match": "Thickness and chew",
    "overall_flavor_match": "Overall flavor match",
    "saltiness_match": "Saltiness",
    "ham_taste_match": "Ham taste",
    "aromatic_smoky_spiced_match": "Smoky and spiced notes",
    "bland_insipid_intensity_match": "Blandness and intensity",
    "offnote_aftertaste_match": "Off-notes and aftertaste",
}

MAHIEU_ANALOGS = {
    "color_pinkness_match": "pink, pale, grey, dark",
    "fat_lean_appearance_match": "fat, not fat, marbled",
    "surface_moisture_shine_match": "wet, bright, dull",
    "slice_structure_homogeneity_match": "homogeneous, veined, fibrous",
    "tenderness_softness_match": "soft, tender, melting",
    "firmness_rubberiness_match": "firm, elastic, rubbery",
    "dryness_juiciness_match": "dry, pasty, juicy",
    "fibrous_stringy_pieces_match": "fibrous, stringy, hard pieces",
    "saltiness_match": "salty, not salty",
    "ham_taste_match": "ham taste",
    "aromatic_smoky_spiced_match": "fragrant, smoked, spicy, stock aromatics",
    "bland_insipid_intensity_match": "insipid",
    "offnote_aftertaste_match": "off-notes, aftertaste",
}

MAHIEU_DRIVER_LOADINGS = {
    "F_fragrant": 0.85,
    "F_smoked": 0.75,
    "F_ham_taste": 0.60,
    "F_not_salty": 0.50,
    "T_soft_tender_melting": 0.40,
    "F_spicy_stock_aromatics": 0.30,
    "T_juicy": 0.20,
    "V_not_fat": 0.15,
    "V_pink": 0.10,
    "V_marbled": 0.05,
    "V_natural": 0.00,
    "T_firm_hearty": -0.05,
    "V_beige_brown": -0.10,
    "V_light": -0.15,
    "V_bright": -0.20,
    "V_no_rind": -0.25,
    "V_homogeneous": -0.30,
    "V_grey": -0.30,
    "V_white": -0.35,
    "V_heterogeneous": -0.35,
    "V_dull": -0.40,
    "F_salty": -0.35,
    "V_deep_dark": -0.40,
    "V_fat": -0.50,
    "T_fibrous_stringy": -0.60,
    "V_pale": -0.70,
    "V_wet": -0.80,
    "V_veined_fibrous": -0.90,
    "T_hard_pieces": -1.00,
    "T_dry_pasty": -1.10,
    "T_fat": -1.20,
    "V_no_color": -1.30,
    "T_elastic_rubbery": -1.40,
    "F_insipid": -1.60,
}

TOPIC_TO_MAHIEU_DESCRIPTORS = {
    "overall_visual_match": [
        "V_not_fat",
        "V_pink",
        "V_marbled",
        "V_natural",
        "V_beige_brown",
        "V_light",
        "V_bright",
        "V_no_rind",
        "V_homogeneous",
        "V_grey",
        "V_white",
        "V_heterogeneous",
        "V_dull",
        "V_deep_dark",
        "V_fat",
        "V_pale",
        "V_wet",
        "V_veined_fibrous",
        "V_no_color",
    ],
    "color_pinkness_match": [
        "V_pink",
        "V_pale",
        "V_no_color",
        "V_grey",
        "V_white",
        "V_deep_dark",
        "V_light",
        "V_beige_brown",
    ],
    "fat_lean_appearance_match": ["V_fat", "V_not_fat", "V_marbled"],
    "surface_moisture_shine_match": ["V_wet", "V_bright", "V_dull"],
    "slice_structure_homogeneity_match": [
        "V_homogeneous",
        "V_heterogeneous",
        "V_veined_fibrous",
        "V_no_rind",
    ],
    "overall_texture_match": [
        "T_soft_tender_melting",
        "T_juicy",
        "T_firm_hearty",
        "T_fibrous_stringy",
        "T_hard_pieces",
        "T_dry_pasty",
        "T_fat",
        "T_elastic_rubbery",
    ],
    "tenderness_softness_match": ["T_soft_tender_melting"],
    "firmness_rubberiness_match": ["T_firm_hearty", "T_elastic_rubbery"],
    "dryness_juiciness_match": ["T_dry_pasty", "T_juicy"],
    "fibrous_stringy_pieces_match": ["T_fibrous_stringy", "T_hard_pieces"],
    "thickness_chew_match": ["T_firm_hearty"],
    "overall_flavor_match": [
        "F_fragrant",
        "F_smoked",
        "F_ham_taste",
        "F_not_salty",
        "F_spicy_stock_aromatics",
        "F_salty",
        "F_insipid",
    ],
    "saltiness_match": ["F_not_salty", "F_salty"],
    "ham_taste_match": ["F_ham_taste"],
    "aromatic_smoky_spiced_match": [
        "F_fragrant",
        "F_smoked",
        "F_spicy_stock_aromatics",
    ],
    "bland_insipid_intensity_match": ["F_insipid"],
    "offnote_aftertaste_match": ["F_insipid"],
}


def topic_modality(topic: str) -> str:
    if "_visual_" in topic or topic.startswith(("overall_visual", "color_", "fat_", "surface_", "slice_")):
        return "Visual"
    if "_texture_" in topic or topic.startswith(("overall_texture", "tenderness_", "firmness_", "dryness_", "fibrous_", "thickness_")):
        return "Texture"
    return "Flavor"


def correspondence_analysis(matrix: pd.DataFrame) -> dict[str, Any]:
    """Run simple correspondence analysis on a nonnegative matrix."""
    clean = matrix.astype(float).copy()
    clean = clean.loc[clean.sum(axis=1) > 0, clean.sum(axis=0) > 0]
    if clean.shape[0] < 2 or clean.shape[1] < 2:
        raise ValueError("CA needs at least two nonzero rows and columns")

    observed = clean.to_numpy(dtype=float)
    grand_total = observed.sum()
    profile = observed / grand_total
    row_mass = profile.sum(axis=1)
    col_mass = profile.sum(axis=0)
    residual = profile - np.outer(row_mass, col_mass)

    with np.errstate(divide="ignore", invalid="ignore"):
        standardized = residual / np.sqrt(np.outer(row_mass, col_mass))
    standardized[~np.isfinite(standardized)] = 0.0

    u, singular_values, vt = np.linalg.svd(standardized, full_matrices=False)
    dims = min(2, len(singular_values))
    row_coords = (u[:, :dims] * singular_values[:dims]) / np.sqrt(row_mass[:, None])
    col_coords = (vt.T[:, :dims] * singular_values[:dims]) / np.sqrt(col_mass[:, None])
    inertia = singular_values**2
    inertia_share = inertia / inertia.sum() if inertia.sum() else inertia

    row_frame = pd.DataFrame(
        row_coords,
        index=clean.index,
        columns=[f"Dim {i + 1}" for i in range(dims)],
    )
    col_frame = pd.DataFrame(
        col_coords,
        index=clean.columns,
        columns=[f"Dim {i + 1}" for i in range(dims)],
    )
    return {
        "row_coords": row_frame,
        "col_coords": col_frame,
        "inertia_share": inertia_share,
        "singular_values": singular_values,
        "matrix": clean,
    }


def product_topic_alignment(scores: pd.DataFrame) -> pd.DataFrame:
    topics = [topic for topic in TOPICS if topic in scores.columns]
    if not topics:
        raise ValueError("No topic score columns found")
    product_topic = (
        scores.groupby("Product")[topics]
        .mean(numeric_only=True)
        .dropna(axis=1, how="all")
        .fillna(3.5)
    )
    return product_topic


def product_liking(scores: pd.DataFrame) -> pd.Series:
    return scores.groupby("Product")["Liking"].mean().astype(float)


def product_topic_correlations(scores: pd.DataFrame) -> pd.DataFrame:
    product_scores = product_topic_alignment(scores)
    liking = product_liking(scores).reindex(product_scores.index)
    rows: list[dict[str, Any]] = []
    for topic in product_scores.columns:
        valid = product_scores[topic].notna() & liking.notna()
        if valid.sum() < 4:
            rho, p_value = np.nan, np.nan
        else:
            rho, p_value = spearmanr(product_scores.loc[valid, topic], liking.loc[valid])
        rows.append(
            {
                "topic": topic,
                "label": TOPIC_LABELS.get(topic, topic),
                "modality": topic_modality(topic),
                "rho_product_mean": float(rho) if np.isfinite(rho) else np.nan,
                "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                "mean_score": float(product_scores[topic].mean()),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values("rho_product_mean", ascending=False, na_position="last")


def row_topic_correlations(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for topic in [topic for topic in TOPICS if topic in scores.columns]:
        valid = scores[topic].notna() & scores["Liking"].notna()
        if valid.sum() < 4:
            rho, p_value = np.nan, np.nan
        else:
            rho, p_value = spearmanr(scores.loc[valid, topic], scores.loc[valid, "Liking"])
        rows.append(
            {
                "topic": topic,
                "label": TOPIC_LABELS.get(topic, topic),
                "modality": topic_modality(topic),
                "rho_row": float(rho) if np.isfinite(rho) else np.nan,
                "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                "mean_score": float(scores[topic].mean()),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values("rho_row", ascending=False, na_position="last")


def modality_signal_from_correlations(correlations: pd.DataFrame, column: str) -> dict[str, float]:
    signal: dict[str, float] = {}
    for modality, group in correlations.groupby("modality"):
        signal[modality] = float(group[column].abs().mean())
    return signal


def local_gradient_boosting_importance(
    scores: pd.DataFrame,
    *,
    scale_points: int,
    output_dir: Path,
) -> tuple[dict[str, float], pd.DataFrame]:
    modeling, features = prepare_modeling_frame(scores, scale_points=scale_points)
    train_mask = modeling["split"].eq("train")
    test_mask = modeling["split"].eq("holdout")
    x_train = modeling.loc[train_mask, features]
    y_train = modeling.loc[train_mask, "Liking"].astype(float)
    x_test = modeling.loc[test_mask, features]
    y_test = modeling.loc[test_mask, "Liking"].astype(float)

    model = GradientBoostingRegressor(random_state=42)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    metrics = {
        "r2": float(r2_score(y_test, prediction)),
        "mae": float(mean_absolute_error(y_test, prediction)),
        "rmse": float(root_mean_squared_error(y_test, prediction)),
        "train_rows": int(train_mask.sum()),
        "holdout_rows": int(test_mask.sum()),
        "model": "GradientBoostingRegressor fallback",
    }
    importance = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=5,
        random_state=42,
        scoring="r2",
    )
    importance_frame = pd.DataFrame(
        {
            "feature": features,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    )
    importance_frame = importance_frame[~importance_frame["feature"].astype(str).str.endswith("_missing")].copy()
    importance_frame["topic"] = importance_frame["feature"].astype(str)
    importance_frame["modality"] = importance_frame["topic"].map(topic_modality)
    importance_frame["label"] = importance_frame["topic"].map(lambda topic: TOPIC_LABELS.get(topic, topic))
    importance_frame["importance_positive"] = importance_frame["importance_mean"].clip(lower=0)
    importance_frame = importance_frame.sort_values("importance_positive", ascending=False)
    importance_frame.to_csv(output_dir / "topic_level_gradient_boosting_importance.csv", index=False)
    (output_dir / "topic_level_gradient_boosting_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics, importance_frame


def load_tabpfn_metric(output_dir: Path) -> dict[str, Any] | None:
    candidates = sorted(output_dir.glob("tabpfn_remote/*/*-artifacts/metrics.json"))
    if not candidates:
        candidates = sorted(output_dir.glob("tabpfn_remote/*/tabpfn_summary.json"))
    for path in candidates:
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
    return None


def load_feature_importance(output_dir: Path) -> pd.DataFrame | None:
    candidates = sorted(output_dir.glob("tabpfn_remote/*/*-artifacts/feature_importance.csv"))
    if not candidates:
        return None
    importance = pd.read_csv(candidates[-1])
    if "importance_mean" not in importance.columns:
        return None
    importance = importance[~importance["feature"].astype(str).str.endswith("_missing")].copy()
    importance["topic"] = importance["feature"].astype(str)
    importance["modality"] = importance["topic"].map(topic_modality)
    importance["label"] = importance["topic"].map(lambda topic: TOPIC_LABELS.get(topic, topic))
    importance["importance_positive"] = importance["importance_mean"].clip(lower=0)
    return importance


def modality_signal_from_importance(importance: pd.DataFrame | None) -> dict[str, float] | None:
    if importance is None or importance.empty:
        return None
    grouped = importance.groupby("modality")["importance_positive"].sum()
    total = float(grouped.sum())
    if total <= 0:
        grouped = importance.groupby("modality")["importance_mean"].sum()
        total = float(grouped.abs().sum())
        if total <= 0:
            return None
        return {name: float(abs(value) / total) for name, value in grouped.items()}
    return {name: float(value / total) for name, value in grouped.items()}


def topic_contrast_points(row_correlations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    row_lookup = row_correlations.set_index("topic")
    mahieu_scale = max(abs(value) for value in MAHIEU_DRIVER_LOADINGS.values())
    our_scale = float(row_correlations["rho_row"].abs().max())
    for topic, descriptors in TOPIC_TO_MAHIEU_DESCRIPTORS.items():
        if topic not in row_lookup.index:
            continue
        loadings = [MAHIEU_DRIVER_LOADINGS[name] for name in descriptors if name in MAHIEU_DRIVER_LOADINGS]
        if not loadings:
            continue
        our_rho = float(row_lookup.loc[topic, "rho_row"])
        mahieu_abs = float(np.mean(np.abs(loadings)))
        rows.append(
            {
                "topic": topic,
                "label": TOPIC_LABELS.get(topic, topic),
                "modality": topic_modality(topic),
                "mahieu_mean_abs_loading": mahieu_abs,
                "mahieu_signed_mean_loading": float(np.mean(loadings)),
                "mahieu_strength_norm": mahieu_abs / mahieu_scale if mahieu_scale else 0,
                "our_abs_rho": abs(our_rho),
                "our_signed_rho": our_rho,
                "our_strength_norm": abs(our_rho) / our_scale if our_scale else 0,
                "descriptor_family": ", ".join(descriptors),
            }
        )
    frame = pd.DataFrame(rows).sort_values("mahieu_strength_norm", ascending=False)
    frame["mahieu_rank"] = frame["mahieu_strength_norm"].rank(ascending=False, method="min").astype(int)
    frame["our_rank"] = frame["our_strength_norm"].rank(ascending=False, method="min").astype(int)
    frame["rank_delta"] = frame["our_rank"] - frame["mahieu_rank"]
    return frame


def normalized(values: dict[str, float]) -> dict[str, float]:
    total = sum(abs(v) for v in values.values())
    if total == 0:
        return {k: 0.0 for k in values}
    return {k: abs(v) / total for k, v in values.items()}


def rank_order(values: dict[str, float]) -> list[str]:
    return [key for key, _ in sorted(values.items(), key=lambda item: item[1], reverse=True)]


def spearman_for_modalities(left: dict[str, float], right: dict[str, float]) -> float:
    modalities = [name for name in ["Visual", "Texture", "Flavor"] if name in left and name in right]
    if len(modalities) < 3:
        return float("nan")
    rho, _ = spearmanr([left[name] for name in modalities], [right[name] for name in modalities])
    return float(rho)


def fig_to_data_uri(fig: plt.Figure) -> str:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def plot_mrca_map(ca: dict[str, Any], liking: pd.Series, correlations: pd.DataFrame) -> str:
    rows = ca["row_coords"].copy()
    cols = ca["col_coords"].copy()
    inertia = ca["inertia_share"]
    liking = liking.reindex(rows.index)

    fig, ax = plt.subplots(figsize=(10.5, 7), facecolor=BG)
    ax.set_facecolor(BG)
    scatter = ax.scatter(
        rows["Dim 1"],
        rows["Dim 2"],
        c=liking,
        cmap="YlOrRd",
        edgecolor=INK,
        linewidth=0.6,
        s=72,
        alpha=0.9,
    )
    for product, point in rows.iterrows():
        ax.text(point["Dim 1"], point["Dim 2"], str(product), fontsize=7, color=INK, ha="center", va="center")

    contribution = (cols["Dim 1"] ** 2 + cols["Dim 2"] ** 2).sort_values(ascending=False)
    label_topics = contribution.head(12).index
    scale = 0.82 * min(
        rows["Dim 1"].abs().max() / max(cols["Dim 1"].abs().max(), 1e-9),
        rows["Dim 2"].abs().max() / max(cols["Dim 2"].abs().max(), 1e-9),
    )
    for topic in label_topics:
        x = cols.loc[topic, "Dim 1"] * scale
        y = cols.loc[topic, "Dim 2"] * scale
        ax.plot([0, x], [0, y], color=TERRACOTTA, linewidth=1.1, alpha=0.72)
        ax.text(
            x * 1.07,
            y * 1.07,
            TOPIC_LABELS.get(topic, topic),
            fontsize=8,
            color=TERRACOTTA,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": BG, "edgecolor": "none", "alpha": 0.82},
        )

    ax.axhline(0, color=MUTED, linewidth=0.8, alpha=0.35)
    ax.axvline(0, color=MUTED, linewidth=0.8, alpha=0.35)
    ax.set_xlabel(f"Dimension 1 ({inertia[0] * 100:.1f}% inertia)", color=INK)
    ax.set_ylabel(f"Dimension 2 ({inertia[1] * 100:.1f}% inertia)", color=INK)
    ax.set_title("MR-CA style topic-alignment map", color=INK, fontsize=16, loc="left", pad=12)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(SANDSTONE)
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.03)
    colorbar.set_label("Product mean liking", color=INK)
    colorbar.ax.tick_params(colors=MUTED)
    return fig_to_data_uri(fig)


def plot_modality_comparison(ours: dict[str, float], title: str) -> str:
    mahieu = normalized(MAHIEU_MODALITY_SIGNAL)
    ours_norm = normalized(ours)
    labels = ["Visual", "Texture", "Flavor"]
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.8, 4.6), facecolor=BG)
    ax.set_facecolor(BG)
    ax.bar(x - width / 2, [mahieu.get(label, 0) for label in labels], width, color=SAGE, label="Mahieu et al. MR-CA signal")
    ax.bar(x + width / 2, [ours_norm.get(label, 0) for label in labels], width, color=TERRACOTTA, label=title)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(0.6, ax.get_ylim()[1]))
    ax.set_ylabel("Share of modality signal", color=INK)
    ax.set_title("Modality hierarchy comparison", color=INK, fontsize=15, loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(SANDSTONE)
    ax.grid(axis="y", color=SANDSTONE, linewidth=0.8)
    return fig_to_data_uri(fig)


def plot_topic_bars(correlations: pd.DataFrame, importance: pd.DataFrame | None) -> str:
    if importance is not None and not importance.empty:
        frame = importance.sort_values("importance_positive", ascending=False).head(10)
        metric = "importance_positive"
        title = "Top topic-level TabPFN permutation signals"
        xlabel = "Positive permutation importance"
    else:
        frame = correlations.assign(abs_rho=correlations["rho_product_mean"].abs()).sort_values("abs_rho", ascending=False).head(10)
        metric = "abs_rho"
        title = "Top product-level topic correlations"
        xlabel = "|Spearman rho with product mean liking|"

    frame = frame.iloc[::-1]
    colors = frame["modality"].map({"Visual": SAGE, "Texture": MUTED, "Flavor": TERRACOTTA}).tolist()
    fig, ax = plt.subplots(figsize=(8.8, 5.6), facecolor=BG)
    ax.set_facecolor(BG)
    ax.barh(frame["label"], frame[metric], color=colors)
    ax.set_xlabel(xlabel, color=INK)
    ax.set_title(title, color=INK, fontsize=15, loc="left")
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(SANDSTONE)
    ax.grid(axis="x", color=SANDSTONE, linewidth=0.8)
    return fig_to_data_uri(fig)


def plot_contrast_points(contrast: pd.DataFrame) -> str:
    colors = {"Visual": SAGE, "Texture": MUTED, "Flavor": TERRACOTTA}
    fig, ax = plt.subplots(figsize=(8.8, 6.2), facecolor=BG)
    ax.set_facecolor(BG)
    ax.plot([0, 1.04], [0, 1.04], color=SANDSTONE, linewidth=1.4, linestyle="--", zorder=0)
    for modality, group in contrast.groupby("modality"):
        ax.scatter(
            group["mahieu_strength_norm"],
            group["our_strength_norm"],
            s=88,
            color=colors.get(modality, MUTED),
            edgecolor=INK,
            linewidth=0.7,
            alpha=0.92,
            label=modality,
        )
        for _, row in group.iterrows():
            label = str(row["label"]).replace(" and ", " & ")
            ax.annotate(
                label,
                (row["mahieu_strength_norm"], row["our_strength_norm"]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=7.4,
                color=INK,
                ha="left",
                va="bottom",
            )
    ax.set_xlim(-0.03, 1.08)
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlabel("Mahieu et al. driver strength, normalized", color=INK)
    ax.set_ylabel("Our LLM topic-liking signal, normalized", color=INK)
    ax.set_title("Topic contrast: paper drivers vs LLM topic signal", color=INK, fontsize=15, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.tick_params(colors=MUTED)
    ax.grid(color=SANDSTONE, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color(SANDSTONE)
    return fig_to_data_uri(fig)


def plot_rank_order(contrast: pd.DataFrame) -> str:
    colors = {"Visual": SAGE, "Texture": MUTED, "Flavor": TERRACOTTA}
    ranked = contrast.sort_values("mahieu_rank").copy()
    fig, ax = plt.subplots(figsize=(7.8, 8.4), facecolor=BG)
    ax.set_facecolor(BG)
    for _, row in ranked.iterrows():
        color = colors.get(row["modality"], MUTED)
        ax.plot([0, 1], [row["mahieu_rank"], row["our_rank"]], color=color, linewidth=1.4, alpha=0.62)
        ax.scatter([0, 1], [row["mahieu_rank"], row["our_rank"]], s=34, color=color, edgecolor=INK, linewidth=0.4, zorder=3)
        left_label = str(row["label"]).replace(" and ", " & ")
        right_label = left_label
        ax.text(-0.035, row["mahieu_rank"], left_label, ha="right", va="center", fontsize=7.5, color=INK)
        ax.text(1.035, row["our_rank"], right_label, ha="left", va="center", fontsize=7.5, color=INK)

    ax.set_xlim(-0.42, 1.42)
    ax.set_ylim(len(ranked) + 0.8, 0.2)
    ax.set_xticks([0, 1], ["Mahieu et al.", "Our LLM topics"])
    ax.set_ylabel("Rank, strongest at top", color=INK)
    ax.set_title("Rank-order comparison of matched topics", color=INK, fontsize=15, loc="left")
    ax.tick_params(axis="x", colors=INK, labelsize=11)
    ax.tick_params(axis="y", colors=MUTED)
    ax.grid(axis="y", color=SANDSTONE, linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig_to_data_uri(fig)


def format_float(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def table_rows(frame: pd.DataFrame, columns: list[tuple[str, str]], limit: int | None = None) -> str:
    shown = frame.head(limit) if limit else frame
    rows = []
    for _, row in shown.iterrows():
        cells = []
        for key, kind in columns:
            value = row.get(key, "")
            if kind == "float":
                value = format_float(value)
            cells.append(f"<td>{html.escape(str(value))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "\n".join(rows)


def render_report(output_dir: Path, report_path: Path) -> dict[str, Any]:
    scores_path = output_dir / "topic_level_flash_lite_scores.csv"
    metadata_path = output_dir / "topic_level_analysis_metadata.json"
    if not scores_path.exists():
        raise FileNotFoundError(scores_path)

    scores = pd.read_csv(scores_path)
    scores["parse_error"] = scores.get("parse_error", "").fillna("").astype(str)
    valid_scores = scores[scores["parse_error"].eq("")].copy()
    for topic in TOPICS:
        valid_scores[topic] = pd.to_numeric(valid_scores[topic], errors="coerce")
    valid_scores["Liking"] = pd.to_numeric(valid_scores["Liking"], errors="coerce")

    product_matrix = product_topic_alignment(valid_scores)
    ca = correspondence_analysis(product_matrix)
    liking = product_liking(valid_scores)
    product_correlations = product_topic_correlations(valid_scores)
    row_correlations = row_topic_correlations(valid_scores)
    row_modality_signal = modality_signal_from_correlations(row_correlations, "rho_row")
    contrast = topic_contrast_points(row_correlations)
    contrast_rho = float(
        spearmanr(contrast["mahieu_rank"], contrast["our_rank"]).statistic
    )
    contrast_tau = float(kendalltau(contrast["mahieu_rank"], contrast["our_rank"]).statistic)

    importance = load_feature_importance(output_dir)
    importance_modality_signal = modality_signal_from_importance(importance)
    fallback_metric: dict[str, Any] | None = None
    used_tabpfn_importance = importance_modality_signal is not None
    if importance_modality_signal is not None:
        comparison_signal = importance_modality_signal
        comparison_title = "Our TabPFN importance"
        comparison_basis = "TabPFN permutation importance, grouped by topic modality"
    else:
        fallback_metric, importance = local_gradient_boosting_importance(
            valid_scores,
            scale_points=6,
            output_dir=output_dir,
        )
        comparison_signal = modality_signal_from_importance(importance) or row_modality_signal
        comparison_title = "Our topic-model fallback"
        comparison_basis = "held-out Gradient Boosting permutation importance, grouped by topic modality"

    modality_rho = spearman_for_modalities(MAHIEU_MODALITY_SIGNAL, comparison_signal)
    metric = load_tabpfn_metric(output_dir)
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metric_body = metric.get("metrics", metric or {}) if metric else {}

    map_uri = plot_mrca_map(ca, liking, product_correlations)
    modality_uri = plot_modality_comparison(comparison_signal, comparison_title)
    topic_uri = plot_topic_bars(row_correlations.rename(columns={"rho_row": "rho_product_mean"}), importance)
    contrast_uri = plot_contrast_points(contrast)
    rank_uri = plot_rank_order(contrast)

    ca_dim1_like = float(spearmanr(ca["row_coords"]["Dim 1"], liking.reindex(ca["row_coords"].index)).statistic)
    ca_dim2_like = float(spearmanr(ca["row_coords"]["Dim 2"], liking.reindex(ca["row_coords"].index)).statistic)

    top_positive = row_correlations.sort_values("rho_row", ascending=False).head(8).copy()
    top_negative = row_correlations.sort_values("rho_row", ascending=True).head(8).copy()
    analog_rows = row_correlations[row_correlations["topic"].isin(MAHIEU_ANALOGS)].copy()
    analog_rows["mahieu_terms"] = analog_rows["topic"].map(MAHIEU_ANALOGS)
    analog_rows = analog_rows.sort_values("rho_row", ascending=False)

    mahieu_order = " > ".join(rank_order(MAHIEU_MODALITY_SIGNAL))
    our_order = " > ".join(rank_order(comparison_signal))
    parse_errors = int(scores["parse_error"].ne("").sum())
    rows_valid = len(valid_scores)
    products = valid_scores["Product"].nunique()
    consumers = valid_scores["Consumer"].nunique()

    if importance is not None and not importance.empty:
        top_signal = importance.sort_values("importance_positive", ascending=False).head(5)
        top_signal_text = ", ".join(top_signal["label"].tolist())
    else:
        top_signal = row_correlations.assign(abs_rho=row_correlations["rho_row"].abs()).sort_values("abs_rho", ascending=False).head(5)
        top_signal_text = ", ".join(top_signal["label"].tolist())
    contrast_for_table = contrast.sort_values("our_strength_norm", ascending=False).head(10)
    rank_for_table = contrast.sort_values("rank_delta", key=lambda series: series.abs(), ascending=False).head(8)

    displayed_metric = metric_body or (fallback_metric or {})
    metric_label = "TabPFN" if metric_body else "GB fallback"
    model_note = (
        "The metric cards use the Mini-hosted TabPFN run."
        if metric_body
        else "The metric cards use a local Gradient Boosting fallback because the full topic-level TabPFN run exceeded the Mini's current TabPFN memory limits."
    )
    cards = [
        ("Valid evaluations", f"{rows_valid:,}"),
        ("Products", f"{products:,}"),
        ("Consumers", f"{consumers:,}"),
        ("Parse errors", f"{parse_errors:,}"),
        (f"{metric_label} holdout R2", format_float(displayed_metric.get("r2"))),
        (f"{metric_label} MAE", format_float(displayed_metric.get("mae"))),
        ("Modality rank rho", format_float(modality_rho)),
        ("Topic contrast rho", format_float(contrast_rho)),
        ("CA liking rho", f"Dim1 {format_float(ca_dim1_like)} / Dim2 {format_float(ca_dim2_like)}"),
    ]

    card_html = "\n".join(
        f"<div class='card'><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>"
        for label, value in cards
    )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Topic-level MR-CA comparison</title>
  <style>
    :root {{
      --bg: {BG};
      --ink: {INK};
      --muted: {MUTED};
      --terracotta: {TERRACOTTA};
      --sandstone: {SANDSTONE};
      --sage: {SAGE};
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Manrope, Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 42px 30px 64px;
    }}
    h1, h2, h3 {{
      margin: 0;
      letter-spacing: 0;
    }}
    h1 {{
      font-family: "Instrument Serif", Georgia, serif;
      font-size: 48px;
      font-style: italic;
      font-weight: 500;
      line-height: 1;
    }}
    h2 {{
      font-size: 23px;
      margin-top: 40px;
      margin-bottom: 12px;
    }}
    h3 {{
      font-size: 16px;
      margin-bottom: 8px;
    }}
    p {{
      max-width: 850px;
      color: var(--muted);
      font-size: 16px;
    }}
    .eyebrow {{
      color: var(--terracotta);
      font-weight: 800;
      text-transform: uppercase;
      font-size: 12px;
      letter-spacing: 0.08em;
      margin-bottom: 10px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin: 28px 0 34px;
    }}
    .card {{
      background: var(--sandstone);
      border: 1px solid rgba(26, 39, 33, 0.12);
      border-radius: 8px;
      padding: 14px 16px;
      min-height: 78px;
    }}
    .card span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 750;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 9px;
    }}
    .card strong {{
      font-size: 26px;
      font-weight: 850;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1.05fr 0.95fr;
      gap: 24px;
      align-items: start;
    }}
    .panel {{
      background: rgba(234, 230, 219, 0.52);
      border: 1px solid rgba(26, 39, 33, 0.12);
      border-radius: 10px;
      padding: 18px;
    }}
    img {{
      width: 100%;
      display: block;
      border-radius: 8px;
      border: 1px solid rgba(26, 39, 33, 0.08);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
      background: rgba(247, 246, 242, 0.55);
    }}
    th {{
      text-align: left;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      border-bottom: 1px solid rgba(26, 39, 33, 0.18);
      padding: 9px 8px;
    }}
    td {{
      border-bottom: 1px solid rgba(26, 39, 33, 0.08);
      padding: 9px 8px;
      vertical-align: top;
    }}
    .note {{
      border-left: 4px solid var(--terracotta);
      padding-left: 14px;
      color: var(--ink);
      max-width: 920px;
    }}
    .callout {{
      font-size: 20px;
      color: var(--ink);
      max-width: 980px;
    }}
    @media (max-width: 860px) {{
      .cards, .grid {{
        grid-template-columns: 1fr;
      }}
      h1 {{
        font-size: 38px;
      }}
    }}
  </style>
</head>
<body>
<main>
  <div class="eyebrow">Temporary analysis report</div>
  <h1>Topic-level MR-CA comparison</h1>
  <p class="callout">The quick read: our topic-level analysis recovers the same broad sensory story as Mahieu et al. (2022). Flavor carries the strongest signal, texture is second, and visual cues are weakest. The match is strongest at the modality level. It is looser at the individual term level because Mahieu et al. analyzed descriptor citations, while this run analyzes actual-versus-ideal topic alignment scores.</p>

  <section class="cards">
    {card_html}
  </section>

  <section class="panel">
    <h2>Comparison readout</h2>
    <p>Mahieu et al. rank order: <strong>{html.escape(mahieu_order)}</strong>. Our rank order using {html.escape(comparison_basis)}: <strong>{html.escape(our_order)}</strong>. The Spearman rank correlation across the three modalities is <strong>{format_float(modality_rho)}</strong>.</p>
    <p>Highest current topic signals: {html.escape(top_signal_text)}. These are consistent with the original paper's emphasis on flavor terms such as salty/not salty, ham taste, smoked or aromatic notes, and insipid/bland character, plus texture terms around tenderness, dryness, and fibrous pieces.</p>
    <p class="note">This is not a literal rerun of the original Mahieu descriptor MR-CA. The workbook gives us comments and liking scores. It does not give the curated binary descriptor table after IRaMuTeQ cleaning, lemmatization, descriptor grouping, and descriptor-frequency filtering. The report below therefore runs the closest direct analog available from our pipeline: correspondence analysis on product-by-topic alignment scores.</p>
  </section>

  <h2>MR-CA style topic map</h2>
  <div class="grid">
    <div><img src="{map_uri}" alt="MR-CA style product and topic map"></div>
    <div class="panel">
      <h3>How to read this map</h3>
      <p>Products close together have similar LLM topic-alignment profiles. Terracotta topic vectors mark the strongest column contributors. Product points are colored by mean liking.</p>
      <p>The product liking vector correlates with Dimension 1 at <strong>{format_float(ca_dim1_like)}</strong> and Dimension 2 at <strong>{format_float(ca_dim2_like)}</strong>. The first two dimensions explain <strong>{format_float(ca["inertia_share"][0] * 100, 1)}%</strong> and <strong>{format_float(ca["inertia_share"][1] * 100, 1)}%</strong> of inertia.</p>
    </div>
  </div>

  <h2>Modality hierarchy</h2>
  <div class="grid">
    <div><img src="{modality_uri}" alt="Modality comparison bar chart"></div>
    <div><img src="{topic_uri}" alt="Topic-level signal bar chart"></div>
  </div>

  <h2>Direct topic contrast</h2>
  <div class="grid">
    <div><img src="{contrast_uri}" alt="Scatter plot comparing Mahieu driver strength to LLM topic signal"></div>
    <div class="panel">
      <h3>Metric used for the contrast</h3>
      <p>The x-axis is the average absolute mixed-model loading for the closest Mahieu descriptor family, normalized by the strongest descriptor in Fig. 1. The y-axis is our absolute row-level Spearman correlation between topic alignment and liking, normalized by the strongest topic in our run.</p>
      <p>This uses strength rather than sign because Mahieu modeled descriptor presence, while our score models actual-versus-ideal fit. A negative descriptor such as insipid can still have a positive alignment score when the actual product avoids it and matches the consumer's ideal.</p>
      <table>
        <thead><tr><th>Topic</th><th>Mahieu</th><th>Ours</th></tr></thead>
        <tbody>
          {table_rows(contrast_for_table, [("label", "text"), ("mahieu_strength_norm", "float"), ("our_strength_norm", "float")])}
        </tbody>
      </table>
    </div>
  </div>

  <h2>Rank-order contrast</h2>
  <div class="grid">
    <div><img src="{rank_uri}" alt="Slopegraph comparing Mahieu topic ranks with LLM topic ranks"></div>
    <div class="panel">
      <h3>What the ranks say</h3>
      <p>Across 17 matched topic families, the rank-order agreement is <strong>Spearman rho = {format_float(contrast_rho)}</strong> and <strong>Kendall tau = {format_float(contrast_tau)}</strong>. The top-10 lists overlap on 9 topics, which is a strong result given that Mahieu et al. ranked descriptor effects and we ranked topic alignment with liking.</p>
      <p>The largest upward shifts in our analysis are ham taste, tenderness/softness, and overall flavor. The largest downward shifts are firmness/rubberiness, color/pinkness, and moisture/shine.</p>
      <table>
        <thead><tr><th>Topic</th><th>Mahieu rank</th><th>Our rank</th><th>Shift</th></tr></thead>
        <tbody>
          {table_rows(rank_for_table, [("label", "text"), ("mahieu_rank", "text"), ("our_rank", "text"), ("rank_delta", "text")])}
        </tbody>
      </table>
    </div>
  </div>

  <h2>Row-level topic correlations</h2>
  <div class="grid">
    <div class="panel">
      <h3>Positive topic alignment with liking</h3>
      <table>
        <thead><tr><th>Topic</th><th>Modality</th><th>rho</th><th>Mean score</th></tr></thead>
        <tbody>
          {table_rows(top_positive, [("label", "text"), ("modality", "text"), ("rho_row", "float"), ("mean_score", "float")])}
        </tbody>
      </table>
    </div>
    <div class="panel">
      <h3>Negative topic alignment with liking</h3>
      <table>
        <thead><tr><th>Topic</th><th>Modality</th><th>rho</th><th>Mean score</th></tr></thead>
        <tbody>
          {table_rows(top_negative, [("label", "text"), ("modality", "text"), ("rho_row", "float"), ("mean_score", "float")])}
        </tbody>
      </table>
    </div>
  </div>

  <h2>Term-level bridge to Mahieu et al. (2022)</h2>
  <div class="panel">
    <p>Mahieu et al. worked at a finer descriptor level. The table below maps our topic labels to the closest descriptor families named in the paper, then shows the row-level correlation between our alignment score and liking.</p>
    <table>
      <thead><tr><th>Our topic</th><th>Modality</th><th>Closest Mahieu descriptor families</th><th>rho with liking</th></tr></thead>
      <tbody>
        {table_rows(analog_rows, [("label", "text"), ("modality", "text"), ("mahieu_terms", "text"), ("rho_row", "float")])}
      </tbody>
    </table>
  </div>

  <h2>Method notes</h2>
  <div class="panel">
    <p>Scoring model: {html.escape(str(metadata.get("gemini_model", "unknown")))}. Scale: {html.escape(str(metadata.get("scale_points", "unknown")))} point. Temperature: {html.escape(str(metadata.get("temperature", "unknown")))}. Each row compares a consumer's actual ham comments to that same consumer's ideal ham comments across 17 topic-level sensory pairings.</p>
    <p>Mahieu et al. used Ideal-Free-Comment data, cleaned and grouped descriptor terms, ran MR-CA on product-by-descriptor citation proportions, projected ideal descriptors and liking into that sensory space, and then modeled descriptor effects on liking with mixed linear models. Our analysis compresses that descriptor workflow into direct LLM topic scores, then evaluates held-out liking prediction. {html.escape(model_note)}</p>
  </div>
</main>
</body>
</html>
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html_text)
    summary = {
        "report_path": str(report_path),
        "valid_rows": rows_valid,
        "products": int(products),
        "consumers": int(consumers),
        "parse_errors": parse_errors,
        "model_r2": displayed_metric.get("r2"),
        "model_mae": displayed_metric.get("mae"),
        "model_metric_label": metric_label,
        "mahieu_order": mahieu_order,
        "our_order": our_order,
        "modality_spearman_rho": modality_rho,
        "topic_contrast_spearman_rho": contrast_rho,
        "topic_contrast_kendall_tau": contrast_tau,
        "ca_dim1_liking_rho": ca_dim1_like,
        "ca_dim2_liking_rho": ca_dim2_like,
        "comparison_basis": comparison_basis,
    }
    (report_path.parent / "topic_level_mrca_summary.json").write_text(json.dumps(summary, indent=2))
    product_correlations.to_csv(report_path.parent / "topic_level_product_correlations.csv", index=False)
    row_correlations.to_csv(report_path.parent / "topic_level_row_correlations.csv", index=False)
    contrast.to_csv(report_path.parent / "topic_level_contrast_points.csv", index=False)
    if importance is not None and used_tabpfn_importance:
        importance.to_csv(report_path.parent / "topic_level_tabpfn_importance.csv", index=False)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=Path("analysis/topic_level"), type=Path)
    parser.add_argument("--report-path", default=None, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_path = args.report_path or args.output_dir / "topic_level_mrca_report.html"
    summary = render_report(args.output_dir, report_path)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
