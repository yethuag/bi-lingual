from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


RESULTS_CSV = Path("llm_persona_results.csv")
RADAR_PNG = Path("radar_persona_profiles.png")
HEATMAP_PNG = Path("persona_drift_heatmap.png")
FAILURE_PNG = Path("parse_failure_rates.png")

TRAITS = ["Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness"]
LANGUAGES = ["english", "burmese"]
MODEL_ORDER = [
    "qwen2.5:3b",
    "llama3.2:3b",
    "aisingapore/Llama-SEA-LION-v3.5-8B-R",
]
MODEL_LABELS = {
    "qwen2.5:3b": "Qwen 2.5 3B",
    "llama3.2:3b": "Llama 3.2 3B",
    "aisingapore/Llama-SEA-LION-v3.5-8B-R": "SEA-LION v3.5 8B-R",
}
LANGUAGE_COLORS = {
    "english": "#E6862A",
    "burmese": "#2F80C1",
}


def set_plot_style() -> None:
    """Apply a consistent academic plotting style."""
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.15,
        rc={
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        },
    )


def load_results(path: Path) -> pd.DataFrame:
    """Load raw item-level model responses and normalize score types."""
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    df = pd.read_csv(path)
    required = {
        "session_id",
        "model_name",
        "language",
        "item_id",
        "trait",
        "keying",
        "raw_output",
        "parsed_score",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {sorted(missing)}")

    df["parsed_score"] = pd.to_numeric(df["parsed_score"], errors="coerce")
    return df


def add_reverse_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Create IPIP reverse-scored values while preserving parse failures as NaN."""
    scored = df.copy()
    scored["adjusted_score"] = np.where(
        scored["keying"].eq("-"),
        6 - scored["parsed_score"],
        scored["parsed_score"],
    )
    return scored


def compute_trait_summary(scored: pd.DataFrame) -> pd.DataFrame:
    """Aggregate item responses into session trait means, then model-language summaries."""
    session_traits = (
        scored.groupby(["model_name", "language", "session_id", "trait"], observed=True)
        .agg(
            trait_mean=("adjusted_score", "mean"),
            valid_items=("adjusted_score", "count"),
            total_items=("adjusted_score", "size"),
        )
        .reset_index()
    )

    # Entirely unparseable trait blocks should not contribute to means or SDs.
    session_traits = session_traits[session_traits["valid_items"] > 0]

    summary = (
        session_traits.groupby(["model_name", "language", "trait"], observed=True)
        .agg(
            mean_score=("trait_mean", "mean"),
            sd_score=("trait_mean", "std"),
            n_sessions=("trait_mean", "count"),
        )
        .reset_index()
    )
    return summary


def compute_parse_failures(scored: pd.DataFrame) -> pd.DataFrame:
    """Calculate parse-failure percentage for each model-language pair."""
    failures = (
        scored.assign(parse_failed=scored["parsed_score"].isna())
        .groupby(["model_name", "language"], observed=True)
        .agg(
            total_items=("parsed_score", "size"),
            failed_items=("parse_failed", "sum"),
        )
        .reset_index()
    )
    failures["failure_rate_pct"] = 100 * failures["failed_items"] / failures["total_items"]
    return failures


def plot_radar_profiles(summary: pd.DataFrame, output_path: Path) -> None:
    """Create one radar chart per model with English and Burmese profiles overlaid."""
    angles = np.linspace(0, 2 * np.pi, len(TRAITS), endpoint=False).tolist()
    closed_angles = angles + angles[:1]

    fig, axes = plt.subplots(
        1,
        len(MODEL_ORDER),
        figsize=(16, 5.8),
        subplot_kw={"projection": "polar"},
    )

    for ax, model in zip(axes, MODEL_ORDER):
        model_summary = summary[summary["model_name"].eq(model)]

        for language in LANGUAGES:
            profile = (
                model_summary[model_summary["language"].eq(language)]
                .set_index("trait")
                .reindex(TRAITS)["mean_score"]
            )

            if profile.notna().sum() == 0:
                continue

            values = profile.to_numpy(dtype=float).tolist()
            closed_values = values + values[:1]

            ax.plot(
                closed_angles,
                closed_values,
                color=LANGUAGE_COLORS[language],
                linewidth=2.2,
                label=language.title(),
            )
            ax.fill(
                closed_angles,
                closed_values,
                color=LANGUAGE_COLORS[language],
                alpha=0.18,
            )

        ax.set_title(MODEL_LABELS.get(model, model), pad=18, fontsize=12, fontweight="bold")
        ax.set_xticks(angles)
        ax.set_xticklabels([trait[:4] for trait in TRAITS], fontsize=10)
        ax.set_ylim(1, 5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=8)
        ax.grid(alpha=0.35)

        if model == "llama3.2:3b":
            missing_burmese = model_summary[
                model_summary["language"].eq("burmese")
            ].empty
            if missing_burmese:
                ax.text(
                    0.5,
                    -0.15,
                    "Burmese: no parseable scores",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#555555",
                )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Cross-Lingual Big Five Persona Profiles", y=0.99, fontsize=16, fontweight="bold")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=2,
        frameon=False,
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88], pad=2.5)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_drift_heatmap(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot Burmese-minus-English trait deltas for each model."""
    wide = summary.pivot_table(
        index=["model_name", "trait"],
        columns="language",
        values="mean_score",
        aggfunc="mean",
    )
    wide["delta"] = wide.get("burmese") - wide.get("english")

    delta = (
        wide["delta"]
        .unstack("trait")
        .reindex(index=MODEL_ORDER, columns=TRAITS)
        .rename(index=MODEL_LABELS)
    )

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    mask = delta.isna()
    cmap = sns.color_palette("coolwarm", as_cmap=True)

    sns.heatmap(
        delta,
        mask=mask,
        cmap=cmap,
        center=0,
        vmin=-2.5,
        vmax=2.5,
        linewidths=0.8,
        linecolor="white",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Delta score: Burmese - English"},
        ax=ax,
    )

    for row_idx, model in enumerate(delta.index):
        for col_idx, trait in enumerate(delta.columns):
            if pd.isna(delta.loc[model, trait]):
                ax.text(
                    col_idx + 0.5,
                    row_idx + 0.5,
                    "NA",
                    ha="center",
                    va="center",
                    color="#555555",
                    fontsize=10,
                    fontweight="bold",
                )

    ax.set_title("Cross-Lingual Persona Drift by Trait", fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("Big Five Trait", fontsize=11)
    ax.set_ylabel("Model", fontsize=11)
    ax.tick_params(axis="x", rotation=25)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_parse_failures(failures: pd.DataFrame, output_path: Path) -> None:
    """Plot parse-failure rates by model and language."""
    plot_df = failures.copy()
    plot_df["model_label"] = plot_df["model_name"].map(MODEL_LABELS).fillna(plot_df["model_name"])
    plot_df["language"] = pd.Categorical(plot_df["language"], categories=LANGUAGES, ordered=True)
    plot_df["model_label"] = pd.Categorical(
        plot_df["model_label"],
        categories=[MODEL_LABELS[m] for m in MODEL_ORDER],
        ordered=True,
    )

    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.barplot(
        data=plot_df,
        x="model_label",
        y="failure_rate_pct",
        hue="language",
        palette=LANGUAGE_COLORS,
        edgecolor="#333333",
        linewidth=0.7,
        ax=ax,
    )

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=3, fontsize=9)

    ax.set_title("Parse-Failure Rate by Model and Language", fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Parse Failure Rate (%)", fontsize=11)
    ax.set_ylim(0, max(105, plot_df["failure_rate_pct"].max() * 1.12))
    ax.legend(title="Language", frameon=False, loc="upper left")
    ax.tick_params(axis="x", rotation=12)
    sns.despine(ax=ax, left=False, bottom=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_plot_style()

    raw = load_results(RESULTS_CSV)
    scored = add_reverse_scores(raw)
    summary = compute_trait_summary(scored)
    failures = compute_parse_failures(scored)

    plot_radar_profiles(summary, RADAR_PNG)
    plot_drift_heatmap(summary, HEATMAP_PNG)
    plot_parse_failures(failures, FAILURE_PNG)

    print(f"Wrote {RADAR_PNG}")
    print(f"Wrote {HEATMAP_PNG}")
    print(f"Wrote {FAILURE_PNG}")


if __name__ == "__main__":
    main()
