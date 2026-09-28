"""Step 7 — Generate final publication-quality figures and LaTeX tables.

Usage:
    python -m gtzan_probe.generate_figures
"""

import pickle

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

from .config import (
    DATA_DIR, FIGURES_DIR, GENRES, GENRE_COLORS,
    setup_plotting, savefig,
)

GENRE_DISPLAY = {
    "blues": "Blues", "classical": "Classical", "country": "Country",
    "disco": "Disco", "hiphop": "Hip-hop", "jazz": "Jazz",
    "metal": "Metal", "pop": "Pop", "reggae": "Reggae", "rock": "Rock",
}

# IEEE / ISMIR layout widths (inches)
IEEE_SINGLE = 7.25
IEEE_DOUBLE = 3.5
ISMIR_FULL = 16.0


def _robust_signed_overlay(sv: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return clipped SHAP values + alpha mask using robust percentile scaling.

    This prevents a few extreme pixels from washing out the colormap and keeps
    positive (red) / negative (blue) regions clearly visible.
    """
    abs_sv = np.abs(sv)
    vmax = float(np.percentile(abs_sv, 99))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(abs_sv.max() if abs_sv.size else 1.0)
    vmax = max(vmax, 1e-8)
    sv_clip = np.clip(sv, -vmax, vmax)
    norm = np.abs(sv_clip) / vmax
    # Hide near-zero values and emphasize salient regions
    alpha = np.clip((norm - 0.15) / 0.85, 0.0, 1.0) ** 0.7
    alpha *= 0.98
    return sv_clip, alpha, vmax


def main():
    print("LEGACY WORKFLOW: read docs/REPRODUCIBILITY.md; this is not the revised independent-test protocol.")
    setup_plotting()

    # ── Load all result files ─────────────────────────────────────────────────
    with open(DATA_DIR / "shap_values.pkl", "rb") as f:
        shap_results = pickle.load(f)

    with open(DATA_DIR / "lime_results.pkl", "rb") as f:
        lime_results = pickle.load(f)

    alignment_df = pd.read_csv(DATA_DIR / "musicological_alignment.csv")
    faith_df = pd.read_csv(DATA_DIR / "faithfulness_results.csv")
    agree_df = pd.read_csv(DATA_DIR / "method_agreement.csv")
    spurious_df = pd.read_csv(DATA_DIR / "spurious_correlation.csv")
    summary_df = pd.read_csv(DATA_DIR / "xai_summary_table.csv")

    print("All result CSVs loaded.")

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL FIGURE 1 — Main SHAP result (hero figure)
    # ══════════════════════════════════════════════════════════════════════════
    shap_cmap = LinearSegmentedColormap.from_list("shap_div", ["#2196F3", "#FFFFFF", "#F44336"], N=256)
    available = [g for g in GENRES if g in shap_results]

    fig, axes = plt.subplots(2, 5, figsize=(ISMIR_FULL, 6))
    axes_flat = axes.flatten()

    for i, genre in enumerate(available):
        ax = axes_flat[i]
        res = shap_results[genre]
        sv = res["shap_values"]
        sv_clip, alpha_mask, vmax = _robust_signed_overlay(sv)

        # Slightly brighter base to improve red/blue contrast.
        ax.imshow(res["spectrogram"], aspect="auto", origin="lower", cmap="gray", alpha=0.52)
        im = ax.imshow(
            sv_clip,
            aspect="auto",
            origin="lower",
            cmap=shap_cmap,
            alpha=alpha_mask,
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_title(genre.capitalize(), fontsize=10, fontweight="bold", color=GENRE_COLORS[genre])
        if i >= 5:
            ax.set_xlabel("Time", fontsize=8)
        if i % 5 == 0:
            ax.set_ylabel("Mel bin", fontsize=8)

    for j in range(len(available), len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Reserve space on the far-right and place a dedicated colorbar axis there.
    fig.subplots_adjust(right=0.90, wspace=0.20, hspace=0.25)
    cax = fig.add_axes([0.92, 0.14, 0.012, 0.72])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("SHAP value\n(blue = negative, red = positive)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.suptitle("SHAP Attribution Overlays Across All GTZAN Genres", fontweight="bold")
    savefig("FINAL_01_shap_hero", fig, tight=False)

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL FIGURE 2 — Quantitative results panel
    # ══════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(ISMIR_FULL, 9))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # A: Musicological alignment
    ax_a = fig.add_subplot(gs[0, 0])
    x = np.arange(len(alignment_df))
    colors_a = [GENRE_COLORS.get(g, "#999") for g in alignment_df["genre"]]
    ax_a.bar(x, alignment_df["spearman_r"], color=colors_a, edgecolor="white")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([GENRE_DISPLAY.get(g, g.capitalize()) for g in alignment_df["genre"]], rotation=45, ha="right", fontsize=8)
    ax_a.set_ylabel("Spearman ρ")
    ax_a.set_title("A  Musicological Alignment", fontweight="bold")
    ax_a.axhline(0, color="black", linewidth=0.5)

    # B: Faithfulness
    ax_b = fig.add_subplot(gs[0, 1])
    for genre in [g for g in GENRES if g in faith_df["genre"].values]:
        gdf = faith_df[faith_df.genre == genre]
        ax_b.plot(gdf["k_fraction"] * 100, gdf["conf_drop"], "-o",
                  color=GENRE_COLORS[genre], label=genre, markersize=3, linewidth=1.5)
    ax_b.set_xlabel("% Pixels Removed")
    ax_b.set_ylabel("Conf. Drop")
    ax_b.set_title("B  Faithfulness (SHAP removal)", fontweight="bold")
    ax_b.legend(fontsize=6, ncol=2)

    # C: Inter-method agreement
    ax_c = fig.add_subplot(gs[1, 0])
    w = 0.35
    x_c = np.arange(len(agree_df))
    ax_c.bar(x_c - w / 2, agree_df["pearson_r"], w, label="Pearson r", color="#2196F3")
    ax_c.bar(x_c + w / 2, agree_df["top10_iou"], w, label="Top-10% IoU", color="#F44336")
    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels([GENRE_DISPLAY.get(g, g.capitalize()) for g in agree_df["genre"]], rotation=45, ha="right", fontsize=8)
    ax_c.set_ylabel("Score")
    ax_c.set_title("C  SHAP vs LIME Agreement", fontweight="bold")
    ax_c.legend(fontsize=8)

    # D: Spurious correlation
    ax_d = fig.add_subplot(gs[1, 1])
    x_d = np.arange(len(spurious_df))
    ax_d.bar(x_d, spurious_df["silence_ratio"], color="#F44336", alpha=0.8, label="Silence/Signal ratio")
    ax_d.set_xticks(x_d)
    ax_d.set_xticklabels([GENRE_DISPLAY.get(g, g.capitalize()) for g in spurious_df["genre"]], rotation=45, ha="right", fontsize=8)
    ax_d.set_ylabel("Ratio")
    ax_d.set_title("D  Spurious: Silence Attribution Ratio", fontweight="bold")

    fig.suptitle("Quantitative XAI Evaluation Results", fontweight="bold", fontsize=14)
    savefig("FINAL_02_quantitative_panel", fig)

    # ══════════════════════════════════════════════════════════════════════════
    # LaTeX table export
    # ══════════════════════════════════════════════════════════════════════════
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\caption{XAI Metrics Summary per Genre}",
        r"\label{tab:xai_summary}",
        r"\centering",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Genre & Alignment $\rho$ & Faith. (50\%) & SHAP-LIME $r$ & Spurious Ratio \\",
        r"\midrule",
    ]
    for _, row in summary_df.iterrows():
        latex_lines.append(
            f"{row['genre'].capitalize()} & "
            f"{row.get('alignment_r', 0):.3f} & "
            f"{row.get('faithfulness_50', 0):.3f} & "
            f"{row.get('shap_lime_pearson', 0):.3f} & "
            f"{row.get('spurious_ratio', 0):.3f} \\\\"
        )
    latex_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    latex_path = DATA_DIR / "paper_results.tex"
    latex_path.write_text("\n".join(latex_lines))
    print(f"LaTeX table saved to {latex_path}")

    # ── Results summary markdown ──────────────────────────────────────────────
    mean_align = alignment_df["spearman_r"].mean()
    mean_faith = faith_df[faith_df.k_fraction == 0.5]["conf_drop"].mean()
    mean_agree = agree_df["pearson_r"].mean()
    mean_spurious = spurious_df["silence_ratio"].mean()

    md_lines = [
        "# XAI Music Genre Classification — Results Summary\n",
        "## Key Findings\n",
        f"### 1. Musicological Alignment",
        f"- Mean Spearman ρ = {mean_align:.3f}",
        f"- Best: {alignment_df.loc[alignment_df['spearman_r'].idxmax(), 'genre']}",
        f"- Worst: {alignment_df.loc[alignment_df['spearman_r'].idxmin(), 'genre']}\n",
        f"### 2. Faithfulness",
        f"- Mean confidence drop at 50% removal: {mean_faith:.3f}\n",
        f"### 3. Inter-method Agreement (SHAP vs LIME)",
        f"- Mean Pearson r: {mean_agree:.3f}\n",
        f"### 4. Spurious Correlation",
        f"- Mean silence/signal attribution ratio: {mean_spurious:.3f}\n",
        "## Conclusion",
        "The model partially learns musicologically meaningful features but also",
        "attends to recording artifacts, particularly in genres with older recordings.",
    ]

    md_path = DATA_DIR / "paper_results_summary.md"
    md_path.write_text("\n".join(md_lines))
    print(f"Markdown summary saved to {md_path}")

    # ── Figure inventory ──────────────────────────────────────────────────────
    all_figs = sorted(FIGURES_DIR.glob("*.pdf"))
    print(f"\nTotal PDF figures generated: {len(all_figs)}")
    for fig_path in all_figs:
        size_kb = fig_path.stat().st_size // 1024
        print(f"  {fig_path.name:<50} {size_kb:>5} KB")

    print("\n✓ Figure generation complete.")
    print("  Use \\includegraphics{figures/<name>.pdf} in your LaTeX document.")


if __name__ == "__main__":
    main()
