"""
Publication-ready visualizations for TFM benchmarking.
Generates bar charts, scaling curves, radar plots, calibration diagrams, and more.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Style configuration
COLORS = {
    # TFMs
    "TabPFN-v2": "#2196F3",
    "TabPFN-v2.5": "#1565C0",
    "Real-TabPFN-2.5": "#0D47A1",
    "TabPFN-v2-finetuned": "#64B5F6",
    "TabPFN-v2.5-finetuned": "#42A5F5",
    "TabICLv2": "#4CAF50",
    "TabICL-v2": "#4CAF50",   # Matches wrapper name format (TabICL-{version})
    "TabICL-v1.1": "#81C784",
    "Mitra": "#FF9800",
    "TabDPT": "#9C27B0",
    # GBDTs
    "XGBoost-Default": "#F44336",
    "XGBoost-Tuned": "#C62828",
    "CatBoost-Default": "#E91E63",
    "CatBoost-Tuned": "#AD1457",
    "LightGBM-Default": "#FF5722",
    # Others
    "RandomForest-Default": "#795548",
    "LogisticRegression": "#607D8B",
}

MODEL_CATEGORIES = {
    "TFM (ICL)": ["TabPFN-v2", "TabPFN-v2.5", "Real-TabPFN-2.5", "TabICLv2", "TabICL-v1.1", "Mitra", "TabDPT"],
    "TFM (Finetuned)": ["TabPFN-v2-finetuned", "TabPFN-v2.5-finetuned"],
    "GBDT": ["XGBoost-Default", "XGBoost-Tuned", "CatBoost-Default", "CatBoost-Tuned", "LightGBM-Default"],
    "Traditional": ["RandomForest-Default", "LogisticRegression"],
}


def set_style():
    """Apply consistent plot styling."""
    plt.rcParams.update({
        "figure.figsize": (12, 6),
        "figure.dpi": 150,
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.framealpha": 0.9,
    })


def _get_color(model_name: str) -> str:
    return COLORS.get(model_name, "#9E9E9E")


# ---------------------------------------------------------------------------
# 1. Leaderboard bar chart
# ---------------------------------------------------------------------------

def plot_leaderboard(
    df: pd.DataFrame,
    metric: str = "auc_roc",
    title: str = "Zero-Shot Performance Leaderboard",
    save_path: Optional[str] = None,
):
    """
    Horizontal bar chart showing all models ranked by a metric.

    Parameters
    ----------
    df : DataFrame with columns ['model_name', metric, 'success']
    """
    set_style()
    successful = df[df["success"] == True].sort_values(metric, ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(6, len(successful) * 0.5)))
    bars = ax.barh(
        successful["model_name"],
        successful[metric],
        color=[_get_color(n) for n in successful["model_name"]],
        edgecolor="white",
        linewidth=0.5,
        height=0.6,
    )

    # Add value labels
    for bar, val in zip(bars, successful[metric]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9, fontweight="bold")

    # Category legend
    legend_patches = []
    for cat_name, cat_models in MODEL_CATEGORIES.items():
        if any(m in successful["model_name"].values for m in cat_models):
            sample_model = next((m for m in cat_models if m in successful["model_name"].values), None)
            if sample_model:
                legend_patches.append(mpatches.Patch(color=_get_color(sample_model), label=cat_name))

    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
    ax.set_xlabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 2. Scaling curves
# ---------------------------------------------------------------------------

def plot_scaling_curves(
    df: pd.DataFrame,
    metric: str = "auc_roc",
    title: str = "Model Performance vs. Dataset Size",
    save_path: Optional[str] = None,
):
    """
    Line chart showing how each model's performance changes with dataset size.

    Parameters
    ----------
    df : DataFrame with columns ['model_name', 'n_rows_requested', metric, 'success']
    """
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Performance
    ax1 = axes[0]
    for model_name in df["model_name"].unique():
        model_df = df[(df["model_name"] == model_name) & (df["success"] == True)]
        if model_df.empty:
            continue
        model_df = model_df.sort_values("n_rows_requested")
        ax1.plot(
            model_df["n_rows_requested"], model_df[metric],
            marker="o", markersize=5, linewidth=2,
            label=model_name, color=_get_color(model_name),
        )

    ax1.set_xscale("log")
    ax1.set_xlabel("Number of Training Rows (log scale)", fontsize=11)
    ax1.set_ylabel(metric.replace("_", " ").title(), fontsize=11)
    ax1.set_title("Accuracy vs. Scale", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8, ncol=2, loc="lower right")

    # Panel 2: Inference time
    ax2 = axes[1]
    for model_name in df["model_name"].unique():
        model_df = df[(df["model_name"] == model_name) & (df["success"] == True)]
        if model_df.empty or "total_time" not in model_df.columns:
            continue
        model_df = model_df.sort_values("n_rows_requested")
        ax2.plot(
            model_df["n_rows_requested"], model_df["total_time"],
            marker="s", markersize=5, linewidth=2,
            label=model_name, color=_get_color(model_name),
        )

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Number of Training Rows (log scale)", fontsize=11)
    ax2.set_ylabel("Total Time (seconds, log scale)", fontsize=11)
    ax2.set_title("Inference Time vs. Scale", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8, ncol=2, loc="upper left")

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 3. Fine-tuning impact
# ---------------------------------------------------------------------------

def plot_finetuning_impact(
    zero_shot_df: pd.DataFrame,
    finetuned_df: pd.DataFrame,
    metric: str = "auc_roc",
    save_path: Optional[str] = None,
):
    """
    Grouped bar chart comparing zero-shot vs fine-tuned performance.
    """
    set_style()

    # Match models that have both zero-shot and fine-tuned versions
    pairs = []
    for _, ft_row in finetuned_df[finetuned_df["success"] == True].iterrows():
        base_name = ft_row["model_name"].replace("-finetuned", "")
        zs_match = zero_shot_df[
            (zero_shot_df["model_name"] == base_name) & (zero_shot_df["success"] == True)
        ]
        if not zs_match.empty:
            pairs.append({
                "model": base_name,
                "zero_shot": zs_match.iloc[0][metric],
                "finetuned": ft_row[metric],
                "delta": ft_row[metric] - zs_match.iloc[0][metric],
                "ft_time": ft_row.get("total_time", 0),
            })

    if not pairs:
        print("No matching zero-shot / fine-tuned pairs found.")
        return None

    pdf = pd.DataFrame(pairs)
    x = np.arange(len(pdf))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, pdf["zero_shot"], width, label="Zero-Shot",
                   color="#90CAF9", edgecolor="white")
    bars2 = ax.bar(x + width / 2, pdf["finetuned"], width, label="Fine-Tuned",
                   color="#1565C0", edgecolor="white")

    # Delta annotations
    for i, row in pdf.iterrows():
        delta = row["delta"]
        sign = "+" if delta >= 0 else ""
        ax.annotate(
            f"{sign}{delta:.4f}",
            xy=(i + width / 2, row["finetuned"]),
            xytext=(0, 8), textcoords="offset points",
            ha="center", fontsize=9, fontweight="bold",
            color="green" if delta > 0 else "red",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(pdf["model"], rotation=15, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title("Impact of Fine-Tuning on TFM Performance", fontsize=14, fontweight="bold")
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 4. Calibration (Reliability) diagrams
# ---------------------------------------------------------------------------

def plot_calibration_diagrams(
    predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    n_bins: int = 10,
    save_path: Optional[str] = None,
):
    """
    Reliability diagrams for multiple models.

    Parameters
    ----------
    predictions : dict of model_name -> (y_true, y_prob)
    """
    set_style()
    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(figsize=(8, 8))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")

    for model_name, (y_true, y_prob) in predictions.items():
        fraction_pos, mean_predicted = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy="uniform"
        )
        ax.plot(
            mean_predicted, fraction_pos,
            marker="o", markersize=5, linewidth=2,
            label=model_name, color=_get_color(model_name),
        )

    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title("Calibration (Reliability) Diagram", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 5. Model limitation heatmap
# ---------------------------------------------------------------------------

def plot_limitation_matrix(
    models_info: List[Dict],
    save_path: Optional[str] = None,
):
    """
    Visual matrix showing which models support which dataset sizes.
    """
    set_style()

    row_thresholds = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]
    labels = ["1K", "5K", "10K", "50K", "100K", "500K", "1M"]

    model_names = [m["name"] for m in models_info]
    matrix = np.zeros((len(models_info), len(row_thresholds)))

    for i, model in enumerate(models_info):
        max_r = model.get("max_rows", float("inf"))
        rec_r = model.get("recommended_max_rows", max_r)
        for j, threshold in enumerate(row_thresholds):
            if max_r and threshold > max_r:
                matrix[i, j] = 0  # Cannot handle
            elif rec_r and threshold > rec_r:
                matrix[i, j] = 0.5  # Degraded
            else:
                matrix[i, j] = 1  # Good

    fig, ax = plt.subplots(figsize=(10, max(5, len(models_info) * 0.5)))

    cmap = plt.cm.RdYlGn
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=10)
    ax.set_xlabel("Dataset Size (rows)", fontsize=12)
    ax.set_title("Model Scalability Matrix", fontsize=14, fontweight="bold")

    # Cell annotations
    for i in range(len(model_names)):
        for j in range(len(labels)):
            val = matrix[i, j]
            text = "✅" if val == 1 else ("⚠️" if val == 0.5 else "❌")
            ax.text(j, i, text, ha="center", va="center", fontsize=12)

    legend_elements = [
        mpatches.Patch(facecolor=cmap(1.0), label="Fully supported"),
        mpatches.Patch(facecolor=cmap(0.5), label="Degraded performance"),
        mpatches.Patch(facecolor=cmap(0.0), label="Exceeds limits"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9, bbox_to_anchor=(1.25, 1))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 6. Time vs accuracy scatter
# ---------------------------------------------------------------------------

def plot_time_vs_accuracy(
    df: pd.DataFrame,
    metric: str = "auc_roc",
    save_path: Optional[str] = None,
):
    """Scatter plot of accuracy vs. inference time (Pareto front analysis)."""
    set_style()
    successful = df[df["success"] == True].copy()

    fig, ax = plt.subplots(figsize=(10, 7))

    for _, row in successful.iterrows():
        ax.scatter(
            row["total_time"], row[metric],
            s=120, c=_get_color(row["model_name"]),
            edgecolors="white", linewidth=1.5, zorder=5,
        )
        ax.annotate(
            row["model_name"],
            (row["total_time"], row[metric]),
            xytext=(8, 4), textcoords="offset points",
            fontsize=8, alpha=0.8,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Total Time (seconds, log scale)", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title("Accuracy vs. Speed Trade-off", fontsize=14, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 7. Generate all plots
# ---------------------------------------------------------------------------

def generate_all_plots(results_dir: str, figures_dir: str):
    """Generate all standard plots from benchmark results."""
    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Zero-shot leaderboard
    for csv in results_dir.glob("zero_shot_*.csv"):
        df = pd.read_csv(csv)
        dataset_name = csv.stem.replace("zero_shot_", "")
        plot_leaderboard(
            df, metric="auc_roc",
            title=f"Zero-Shot AUC-ROC: {dataset_name}",
            save_path=str(figures_dir / f"leaderboard_{dataset_name}.png"),
        )
        plot_time_vs_accuracy(
            df, metric="auc_roc",
            save_path=str(figures_dir / f"time_vs_accuracy_{dataset_name}.png"),
        )
        plt.close("all")

    # Scaling curves
    for csv in results_dir.glob("scaling_*.csv"):
        df = pd.read_csv(csv)
        dataset_name = csv.stem.replace("scaling_", "")
        plot_scaling_curves(
            df, metric="auc_roc",
            title=f"Scaling Behavior: {dataset_name}",
            save_path=str(figures_dir / f"scaling_{dataset_name}.png"),
        )
        plt.close("all")

    # Fine-tuning impact
    for zs_csv in results_dir.glob("zero_shot_*.csv"):
        dataset_name = zs_csv.stem.replace("zero_shot_", "")
        ft_csv = results_dir / f"finetuned_{dataset_name}.csv"
        if ft_csv.exists():
            zs_df = pd.read_csv(zs_csv)
            ft_df = pd.read_csv(ft_csv)
            plot_finetuning_impact(
                zs_df, ft_df, metric="auc_roc",
                save_path=str(figures_dir / f"finetuning_{dataset_name}.png"),
            )
            plt.close("all")

    print(f"📊 All plots saved to {figures_dir}/")
