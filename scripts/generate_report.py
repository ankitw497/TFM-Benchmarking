#!/usr/bin/env python3
"""
generate_report.py — Auto-generate a markdown report from benchmark results.

Usage:
    python scripts/generate_report.py
    python scripts/generate_report.py --results-dir results --output results/report.md
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def generate_report(results_dir: str, output_path: str):
    results_dir = Path(results_dir)
    sections = []

    # Header
    sections.append(f"""# TFM Benchmark Report
_Auto-generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}_

---
""")

    # ---------------------------------------------------------------------------
    # Zero-shot results
    # ---------------------------------------------------------------------------
    for csv in sorted(results_dir.glob("zero_shot_*.csv")):
        dataset = csv.stem.replace("zero_shot_", "")
        df = pd.read_csv(csv)
        successful = df[df["success"] == True].sort_values("auc_roc", ascending=False)
        failed = df[df["success"] == False]

        sections.append(f"## Zero-Shot Results: {dataset}\n")

        if not successful.empty:
            sections.append("### Leaderboard\n")
            sections.append("| Rank | Model | AUC-ROC | Log-Loss | Brier | ECE | F1 (macro) | Time (s) |")
            sections.append("|------|-------|---------|----------|-------|-----|-----------|----------|")
            for rank, (_, row) in enumerate(successful.iterrows(), 1):
                sections.append(
                    f"| {rank} | **{row['model_name']}** | "
                    f"{row['auc_roc']:.4f} | {row['log_loss_val']:.4f} | "
                    f"{row['brier_score']:.4f} | {row['ece']:.4f} | "
                    f"{row['f1_macro']:.4f} | {row['total_time']:.2f} |"
                )
            sections.append("")

            # Winner analysis
            best = successful.iloc[0]
            sections.append(f"**Winner:** {best['model_name']} with AUC-ROC = {best['auc_roc']:.4f}\n")

            # TFM vs GBDT comparison
            tfm_names = ["TabPFN", "TabICL", "Mitra", "TabDPT"]
            gbdt_names = ["XGBoost", "CatBoost", "LightGBM"]

            tfm_results = successful[successful["model_name"].apply(
                lambda x: any(t in x for t in tfm_names)
            )]
            gbdt_results = successful[successful["model_name"].apply(
                lambda x: any(t in x for t in gbdt_names)
            )]

            if not tfm_results.empty and not gbdt_results.empty:
                best_tfm = tfm_results.iloc[0]
                best_gbdt = gbdt_results.iloc[0]
                delta = best_tfm["auc_roc"] - best_gbdt["auc_roc"]
                winner = "TFMs" if delta > 0 else "GBDTs"
                sections.append(
                    f"**TFM vs GBDT:** Best TFM ({best_tfm['model_name']}: {best_tfm['auc_roc']:.4f}) "
                    f"vs Best GBDT ({best_gbdt['model_name']}: {best_gbdt['auc_roc']:.4f}) → "
                    f"{winner} win by {abs(delta):.4f}\n"
                )

        if not failed.empty:
            sections.append("### Failed Models\n")
            for _, row in failed.iterrows():
                sections.append(f"- **{row['model_name']}**: {row['error_message']}")
            sections.append("")

    # ---------------------------------------------------------------------------
    # Fine-tuning results
    # ---------------------------------------------------------------------------
    ft_files = list(results_dir.glob("finetuned_*.csv"))
    if ft_files:
        sections.append("## Fine-Tuning Impact\n")
        for csv in sorted(ft_files):
            dataset = csv.stem.replace("finetuned_", "")
            ft_df = pd.read_csv(csv)
            zs_csv = results_dir / f"zero_shot_{dataset}.csv"

            if zs_csv.exists():
                zs_df = pd.read_csv(zs_csv)
                sections.append(f"### {dataset}\n")
                sections.append("| Model | Zero-Shot AUC | Fine-Tuned AUC | Δ AUC | FT Time (s) |")
                sections.append("|-------|---------------|----------------|-------|-------------|")

                for _, ft_row in ft_df[ft_df["success"] == True].iterrows():
                    base_name = ft_row["model_name"].replace("-finetuned", "")
                    zs_match = zs_df[
                        (zs_df["model_name"] == base_name) & (zs_df["success"] == True)
                    ]
                    if not zs_match.empty:
                        zs_auc = zs_match.iloc[0]["auc_roc"]
                        ft_auc = ft_row["auc_roc"]
                        delta = ft_auc - zs_auc
                        sign = "+" if delta >= 0 else ""
                        sections.append(
                            f"| {base_name} | {zs_auc:.4f} | {ft_auc:.4f} | "
                            f"{sign}{delta:.4f} | {ft_row['total_time']:.1f} |"
                        )
                sections.append("")

    # ---------------------------------------------------------------------------
    # Scaling results
    # ---------------------------------------------------------------------------
    for csv in sorted(results_dir.glob("scaling_*.csv")):
        dataset = csv.stem.replace("scaling_", "")
        df = pd.read_csv(csv)
        sections.append(f"## Scaling Results: {dataset}\n")

        # Create pivot table
        pivot = df.pivot_table(
            index="model_name", columns="n_rows_requested",
            values="auc_roc", aggfunc="first"
        )

        if not pivot.empty:
            sections.append("### AUC-ROC at Each Scale\n")
            cols = sorted(pivot.columns)
            header = "| Model | " + " | ".join(f"{c:,}" for c in cols) + " |"
            sep = "|-------|" + "|".join("------" for _ in cols) + "|"
            sections.append(header)
            sections.append(sep)

            for model in pivot.index:
                row_vals = []
                for c in cols:
                    val = pivot.loc[model, c]
                    if pd.isna(val):
                        row_vals.append("❌")
                    else:
                        row_vals.append(f"{val:.3f}")
                sections.append(f"| {model} | " + " | ".join(row_vals) + " |")
            sections.append("")

        # Find breakpoints
        sections.append("### Breakpoint Analysis\n")
        for model in df["model_name"].unique():
            model_df = df[df["model_name"] == model]
            succeeded = model_df[model_df["success"] == True]
            failed = model_df[model_df["success"] == False]

            if not succeeded.empty:
                max_ok = succeeded["n_rows_requested"].max()
                if not failed.empty:
                    min_fail = failed["n_rows_requested"].min()
                    sections.append(f"- **{model}**: Works up to {max_ok:,} rows, fails at {min_fail:,} rows")
                else:
                    sections.append(f"- **{model}**: Works at all tested sizes (up to {max_ok:,} rows)")
            else:
                sections.append(f"- **{model}**: Failed at all sizes")
        sections.append("")

    # ---------------------------------------------------------------------------
    # License summary
    # ---------------------------------------------------------------------------
    sections.append("""## License Summary for Enterprise Use

| Model | License | Commercial Use | Notes |
|-------|---------|----------------|-------|
| TabPFN v1 | Apache 2.0 | ✅ Yes | Numerical only, 1K row limit |
| TabPFN v2 | Prior Labs License | ✅ With attribution | Must include "Built with PriorLabs-TabPFN" |
| TabPFN v2.5 | TabPFN-2.5 License v1.0 | ❌ Non-commercial | Contact sales@priorlabs.ai for commercial |
| TabICLv2 | BSD-3-Clause | ✅ Yes | Best open-source option |
| Mitra | Apache 2.0 | ✅ Yes | Via AutoGluon |
| TabDPT | MIT | ✅ Yes | |
| TabNet | Apache 2.0 | ✅ Yes | |
| FT-Transformer | MIT | ✅ Yes | |
| XGBoost | Apache 2.0 | ✅ Yes | |
| CatBoost | Apache 2.0 | ✅ Yes | |
| LightGBM | MIT | ✅ Yes | |

""")

    # ---------------------------------------------------------------------------
    # Recommendations
    # ---------------------------------------------------------------------------
    sections.append("""## Recommendations

### For Enterprise Production
1. **< 10K rows:** Start with TabICLv2 (BSD-3) or Mitra (Apache 2.0) for best accuracy + open license
2. **10K-50K rows:** TabICLv2 is the clear winner (scales well, fully open)
3. **50K-1M rows:** TabICLv2 with disk offloading, or fall back to tuned CatBoost/XGBoost
4. **> 1M rows:** Use XGBoost/CatBoost/LightGBM with proper HPO

### For Maximum Accuracy (research, non-commercial)
1. Start with TabPFN v2.5 zero-shot
2. Fine-tune if improvement needed
3. Ensemble TabPFN v2.5 + XGBoost for best results

### Universal Advice
- **Always ensemble a TFM with a GBDT** — their errors are uncorrelated
- **TFMs excel at small data** — the advantage shrinks as data grows
- **Calibration matters for credit scoring** — check ECE, not just AUC
""")

    # Write report
    report_text = "\n".join(sections)
    output = Path(output_path)
    output.write_text(report_text)
    print(f"📝 Report generated: {output}")
    print(f"   {len(sections)} sections, {len(report_text):,} characters")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark report")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output", type=str, default="results/report.md")
    args = parser.parse_args()
    generate_report(args.results_dir, args.output)


if __name__ == "__main__":
    main()
