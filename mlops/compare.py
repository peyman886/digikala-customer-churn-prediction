"""
Model Comparison Script

مقایسه مدل‌ها و تحلیل نتایج

Usage:
    # Compare all models
    python mlops/compare.py

    # Compare specific runs
    python mlops/compare.py --runs run1_id run2_id

    # Generate comparison report
    python mlops/compare.py --report

Author: Peyman
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from mlops.experiment import ExperimentTracker


# =============================================================================
# Comparison Functions
# =============================================================================

def compare_all(
        metric: str = "roc_auc",
        top_n: int = 20,
        show_timing: bool = True
) -> pd.DataFrame:
    """
    Compare all experiment runs.

    Args:
        metric: Primary metric to sort by
        top_n: Number of top runs to show
        show_timing: Whether to show timing columns

    Returns:
        DataFrame with comparison
    """
    tracker = ExperimentTracker("churn-prediction")
    df = tracker.get_all_runs()

    if df.empty:
        print("❌ No experiments found.")
        return df

    # Select and rename columns
    col_mapping = {
        "run_id": "Run ID",
        "tags.mlflow.runName": "Name",
        "params.model_type": "Model",
        "params.feature_count": "Features",
        "metrics.roc_auc": "ROC-AUC",
        "metrics.f1": "F1",
        "metrics.precision": "Precision",
        "metrics.recall": "Recall",
        "metrics.train_time_seconds": "Train (s)",
        "metrics.inference_time_ms": "Infer (ms)",
        "start_time": "Date"
    }

    # Filter available columns
    available = [c for c in col_mapping.keys() if c in df.columns]
    df = df[available].copy()
    df.columns = [col_mapping[c] for c in available]

    # Sort by metric
    metric_col = {
        "roc_auc": "ROC-AUC",
        "f1": "F1",
        "precision": "Precision",
        "recall": "Recall"
    }.get(metric, "ROC-AUC")

    if metric_col in df.columns:
        df = df.sort_values(metric_col, ascending=False)

    # Top N
    df = df.head(top_n)

    # Format
    for col in ["ROC-AUC", "F1", "Precision", "Recall"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "-")

    if "Train (s)" in df.columns:
        df["Train (s)"] = df["Train (s)"].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "-")

    if "Infer (ms)" in df.columns:
        df["Infer (ms)"] = df["Infer (ms)"].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "-")

    return df


def compare_baseline(baseline_run: str, new_run: str) -> dict:
    """
    Compare a new run against baseline.

    Args:
        baseline_run: Run ID or name of baseline
        new_run: Run ID or name of new run

    Returns:
        dict with comparison results
    """
    tracker = ExperimentTracker("churn-prediction")
    df = tracker.get_all_runs()

    if df.empty:
        return {"error": "No runs found"}

    # Find runs (by ID or name)
    def find_run(identifier):
        if identifier in df["run_id"].values:
            return df[df["run_id"] == identifier].iloc[0]

        name_col = "tags.mlflow.runName"
        if name_col in df.columns and identifier in df[name_col].values:
            return df[df[name_col] == identifier].iloc[0]

        return None

    base = find_run(baseline_run)
    new = find_run(new_run)

    if base is None:
        return {"error": f"Baseline run not found: {baseline_run}"}
    if new is None:
        return {"error": f"New run not found: {new_run}"}

    # Compare metrics
    metrics = ["roc_auc", "f1", "precision", "recall"]
    comparison = {
        "baseline": baseline_run,
        "new": new_run,
        "improvements": {},
        "is_better": False
    }

    better_count = 0
    for m in metrics:
        col = f"metrics.{m}"
        if col in df.columns:
            base_val = base[col]
            new_val = new[col]

            if pd.notnull(base_val) and pd.notnull(new_val):
                diff = new_val - base_val
                pct = (diff / base_val) * 100 if base_val != 0 else 0

                comparison["improvements"][m] = {
                    "baseline": round(base_val, 4),
                    "new": round(new_val, 4),
                    "diff": round(diff, 4),
                    "pct_change": round(pct, 2),
                    "is_improvement": diff > 0
                }

                if diff > 0:
                    better_count += 1

    # Overall verdict
    comparison["is_better"] = better_count > len(metrics) / 2
    comparison["better_in"] = better_count
    comparison["total_metrics"] = len(metrics)

    return comparison


def generate_report(output_path: str = "reports/comparison_report.md") -> str:
    """
    Generate markdown comparison report.
    """
    df = compare_all(top_n=50)

    if df.empty:
        return "No experiments to report."

    # Build report
    report = []
    report.append("# 📊 Model Comparison Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Summary
    report.append("## 📈 Summary\n")
    report.append(f"- Total experiments: {len(df)}")

    if "ROC-AUC" in df.columns:
        best_auc = df["ROC-AUC"].iloc[0]
        best_name = df["Name"].iloc[0] if "Name" in df.columns else df.index[0]
        report.append(f"- Best ROC-AUC: {best_auc} ({best_name})")

    # Top 10 table
    report.append("\n## 🏆 Top 10 Models\n")
    report.append(df.head(10).to_markdown(index=False))

    # Model type comparison
    if "Model" in df.columns:
        report.append("\n## 🔧 By Model Type\n")
        for model_type in df["Model"].unique():
            if pd.notnull(model_type):
                subset = df[df["Model"] == model_type]
                report.append(f"\n### {model_type}\n")
                report.append(f"- Count: {len(subset)}")
                if "ROC-AUC" in subset.columns:
                    best = subset["ROC-AUC"].iloc[0]
                    report.append(f"- Best ROC-AUC: {best}")

    # Timing analysis
    report.append("\n## ⏱️ Performance (Timing)\n")
    if "Train (s)" in df.columns:
        report.append("\n| Model | Avg Train Time | Avg Inference Time |")
        report.append("|-------|----------------|-------------------|")

        for model_type in df["Model"].unique() if "Model" in df.columns else [None]:
            if model_type is None:
                subset = df
                model_name = "All"
            else:
                if pd.isnull(model_type):
                    continue
                subset = df[df["Model"] == model_type]
                model_name = model_type

            try:
                avg_train = subset["Train (s)"].apply(lambda x: float(x) if x != "-" else np.nan).mean()
                avg_infer = subset["Infer (ms)"].apply(lambda x: float(x) if x != "-" else np.nan).mean()
                report.append(f"| {model_name} | {avg_train:.2f}s | {avg_infer:.4f}ms |")
            except:
                pass

    # Save report
    report_content = "\n".join(report)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        f.write(report_content)

    print(f"📝 Report saved: {output_file}")

    return report_content


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compare ML experiments")

    parser.add_argument("--metric", type=str, default="roc_auc",
                        help="Metric to sort by")
    parser.add_argument("--top", type=int, default=20,
                        help="Top N runs to show")
    parser.add_argument("--report", action="store_true",
                        help="Generate markdown report")
    parser.add_argument("--baseline", type=str,
                        help="Baseline run to compare against")
    parser.add_argument("--new", type=str,
                        help="New run to compare")

    args = parser.parse_args()

    print("=" * 70)
    print("📊 Model Comparison")
    print("=" * 70)

    # Generate report
    if args.report:
        generate_report()
        return

    # Compare specific runs
    if args.baseline and args.new:
        result = compare_baseline(args.baseline, args.new)

        if "error" in result:
            print(f"❌ {result['error']}")
            return

        print(f"\n🔍 Comparing: {result['baseline']} → {result['new']}\n")
        print("-" * 50)

        for metric, data in result["improvements"].items():
            symbol = "✅" if data["is_improvement"] else "❌"
            sign = "+" if data["diff"] > 0 else ""
            print(
                f"{symbol} {metric.upper()}: {data['baseline']:.4f} → {data['new']:.4f} ({sign}{data['pct_change']:.1f}%)")

        print("-" * 50)
        verdict = "✅ NEW MODEL IS BETTER" if result["is_better"] else "❌ BASELINE IS BETTER"
        print(f"\n{verdict} ({result['better_in']}/{result['total_metrics']} metrics improved)")
        return

    # Default: show comparison table
    df = compare_all(metric=args.metric, top_n=args.top)

    if not df.empty:
        print(df.to_string(index=False))
        print(f"\n📊 Showing top {len(df)} runs sorted by {args.metric}")


if __name__ == "__main__":
    main()