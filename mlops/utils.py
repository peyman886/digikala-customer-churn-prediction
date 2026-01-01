#!/usr/bin/env python3
"""
Compare ML Experiments

CLI tool for comparing experiments and generating reports.

Usage:
    python mlops/compare.py                    # Show top 10 by ROC-AUC
    python mlops/compare.py --metric f1        # Sort by F1
    python mlops/compare.py --top 20           # Show top 20
    python mlops/compare.py --promote best     # Promote best model
    python mlops/compare.py --report           # Generate markdown report
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlops.tracker import ExperimentTracker, promote_best
from mlops.config import PRIMARY_METRIC


def generate_report(output_path: str = "reports/comparison_report.md") -> str:
    """
    Generate a markdown comparison report.

    Args:
        output_path: Where to save the report

    Returns:
        Report content as string
    """
    tracker = ExperimentTracker()

    # Get all runs
    import mlflow
    experiment = mlflow.get_experiment_by_name(tracker.experiment_name)

    if not experiment:
        return "No experiments found."

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.roc_auc DESC"],
    )

    if runs.empty:
        return "No runs found."

    # Build report
    lines = []
    lines.append("# 📊 Model Comparison Report")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Summary
    lines.append("## Summary\n")
    lines.append(f"- Total experiments: {len(runs)}")

    if "metrics.roc_auc" in runs.columns:
        best_auc = runs["metrics.roc_auc"].iloc[0]
        best_name = runs["tags.mlflow.runName"].iloc[0]
        lines.append(f"- Best ROC-AUC: {best_auc:.4f} ({best_name})")

    # Top 10 table
    lines.append("\n## Top 10 Models\n")

    col_map = {
        "tags.mlflow.runName": "Name",
        "params.model_type": "Model",
        "metrics.roc_auc": "ROC-AUC",
        "metrics.f1": "F1",
        "metrics.train_time_seconds": "Train(s)",
        "metrics.inference_time_ms": "Infer(ms)",
    }

    available = [c for c in col_map.keys() if c in runs.columns]
    df = runs[available].head(10).copy()
    df.columns = [col_map[c] for c in available]

    # Format numbers
    for col in ["ROC-AUC", "F1"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "-")

    lines.append(df.to_markdown(index=False))

    # By model type
    if "params.model_type" in runs.columns:
        lines.append("\n## Performance by Model Type\n")

        for model_type in runs["params.model_type"].dropna().unique():
            subset = runs[runs["params.model_type"] == model_type]
            if len(subset) > 0 and "metrics.roc_auc" in subset.columns:
                best = subset["metrics.roc_auc"].max()
                avg = subset["metrics.roc_auc"].mean()
                lines.append(f"- **{model_type}**: Best={best:.4f}, Avg={avg:.4f}, Count={len(subset)}")

    # Save report
    report_content = "\n".join(lines)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        f.write(report_content)

    print(f"📝 Report saved: {output_file}")

    return report_content


def main():
    parser = argparse.ArgumentParser(
        description="Compare ML experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python mlops/compare.py                    # Show top 10 by ROC-AUC
    python mlops/compare.py --metric f1        # Sort by F1
    python mlops/compare.py --top 20           # Show top 20
    python mlops/compare.py --promote best     # Promote best model
    python mlops/compare.py --promote "my_run" # Promote specific run
    python mlops/compare.py --report           # Generate markdown report
        """
    )

    parser.add_argument(
        "--metric", "-m",
        type=str,
        default=PRIMARY_METRIC,
        help=f"Metric to sort by (default: {PRIMARY_METRIC})"
    )

    parser.add_argument(
        "--top", "-n",
        type=int,
        default=10,
        help="Number of top runs to show"
    )

    parser.add_argument(
        "--promote", "-p",
        type=str,
        metavar="RUN_NAME",
        help="Promote a run to production ('best' for best model)"
    )

    parser.add_argument(
        "--report", "-r",
        action="store_true",
        help="Generate markdown comparison report"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("📊 Experiment Comparison")
    print("=" * 60)

    # Generate report
    if args.report:
        generate_report()
        return

    # Promote model
    if args.promote:
        if args.promote.lower() == "best":
            promote_best(metric=args.metric)
        else:
            tracker = ExperimentTracker()
            tracker.promote(args.promote)
        return

    # Default: show comparison table
    tracker = ExperimentTracker()
    tracker.compare(metric=args.metric, top_n=args.top)


if __name__ == "__main__":
    main()