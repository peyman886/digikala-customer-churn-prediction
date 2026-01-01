"""
Visualization Module for Churn Prediction.

Provides:
- Confusion matrix plots
- ROC and PR curves
- Feature importance plots
- Metrics comparison charts
- SHAP visualization wrappers
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score

# Color palettes
SEGMENT_COLORS = {
    '1 Order': '#e74c3c',
    '2-4 Orders': '#3498db',
    '5-10 Orders': '#2ecc71',
    '11-30 Orders': '#9b59b6',
    '30+ Orders': '#f39c12',
    'Cluster_0': '#e74c3c',
    'Cluster_1': '#3498db',
    'Cluster_2': '#2ecc71',
    'Cluster_3': '#9b59b6',
    'Cluster_4': '#f39c12',
}

MODEL_COLORS = {
    'XGBoost': '#3498db',
    'LightGBM': '#2ecc71',
    'RandomForest': '#e74c3c',
    'LogisticRegression': '#9b59b6',
    'GradientBoosting': '#f39c12',
    'RuleBasedBaseline': '#95a5a6',
}


def set_plot_style():
    """Set consistent plot style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['axes.labelsize'] = 11


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          title: str = 'Confusion Matrix',
                          ax: plt.Axes = None,
                          cmap: str = 'Blues') -> plt.Axes:
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
        ax: Matplotlib axes (creates new if None)
        cmap: Color map

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                xticklabels=['Active', 'Churned'],
                yticklabels=['Active', 'Churned'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)

    return ax


def plot_confusion_matrices_grid(results: Dict[str, Tuple[np.ndarray, np.ndarray]],
                                 figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Plot confusion matrices for multiple models/segments in a grid.

    Args:
        results: Dict mapping name to (y_true, y_pred) tuples
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    n = len(results)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if n > 1 else [axes]

    for idx, (name, (y_true, y_pred)) in enumerate(results.items()):
        plot_confusion_matrix(y_true, y_pred, title=name, ax=axes[idx])

    # Hide empty subplots
    for idx in range(n, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    return fig


def plot_roc_curves(results: Dict[str, Tuple[np.ndarray, np.ndarray]],
                    title: str = 'ROC Curves',
                    ax: plt.Axes = None) -> plt.Axes:
    """
    Plot ROC curves for multiple models.

    Args:
        results: Dict mapping name to (y_true, y_prob) tuples
        title: Plot title
        ax: Matplotlib axes

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    for name, (y_true, y_prob) in results.items():
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        color = MODEL_COLORS.get(name, SEGMENT_COLORS.get(name, None))
        ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC={auc:.3f})', color=color)

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    return ax


def plot_pr_curves(results: Dict[str, Tuple[np.ndarray, np.ndarray]],
                   title: str = 'Precision-Recall Curves',
                   ax: plt.Axes = None) -> plt.Axes:
    """
    Plot Precision-Recall curves for multiple models.

    Args:
        results: Dict mapping name to (y_true, y_prob) tuples
        title: Plot title
        ax: Matplotlib axes

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    for name, (y_true, y_prob) in results.items():
        if len(np.unique(y_true)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(y_true, y_prob)
        color = MODEL_COLORS.get(name, SEGMENT_COLORS.get(name, None))
        ax.plot(recall, precision, lw=2, label=f'{name} (AP={ap:.3f})', color=color)

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    return ax


def plot_feature_importance(importance: pd.DataFrame,
                            top_n: int = 20,
                            title: str = 'Feature Importance',
                            ax: plt.Axes = None,
                            color: str = 'steelblue') -> plt.Axes:
    """
    Plot horizontal bar chart of feature importance.

    Args:
        importance: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to show
        title: Plot title
        ax: Matplotlib axes
        color: Bar color

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    top_features = importance.nlargest(top_n, 'importance')

    ax.barh(range(len(top_features)), top_features['importance'],
            color=color, alpha=0.8, edgecolor='black')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(title)

    return ax


def plot_metrics_comparison(metrics_df: pd.DataFrame,
                            metrics_to_plot: List[str] = None,
                            title: str = 'Model Comparison',
                            figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot bar chart comparing metrics across models.

    Args:
        metrics_df: DataFrame with model names as index and metrics as columns
        metrics_to_plot: List of metric columns to plot
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    if metrics_to_plot is None:
        metrics_to_plot = ['recall', 'precision', 'f1', 'roc_auc']

    # Filter to available metrics
    metrics_to_plot = [m for m in metrics_to_plot if m in metrics_df.columns]

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(metrics_to_plot))
    width = 0.8 / len(metrics_df)

    for i, (model_name, row) in enumerate(metrics_df.iterrows()):
        values = [row[m] for m in metrics_to_plot]
        color = MODEL_COLORS.get(model_name, None)
        bars = ax.bar(x + i * width, values, width, label=model_name,
                      color=color, alpha=0.8)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=45)

    ax.set_xticks(x + width * (len(metrics_df) - 1) / 2)
    ax.set_xticklabels([m.upper() for m in metrics_to_plot])
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1.15])

    plt.tight_layout()
    return fig


def plot_segment_metrics_heatmap(segment_metrics: Dict[str, Dict[str, float]],
                                 metric: str = 'recall',
                                 title: str = None,
                                 figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot heatmap of a metric across segments and models.

    Args:
        segment_metrics: Nested dict {model: {segment: {metric: value}}}
        metric: Which metric to plot
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Convert to DataFrame
    data = {}
    for model_name, segments in segment_metrics.items():
        data[model_name] = {seg: metrics.get(metric, 0)
                            for seg, metrics in segments.items()}

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax,
                vmin=0, vmax=1, cbar_kws={'label': metric.upper()})
    ax.set_title(title or f'{metric.upper()} by Segment and Model')
    ax.set_xlabel('Model')
    ax.set_ylabel('Segment')

    plt.tight_layout()
    return fig


def plot_threshold_analysis(y_true: np.ndarray, y_prob: np.ndarray,
                            thresholds: np.ndarray = None,
                            figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
    """
    Plot how metrics change with classification threshold.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        thresholds: Array of thresholds to evaluate
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import precision_score, recall_score, f1_score

    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 81)

    metrics = {'threshold': [], 'precision': [], 'recall': [], 'f1': []}

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        metrics['threshold'].append(thresh)
        metrics['precision'].append(precision_score(y_true, y_pred, zero_division=0))
        metrics['recall'].append(recall_score(y_true, y_pred, zero_division=0))
        metrics['f1'].append(f1_score(y_true, y_pred, zero_division=0))

    df = pd.DataFrame(metrics)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Metrics vs Threshold
    ax = axes[0]
    ax.plot(df['threshold'], df['precision'], 'b-', lw=2, label='Precision')
    ax.plot(df['threshold'], df['recall'], 'r-', lw=2, label='Recall')
    ax.plot(df['threshold'], df['f1'], 'g-', lw=2, label='F1')
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Default (0.5)')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Metrics vs Classification Threshold')
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Precision-Recall Tradeoff
    ax = axes[1]
    ax.plot(df['recall'], df['precision'], 'purple', lw=2)
    ax.scatter([df.loc[df['threshold'].sub(0.5).abs().idxmin(), 'recall']],
               [df.loc[df['threshold'].sub(0.5).abs().idxmin(), 'precision']],
               s=100, c='red', marker='*', label='Threshold=0.5', zorder=5)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Tradeoff')
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    return fig


def plot_segment_distribution(df: pd.DataFrame,
                              segment_col: str = 'segment',
                              churn_col: str = 'is_churned',
                              figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
    """
    Plot segment distribution and churn rates.

    Args:
        df: DataFrame with segment and churn columns
        segment_col: Name of segment column
        churn_col: Name of churn column
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Segment counts
    ax = axes[0]
    segment_counts = df[segment_col].value_counts()
    colors = [SEGMENT_COLORS.get(seg, '#95a5a6') for seg in segment_counts.index]
    bars = ax.bar(range(len(segment_counts)), segment_counts.values, color=colors, alpha=0.8)
    ax.set_xticks(range(len(segment_counts)))
    ax.set_xticklabels(segment_counts.index, rotation=15, ha='right')
    ax.set_ylabel('Count')
    ax.set_title('Users per Segment')

    # Add count labels
    for bar, count in zip(bars, segment_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                f'{count:,}', ha='center', va='bottom', fontsize=9)

    # Churn rates
    ax = axes[1]
    churn_rates = df.groupby(segment_col)[churn_col].mean().reindex(segment_counts.index)
    bars = ax.bar(range(len(churn_rates)), churn_rates.values, color=colors, alpha=0.8)
    ax.set_xticks(range(len(churn_rates)))
    ax.set_xticklabels(churn_rates.index, rotation=15, ha='right')
    ax.set_ylabel('Churn Rate')
    ax.set_title('Churn Rate per Segment')
    ax.set_ylim([0, 1])

    # Add rate labels
    for bar, rate in zip(bars, churn_rates.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{rate:.1%}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig


def create_results_summary_table(results: Dict[str, Dict[str, float]],
                                 highlight_best: bool = True) -> pd.DataFrame:
    """
    Create a formatted summary table of results.

    Args:
        results: Dict mapping model name to metrics dict
        highlight_best: Whether to highlight best values

    Returns:
        Styled DataFrame
    """
    df = pd.DataFrame(results).T
    df = df.round(4)

    if highlight_best:
        # Metrics where higher is better
        higher_better = ['recall', 'precision', 'f1', 'roc_auc', 'pr_auc', 'accuracy']

        def highlight_max(s):
            if s.name in higher_better:
                is_max = s == s.max()
                return ['background-color: lightgreen' if v else '' for v in is_max]
            return ['' for _ in s]

        return df.style.apply(highlight_max)

    return df