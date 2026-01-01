"""
Evaluation Module.

Provides metrics calculation, visualization, and reporting tools.
"""

from .metrics import (
    MetricsCalculator,
    SegmentWeights,
    RecallFocusedMetrics,
    get_optimal_threshold,
    print_classification_report
)

__all__ = [
    'MetricsCalculator',
    'SegmentWeights',
    'RecallFocusedMetrics',
    'get_optimal_threshold',
    'print_classification_report'
]