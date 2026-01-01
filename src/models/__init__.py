"""
Models for Churn Prediction.

Neural Network Models:
- MLP: Simple Multi-Layer Perceptron
- TabNet: Attentive Interpretable Tabular Learning
- FTTransformer: Feature Tokenizer + Transformer

Traditional ML Models:
- XGBoost, LightGBM, RandomForest, LogisticRegression, GradientBoosting
- RuleBasedBaseline: Simple recency-based rules
"""

from .mlp import MLP
from .tabnet import TabNet
from .ft_transformer import FTTransformer
from .ml_models import (
    MLModelFactory,
    RuleBasedBaseline,
    Segmenter,
    ModelConfig,
    train_model_with_early_stopping,
    calculate_class_weight,
    # Per-segment training utilities
    train_per_segment_models,
    calculate_weighted_recall,
    combine_segment_predictions,
    print_segment_results,
    find_optimal_threshold_per_segment
)

__all__ = [
    # Neural Networks
    'MLP', 'TabNet', 'FTTransformer',
    # ML Models
    'MLModelFactory', 'RuleBasedBaseline', 'Segmenter', 'ModelConfig',
    'train_model_with_early_stopping', 'calculate_class_weight',
    # Per-segment utilities
    'train_per_segment_models', 'calculate_weighted_recall',
    'combine_segment_predictions', 'print_segment_results',
    'find_optimal_threshold_per_segment'
]


def get_model(model_name, input_dim=None, **kwargs):
    """
    Factory function to create models by name.

    Args:
        model_name: Model name. Options:
            - Neural Networks: 'MLP', 'TabNet', 'FTTransformer'
            - ML Models: 'XGBoost', 'LightGBM', 'RandomForest',
                        'LogisticRegression', 'GradientBoosting'
        input_dim: Number of input features (required for NN models)
        **kwargs: Model-specific parameters

    Returns:
        Instantiated model
    """
    # Neural Network models
    nn_models = {
        'MLP': MLP,
        'TabNet': TabNet,
        'FTTransformer': FTTransformer
    }

    # ML models
    ml_models = ['XGBoost', 'LightGBM', 'RandomForest',
                 'LogisticRegression', 'GradientBoosting']

    if model_name in nn_models:
        # Default configurations for NN models
        default_configs = {
            'MLP': {'hidden_dims': [256, 128, 64], 'dropout': 0.3},
            'TabNet': {'n_steps': 3, 'n_d': 64, 'n_a': 64, 'gamma': 1.3},
            'FTTransformer': {'d_token': 64, 'n_blocks': 3, 'n_heads': 4,
                             'd_ff_multiplier': 2, 'dropout': 0.2}
        }

        config = {**default_configs.get(model_name, {}), **kwargs}

        if model_name == 'FTTransformer':
            return nn_models[model_name](num_features=input_dim, **config)
        else:
            return nn_models[model_name](input_dim=input_dim, **config)

    elif model_name in ml_models:
        # Use MLModelFactory for ML models
        is_imbalanced = kwargs.pop('is_imbalanced', False)
        scale_pos_weight = kwargs.pop('scale_pos_weight', 1.0)
        config = ModelConfig(**kwargs) if kwargs else None
        return MLModelFactory.create_model(
            model_name, config, is_imbalanced, scale_pos_weight
        )

    else:
        all_models = list(nn_models.keys()) + ml_models
        raise ValueError(f"Unknown model: {model_name}. Available: {all_models}")