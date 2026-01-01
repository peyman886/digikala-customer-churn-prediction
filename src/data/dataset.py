"""
Data Module.

Provides dataset and dataloader utilities with:
- Segment-aware data splitting
- Weighted sampling for imbalanced data
- Efficient data loading
"""

import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DataConfig:
    """Data configuration."""
    batch_size: int = 256
    val_size: float = 0.15
    use_weighted_sampler: bool = True
    random_state: int = 42


class ChurnDataset:
    """
    Dataset handler for churn prediction.

    Handles:
    - Loading and preprocessing
    - Splitting 1-order vs multi-order users
    - Creating train/val/test DataLoaders
    - Segment tracking
    """

    def __init__(self, config: DataConfig = None):
        self.config = config or DataConfig()
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.segment_encoder = None

        # Data containers
        self.train_df = None
        self.test_df = None
        self.feature_cols = None

        # Split data
        self.train_1order = None
        self.test_1order = None
        self.train_multi = None
        self.test_multi = None

        # Processed tensors
        self.X_train_nn = None
        self.X_val_nn = None
        self.X_test_nn = None
        self.y_train_nn = None
        self.y_val_nn = None
        self.y_test_nn = None

        # Segment arrays
        self.segments_train = None
        self.segments_val = None
        self.segments_test = None

    def load_data(self, data_dir: str, feature_cols_file: str = 'feature_columns.txt'):
        """
        Load train and test data.

        Args:
            data_dir: Directory containing processed data
            feature_cols_file: Name of feature columns file
        """
        print("📂 Loading data...")

        self.train_df = pd.read_csv(f'{data_dir}/train_features.csv')
        self.test_df = pd.read_csv(f'{data_dir}/test_features.csv')

        with open(f'{data_dir}/{feature_cols_file}', 'r') as f:
            self.feature_cols = [line.strip() for line in f.readlines()]

        print(f"   Train: {len(self.train_df):,} users")
        print(f"   Test:  {len(self.test_df):,} users")
        print(f"   Features: {len(self.feature_cols)}")

    def split_by_order_count(self):
        """Split data into 1-order and multi-order users."""
        print("\n📦 Splitting by order count...")

        # 1-Order users
        self.train_1order = self.train_df[
            self.train_df['frequency_segment'] == '1 Order'
            ].copy()
        self.test_1order = self.test_df[
            self.test_df['frequency_segment'] == '1 Order'
            ].copy()

        # Multi-order users
        self.train_multi = self.train_df[
            self.train_df['frequency_segment'] != '1 Order'
            ].copy()
        self.test_multi = self.test_df[
            self.test_df['frequency_segment'] != '1 Order'
            ].copy()

        print(f"   1-Order: Train={len(self.train_1order):,}, Test={len(self.test_1order):,}")
        print(f"   Multi:   Train={len(self.train_multi):,}, Test={len(self.test_multi):,}")

    def encode_categoricals(self):
        """Encode categorical features."""
        print("\n🔧 Encoding categorical features...")

        categorical_cols = []
        for col in self.feature_cols:
            if col in self.train_df.columns and self.train_df[col].dtype == 'object':
                categorical_cols.append(col)

        for col in categorical_cols:
            le = LabelEncoder()
            combined = pd.concat([
                self.train_df[col],
                self.test_df[col]
            ]).astype(str)
            le.fit(combined)

            # Apply to all splits
            for df in [self.train_1order, self.test_1order,
                       self.train_multi, self.test_multi]:
                if df is not None:
                    df[col] = le.transform(df[col].astype(str))

            self.label_encoders[col] = le

        print(f"   Encoded: {categorical_cols}")

    def prepare_nn_data(self) -> Tuple:
        """
        Prepare data for neural networks (multi-order users only).

        Returns:
            Tuple of DataLoaders and segment arrays
        """
        print("\n🔧 Preparing data for Neural Networks...")

        # Segment encoder
        self.segment_encoder = LabelEncoder()
        self.segment_encoder.fit(self.train_multi['frequency_segment'])

        # Extract features and labels
        X_train = self.train_multi[self.feature_cols].fillna(0).values
        y_train = self.train_multi['is_churned'].values
        segments_train_full = self.train_multi['frequency_segment'].values

        X_test = self.test_multi[self.feature_cols].fillna(0).values
        y_test = self.test_multi['is_churned'].values
        segments_test = self.test_multi['frequency_segment'].values

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Split train into train/val (stratified)
        X_train_nn, X_val_nn, y_train_nn, y_val_nn, seg_train, seg_val = train_test_split(
            X_train_scaled, y_train, segments_train_full,
            test_size=self.config.val_size,
            random_state=self.config.random_state,
            stratify=y_train
        )

        # Store processed data
        self.X_train_nn = X_train_nn
        self.X_val_nn = X_val_nn
        self.X_test_nn = X_test_scaled
        self.y_train_nn = y_train_nn
        self.y_val_nn = y_val_nn
        self.y_test_nn = y_test
        self.segments_train = seg_train
        self.segments_val = seg_val
        self.segments_test = segments_test

        print(f"   Train: {len(X_train_nn):,}")
        print(f"   Val:   {len(X_val_nn):,}")
        print(f"   Test:  {len(X_test_scaled):,}")

        return self._create_dataloaders()

    def _create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create PyTorch DataLoaders."""
        # Convert to tensors
        X_train_t = torch.FloatTensor(self.X_train_nn)
        y_train_t = torch.FloatTensor(self.y_train_nn)
        X_val_t = torch.FloatTensor(self.X_val_nn)
        y_val_t = torch.FloatTensor(self.y_val_nn)
        X_test_t = torch.FloatTensor(self.X_test_nn)
        y_test_t = torch.FloatTensor(self.y_test_nn)

        # Create datasets
        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        test_dataset = TensorDataset(X_test_t, y_test_t)

        # Weighted sampler for training
        if self.config.use_weighted_sampler:
            class_counts = np.bincount(self.y_train_nn.astype(int))
            class_weights = 1.0 / class_counts
            sample_weights = class_weights[self.y_train_nn.astype(int)]

            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                sampler=sampler
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True
            )

        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def get_1order_data(self) -> Tuple:
        """
        Get data for 1-order users (for XGBoost).

        Returns:
            X_train, y_train, X_test, y_test
        """
        X_train = self.train_1order[self.feature_cols].fillna(0)
        y_train = self.train_1order['is_churned']
        X_test = self.test_1order[self.feature_cols].fillna(0)
        y_test = self.test_1order['is_churned']

        return X_train, y_train, X_test, y_test

    def get_input_dim(self) -> int:
        """Get number of input features."""
        return len(self.feature_cols)

    def get_segment_names(self) -> List[str]:
        """Get list of segment names for multi-order users."""
        return list(self.train_multi['frequency_segment'].unique())


def prepare_all_data(data_dir: str, config: DataConfig = None) -> ChurnDataset:
    """
    Convenience function to prepare all data.

    Args:
        data_dir: Directory containing processed data
        config: Data configuration

    Returns:
        Prepared ChurnDataset
    """
    dataset = ChurnDataset(config)
    dataset.load_data(data_dir)
    dataset.split_by_order_count()
    dataset.encode_categoricals()

    return dataset