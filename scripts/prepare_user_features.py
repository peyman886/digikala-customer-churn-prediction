"""
Prepare user_features.csv for API from processed train/test data.

Run this script to create the correct feature file with 98 features.

Usage:
    python scripts/prepare_user_features.py
"""

import pandas as pd
from pathlib import Path


def main():
    # Paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    data_dir =  BASE_DIR / "data/processed"
    output_path = BASE_DIR / "app/user_features.csv"

    # Also copy to models_v2 for Docker
    output_path_docker = Path("../models_v2/user_features.csv")
    
    print("=" * 60)
    print("Preparing user_features.csv for API")
    print("=" * 60)
    
    # Load train and test
    train_path = data_dir / "train_features.csv"
    test_path = data_dir / "test_features.csv"
    feature_cols_path = data_dir / "feature_columns.txt"
    
    if not train_path.exists():
        print(f"❌ Not found: {train_path}")
        print("   Run notebook 02_preprocessing_feature_engineering_final.ipynb first!")
        return
    
    print(f"\n📂 Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"   Train: {len(train_df):,} users")
    print(f"   Test:  {len(test_df):,} users")
    
    # Load feature columns
    with open(feature_cols_path, 'r') as f:
        feature_cols = [line.strip() for line in f.readlines()]
    
    print(f"   Features: {len(feature_cols)}")
    
    # Combine train and test
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Remove duplicates by user_id (keep last = test version if exists in both)
    combined_df = combined_df.drop_duplicates(subset=['user_id'], keep='last')
    
    print(f"\n📊 Combined: {len(combined_df):,} unique users")
    
    # Verify columns
    required_cols = ['user_id', 'total_orders', 'recency'] + feature_cols
    missing = [c for c in required_cols if c not in combined_df.columns]
    
    if missing:
        print(f"⚠️  Missing columns: {missing[:5]}...")
    
    # Select columns for API
    # We need: user_id + all feature_cols (for prediction)
    # Plus some extra for display: total_orders, recency, etc.
    cols_to_keep = ['user_id'] + feature_cols
    
    # Add is_churned if exists (for validation)
    if 'is_churned' in combined_df.columns:
        cols_to_keep.append('is_churned')
    
    output_df = combined_df[cols_to_keep].copy()
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved: {output_path}")
    print(f"   Users: {len(output_df):,}")
    print(f"   Features: {len(feature_cols)}")
    
    # Also save to models_v2 directory
    output_path_docker.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path_docker, index=False)
    print(f"✅ Saved: {output_path_docker}")
    
    # Print feature count verification
    print(f"\n📋 Verification:")
    print(f"   Expected features: 98")
    print(f"   Actual features: {len(feature_cols)}")
    
    if len(feature_cols) == 98:
        print("   ✅ Feature count matches!")
    else:
        print(f"   ⚠️  Feature count mismatch! Check your preprocessing.")


if __name__ == "__main__":
    main()
