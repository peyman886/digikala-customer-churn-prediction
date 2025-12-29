#!/usr/bin/env python3
"""
Data Loading Script for Digikala Churn Prediction Database

This script loads CSV data into PostgreSQL with proper handling of:
- Data type conversions
- Orphan comment filtering (comments without matching orders)
- Redundant column removal (CRM order_date)

Usage:
    python db/load_data.py

Requirements:
    - PostgreSQL database must be running
    - CSV files must exist in data/ directory
    - Environment variables or .env file for DB credentials

Author: Peyman
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from pathlib import Path
import sys
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

# ============================================================================
# Configuration
# ============================================================================

# Load environment variables from .env file
load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'churn_db'),
    'user': os.getenv('DB_USER', 'ds_user'),
    'password': os.getenv('DB_PASSWORD', 'ds_pass')
}

# Build database URL
DATABASE_URL = os.getenv(
    'DATABASE_URL',
    f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
    f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'

# File names (as they appear in data/ directory)
FILES = {
    'orders': 'orders.csv',
    'crm': 'crm.csv',
    'comments': 'order_comments.csv'
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / 'data_loading.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================

def check_file_exists(filepath: Path) -> bool:
    """Check if file exists and log result."""
    exists = filepath.exists()
    if exists:
        size_mb = filepath.stat().st_size / (1024 * 1024)
        logger.info(f"  Found: {filepath.name} ({size_mb:.1f} MB)")
    else:
        logger.error(f"  Missing: {filepath}")
    return exists


def test_connection(engine) -> bool:
    """Test database connection."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        return False


def truncate_tables(engine) -> None:
    """Truncate all tables in correct order (respecting FK constraints)."""
    logger.info("Truncating existing data...")
    with engine.connect() as conn:
        # Disable FK checks temporarily, truncate, re-enable
        conn.execute(text("TRUNCATE TABLE comments, crm, orders RESTART IDENTITY CASCADE"))
        conn.commit()
    logger.info("  Tables truncated")


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_orders(engine) -> set:
    """
    Load orders data into database.

    Returns:
        Set of valid order_ids (for filtering orphan comments)
    """
    filepath = DATA_DIR / FILES['orders']
    logger.info(f"Loading orders from {filepath.name}...")

    # Read CSV
    df = pd.read_csv(filepath)
    initial_count = len(df)
    logger.info(f"  Read {initial_count:,} rows from CSV")

    # Data type conversions
    df['order_id'] = df['order_id'].astype('int64')
    df['user_id'] = df['user_id'].astype('int64')
    df['is_otd'] = df['is_otd'].astype('int16')  # Keep -1, 0, 1 as-is
    df['order_date'] = pd.to_datetime(df['order_date']).dt.date
    df['delivery_status'] = df['delivery_status'].astype(str).replace('nan', None)

    # Check for duplicates (should be 0 based on EDA)
    duplicates = df.duplicated(subset=['order_id']).sum()
    if duplicates > 0:
        logger.warning(f"  Found {duplicates} duplicate order_ids, removing...")
        df = df.drop_duplicates(subset=['order_id'], keep='first')

    # Load to database
    df.to_sql('orders', engine, if_exists='append', index=False, method='multi', chunksize=10000)
    logger.info(f"  Loaded {len(df):,} orders to database")

    # Return set of valid order_ids for later use
    return set(df['order_id'].unique())


def load_crm(engine) -> None:
    """
    Load CRM data into database.

    Note: order_date column is intentionally skipped (redundant with orders.order_date)
    """
    filepath = DATA_DIR / FILES['crm']
    logger.info(f"Loading CRM from {filepath.name}...")

    # Read CSV
    df = pd.read_csv(filepath)
    initial_count = len(df)
    logger.info(f"  Read {initial_count:,} rows from CSV")

    # Select only needed columns (skip redundant order_date)
    columns_to_keep = [
        'order_id',
        'crm_delivery_request_count',
        'crm_fake_delivery_request_count',
        'rate_to_shop',
        'rate_to_courier'
    ]
    df = df[columns_to_keep]
    logger.info(f"  Skipped redundant 'order_date' column")

    # Data type conversions
    df['order_id'] = df['order_id'].astype('int64')
    df['crm_delivery_request_count'] = df['crm_delivery_request_count'].fillna(0).astype('int16')
    df['crm_fake_delivery_request_count'] = df['crm_fake_delivery_request_count'].fillna(0).astype('int16')
    # Ratings: keep as float, NULL stays NULL

    # Check for duplicates (should be 0 based on EDA)
    duplicates = df.duplicated(subset=['order_id']).sum()
    if duplicates > 0:
        logger.warning(f"  Found {duplicates} duplicate order_ids, removing...")
        df = df.drop_duplicates(subset=['order_id'], keep='first')

    # Load to database
    df.to_sql('crm', engine, if_exists='append', index=False, method='multi', chunksize=10000)
    logger.info(f"  Loaded {len(df):,} CRM records to database")


def load_comments(engine, valid_order_ids: set) -> None:
    """
    Load comments data into database.

    Filters out orphan comments (order_id not in orders table).

    Args:
        valid_order_ids: Set of order_ids that exist in orders table
    """
    filepath = DATA_DIR / FILES['comments']
    logger.info(f"Loading comments from {filepath.name}...")

    # Read CSV
    df = pd.read_csv(filepath)
    initial_count = len(df)
    logger.info(f"  Read {initial_count:,} rows from CSV")

    # Data type conversions
    df['order_id'] = df['order_id'].astype('int64')

    # Filter out orphan comments
    df_valid = df[df['order_id'].isin(valid_order_ids)]
    orphan_count = initial_count - len(df_valid)

    if orphan_count > 0:
        logger.info(f"  Filtered {orphan_count:,} orphan comments (order_id not in orders)")

    # Don't include 'id' column - let PostgreSQL auto-generate it
    df_valid = df_valid[['order_id', 'description']]

    # Load to database
    df_valid.to_sql('comments', engine, if_exists='append', index=False, method='multi', chunksize=10000)
    logger.info(f"  Loaded {len(df_valid):,} comments to database")


def verify_data(engine) -> None:
    """Run verification queries after data load."""
    logger.info("Verifying loaded data...")

    with engine.connect() as conn:
        # Row counts
        orders_count = conn.execute(text("SELECT COUNT(*) FROM orders")).scalar()
        crm_count = conn.execute(text("SELECT COUNT(*) FROM crm")).scalar()
        comments_count = conn.execute(text("SELECT COUNT(*) FROM comments")).scalar()

        logger.info(f"  orders:   {orders_count:,} rows")
        logger.info(f"  crm:      {crm_count:,} rows")
        logger.info(f"  comments: {comments_count:,} rows")

        # Unique users
        users_count = conn.execute(text("SELECT COUNT(DISTINCT user_id) FROM orders")).scalar()
        logger.info(f"  unique users: {users_count:,}")

        # Date range
        result = conn.execute(text(
            "SELECT MIN(order_date), MAX(order_date) FROM orders"
        )).fetchone()
        logger.info(f"  date range: {result[0]} to {result[1]}")

        # is_otd distribution
        result = conn.execute(text("""
                                   SELECT is_otd, COUNT(*) as cnt
                                   FROM orders
                                   GROUP BY is_otd
                                   ORDER BY is_otd
                                   """)).fetchall()
        logger.info(f"  is_otd distribution: {dict(result)}")

        # FK integrity check
        orphan_crm = conn.execute(text("""
                                       SELECT COUNT(*)
                                       FROM crm c
                                       WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.order_id = c.order_id)
                                       """)).scalar()
        orphan_comments = conn.execute(text("""
                                            SELECT COUNT(*)
                                            FROM comments c
                                            WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.order_id = c.order_id)
                                            """)).scalar()

        if orphan_crm == 0 and orphan_comments == 0:
            logger.info("  FK integrity: OK (no orphan records)")
        else:
            logger.warning(f"  FK integrity issue: {orphan_crm} orphan CRM, {orphan_comments} orphan comments")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    start_time = datetime.now()

    logger.info("=" * 60)
    logger.info("DIGIKALA CHURN - DATA LOADING")
    logger.info("=" * 60)
    logger.info(f"Database: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info("")

    # Step 1: Check if all files exist
    logger.info("Step 1: Checking data files...")
    all_files_exist = True
    for name, filename in FILES.items():
        if not check_file_exists(DATA_DIR / filename):
            all_files_exist = False

    if not all_files_exist:
        logger.error("Missing data files. Exiting.")
        sys.exit(1)
    logger.info("")

    # Step 2: Connect to database
    logger.info("Step 2: Connecting to database...")
    engine = create_engine(DATABASE_URL)

    if not test_connection(engine):
        logger.error("Cannot connect to database. Exiting.")
        sys.exit(1)
    logger.info("  Connection successful")
    logger.info("")

    # Step 3: Truncate existing data (fresh load)
    logger.info("Step 3: Preparing tables...")
    try:
        truncate_tables(engine)
    except Exception as e:
        logger.warning(f"  Could not truncate (tables might not exist): {e}")
    logger.info("")

    # Step 4: Load data in correct order (orders first, then children)
    logger.info("Step 4: Loading data...")
    try:
        # Load orders first (returns set of valid order_ids)
        valid_order_ids = load_orders(engine)

        # Load CRM (1:1 with orders)
        load_crm(engine)

        # Load comments (filter orphans using valid_order_ids)
        load_comments(engine, valid_order_ids)

    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        raise
    logger.info("")

    # Step 5: Verify data
    logger.info("Step 5: Verification...")
    verify_data(engine)
    logger.info("")

    # Done
    elapsed = datetime.now() - start_time
    logger.info("=" * 60)
    logger.info(f"DATA LOADING COMPLETED in {elapsed.total_seconds():.1f} seconds")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()