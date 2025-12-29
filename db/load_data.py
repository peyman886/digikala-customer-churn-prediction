#!/usr/bin/env python3
"""
Data Loading Script for Digikala Churn Prediction Database

This script loads CSV data files into PostgreSQL database.
It handles data type conversions, missing values, and provides logging.

Author: Peyman
Date: 2025
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import os
import sys
from pathlib import Path
from datetime import datetime
import logging
from typing import Optional
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('db_loading.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'churn_db'),
    'user': os.getenv('DB_USER', 'ds_user'),
    'password': os.getenv('DB_PASSWORD', 'ds_pass')
}

# Construct database URL
DATABASE_URL = os.getenv(
    'DATABASE_URL',
    f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
    f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)

# Data directory
DATA_DIR = Path(__file__).parent.parent / 'data'


class DataLoader:
    """Handle data loading operations for the churn prediction database."""
    
    def __init__(self, database_url: str, data_dir: Path):
        """
        Initialize DataLoader.
        
        Args:
            database_url: PostgreSQL connection string
            data_dir: Path to directory containing CSV files
        """
        self.database_url = database_url
        self.data_dir = data_dir
        self.engine = None
        
    def connect(self) -> bool:
        """
        Establish database connection.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.engine = create_engine(self.database_url)
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("✅ Database connection established successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            return False
    
    def load_orders(self) -> bool:
        """
        Load orders data from CSV to database.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("📦 Loading orders data...")
            
            # Read CSV file
            orders_path = self.data_dir / 'orders.csv'
            if not orders_path.exists():
                logger.error(f"❌ File not found: {orders_path}")
                return False
                
            df = pd.read_csv(orders_path)
            logger.info(f"   Loaded {len(df)} rows from CSV")
            
            # Data cleaning and type conversion
            df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
            df['is_otd'] = df['is_otd'].astype(bool)
            df['order_id'] = df['order_id'].astype(str)
            df['user_id'] = df['user_id'].astype(str)
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['order_id'])
            logger.info(f"   After deduplication: {len(df)} rows")
            
            # Load to database
            df.to_sql('orders', self.engine, if_exists='append', index=False)
            logger.info(f"✅ Successfully loaded {len(df)} orders")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading orders: {e}")
            return False
    
    def load_crm(self) -> bool:
        """
        Load CRM data from CSV to database.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("📦 Loading CRM data...")
            
            # Read CSV file
            crm_path = self.data_dir / 'crm.csv'
            if not crm_path.exists():
                logger.error(f"❌ File not found: {crm_path}")
                return False
                
            df = pd.read_csv(crm_path)
            logger.info(f"   Loaded {len(df)} rows from CSV")
            
            # Data cleaning
            df['order_id'] = df['order_id'].astype(str)
            df['crm_delivery_request_count'] = df['crm_delivery_request_count'].fillna(0).astype(int)
            df['crm_fake_delivery_request_count'] = df['crm_fake_delivery_request_count'].fillna(0).astype(int)
            
            # Ensure ratings are within valid range (0-5)
            df['rate_to_shop'] = df['rate_to_shop'].clip(0, 5)
            df['rate_to_courier'] = df['rate_to_courier'].clip(0, 5)
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['order_id'])
            logger.info(f"   After deduplication: {len(df)} rows")
            
            # Load to database
            df.to_sql('crm', self.engine, if_exists='append', index=False)
            logger.info(f"✅ Successfully loaded {len(df)} CRM records")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading CRM data: {e}")
            return False
    
    def load_comments(self) -> bool:
        """
        Load comments data from CSV to database.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("📦 Loading comments data...")
            
            # Read CSV file
            comments_path = self.data_dir / 'comments.csv'
            if not comments_path.exists():
                logger.error(f"❌ File not found: {comments_path}")
                return False
                
            df = pd.read_csv(comments_path)
            logger.info(f"   Loaded {len(df)} rows from CSV")
            
            # Data cleaning
            df['order_id'] = df['order_id'].astype(str)
            df['description'] = df['description'].fillna('')
            
            # Load to database
            df.to_sql('comments', self.engine, if_exists='append', index=False)
            logger.info(f"✅ Successfully loaded {len(df)} comments")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading comments: {e}")
            return False
    
    def verify_data(self) -> None:
        """
        Verify loaded data by running basic queries.
        """
        try:
            logger.info("\n🔍 Verifying loaded data...")
            
            with self.engine.connect() as conn:
                # Count records in each table
                orders_count = conn.execute(text("SELECT COUNT(*) FROM orders")).scalar()
                crm_count = conn.execute(text("SELECT COUNT(*) FROM crm")).scalar()
                comments_count = conn.execute(text("SELECT COUNT(*) FROM comments")).scalar()
                
                logger.info(f"\n📊 Database Statistics:")
                logger.info(f"   Orders: {orders_count:,} records")
                logger.info(f"   CRM: {crm_count:,} records")
                logger.info(f"   Comments: {comments_count:,} records")
                
                # Sample query
                logger.info(f"\n📋 Sample Data:")
                result = conn.execute(text("""
                    SELECT 
                        COUNT(DISTINCT user_id) as unique_users,
                        AVG(CASE WHEN is_otd THEN 1 ELSE 0 END) as avg_otd_rate
                    FROM orders
                """))
                row = result.fetchone()
                logger.info(f"   Unique Users: {row[0]:,}")
                logger.info(f"   Average On-Time Delivery Rate: {row[1]:.2%}")
                
        except Exception as e:
            logger.error(f"❌ Error verifying data: {e}")
    
    def close(self) -> None:
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            logger.info("🔌 Database connection closed")


def main():
    """
    Main execution function.
    """
    logger.info("="*60)
    logger.info("🚀 Starting Data Loading Process")
    logger.info("="*60)
    logger.info(f"Data Directory: {DATA_DIR}")
    logger.info(f"Database: {DB_CONFIG['database']}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60)
    
    # Initialize loader
    loader = DataLoader(DATABASE_URL, DATA_DIR)
    
    # Connect to database
    if not loader.connect():
        logger.error("❌ Failed to establish database connection. Exiting.")
        sys.exit(1)
    
    try:
        # Load data in sequence
        success = True
        
        # Load orders first (parent table)
        success = success and loader.load_orders()
        
        # Load CRM data (child table)
        success = success and loader.load_crm()
        
        # Load comments data (child table)
        success = success and loader.load_comments()
        
        # Verify data
        if success:
            loader.verify_data()
            logger.info("\n" + "="*60)
            logger.info("✅ Data loading completed successfully!")
            logger.info("="*60)
        else:
            logger.error("\n" + "="*60)
            logger.error("❌ Data loading completed with errors!")
            logger.error("="*60)
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Unexpected error during data loading: {e}")
        sys.exit(1)
    finally:
        loader.close()


if __name__ == "__main__":
    main()
