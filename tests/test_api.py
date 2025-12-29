#!/usr/bin/env python3
"""
Quick API Test Script

Usage:
    python scripts/test_api.py
    python scripts/test_api.py --user 1385028
"""

import argparse
import requests
import sys

API_URL = "http://localhost:9000"


def test_health():
    """Test health endpoint."""
    print("Testing /health...")
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        if r.status_code == 200:
            data = r.json()
            print(f"  ✅ Status: {data['status']}")
            print(f"     Model loaded: {data['model_loaded']}")
            print(f"     Users loaded: {data['users_loaded']:,}")
            return True
        else:
            print(f"  ❌ Status code: {r.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("  ❌ Cannot connect to API. Is it running?")
        return False


def test_predict(user_id: str):
    """Test predict endpoint."""
    print(f"\nTesting /predict for user {user_id}...")
    try:
        r = requests.post(
            f"{API_URL}/predict",
            json={"user_id": user_id},
            timeout=10
        )
        
        if r.status_code == 200:
            data = r.json()
            print(f"  ✅ Prediction successful:")
            print(f"     User ID:     {data['user_id']}")
            print(f"     Will Churn:  {data['will_churn']}")
            print(f"     Probability: {data['probability']:.2%}")
            print(f"     Risk Level:  {data['risk_level']}")
            return True
        elif r.status_code == 404:
            print(f"  ⚠️ User not found: {user_id}")
            return False
        else:
            print(f"  ❌ Error: {r.status_code} - {r.text}")
            return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Churn Prediction API")
    parser.add_argument("--user", type=str, default="1385028", help="User ID to test")
    args = parser.parse_args()
    
    print("=" * 50)
    print("Churn Prediction API Test")
    print("=" * 50)
    
    # Test health
    if not test_health():
        print("\n❌ API not healthy. Exiting.")
        sys.exit(1)
    
    # Test prediction
    test_predict(args.user)
    
    # Test a few more users
    print("\nTesting additional users...")
    for user_id in ["54227", "30492532", "999999999"]:
        test_predict(user_id)
    
    print("\n" + "=" * 50)
    print("Tests complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()