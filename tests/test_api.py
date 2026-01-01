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

# !/usr/bin/env python3
"""
Test script for Churn Prediction API

Usage:
    python scripts/test_api.py
"""

import requests
import json
import sys
from typing import Dict, Any

# API Configuration
API_BASE_URL = "http://localhost:9000"


def test_health_endpoint() -> bool:
    """
Test the health check endpoint
    """
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error connecting to API: {e}")
        return False


def test_root_endpoint() -> bool:
    """
    Test the root endpoint
    """
    try:
        response = requests.get(f"{API_BASE_URL}/")
        if response.status_code == 200:
            print("✅ Root endpoint passed")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_predict_endpoint(user_id: str) -> Dict[str, Any]:
    """
    Test the prediction endpoint

    Args:
        user_id: User ID to predict churn for

    Returns:
        Prediction response or error dict
    """
    try:
        payload = {"user_id": user_id}
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction for user {user_id}:")
            print(f"   Will Churn: {result['will_churn']}")
            print(f"   Probability: {result['probability']:.4f}")
            print(f"   Risk Level: {result['risk_level']}")
            return result
        elif response.status_code == 404:
            print(f"⚠️ User {user_id} not found")
            return {"error": "User not found"}
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return {"error": response.text}
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"error": str(e)}


def run_all_tests():
    """
    Run comprehensive API tests
    """
    print("=" * 60)
    print("🧪 Testing Churn Prediction API")
    print("=" * 60)

    # Test 1: Health check
    print("\n1️⃣ Testing Health Endpoint...")
    if not test_health_endpoint():
        print("❌ API is not healthy. Exiting.")
        sys.exit(1)

    # Test 2: Root endpoint
    print("\n2️⃣ Testing Root Endpoint...")
    test_root_endpoint()

    # Test 3: Prediction endpoint (example user IDs)
    print("\n3️⃣ Testing Prediction Endpoint...")
    test_user_ids = ["12345", "67890", "11111"]

    for user_id in test_user_ids:
        print(f"\n   Testing user: {user_id}")
        test_predict_endpoint(user_id)

    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
