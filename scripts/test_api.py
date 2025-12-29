#!/usr/bin/env python3
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
    print("="*60)
    print("🧪 Testing Churn Prediction API")
    print("="*60)
    
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
    
    print("\n" + "="*60)
    print("✅ All tests completed!")
    print("="*60)

if __name__ == "__main__":
    run_all_tests()
