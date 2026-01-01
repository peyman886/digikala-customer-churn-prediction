# 🛠️ Scripts

Utility scripts for testing and data processing.

## Available Scripts

### `test_api.py`
Comprehensive API testing script.

**Usage:**
```bash
python scripts/test_api.py
```

**What it tests:**
- ✅ Health check endpoint
- ✅ Root endpoint
- ✅ Prediction endpoint with multiple users

**Example output:**
```
🧪 Testing Churn Prediction API
✅ Health check passed
✅ Prediction for user 12345:
   Will Churn: True
   Probability: 0.8723
   Risk Level: HIGH
```

### `load_sample_data.py`
*(To be added)* Load sample data into the database.

---

**Note:** Make sure the API is running before executing test scripts:
```bash
docker-compose up
```
