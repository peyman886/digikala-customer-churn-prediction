# 🧪 Tests

Unit tests for the Churn Prediction System.

## Running Tests

### Run all tests
```bash
pytest
```

### Run with coverage
```bash
pytest --cov=app --cov=db --cov-report=html
```

### Run specific test file
```bash
pytest tests/test_api.py -v
```

### Run specific test function
```bash
pytest tests/test_api.py::test_health_endpoint -v
```

## Test Structure

```
tests/
├── __init__.py
├── test_api.py                 # API endpoint tests
├── test_feature_engineering.py # Feature engineering tests
└── README.md                   # This file
```

## Writing Tests

### API Tests
```python
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
```

### Feature Tests
```python
import pandas as pd

def test_feature_calculation():
    df = pd.DataFrame({...})
    result = calculate_features(df)
    assert 'total_orders' in result.columns
    assert result['total_orders'].min() >= 0
```

## Test Coverage Goals

- ✅ **API Endpoints:** 100%
- ✅ **Feature Engineering:** 80%+
- ✅ **Data Loading:** 80%+
- ✅ **Model Inference:** 90%+

## CI/CD Integration

Tests run automatically on:
- 🚀 Every push to `main` branch
- 🔄 Every pull request
- 🏷️ Every release tag

See `.github/workflows/ci.yml` for details.

---

**Note:** Some tests are commented out until model artifacts are generated. Run notebooks first!
