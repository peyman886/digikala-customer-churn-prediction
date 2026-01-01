

# 🔬 MLOps Module

A simple and practical system for tracking and comparing ML experiments.

---

## 🚀 Quick Start

### In Notebook:

```python
from mlops import ExperimentTracker

tracker = ExperimentTracker()

# New experiment
with tracker.run("my_experiment"):
    model = XGBClassifier()
    model.fit(X_train, y_train)
    
    tracker.log_params({"n_estimators": 100, "max_depth": 6})
    tracker.log_metrics(model, X_test, y_test)
    tracker.log_model(model, feature_names=feature_cols)

# Compare experiments
tracker.compare()

# Promote to production
tracker.promote("my_experiment")
```

---

## 📊 CLI Commands

```bash
# Compare all experiments
make compare

# Promote the best model to production
make promote

# Run MLflow UI
make mlflow
```

---

## 🔧 Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  1. Make a change (new feature, hyperparameter, model, ...) │
│                          ↓                                  │
│  2. Start an experiment run with a descriptive name         │
│     with tracker.run("added_sentiment"):                    │
│                          ↓                                  │
│  3. Compare                                                 │
│     tracker.compare()                                       │
│                          ↓                                  │
│  4. If it performs better → promote it                      │
│     tracker.promote("added_sentiment")                      │
│                          ↓                                  │
│  5. The model is copied to app/model.pkl                    │
│     FastAPI uses this model                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 File Structure

```
mlops/
├── __init__.py       # Exports
├── config.py         # Settings & paths
├── tracker.py        # Main ExperimentTracker class
└── compare.py        # CLI comparison utilities

models/
├── production/       # Current production model (versioned backup)
│   ├── model_YYYYMMDD_HHMMSS.pkl
│   └── metadata_YYYYMMDD_HHMMSS.json
└── experiments/      # Experimental models

app/
├── model.pkl         # ← Production model used by the API
└── model_metadata.json

mlruns/               # MLflow tracking data
```

---

## 📈 Logged Metrics

| Metric               | Description               |
| -------------------- | ------------------------- |
| `roc_auc`            | ROC-AUC Score             |
| `f1`                 | F1 Score                  |
| `precision`          | Precision                 |
| `recall`             | Recall                    |
| `accuracy`           | Accuracy                  |
| `train_time_seconds` | Training time             |
| `inference_time_ms`  | Per-sample inference time |

---

## 🖥️ MLflow UI

To view experiments in a graphical UI:

```bash
# Using make
make mlflow

# Or directly
mlflow ui --port 5000 --backend-store-uri ./mlruns
```

Then open: [http://localhost:5000](http://localhost:5000)

---

## 🐳 Docker

```bash
# Run all services
docker-compose up -d

# With Jupyter
docker-compose --profile dev up -d
```

Services:

* **API:** [http://localhost:8000/docs](http://localhost:8000/docs)
* **MLflow:** [http://localhost:5000](http://localhost:5000)
* **Jupyter:** [http://localhost:8888](http://localhost:8888) (token: churn123)

---

## 📝 Usage Examples

### Experiment with different hyperparameters:

```python
with tracker.run("deeper_trees"):
    model = XGBClassifier(n_estimators=200, max_depth=10)
    tracker.log_params({"n_estimators": 200, "max_depth": 10})
    model.fit(X_train, y_train)
    tracker.log_metrics(model, X_test, y_test)
    tracker.log_model(model)
```

### Experiment with a new feature:

```python
# Add new feature
X_train['sentiment_score'] = calculate_sentiment(comments)
X_test['sentiment_score'] = calculate_sentiment(test_comments)

with tracker.run("added_sentiment"):
    model = XGBClassifier()
    tracker.log_params({"note": "added sentiment feature"})
    model.fit(X_train, y_train)
    tracker.log_metrics(model, X_test, y_test)
    tracker.log_model(model, feature_names=X_train.columns.tolist())
```

### Compare and choose the best:

```python
# Compare by ROC-AUC
tracker.compare(metric="roc_auc")

# Compare by speed
tracker.compare(metric="inference_time_ms")

# Promote the best model
from mlops import promote_best
promote_best(metric="roc_auc")
```

---

## ⚙️ Configuration

All configurations are in `mlops/config.py`:

```python
# Data path
FEATURES_PATH = "data/user_features.csv"

# Primary comparison metric
PRIMARY_METRIC = "roc_auc"

# Default hyperparameters
MODEL_DEFAULTS = {
    "xgboost": {"n_estimators": 100, "max_depth": 6, ...},
    "rf": {"n_estimators": 100, "max_depth": 10, ...},
}
```

---


