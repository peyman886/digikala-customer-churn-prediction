# 📚 Notebooks Directory

This directory contains Jupyter notebooks for data analysis, feature engineering, and model training.

## 📝 Notebooks Overview

### 1️⃣ `01_eda.ipynb` - Exploratory Data Analysis
**Purpose:** Initial data exploration and understanding

**Contents:**
- Load data from PostgreSQL database
- Analyze data distributions and patterns
- Identify missing values and outliers
- Visualize key metrics
- Understand user behavior patterns

**Key Insights:**
- User ordering frequency
- Delivery performance metrics
- CRM complaint patterns
- Rating distributions

---

### 2️⃣ `02_feature_engineering.ipynb` - Feature Engineering
**Purpose:** Create user-level features for churn prediction

**Contents:**
- Merge orders, CRM, and comments data
- Engineer temporal features (recency, frequency)
- Calculate delivery performance metrics
- Extract sentiment from comments
- Define churn label (30-day window)

**Features Created:**
1. **Behavioral Features:**
   - Total orders
   - Days since last order
   - Average order frequency
   - On-time delivery ratio

2. **CRM Features:**
   - Total complaints
   - Fake delivery requests
   - Average shop rating
   - Average courier rating

3. **Text Features:**
   - Average sentiment score
   - Minimum sentiment score

**Output:** `../data/user_features.csv`

---

### 3️⃣ `03_model_training.ipynb` - Model Training & Evaluation
**Purpose:** Train and evaluate churn prediction models

**Contents:**
- Load engineered features
- Train/test split
- Train multiple models:
  - Logistic Regression (baseline)
  - Random Forest
  - XGBoost (best)
- Model evaluation (ROC-AUC, Precision, Recall, F1)
- Feature importance analysis
- SHAP interpretability

**Models Trained:**
| Model | ROC-AUC | Best For |
|-------|---------|----------|
| Logistic Regression | ~0.78 | Baseline |
| Random Forest | ~0.83 | Interpretability |
| XGBoost | ~0.87 | Performance |

**Output:** 
- `../app/model.pkl` (trained XGBoost model)
- `../app/scaler.pkl` (feature scaler)
- `../reports/shap_summary.png` (SHAP plots)

---

## 🚀 How to Run

### Option 1: Using Docker Jupyter Service
```bash
# Start Jupyter with Docker Compose
docker-compose --profile dev up jupyter

# Access at: http://localhost:8888
# Token: churn123 (or check .env)
```

### Option 2: Local Installation
```bash
# Install dependencies
pip install -r ../app/requirements.txt

# Install additional notebook dependencies
pip install jupyter ipykernel

# Start Jupyter
jupyter notebook
```

---

## 📊 Expected Workflow

1. **EDA First** → `01_eda.ipynb`
   - Understand data structure
   - Identify data quality issues
   - Get initial insights

2. **Feature Engineering** → `02_feature_engineering.ipynb`
   - Create user-level features
   - Define churn labels
   - Save features to CSV

3. **Model Training** → `03_model_training.ipynb`
   - Train multiple models
   - Evaluate performance
   - Select best model
   - Save for deployment

---

## 📁 Output Files

Notebooks generate the following files:

```
data/
└── user_features.csv         # Engineered features

app/
├── model.pkl                 # Trained XGBoost model
└── scaler.pkl                # Feature scaler

reports/
└── shap_summary.png          # SHAP feature importance
```

---

## ⚠️ Prerequisites

1. **Database must be running:**
   ```bash
   docker-compose up db
   ```

2. **Data must be loaded:**
   ```bash
   python db/load_data.py
   ```

3. **CSV data files must exist in `data/` directory**

---

## 📝 Notes

- Notebooks use relative paths
- Database connection: `postgresql://ds_user:ds_pass@localhost:5432/churn_db`
- All visualizations use seaborn and matplotlib
- Run notebooks in order (01 → 02 → 03)

---

## 👤 Author

Peyman - [@peyman886](https://github.com/peyman886)
