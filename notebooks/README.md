# 📚 Notebooks Directory

This directory contains Jupyter notebooks for data analysis, feature engineering, and model training.

## 📝 Notebooks Overview


## 📁 Output Files

Notebooks generate the following files:

```
data/
└── user_features.csv         # Engineered features

app/
├── model.pkl                 # Trained XGBoost model
└── scaler.pkl                # Feature scaler


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
- Run notebooks in order 

---

## 👤 Author

Peyman - [@peyman886](https://github.com/peyman886)
