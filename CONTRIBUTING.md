# 🤝 Contributing to Digikala Churn Prediction

Thank you for considering contributing to this project! 🎉

## 📌 How to Contribute

### 1️⃣ Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/digikala-customer-churn-prediction.git
cd digikala-customer-churn-prediction
```

### 2️⃣ Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 3️⃣ Set Up Development Environment

```bash
# Install dependencies
make install

# Start services
make start
```

### 4️⃣ Make Your Changes

- Write clean, readable code
- Follow PEP 8 style guide
- Add docstrings to functions
- Update documentation if needed

### 5️⃣ Test Your Changes

```bash
# Run tests
make test

# Run notebooks to ensure they work
jupyter notebook notebooks/
```

### 6️⃣ Commit and Push

```bash
git add .
git commit -m "feat: Add your feature description"
git push origin feature/your-feature-name
```

### 7️⃣ Create Pull Request

1. Go to GitHub
2. Click "New Pull Request"
3. Describe your changes
4. Wait for review

## 📝 Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting)
- `refactor:` Code refactoring
- `test:` Adding tests
- `chore:` Maintenance tasks

**Examples:**
```bash
feat: Add SHAP explainability to model
fix: Correct missing value handling in feature engineering
docs: Update README with deployment instructions
```

## 🛠️ Development Guidelines

### Code Style

- Use **type hints** for function parameters and return values
- Write **docstrings** (Google style preferred)
- Keep functions **small and focused**
- Use **meaningful variable names**

**Example:**
```python
def calculate_churn_probability(user_id: str, features: pd.DataFrame) -> float:
    """
    Calculate churn probability for a given user.
    
    Args:
        user_id: Unique user identifier
        features: DataFrame with user features
        
    Returns:
        Churn probability between 0 and 1
        
    Raises:
        ValueError: If user_id not found in features
    """
    # Implementation...
```

### Project Structure

When adding new files, follow this structure:

```
├── data/          # Data files (not committed)
├── notebooks/     # Jupyter notebooks
├── app/           # FastAPI application
├── db/            # Database scripts
├── scripts/       # Utility scripts
├── reports/       # Generated reports
└── tests/         # Unit tests (to be added)
```

### Adding New Features

If adding a new feature:

1. **Update notebooks** if it affects data processing
2. **Update API** if it affects predictions
3. **Update README** with usage examples
4. **Add tests** for new functionality

### Adding New Dependencies

```bash
# Add to requirements.txt
echo "new-package==1.0.0" >> requirements.txt

# Update app requirements if API-related
echo "new-package==1.0.0" >> app/requirements.txt
```

## ❓ Questions?

Feel free to:
- Open an **Issue** for questions
- Start a **Discussion** for ideas
- Contact maintainer: [@peyman886](https://github.com/peyman886)

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing! 🚀**
