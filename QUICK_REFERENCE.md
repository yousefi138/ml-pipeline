# ML Pipeline - Quick Reference Card

## Quick Start (One Command)

```bash
cd working/scripts/ml-pipeline
python ml_pipeline.py
```

---

## What Gets Created

| File | Purpose | Location |
|------|---------|----------|
| `elastic_net_final.pkl` | Trained Elastic Net model | `results/models/` |
| `random_forest_final.pkl` | Trained Random Forest model | `results/models/` |
| `model_evaluation_report.json` | Complete evaluation metrics | `results/reports/` |
| `cv_summary.csv` | Model comparison table | `results/reports/` |
| `cv_scores_comparison.png` | Per-fold performance plot | `results/reports/` |
| `model_comparison.png` | Model comparison chart | `results/reports/` |
| `*.log` files | Execution logs | `results/logs/` |

---

## Pipeline Flow

```
Data (198 samples, 80 features)
    ↓
[Data Preparation]
  - Load & validate
  - Encode categorical
  - Scale features
    ↓
[Nested 5-fold CV Training] (5 × 5 = 25 configurations per model)
  ├→ Elastic Net (tuning alpha, l1_ratio)
  └→ Random Forest (tuning n_estimators, max_depth, etc)
    ↓
[Evaluation]
  - Calculate mean CV AUC ± std
  - Per-fold scores
  - Feature importance
    ↓
[Reports]
  - JSON/CSV summaries
  - Plots & visualizations
  - Best model parameters
```

---

## Expected Results

| Model | Typical CV AUC | Runtime |
|-------|----------------|---------|
| Elastic Net | 0.82-0.86 | 1-2 min |
| Random Forest | 0.84-0.88 | 3-8 min |

**Total runtime**: 5-10 minutes

---

## Module Functions

### `data_prep.py`
```python
data = prepare_pipeline_data()
# Returns: X, y, feature_groups, encoders, scaler, validation_report
```

### `model_training.py`
```python
results = run_full_pipeline(X, y)
# Returns: elastic_net result, random_forest results, best_model_name
```

### `evaluation.py`
```python
generate_full_evaluation_report(model_results)
# Generates: JSON/CSV reports and plots
```

---

## Access Results After Running

### Load Trained Models
```python
import joblib

# Load best model
en_model = joblib.load('results/models/elastic_net_final.pkl')
rf_model = joblib.load('results/models/random_forest_final.pkl')

# Make predictions
predictions = rf_model.predict(X_new)
probabilities = rf_model.predict_proba(X_new)
```

### Read Report
```python
import json

with open('results/reports/model_evaluation_report.json') as f:
    report = json.load(f)
    
print(f"Best model: {report['best_model']}")
print(f"CV AUC: {report['cv_summary']}")
```

---

## Key Hyperparameters

### Elastic Net
- `alpha`: [0.0001, 0.001, 0.01, 0.1, 1.0] (regularization strength)
- `l1_ratio`: [0.1, 0.3, 0.5, 0.7, 0.9] (L1 vs L2 balance)

### Random Forest
- `n_estimators`: [50, 100, 200] (number of trees)
- `max_depth`: [5, 10, 15, 20, None] (tree depth)
- `min_samples_split`: [2, 5, 10] (min samples to split)
- `min_samples_leaf`: [1, 2, 4] (min samples per leaf)

**To modify**: Edit `config.py`

---

## Understanding Results

### CV AUC Interpretation
- **0.50**: Random classifier
- **0.70-0.80**: Good classifier
- **0.80-0.90**: Very good classifier
- **0.90+**: Excellent classifier

### Std Dev Interpretation
- **Low std** (±0.02-0.03): Stable across folds
- **High std** (±0.05+): Inconsistent performance

### Best Model Selection
Chosen by **highest mean CV AUC** from cross-validation

---

## Feature Information

**Clinical features** (4 total):
- `age`: Patient age (continuous)
- `er`: Estrogen receptor (categorical: negative/positive)
- `grade`: Tumor grade (categorical: 4 levels)
- `size`: Tumor size (continuous)

**Gene expression features** (76 total):
- Names: `X200726_at`, `X200965_s_at`, etc.
- All continuous, normalized log2 values

---

## Logs Location

All logs automatically saved with timestamps:
```
results/logs/
├── ml_pipeline_main_20260213_101034.log
├── data_prep_20260213_101034.log
├── model_training_20260213_101034.log
└── evaluation_20260213_101034.log
```

**View last 50 lines**:
```bash
tail -50 results/logs/model_training_20260213_101034.log
```

---

## If Pipeline Takes Too Long

1. **Reduce inner CV folds**: Edit `config.py`
   ```python
   N_SPLITS_INNER = 3  # Instead of 5
   ```

2. **Reduce hyperparameter grid**:
   ```python
   ELASTIC_NET_PARAMS = {
       'alpha': [0.01, 0.1, 1.0],      # Fewer values
       'l1_ratio': [0.3, 0.7]           # Fewer values
   }
   ```

3. **Reduce models tested**: Comment out one model in `model_training.py`

---

## Common Issues & Fixes

| Issue | Solution |
|-------|----------|
| "Config file not found" | Run from `scripts/ml-pipeline/` directory |
| Very low AUC (<0.6) | Dataset may be too noisy; try feature selection |
| Very high AUC (>0.99) | Check for data leakage or lab data bias |
| Out of memory | Reduce `N_JOBS` in `config.py` |
| Slow performance | Reduce CV folds or hyperparameter grid |

---

## Files You'll See

### Created by Pipeline
- `results/models/elastic_net_final.pkl`
- `results/models/random_forest_final.pkl`
- `results/reports/model_evaluation_report.json`
- `results/reports/cv_summary.csv`
- `results/reports/cv_scores_comparison.png`
- `results/reports/model_comparison.png`

### Pipeline Source Code
- `config.py` - Settings
- `utils.py` - Helpers
- `data_prep.py` - Data loading
- `model_training.py` - Models
- `evaluation.py` - Metrics
- `ml_pipeline.py` - Orchestrator

---

## Next: Using Predictions

```python
# After running pipeline:
import joblib
from data_prep import prepare_pipeline_data

# Load model
model = joblib.load('results/models/random_forest_final.pkl')

# Get data
data = prepare_pipeline_data()
X_test = data['X'][-20:]  # Last 20 samples as example

# Predict
predictions = model.predict(X_test)  # Binary 0/1
probabilities = model.predict_proba(X_test)  # [P(class=0), P(class=1)]

print(f"Predictions: {predictions}")
print(f"Probabilities: {probabilities}")
```

---

# That's It! 🚀

Run `python ml_pipeline.py` and your complete nested CV ML pipeline will execute automatically.
