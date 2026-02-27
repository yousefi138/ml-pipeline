# ML Pipeline - Implementation Guide

Complete technical documentation of the ML Pipeline for developers and advanced users. For a quick overview, see [QUICK_GUIDE.md](QUICK_GUIDE.md).

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [System Architecture](#system-architecture)
3. [Core Modules](#core-modules)
4. [Data Preparation](#data-preparation)
5. [Imputation Strategies](#imputation-strategies)
6. [Model Training](#model-training)
7. [Evaluation & Reporting](#evaluation--reporting)
8. [Configuration](#configuration)
9. [Advanced Usage](#advanced-usage)

## Pipeline Overview

The ML Pipeline is a comprehensive machine learning framework designed for binary classification with robust cross-validation and missing data handling.

**Key Characteristics**:
- **Nested Cross-Validation**: Outer CV for honest performance estimation, inner CV for hyperparameter tuning
- **Data Leakage Prevention**: All preprocessing fitted on training data only
- **Multiple Imputation Strategies**: Tests both simple (median) and sophisticated (KNN) approaches
- **Model Caching**: Trained models cached to avoid redundant computation
- **Comprehensive Reporting**: JSON/CSV results, visualizations, and feature importance

**Supported Models**:
1. **Elastic Net Logistic Regression**: Simple, interpretable linear model with L1/L2 regularization
2. **Random Forest**: Ensemble model with built-in feature importance

## System Architecture

### Component Diagram

```
┌─────────────────────────────────────────┐
│          ml_pipeline.py (Main)          │
│     (Orchestrates full workflow)        │
└─────────┬─────────────────────────────┬─┘
          │                             │
    ┌─────▼──────┐              ┌───────▼────────┐
    │ data_prep  │              │ model_training │
    │ (Step 1)   │◄────────────►│  (Step 2)      │
    └─────┬──────┘              └───────┬────────┘
          │                             │
          │        ┌──────────────┐     │
          └───────►│  imputation  │◄────┘
                   │  (Strategies)│
                   └──────────────┘
                         │
                    ┌────▼─────┐
                    │evaluation │
                    │ (Step 3)  │
                    └────┬─────┘
                         │
                    ┌────▼──────────┐
                    │ reports/      │
                    │ visualizations│
                    │ saved models  │
                    └───────────────┘

Configuration Layer (config.py):
- Paths, seeds, CV splits, hyperparameters

Utility Layer (utils.py):
- Logging, validation, feature processing
```

## Core Modules

### Module Overview

| Module | Purpose | Main Functions |
|--------|---------|----------------|
| `config.py` | Configuration & constants | Path/seed/CV settings |
| `utils.py` | Shared utilities | Logging, validation, formatting |
| `data_prep.py` | Data loading & preparation | Load, explore, validate, preprocess |
| `imputation.py` | Missing data handling | Median & KNN imputation |
| `model_training.py` | Model training & CV | Nested CV, hyperparameter tuning, caching |
| `evaluation.py` | Performance evaluation | Metrics, reports, visualizations |
| `ml_pipeline.py` | Main orchestrator | Coordinates all steps |

---

## Data Preparation

Data preparation handles loading, exploration, validation, and preprocessing of raw data.

### Module: `data_prep.py`

#### Function: `prepare_pipeline_data()`

Main entry point for data preparation.

**Signature**:
```python
def prepare_pipeline_data(
    filepath=DATA_FILE,
    target_column=TARGET_COLUMN,
    time_column=TIME_COLUMN
)
```

**Returns**: Dictionary with:
```python
{
    'X': pd.DataFrame,                 # Features
    'y': pd.Series,                    # Target
    'feature_groups': dict,            # Feature classification
    'original_feature_names': list,    # Original names
    'transformed_feature_names': list, # After encoding
    'encoders': dict,                  # Categorical encoders
    'validation_report': dict          # Validation results
}
```

**Example**:
```python
from data_prep import prepare_pipeline_data

data = prepare_pipeline_data()
X = data['X']
y = data['y']
```

#### Function: `explore_data(df)`

Generate exploratory statistics about dataset.

#### Function: `validate_data(df, target_column)`

Perform comprehensive data validation checks.

---

## Imputation Strategies

Handles missing data using different imputation approaches within sklearn Pipeline framework.

### Module: `imputation.py`

#### Class: `MissingnessAnalyzer`

Analyzes missing data patterns.

**Methods**:

##### `assess_missingness(X) → dict`

Calculate missing value statistics.

**Example**:
```python
from imputation import MissingnessAnalyzer

analysis = MissingnessAnalyzer.assess_missingness(X)
print(f"Total missing: {analysis['total_missing_values']}")
```

#### Class: `MedianImputationTransformer`

Custom scikit-learn transformer for median/mode imputation.

**Purpose**: Impute missing values fitted only on training data to prevent leakage.

**Usage in sklearn Pipeline**:
```python
from imputation import MedianImputationTransformer
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('impute', MedianImputationTransformer()),
    ('scale', StandardScaler()),
    ('clf', LogisticRegression())
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

#### Function: `get_imputation_transformer(strategy)`

Get the appropriate transformer.

**Parameters**: `strategy` ∈ {'median', 'knn'}

---

## Model Training

Implements nested cross-validation with hyperparameter tuning and model caching.

### Module: `model_training.py`

#### Function: `run_full_pipeline(X, y, imputation_strategy='median')`

Trains both models with nested cross-validation.

**Signature**:
```python
def run_full_pipeline(
    X,
    y,
    imputation_strategy='median'
) -> dict
```

**Returns**: Dictionary with results:
```python
{
    'elastic_net': {
        'model_name': str,
        'mean_score': float,           # Mean CV AUC
        'std_score': float,            # Std CV AUC
        'cv_scores': np.ndarray,       # Per-fold scores
        'best_params': dict,           # Hyperparameters
        'best_model': estimator,       # Best model
        'final_model': Pipeline,       # Fit on full data
        'imputation_strategy': str
    },
    'random_forest': {...},
    'best_model_name': str,            # 'Elastic Net' or 'Random Forest'
    'feature_names': list,
    'transformed_feature_names': list,
    'imputation_strategy': str
}
```

**Example**:
```python
from model_training import run_full_pipeline

results = run_full_pipeline(X, y, imputation_strategy='median')

print(f"Elastic Net CV AUC: {results['elastic_net']['mean_score']:.4f}")
print(f"Random Forest CV AUC: {results['random_forest']['mean_score']:.4f}")
```

#### Function: `check_cached_models_exist(strategies=None) → dict`

Check if trained model results are cached.

**Example**:
```python
cache_status = check_cached_models_exist()
for strategy, exists in cache_status.items():
    print(f"{strategy}: {'✓ Found' if exists else '✗ Not found'}")
```

#### Function: `load_cached_training_results(strategy) → dict or None`

Load previously trained model results from cache.

#### Function: `save_training_results(results, strategy)`

Save training results to cache.

---

## Evaluation & Reporting

Calculates performance metrics and generates comprehensive reports.

### Module: `evaluation.py`

#### Class: `ModelEvaluator`

Comprehensive model evaluator with metrics and reporting.

**Initialization**:
```python
evaluator = ModelEvaluator(
    model_results_dict,
    feature_names=None
)
```

#### Methods

##### `generate_cv_summary() → pd.DataFrame`

Generate summary statistics from cross-validation.

**Example**:
```python
evaluator = ModelEvaluator(results)
summary = evaluator.generate_cv_summary()
print(summary.to_string())
```

##### `get_final_model_parameters() → dict`

Extract final model parameters and coefficients.

##### `plot_cv_scores() → plt.Figure`

Line plot of per-fold performance.

##### `plot_model_comparison() → plt.Figure`

Bar chart comparing model AUC.

#### Class: `ConsolidatedReportGenerator`

Generates consolidated reports across multiple imputation strategies.

#### Function: `generate_full_evaluation_report(model_results_dict)`

Main evaluation entry point.

**Example**:
```python
from evaluation import generate_full_evaluation_report

results = run_full_pipeline(X, y)
report = generate_full_evaluation_report(results)
```

---

## Configuration

All settings centralized in `config.py` and loaded from `config.yml`.

### Module: `config.py`

#### File Paths

```python
PROJECT_PATH = config['default']['project']
DATA_DIR = os.path.join(PROJECT_PATH, 'data')
DATA_FILE = os.path.join(DATA_DIR, 'breast_cancer_survival_with_missingness.csv')
RESULTS_DIR = os.path.join(PROJECT_PATH, 'results')
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
REPORTS_DIR = os.path.join(RESULTS_DIR, 'reports')
LOGS_DIR = os.path.join(RESULTS_DIR, 'logs')
```

#### Reproducibility

```python
RANDOM_SEED = 42  # Set across numpy, sklearn, etc.
```

#### Cross-Validation Settings

```python
N_SPLITS_OUTER = 5      # Outer CV folds
N_SPLITS_INNER = 5      # Inner CV folds
STRATIFIED_CV = True    # Preserve class distribution
```

#### Computational Settings

```python
N_JOBS = -1             # -1 = all cores, N = use N cores
```

#### Model Hyperparameters

```python
ELASTIC_NET_PARAMS = {
    'C': [0.01, 0.1, 1.0, 10.0, 100.0],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
```

#### Feature Columns

```python
TARGET_COLUMN = 'e.tdm'     # Binary outcome
TIME_COLUMN = 't.tdm'       # Survival time (reference)
```

### Modifying Configuration

#### Approach 1: Edit `config.py` directly

```python
N_SPLITS_OUTER = 3          # Speed up
ELASTIC_NET_PARAMS = {      # Reduce search space
    'C': [0.1, 1.0, 10.0],
    'l1_ratio': [0.3, 0.7]
}
```

#### Approach 2: Command-line Overrides

```bash
python ml_pipeline.py --dataset my_data.csv --target-column outcome
```

#### Approach 3: Programmatic

```python
import config
config.N_SPLITS_OUTER = 3
results = run_full_pipeline(X, y)
```

---

## Advanced Usage

### Custom Dataset with Different Column Names

```bash
python ml_pipeline.py \
    --dataset my_patients.csv \
    --target-column survival_event \
    --time-column followup_months
```

### Re-train All Models from Scratch

```bash
python ml_pipeline.py --retrain
```

### Extract and Use Trained Model

```python
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score

# Load trained model
model = joblib.load('results/models/random_forest_final.pkl')

# Load new data
new_data = pd.read_csv('new_patients.csv')

# Make predictions
predictions = model.predict(new_data)
probabilities = model.predict_proba(new_data)[:, 1]

# Evaluate if labels available
labels = pd.read_csv('labels.csv')['outcome']
auc = roc_auc_score(labels, probabilities)
print(f"AUC: {auc:.4f}")
```

### Programmatic Pipeline Execution

```python
import sys
sys.path.insert(0, '/path/to/ml-pipeline/working/scripts/ml-pipeline')

from data_prep import prepare_pipeline_data
from model_training import run_full_pipeline
from evaluation import generate_full_evaluation_report

# Step 1: Prepare data
data = prepare_pipeline_data('data/cancer.csv')
X = data['X']
y = data['y']

# Step 2: Train models
all_results = {}
for strategy in ['median', 'knn']:
    print(f"Training with {strategy} imputation...")
    results = run_full_pipeline(X, y, imputation_strategy=strategy)
    all_results[strategy] = results

# Step 3: Report results
for strategy, results in all_results.items():
    print(f"\n{strategy.upper()} Results:")
    print(f"  Elastic Net CV AUC: {results['elastic_net']['mean_score']:.4f}")
    print(f"  Random Forest CV AUC: {results['random_forest']['mean_score']:.4f}")
```

---

## Understanding Cross-Validation

### Nested Cross-Validation

The pipeline uses **nested CV** for valid performance estimates:

```
OUTER CV LOOP (Honest Estimation)
│
├─ Fold 1:
│  ├─ Training set: 80% of data
│  │  └─ INNER CV LOOP (Hyperparameter Tuning)
│  │     └─ Grid search over parameters
│  │     └─ Return best parameters
│  ├─ Test set: 20% of data
│  └─ Evaluate with best params → AUC₁
│
├─ Fold 2-5: Repeat above
│  └─ AUC₂, AUC₃, AUC₄, AUC₅
│
└─ Report mean ± std across folds
```

**Why two levels?**
- Inner CV finds best hyperparameters safely
- Outer CV estimates unbiased generalization
- Single CV would overestimate performance

### Data Leakage Prevention

All preprocessing fitted on training data only:

```
For each outer CV fold:
  1. Split into train + test
  2. Fit imputation on TRAINING set only
  3. Apply to test set
  4. Fit scaling on TRAINING set only
  5. Apply to test set
  6. Train model on scaled training data
  7. Evaluate on scaled test data
```

---

## Troubleshooting & Performance Tips

### Pipeline Takes Too Long

1. **Use cached models** (default):
   ```bash
   python ml_pipeline.py
   ```

2. **Reduce CV splits**:
   ```python
   # config.py
   N_SPLITS_OUTER = 3
   N_SPLITS_INNER = 3
   ```

3. **Reduce hyperparameter grid**:
   ```python
   # config.py
   ELASTIC_NET_PARAMS = {
       'C': [0.1, 1.0, 10.0],
       'l1_ratio': [0.3, 0.7]
   }
   ```

4. **Limit CPU cores**:
   ```python
   # config.py
   N_JOBS = 4  # Use 4 cores instead of all
   ```

### Out of Memory

```python
# config.py
N_JOBS = 2  # Fewer parallel processes
```

### Inconsistent Results

Check `RANDOM_SEED` in `config.py`:
```python
RANDOM_SEED = 42  # Ensures reproducibility
```

### Model Performance Low

1. Check class imbalance in logs
2. Verify feature quality 
3. Try different imputation strategies
4. Expand hyperparameter grid

---

## Logging and Debugging

### Log Files Location

```
results/logs/
├── ml_pipeline_main_*.log      # Main orchestrator
├── data_prep_*.log             # Data preparation
├── model_training_*.log        # Training details
└── evaluation_*.log            # Evaluation
```

### Reading Logs

```bash
# Last 50 lines
tail -50 results/logs/model_training_*.log

# Watch in real-time
tail -f results/logs/ml_pipeline_main_*.log

# Search for errors
grep ERROR results/logs/*.log
```

### Increase Verbosity

```python
# config.py
LOG_LEVEL = 'DEBUG'  # More detailed logging
```

---

## API Reference Summary

### Main Entry Points

| Function | Module | Purpose |
|----------|--------|---------|
| `prepare_pipeline_data()` | data_prep | Load & prepare data |
| `run_full_pipeline()` | model_training | Train & evaluate models |
| `generate_full_evaluation_report()` | evaluation | Generate reports |

### Key Classes

| Class | Module | Purpose |
|-------|--------|---------|
| `MissingnessAnalyzer` | imputation | Analyze missing data |
| `MedianImputationTransformer` | imputation | Imputation transformer |
| `ModelEvaluator` | evaluation | Evaluate models |
| `ConsolidatedReportGenerator` | evaluation | Cross-strategy comparisons |

---

## References

- scikit-learn: https://scikit-learn.org/stable/
- Nested CV: https://scikit-learn.org/stable/modules/cross_validation.html
- SHAP: https://shap.readthedocs.io/
- Scikit-Survival: https://scikit-survival.readthedocs.io/

For additional help, check log files in `results/logs/` after running the pipeline.
