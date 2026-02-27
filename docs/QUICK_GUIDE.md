# ML Pipeline - Quick Guide

A fast introduction to running the ML Pipeline. For more detail, see [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md).

## What Does This Pipeline Do?

The ML Pipeline trains machine learning models to predict binary outcomes (classification) from tabular data. It uses **nested cross-validation** for robust evaluation and supports **multiple imputation strategies** for handling missing values.

**Key capabilities**:
- Trains two models: Elastic Net and Random Forest
- Uses nested 5×5 cross-validation for honest performance estimates
- Handles missing data with median or KNN imputation
- Caches trained models to avoid re-training
- Automatically generates evaluation reports with visualizations

## One-Minute Setup

```bash
# 1. Create environment (5 min, first time only)
mamba create -n ml-pipeline python=3.10 pandas numpy scikit-learn scikit-survival pyyaml matplotlib seaborn joblib shap -y

# 2. Activate environment
mamba activate ml-pipeline

# 3. Go to pipeline directory
cd /path/to/ml-pipeline/working/scripts/ml-pipeline

# 4. Run pipeline
python ml_pipeline.py
```

See [SETUP.md](SETUP.md) for detailed setup instructions.

## Running the Pipeline

### Basic Usage

```bash
cd working/scripts/ml-pipeline
mamba activate ml-pipeline
python ml_pipeline.py
```

**Runtime**: 5-15 minutes (depending on your hardware)

### Run with Options

```bash
# Use custom dataset
python ml_pipeline.py --dataset my_data.csv

# Specify target and time columns
python ml_pipeline.py --dataset flchain.csv --target-column death --time-column age

# Force re-training (ignores cached models)
python ml_pipeline.py --retrain

# Combine options
python ml_pipeline.py --dataset breast_cancer.csv --target-column event --retrain
```

## What the Pipeline Produces

After running, check `results/` for outputs:

```
results/
├── models/
│   ├── elastic_net_final.pkl       # Trained model 1
│   ├── random_forest_final.pkl     # Trained model 2
│   ├── training_results_median.pkl # Cached results (median imputation)
│   └── training_results_knn.pkl    # Cached results (KNN imputation)
│
├── reports/
│   ├── consolidated_results.json   # Complete metrics & parameters
│   ├── consolidated_results.csv    # Summary table
│   ├── cv_summary_median.csv       # Per-model CV scores (median)
│   ├── cv_summary_knn.csv          # Per-model CV scores (KNN)
│   ├── imputation_strategy_comparison.csv  # Strategy comparison
│   └── shap_plots/                 # SHAP explanability visualizations
│
└── logs/
    ├── ml_pipeline_main_*.log      # Main orchestrator log
    ├── data_prep_*.log             # Data preparation log
    ├── model_training_*.log        # Training log
    └── evaluation_*.log            # Evaluation log
```

## Understanding Results

### Cross-Validation AUC (Area Under Curve)

The CV AUC score ranges from 0 to 1:

- **0.50**: Random guessing
- **0.70-0.80**: Good model
- **0.80-0.90**: Very good model
- **0.90+**: Excellent model

### Comparing Models

The pipeline trains two models and reports:

| Model | Mean CV AUC | Std Dev | Interpretation |
|-------|-------------|---------|-----------------|
| Elastic Net | 0.85 | ±0.03 | Simple, stable predictions |
| Random Forest | 0.88 | ±0.04 | Complex, slightly better |

**Lower standard deviation** = More consistent across data folds

### Comparing Imputation Strategies

The pipeline tests two approaches for handling missing values:

- **Median**: Fast, simple (impute with median of available values)
- **KNN**: More sophisticated (impute using K-Nearest Neighbors)

Results show if imputation strategy matters for your data.

## Making Predictions with Trained Models

```python
import joblib
import pandas as pd

# Load trained model
model = joblib.load('results/models/random_forest_final.pkl')

# Prepare new data (same format as training data)
new_data = pd.read_csv('new_patients.csv')

# Make predictions
predictions = model.predict(new_data)              # 0 or 1
probabilities = model.predict_proba(new_data)     # [P(class 0), P(class 1)]

print(f"Predictions: {predictions}")
print(f"Probabilities: {probabilities}")
```

## Pipeline Workflow

```
1. Load Data
   ↓
2. Prepare & Validate (encode, scale)
   ↓
3. Analyze Missingness
   ↓
4. Train Models with Multiple Imputation Strategies
   │  ├─ Median imputation: Elastic Net + Random Forest
   │  └─ KNN imputation: Elastic Net + Random Forest
   ↓
5. Evaluate All Models
   ├─ Cross-validation scores
   ├─ Per-fold stability
   ├─ Feature importance
   └─ SHAP explanations
   ↓
6. Generate Reports & Save Models
```

## Configuration

To customize the pipeline, edit `config.py`:

```python
# Change cross-validation splits
N_SPLITS_OUTER = 5    # Outer CV folds
N_SPLITS_INNER = 5    # Inner CV folds (for hyperparameter tuning)

# Adjust computational resources
N_JOBS = -1           # -1 = all cores, 4 = use 4 cores

# Modify model hyperparameter search space
ELASTIC_NET_PARAMS = {
    'C': [0.01, 0.1, 1.0, 10.0, 100.0],      # Regularization strength
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]    # L1 vs L2 balance
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': [50, 100, 200],           # Number of trees
    'max_depth': [5, 10, 15, 20, None],       # Tree depth
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
```

## Checking Logs

During execution, logs are written to `results/logs/`. View them in real-time:

```bash
# Last 50 lines of main log
tail -50 results/logs/ml_pipeline_main_*.log

# Watch logs as they're written
tail -f results/logs/ml_pipeline_main_*.log

# Count total log lines
wc -l results/logs/*.log
```

## Common Questions

### How long does it take?

Typical runtime: **5-15 minutes** depending on:
- Dataset size
- Number of features
- Hardware (CPU cores, RAM)
- How many times the pipeline has run (cachespeeds it up)

### Can I speed it up?

1. **Use cached models** (default): Pipeline skips re-training if models exist
2. **Reduce CV splits**: Change `N_SPLITS_INNER` from 5 to 3 in `config.py`
3. **Reduce hyperparameter grid**: Fewer values in `ELASTIC_NET_PARAMS` and `RANDOM_FOREST_PARAMS`
4. **Limit CPU cores**: Set `N_JOBS = 4` instead of `-1`

### What if my dataset has missing values?

The pipeline automatically handles this! It tests two strategies:
- **Median imputation** (simple, fast)
- **KNN imputation** (sophisticated, preserves relationships)

Results show which works better for your data.

### How do I use a different dataset?

```bash
# Option 1: Place file in data/ directory, use filename
python ml_pipeline.py --dataset my_data.csv

# Option 2: Use full path
python ml_pipeline.py --dataset /full/path/to/my_data.csv

# Option 3: Specify target and time columns if different from config
python ml_pipeline.py --dataset my_data.csv --target-column outcome --time-column days
```

## For More Information

- **Setup & Dependencies**: See [SETUP.md](SETUP.md)
- **Detailed Implementation**: See [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)
- **Architecture & Design**: See docs/ directory
- **Code Comments**: Check docstrings in `*.py` files

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `mamba activate ml-pipeline` |
| Pipeline takes too long | Reduce CV splits or hyperparameter grid in `config.py` |
| No cached models used | Use `--retrain` flag to force re-training |
| Out of memory | Reduce `N_JOBS` or dataset size |
| Different random results | Seed is set `RANDOM_SEED = 42`, ensure this in `config.py` |

## Next Steps

1. **New user?** Start with [SETUP.md](SETUP.md) → Run the pipeline → Read [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)
2. **Want more detail?** See [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)
3. **Need help?** Check `results/logs/` for detailed execution logs
