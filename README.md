# ML Pipeline - Complete Documentation Index

Welcome to the **Scalable & Reproducible ML Pipeline** for breast cancer survival prediction!

This document serves as the main index for all pipeline documentation and code.

---

## 📋 Quick Navigation

### For First-Time Users
1. **Start here**: [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - 2-minute overview
2. **Then read**: [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) - Detailed walkthrough
3. **Then run**: `python ml_pipeline.py` in this directory

### For Developers
1. **Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md) - System design and data flow
2. **Implementation**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - What was built
3. **Code**: Review individual `.py` files in this directory

### For Reference
- **Configuration**: See [config.py](config.py) for all settings
- **API Docs**: See docstrings in each module
- **Reports**: Output reports go to `../results/`

---

## 📚 Documentation Files

### Quick Start
| File | Purpose | Read Time |
|------|---------|-----------|
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | One-page cheat sheet with commands and quick answers | 2 min |
| [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) | Summary of what was implemented and status | 5 min |

### Detailed Guides
| File | Purpose | Read Time |
|------|---------|-----------|
| [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) | Complete walkthrough of pipeline, components, and usage | 20 min |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture, data flow, and visual diagrams | 15 min |

### This File
| File | Purpose |
|------|---------|
| [README.md](README.md) ← You are here | Documentation index and navigation |

---

## 🐍 Python Modules

### Core Pipeline

#### [config.py](config.py)
**Purpose**: Centralized configuration for entire pipeline

**Key Exports**:
- `PROJECT_PATH`, `DATA_FILE` - Data locations
- `RESULTS_DIR`, `MODELS_DIR`, `REPORTS_DIR` - Output paths
- `N_SPLITS_OUTER`, `N_SPLITS_INNER` - CV settings
- `ELASTIC_NET_PARAMS`, `RANDOM_FOREST_PARAMS` - Hyperparameter grids
- `RANDOM_SEED` - For reproducibility

**Usage**:
```python
from config import DATA_FILE, MODELS_DIR, RANDOM_SEED
```

---

#### [utils.py](utils.py)
**Purpose**: Shared utility functions for logging, validation, and evaluation

**Key Functions**:
- `setup_logging(script_name)` - Configure logging
- `validate_target_variable(df, target)` - Check target quality
- `assess_class_imbalance(y)` - Evaluate stratification need
- `get_feature_groups(df)` - Separate clinical vs gene features
- `format_cv_results(cv_scores, model_name)` - Format results

**Usage**:
```python
from utils import setup_logging, validate_target_variable
logger = setup_logging('my_script')
```

---

#### [data_prep.py](data_prep.py)
**Purpose**: Data loading, exploration, validation, and preprocessing

**Main Entry Point**:
```python
data = prepare_pipeline_data()
```

**Returns**:
```python
{
    'X': Feature matrix (198 × 80),
    'y': Target variable (198,),
    'df_raw': Original dataframe,
    'feature_groups': {'clinical': [...], 'gene_expression': [...]},
    'encoders': Categorical encoders (er, grade),
    'scaler': StandardScaler object,
    'validation_report': Data quality report
}
```

**Key Functions**:
- `load_data(filepath)` - Load CSV
- `explore_data(df)` - Generate statistics
- `validate_data(df)` - Check data quality
- `preprocess_features(df)` - Encode & scale
- `prepare_pipeline_data()` - End-to-end preparation

**Usage**:
```python
from data_prep import prepare_pipeline_data
data = prepare_pipeline_data()
X, y = data['X'], data['y']
```

---

#### [model_training.py](model_training.py)
**Purpose**: Model training with nested cross-validation

**Main Entry Point**:
```python
results = run_full_pipeline(X, y)
```

**Returns**:
```python
{
    'elastic_net': {
        'cv_scores': array([...]),       # 5 fold scores
        'mean_score': 0.843,             # Mean AUC
        'std_score': 0.045,              # Std AUC
        'best_params': {...},            # Best hyperparams
        'final_model': LogisticRegression(...),  # Model refitted on full data
        'fold_models': [...],            # Models from each fold
    },
    'random_forest': {...},              # Same as elastic_net
    'best_model_name': 'Random Forest'   # Best by mean AUC
}
```

**Key Class**: `NestedCVTrainer`
- `train_elastic_net(X, y)` - Train Elastic Net with nested CV
- `train_random_forest(X, y)` - Train Random Forest with nested CV

**Usage**:
```python
from model_training import run_full_pipeline
results = run_full_pipeline(X, y)
```

---

#### [evaluation.py](evaluation.py)
**Purpose**: Performance evaluation, metrics calculation, and reporting

**Main Entry Point**:
```python
generate_full_evaluation_report(model_results)
```

**Outputs**:
- `results/reports/model_evaluation_report.json` - Complete metrics
- `results/reports/cv_summary.csv` - Comparison table
- `results/reports/cv_scores_comparison.png` - Per-fold plot
- `results/reports/model_comparison.png` - Model comparison chart

**Key Class**: `ModelEvaluator`
- `generate_cv_summary()` - CV results table
- `get_final_model_parameters()` - Extract final model info
- `plot_cv_scores()` - Fold-by-fold performance
- `plot_model_comparison()` - Model comparison

**Usage**:
```python
from evaluation import generate_full_evaluation_report
generate_full_evaluation_report(model_results)
```

---

#### [ml_pipeline.py](ml_pipeline.py)
**Purpose**: Main orchestration script tying all components together

**Main Entry Point**:
```bash
python ml_pipeline.py
```

**This script**:
1. Calls `prepare_pipeline_data()` from data_prep
2. Calls `run_full_pipeline(X, y)` from model_training
3. Calls `generate_full_evaluation_report()` from evaluation
4. Prints formatted summary
5. Saves all outputs to results/

**Usage**:
```bash
cd scripts/ml-pipeline
python ml_pipeline.py
```

---

## 🚀 How to Use

### 1. Run Complete Pipeline
```bash
cd working/scripts/ml-pipeline
python ml_pipeline.py
```
**Time**: 5-10 minutes  
**Output**: Models, reports, visualizations in results/

### 2. Use Individual Components
```python
# Step 1: Prepare data
from data_prep import prepare_pipeline_data
data = prepare_pipeline_data()
X, y = data['X'], data['y']

# Step 2: Train models
from model_training import run_full_pipeline
results = run_full_pipeline(X, y)

# Step 3: Evaluate
from evaluation import generate_full_evaluation_report
generate_full_evaluation_report(results)
```

### 3. Load and Use Trained Models
```python
import joblib

# Load model
model = joblib.load('results/models/random_forest_final.pkl')

# Make predictions
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)
```

---

## 📊 Data Specifications

**Input File**: `breast_cancer_survival.csv` (198 × 82)
- **Target**: `e.tdm` (binary: True=event, False=no event)
- **Time**: `t.tdm` (not used in current phase)

**Features** (80 total):
- **Clinical** (4): age, er (cat), grade (cat), size
- **Gene Expression** (76): X200726_at, X200965_s_at, ...

**After Preprocessing** (X, y):
- X shape: (198 × 80) - scaled and encoded
- y shape: (198,) - binary 0/1
- Encoding: er (negative→0, positive→1), grade (4 categories)

---

## 🔧 Configuration

All settings in [config.py](config.py). Key configurations:

**Cross-Validation**:
```python
N_SPLITS_OUTER = 5   # Outer CV folds (performance estimation)
N_SPLITS_INNER = 5   # Inner CV folds (hyperparameter tuning)
STRATIFIED_CV = True # Use stratified CV (yes, data is imbalanced)
```

**Reproducibility**:
```python
RANDOM_SEED = 42  # Fixed for deterministic results
```

**Elastic Net Hyperparameters**:
```python
ELASTIC_NET_PARAMS = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
}
```

**Random Forest Hyperparameters**:
```python
RANDOM_FOREST_PARAMS = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
```

To customize, edit [config.py](config.py) directly.

---

## 📁 Output Files

After running `python ml_pipeline.py`:

```
results/
├── models/
│   ├── elastic_net_final.pkl           ← Trained model
│   └── random_forest_final.pkl         ← Trained model
│
├── reports/
│   ├── model_evaluation_report.json    ← Metrics & params
│   ├── cv_summary.csv                  ← Comparison table
│   ├── cv_scores_comparison.png        ← Per-fold plot
│   └── model_comparison.png            ← Model comparison
│
└── logs/
    ├── ml_pipeline_main_*.log          ← Main log
    ├── data_prep_*.log                 ← Data prep log
    ├── model_training_*.log            ← Training log
    └── evaluation_*.log                ← Evaluation log
```

---

## 📈 Expected Results

### Performance
| Model | Mean CV AUC | Std AUC | Runtime |
|-------|-------------|---------|---------|
| Elastic Net | 0.82-0.86 | 0.03-0.05 | 1-2 min |
| Random Forest | 0.84-0.88 | 0.02-0.04 | 3-8 min |

### Interpretation
- **AUC 0.50**: Random classifier
- **AUC 0.70-0.80**: Good
- **AUC 0.80-0.90**: Very good
- **AUC 0.90+**: Excellent

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: Config file not found
```bash
# Solution: Run from correct directory
cd working/scripts/ml-pipeline
python ml_pipeline.py
```

**Issue**: Pipeline runs very slowly
```python
# Solution: Reduce CV folds in config.py
N_SPLITS_INNER = 3  # Instead of 5 (faster, less thorough)
```

**Issue**: Low performance (AUC < 0.6)
- Check class balance (may need stratification - already enabled)
- Features may lack predictive power
- Try feature engineering or selection

See [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for more troubleshooting.

---

## 📞 Support

### For Questions About:
- **Quick start**: See [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
- **Components**: See [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)
- **Architecture**: See [ARCHITECTURE.md](ARCHITECTURE.md)
- **Code details**: See docstrings in `.py` files

### For Customization:
1. Read [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) "Customization Guide"
2. Edit [config.py](config.py) for settings
3. Edit specific modules for code changes

---

## 🎯 Next Steps

1. **First time?** → Read [QUICK_REFERENCE.md](QUICK_REFERENCE.md) (2 min)
2. **Want details?** → Read [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) (20 min)
3. **Need architecture?** → Read [ARCHITECTURE.md](ARCHITECTURE.md) (15 min)
4. **Ready to run?** → `python ml_pipeline.py` (5-10 min)
5. **Want to use models?** → See examples in [QUICK_REFERENCE.md](QUICK_REFERENCE.md)

---

## 📋 File Checklist

### Documentation
- ✓ [README.md](README.md) - This file
- ✓ [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Quick start
- ✓ [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) - Detailed guide
- ✓ [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Summary
- ✓ [ARCHITECTURE.md](ARCHITECTURE.md) - Architecture diagrams

### Code
- ✓ [config.py](config.py) - Configuration
- ✓ [utils.py](utils.py) - Utilities
- ✓ [data_prep.py](data_prep.py) - Data preparation
- ✓ [model_training.py](model_training.py) - Model training
- ✓ [evaluation.py](evaluation.py) - Evaluation
- ✓ [ml_pipeline.py](ml_pipeline.py) - Orchestrator

### Configuration
- ✓ [config.yml](config.yml) - Project paths
- ✓ [requirements.txt](requirements.txt) - Dependencies

---

## 🎓 Learning Resources

### Understanding the Pipeline
1. Start with [QUICK_REFERENCE.md](QUICK_REFERENCE.md) diagram
2. Study [ARCHITECTURE.md](ARCHITECTURE.md) data flow
3. Read [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) explanations
4. Review code in [model_training.py](model_training.py) for details

### ML Concepts
- **Nested CV**: See "Nested Cross-Validation" in [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)
- **Elastic Net**: Linear model with L1+L2 regularization
- **Random Forest**: Ensemble of decision trees
- **Hyperparameter Tuning**: GridSearchCV with inner CV loop

---

## 📝 Version History

**Created**: February 13, 2026
**Status**: Production ready
**Last Updated**: February 13, 2026

---

**Happy modeling! 🚀**

For questions or issues, refer to the documentation files above or check the code docstrings.
