"""
MISSINGNESS IMPUTATION IMPLEMENTATION GUIDE
============================================

This guide documents the robust missingness imputation capabilities added to the ML pipeline.

## Overview

The ML pipeline now implements multiple imputation strategies within a cross-validation framework
to robustly handle missing data while preventing data leakage. This ensures that imputation
statistics are learned only from training data in each fold.

## Imputation Strategies Implemented

### 1. Median Imputation
- **Class**: `MedianImputationTransformer`
- **Strategy**: Replaces missing values with the median of each feature
- **Use Case**: Simple, interpretable, works well for continuous features
- **Advantages**:
  - Fast and computationally efficient
  - Preserves the distribution of the data well
  - Works well with features that have outliers
- **Disadvantages**:
  - Ignores relationships between features
  - May reduce variance artificially

### 2. KNN Imputation
- **Class**: `KNNImputationTransformer`
- **Strategy**: Imputes missing values using the k nearest neighbors
- **Configuration**:
  - `n_neighbors`: Number of neighbors to use (default: 5)
  - `weights`: 'distance' (weighted by distance) or 'uniform'
  - `metric`: Distance metric, 'nan_euclidean' recommended for missing values
- **Use Case**: Preserves local data structure and relationships
- **Advantages**:
  - Captures relationships between features
  - Adapts to local data structure
  - Better for multivariate missingness patterns
- **Disadvantages**:
  - More computationally expensive
  - Sensitive to the choice of k and distance metric
  - Can be unstable with high-dimensional data

## Architecture: Data Leakage Prevention

The imputation transformers are integrated into sklearn Pipelines within the nested
cross-validation framework:

```
Nested CV Loop:
  Outer Fold (Performance Estimation):
    Train-Test Split
    │
    └─> Inner CV Loop (Hyperparameter Tuning):
        Train-Validation Split
        │
        └─> Pipeline:
            1. Imputation (fit on train, transform on train+validation)
            2. Scaling (fit on train, transform on train+validation)
            3. Encoding (fit on train, transform on train+validation)
            4. Model (fit on train, predict on validation)
    
    Best hyperparameters selected → evaluate on test set
```

This ensures imputation statistics are ONLY learned from training data in each fold,
preventing information leakage from validation/test data.

## Usage

### Running the Pipeline with Multiple Imputation Strategies

```bash
cd /Users/py16069/repos/ml-pipeline/working/scripts/ml-pipeline
python ml_pipeline.py
```

This default execution:
1. Loads and analyzes missing data patterns
2. Trains models with MEDIAN imputation
3. Trains models with KNN imputation
4. Generates reports for each strategy
5. Creates a comparative analysis report

### Running with a Single Imputation Strategy

```python
from ml_pipeline import main_single_strategy

# Run with median imputation only
results = main_single_strategy('median')

# Run with KNN imputation only
results = main_single_strategy('knn')
```

### Custom Imputation in Training

```python
from model_training import NestedCVTrainer, run_full_pipeline

# Train with specific imputation strategy
results = run_full_pipeline(X, y, imputation_strategy='knn')

# Or use the trainer directly
trainer = NestedCVTrainer(imputation_strategy='median')
en_results = trainer.train_elastic_net(X, y)
rf_results = trainer.train_random_forest(X, y)
```

## Output Files

When running the pipeline, the following files are generated:

### Reports Directory (`results/reports/`)

For each imputation strategy:
- `cv_summary_[strategy].csv` - Cross-validation summary statistics
- `model_evaluation_report_[strategy].json` - Complete evaluation report in JSON
- `model_evaluation_report_[strategy].html` - Human-readable HTML report
- `cv_scores_comparison_[strategy].png` - Visualization of CV scores by fold
- `model_comparison_[strategy].png` - Bar plot comparing model performance

Overall comparison:
- `imputation_strategy_comparison.csv` - Performance comparison across all strategies

### Models Directory (`results/models/`)

For each imputation strategy:
- `elastic_net_[strategy]_final.pkl` - Final Elastic Net model
- `random_forest_[strategy]_final.pkl` - Final Random Forest model

## Interpreting Results

### Comparing Imputation Strategies

Look at `imputation_strategy_comparison.csv` for:
1. **Mean CV AUC**: Average model performance (higher is better)
2. **Std CV AUC**: Stability across folds (lower is more stable)
3. **Best Model**: Which algorithm performed best with this strategy

### Strategy Selection

Consider these factors when choosing an imputation strategy:

**Choose Median Imputation if:**
- You want simplicity and interpretability
- Computational efficiency is important
- Missing data is random and not related to other features
- The dataset has noisy or outlier-prone features

**Choose KNN Imputation if:**
- You suspect relationships between features
- You have sufficient samples (KNN requires representative neighborhood)
- Missing data mechanisms are related to observed values
- You want to preserve local data structure

## Extending with New Imputation Strategies

To add a new imputation strategy:

1. Create a new transformer class in `imputation.py`:

```python
class MyImputationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, **params):
        self.imputer = MyImputation(**params)
    
    def fit(self, X, y=None):
        self.imputer.fit(X)
        return self
    
    def transform(self, X):
        return self.imputer.transform(X)
    
    def get_feature_names_out(self, input_features=None):
        return input_features
```

2. Update `get_imputation_transformer()` function:

```python
def get_imputation_transformer(strategy='median', **kwargs):
    if strategy == 'my_strategy':
        return MyImputationTransformer(**kwargs)
    # ... existing strategies ...
```

3. Add to pipeline in `ml_pipeline.py`:

```python
imputation_strategies = ['median', 'knn', 'my_strategy']
```

## Performance Notes

- **Median Imputation**: ~1-2 seconds per strategy (very fast)
- **KNN Imputation**: ~10-30 seconds per strategy (slower, depends on dataset size)
- **Total Pipeline Time**: Approximately 5-15 minutes for full multi-strategy run
  (5 outer folds × 5 inner folds × 2 models × 2 strategies)

## Troubleshooting

### Issue: "No missing values detected"
**Solution**: The dataset has no missing values. The pipeline will run successfully with any
imputation strategy as they act as pass-through. This is fine for evaluating robustness.

### Issue: KNN imputation is very slow
**Solution**: Reduce number of samples for testing or increase n_neighbors parameter.
KNN is O(n²) in the number of samples.

### Issue: Low performance with KNN
**Solution**: Try reducing n_neighbors or changing to median imputation. KNN can be unstable
with small datasets or high feature dimensionality.

### Issue: Results differ significantly between runs
**Solution**: Ensure RANDOM_SEED is set consistently in config.py. Some variation is expected
due to k-fold splitting, but results should be reproducible with the same seed.

## References

- Scikit-learn SimpleImputer: https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
- Scikit-learn KNNImputer: https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html
- Little & Rubin (2002) - Statistical Analysis with Missing Data
- Multiple Imputation for Nonresponse in Surveys (Rubin, 1987)
"""
