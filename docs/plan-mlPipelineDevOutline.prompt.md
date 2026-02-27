# ML Pipeline Development Outline: Scalable & Reproducible Supervised Learning

## Plan: ML Pipeline Development Outline

**TL;DR:** Create a reference-style markdown guide that walks through building a scalable, reproducible supervised ML pipeline for binary survival prediction. The guide will explain nested 5-fold cross-validation theory, cover elastic net regression and random forest models with hyperparameter tuning, report classification metrics (ROC-AUC with std dev, confusion matrix, precision/recall), and document final model parameters when refitted on full data. The outline will pair theoretical background with practical implementation guidance specific to your breast cancer survival project.

## Steps

### 1. Create the markdown file
Location: `scripts/ml-pipeline/ML_PIPELINE_GUIDE.md`

Content sections to include:
- **Introduction**: Problem statement, pipeline overview, why this approach for binary outcome prediction
- **Data Loading & Exploration**: Reading `breast_cancer_survival.csv`, data structure, feature categories (age, categorical variables `er`/`grade`, continuous `size`, 76 X-prefixed gene expression features)
- **Data Validation & Preprocessing**: Missing value checks, feature scaling considerations, class imbalance assessment, rationale for feature groups

### 2. Include the Cross-Validation Framework section
With:
- Theoretical explanation of nested CV (outer 5-fold for performance estimation, inner 5-fold for hyperparameter optimization)
- Pseudocode showing the nested loop structure
- Rationale for this approach (honest performance estimates, avoiding selection bias)

### 3. Explain the two models
With:
- **Elastic Net (Linear Model)**: Theory, use case for this data, hyperparameters to tune (alpha/regularization strength, l1_ratio for L1/L2 balance), expected behavior with gene expression data
- **Random Forest**: Theory, use case, hyperparameters to tune (n_estimators, max_depth, min_samples_split, min_samples_leaf), feature importance capabilities

### 4. Detail the training workflow
With:
- How to structure the nested CV loops in scikit-learn
- Hyperparameter search spaces for each model (with recommendations)
- Cross-validation scoring strategy (ROC-AUC for threshold-independent binary classification)
- Data leakage prevention best practices

### 5. Specify performance metrics to report
With:
- Mean CV ROC-AUC and standard deviation (primary metric per requirements)
- Confusion matrices, sensitivity, specificity, precision, recall
- Feature importance rankings (especially for Random Forest)
- Cross-fold variability assessment

### 6. Include final model refitting section
Explaining:
- Why retrain on full data after CV evaluation
- How to report final hyperparameters for the best-performing model (by CV AUC)
- Documenting coefficients (Elastic Net) or feature importances (Random Forest)
- Model serialization and reproducibility

### 7. Add governance/reproducibility best practices
Covering:
- Random seed management for reproducibility
- Directory structure recommendations (for logs, models, results)
- Configuration-driven approach (leveraging existing `config.yml` pattern)
- Documentation standards for results interpretation

## Verification Checklist

- [ ] Markdown is readable and technically accurate with clear section structure
- [ ] All model hyperparameters are realistic for dataset size (200 samples, 80 features)
- [ ] Nested CV explanation is clear and implementable in scikit-learn
- [ ] Metric selection aligns with binary classification best practices
- [ ] Examples reference the specific data columns correctly (e.tdm, age, er, grade, size, X*)

## Key Decisions Made

- **Academic + Practical balance**: Includes theory to justify approaches while remaining implementation-focused
- **Nested 5-fold CV**: Full symmetric design (both loops) as requested, despite computational cost (~250 model trains per metric)
- **Classification metrics**: Treats `e.tdm` as binary classification outcome with ROC-AUC as primary metric
- **Output format**: Reference-style documentation with both conceptual and practical implementation guidance
- **Configuration pattern**: Leverage existing YAML config infrastructure for reproducibility

## Dataset Context (breast_cancer_survival.csv)

- **Size**: 200 samples × 83 columns
- **Target variable**: `e.tdm` (binary event indicator: TRUE/FALSE)
- **Time variable** (not used in this phase): `t.tdm` (time to event)
- **Feature groups**:
  - Clinical: age (continuous), er (categorical), grade (categorical), size (continuous)
  - Gene expression: 76 features with names starting with X (continuous normalized log2 values)
- **Total predictive features**: 80 variables
