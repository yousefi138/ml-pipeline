# ML Pipeline Architecture

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ML PIPELINE ORCHESTRATION                        │
│                          ml_pipeline.py                                 │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  Console Output + Formatted Results                               │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└──────────────┬──────────────┬──────────────┬───────────────┬────────────┘
               │              │              │               │
       ┌───────▼──────┐   ┌───▼──────┐   ┌──▼─────────┐   ┌─▼──────────┐
       │  Step 1      │   │  Step 2  │   │  Step 3    │   │ Outputs    │
       │ DATA PREP    │   │ TRAINING │   │ EVALUATION │   │ & REPORTS  │
       └──────────────┘   └──────────┘   └────────────┘   └────────────┘
             │                  │               │              │
             │                  │               │              │
       ┌─────▼────────────────────────────────────────────────────────┐
       │                   CONFIGURATION                              │
       │  ┌─────────────────────────────────────────────────────────┐ │
       │  │ config.py: Seeds, paths, CV splits, hyperparams       │ │
       │  └─────────────────────────────────────────────────────────┘ │
       └────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════

STEP 1: DATA PREPARATION (data_prep.py)
─────────────────────────────────────────
Input: breast_cancer_survival.csv (200 × 83)
         │
         ├─ Load data from CSV
         │
         ├─ Explore (shape, dtypes, stats)
         │
         ├─ Validate (binary target, no missing, class dist)
         │
         ├─ Get feature groups (clinical vs gene expression)
         │
         ├─ Encode categorical (er, grade → 0,1,2,3)
         │
         ├─ Scale continuous features (StandardScaler)
         │
         └─→ Output: X (198 × 80), y (198,)
                    encoders, scaler, metadata

═══════════════════════════════════════════════════════════════════════

STEP 2: MODEL TRAINING (model_training.py)
──────────────────────────────────────────
Input: X (198 × 80), y (198,)

NestedCVTrainer:

    OUTER CV LOOP ══════════════════════════════════════════════════
    │ (5 folds for honest performance estimation)
    │
    ├─ Outer Fold 1: Split into Train (158) + Test (40)
    │  │
    │  ├─ Model 1: Elastic Net ─────────────────────────────────┐
    │  │  │  INNER CV LOOP (5 subfolders within 158)            │
    │  │  │  GridSearch(                                         │
    │  │  │    estimator=LogisticRegression(penalty='elasticnet')│
    │  │  │    param_grid={'alpha': [.0001, .001, ..., 1.0],    │
    │  │  │               'l1_ratio': [.1, .3, .5, .7, .9]}     │
    │  │  │    cv=5)                                             │
    │  │  │                                                       │
    │  │  │  Returns: best_params, best_model                   │
    │  │  │  Evaluate on test fold (40) → AUC score             │
    │  │  │                                                       │
    │  │  └─→ Fold 1 AUC_EN1 = 0.862
    │  │
    │  └─ Model 2: Random Forest ──────────────────────────────┐
    │     │  INNER CV LOOP (5 subfolders within 158)            │
    │     │  GridSearch(                                         │
    │     │    estimator=RandomForestClassifier(),              │
    │     │    param_grid={'n_estimators': [50, 100, 200],      │
    │     │               'max_depth': [5, 10, 15, 20, None],   │
    │     │               'min_samples_*': ...}                 │
    │     │    cv=5)                                             │
    │     │                                                       │
    │     │  Returns: best_params, best_model                   │
    │     │  Evaluate on test fold (40) → AUC score             │
    │     │                                                       │
    │     └─→ Fold 1 AUC_RF1 = 0.879
    │
    ├─ Outer Fold 2-5: ...repeat above...
    │  AUC_EN: [0.862, 0.845, 0.851, 0.833, 0.824]
    │  AUC_RF: [0.879, 0.861, 0.875, 0.850, 0.841]
    │
    ├─ Calculate mean & std:
    │  EN: Mean=0.843, Std=0.045
    │  RF: Mean=0.861, Std=0.038
    │
    └─ FINAL REFIT: Train on full 198 samples with best params
       Elastic Net final model
       Random Forest final model

Output:
├─ elastic_net:
│  ├─ cv_scores: [0.862, 0.845, 0.851, 0.833, 0.824]
│  ├─ mean_score: 0.843
│  ├─ std_score: 0.045
│  ├─ best_params: {alpha: 0.01, l1_ratio: 0.5}
│  └─ final_model: fitted on 198 samples
├─ random_forest:
│  ├─ cv_scores: [0.879, 0.861, 0.875, 0.850, 0.841]
│  ├─ mean_score: 0.861
│  ├─ std_score: 0.038
│  ├─ best_params: {n_estimators: 100, max_depth: 15, ...}
│  └─ final_model: fitted on 198 samples
└─ best_model_name: "Random Forest"

═══════════════════════════════════════════════════════════════════════

STEP 3: EVALUATION & REPORTING (evaluation.py)
──────────────────────────────────────────────
Input: Training results from Step 2

ModelEvaluator:
  │
  ├─ generate_cv_summary()
  │  └─→ DataFrame with model comparison
  │      │ Model           │ Mean AUC │ Std AUC │ Best Params  │
  │      ├─ Elastic Net    │  0.843   │  0.045  │ {...}        │
  │      └─ Random Forest  │  0.861   │  0.038  │ {...}        │
  │
  ├─ generate_fold_scores_report()
  │  └─→ Per-fold scores for stability assessment
  │
  ├─ get_final_model_parameters()
  │  └─→ Extract coefficients/importances from final models
  │
  ├─ plot_cv_scores()
  │  └─→ Line plots: Fold-by-fold performance with error bands
  │
  ├─ plot_model_comparison()
  │  └─→ Bar chart: Mean AUC with error bars
  │
  ├─ save_report_to_json()
  │  └─→ model_evaluation_report.json
  │
  └─ save_summary_to_csv()
     └─→ cv_summary.csv

═══════════════════════════════════════════════════════════════════════

OUTPUT FILES STRUCTURE
──────────────────────
results/
├── models/
│   ├── elastic_net_final.pkl        ← Saved model object
│   └── random_forest_final.pkl      ← Saved model object
│
├── reports/
│   ├── model_evaluation_report.json  ← Complete metrics & params
│   ├── cv_summary.csv               ← Comparison table
│   ├── cv_scores_comparison.png     ← Per-fold plot
│   └── model_comparison.png         ← Model AUC comparison
│
└── logs/
    ├── ml_pipeline_main_*.log       ← Main orchestrator log
    ├── data_prep_*.log              ← Data prep log
    ├── model_training_*.log         ← Training log
    └── evaluation_*.log             ← Evaluation log

═══════════════════════════════════════════════════════════════════════

UTILITY LAYER (utils.py)
────────────────────────
┌──────────────────────────────────────────────────────────────┐
│ setup_logging()                → Creates logger with file/console     │
│ validate_target_variable()     → Checks binary, no missing values    │
│ assess_class_imbalance()       → Evaluates stratified CV need        │
│ get_feature_groups()           → Separates clinical vs genes         │
│ check_feature_coverage()       → Validates feature availability      │
│ format_cv_results()            → Formats scores for reporting        │
└──────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════

CONFIGURATION LAYER (config.py)
───────────────────────────────
┌──────────────────────────────────────────────────────────────┐
│ PATHS:                                                       │
│   DATA_FILE, RESULTS_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR│
│                                                              │
│ CV SETTINGS:                                                 │
│   N_SPLITS_OUTER=5, N_SPLITS_INNER=5                        │
│   STRATIFIED_CV=True                                         │
│                                                              │
│ MODEL HYPERPARAMS:                                          │
│   ELASTIC_NET_PARAMS = {alpha: [...], l1_ratio: [...]}      │
│   RANDOM_FOREST_PARAMS = {n_estimators: [...], ...}         │
│                                                              │
│ REPRODUCIBILITY:                                            │
│   RANDOM_SEED = 42 (used everywhere)                        │
└──────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════

DATA FLOW SUMMARY
─────────────────

breast_cancer_survival.csv
    ↓
data_prep.py (validate, encode, scale)
    ↓ X (198×80), y (198,)
    ↓
model_training.py (nested CV: 5×5 = 250 model trains)
    ├─ Elastic Net model
    └─ Random Forest model
    ↓ CV scores, best params
    ↓
evaluation.py (metrics, plots, reports)
    ├─ model_evaluation_report.json
    ├─ cv_summary.csv
    ├─ cv_scores_comparison.png
    └─ model_comparison.png
    ↓
results/ directory
    ├─ models/ (neural.pkl, rf.pkl)
    ├─ reports/ (all outputs)
    └─ logs/ (execution logs)

═══════════════════════════════════════════════════════════════════════

RUNTIME BREAKDOWN (Approx.)
───────────────────────────
├─ Data Preparation: 5-10 seconds
├─ Elastic Net Training (5×5 CV): 1-2 minutes
├─ Random Forest Training (5×5 CV): 3-8 minutes
├─ Evaluation & Reporting: 30-60 seconds
└─ TOTAL: 5-10 minutes

═══════════════════════════════════════════════════════════════════════

REPRODUCIBILITY MECHANISMS
──────────────────────────
✓ Fixed random seed (42) → Same results every run
✓ Stratified KFold → Balanced class distribution in folds
✓ Data scaling within CV loops → Prevents leakage
✓ Configuration-driven → Easy auditing of parameters
✓ Complete logging → Track every step
✓ Final refit on full data → Honest CV + best model

═══════════════════════════════════════════════════════════════════════
```

---

## Component Dependency Graph

```
ml_pipeline.py (Main Orchestrator)
    │
    ├─ data_prep.py
    │    ├─ config.py (paths, settings)
    │    └─ utils.py (logging, validation)
    │
    ├─ model_training.py
    │    ├─ config.py (CV settings, hyperparams, seed)
    │    ├─ utils.py (logging, formatting)
    │    └─ sklearn (GridSearchCV, StratifiedKFold)
    │
    └─ evaluation.py
         ├─ config.py (paths)
         ├─ utils.py (logging)
         └─ matplotlib/seaborn (plotting)

All modules depend on:
├─ config.py (centralized settings)
└─ utils.py (common functions)
```

---

## Execution Flow

```
START
  │
  ├─→ Parse arguments & setup logging
  │
  ├─→ STEP 1: prepare_pipeline_data()
  │    ├─ Load CSV
  │    ├─ Validate data
  │    ├─ Preprocess features
  │    └─ Return X, y
  │
  ├─→ STEP 2: run_full_pipeline(X, y)
  │    ├─ Train Elastic Net with nested CV
  │    ├─ Train Random Forest with nested CV
  │    ├─ Identify best model
  │    └─ Return results dict
  │
  ├─→ STEP 3: generate_full_evaluation_report(results)
  │    ├─ Create summary table
  │    ├─ Generate plots
  │    ├─ Save JSON report
  │    └─ Save CSV summary
  │
  ├─→ Print formatted results
  │
  └─→ END
     (Models & reports saved to results/)
```
