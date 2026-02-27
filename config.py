"""
Configuration and constants for the ML pipeline.
Provides centralized settings for reproducibility, paths, and model parameters.
"""

import os
import yaml

# Load project configuration
CONFIG_PATH = "config.yml"
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

PROJECT_PATH = config['default']['project']

# Data paths
DATA_DIR = os.path.join(PROJECT_PATH, 'data')
DATA_FILE = os.path.join(DATA_DIR, 'breast_cancer_survival_with_missingness.csv')

# Results paths
RESULTS_DIR = os.path.join(PROJECT_PATH, 'results')
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
REPORTS_DIR = os.path.join(RESULTS_DIR, 'reports')
LOGS_DIR = os.path.join(RESULTS_DIR, 'logs')

# Create directories if they don't exist
for directory in [RESULTS_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Random seed for reproducibility
RANDOM_SEED = 42

# Cross-validation settings
N_SPLITS_OUTER = 5  # Outer CV for honest performance estimation
N_SPLITS_INNER = 5  # Inner CV for hyperparameter tuning
TEST_SIZE = 0.2
N_JOBS = -1  # Use all available cores

# ==============================================================================
# OUTCOME TYPE CONFIGURATION
# ==============================================================================
# Specify whether to run 'binary' classification or 'survival' analysis
# This controls which models, metrics, and evaluation approaches are used
OUTCOME_TYPE = 'binary'  # Options: 'binary' or 'survival'

# Target and time columns
# All other columns in the dataset will be treated as predictive features
TARGET_COLUMN = 'e.tdm'  # Event column (for survival: censoring status; for classification: class label)
TIME_COLUMN = 't.tdm'    # Time column (required for 'survival', ignored for 'binary')

# ==============================================================================
# SURVIVAL ANALYSIS CONFIGURATION
# ==============================================================================
# Minimum time threshold for survival analysis
# If observations have time values ≤ this threshold, add SURVIVAL_TIME_EPSILON to make them positive
# This is standard practice for handling zero or near-zero time-to-event values in survival data
SURVIVAL_MINIMUM_TIME = 0.0  # Allows zero times; can be set to > 0 to enforce minimum
SURVIVAL_TIME_EPSILON = 1e-6  # Small constant added to times at or below SURVIVAL_MINIMUM_TIME
# Example: if you want to exclude times < 1, set SURVIVAL_MINIMUM_TIME = 1.0

# Scoring metric for CV (outcome-specific defaults)
CV_SCORING = 'roc_auc' if OUTCOME_TYPE == 'binary' else 'concordance_index'

# ==============================================================================
# BINARY CLASSIFICATION MODEL HYPERPARAMETERS
# ==============================================================================
# For LogisticRegression with elastic net penalty, use C (inverse regularization strength)
ELASTIC_NET_PARAMS_BINARY = {
    'C': [0.01, 0.1, 1.0, 10.0, 100.0],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
}

RANDOM_FOREST_PARAMS_BINARY = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Backward compatibility: use binary params by default
ELASTIC_NET_PARAMS = ELASTIC_NET_PARAMS_BINARY
RANDOM_FOREST_PARAMS = RANDOM_FOREST_PARAMS_BINARY

# ==============================================================================
# SURVIVAL ANALYSIS MODEL HYPERPARAMETERS
# ==============================================================================
# Cox Proportional Hazards with Elastic Net regularization (lifelines or sksurv)
# Note: penalizer must be > 0 for numerical stability with complex datasets
# Start from 0.01 minimum to ensure convergence
COXPH_PARAMS_SURVIVAL = {
    'penalizer': [0.01, 0.05, 0.1, 0.5, 1.0],
    'l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0]  # 0 = L2 (Ridge), 1 = L1 (Lasso)
}

# Random Survival Forest (scikit-survival)
RANDOM_SURVIVAL_FOREST_PARAMS = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2]
}

# Select model hyperparams based on outcome type
if OUTCOME_TYPE == 'survival':
    ELASTIC_NET_PARAMS = COXPH_PARAMS_SURVIVAL
    RANDOM_FOREST_PARAMS = RANDOM_SURVIVAL_FOREST_PARAMS

# Stratified K-Fold settings (for balanced distribution)
# For survival: stratify by event status; for binary: stratify by class
STRATIFIED_CV = True

# Performance metrics to report
METRICS_BINARY = [
    'roc_auc',
    'accuracy',
    'precision',
    'recall',
    'f1'
]

METRICS_SURVIVAL = [
    'concordance_index',
    'integrated_brier_score'  # Integrated Brier Score for survival
]

# Select metrics based on outcome type
METRICS = METRICS_SURVIVAL if OUTCOME_TYPE == 'survival' else METRICS_BINARY

# Logging settings
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
