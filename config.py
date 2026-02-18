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

# Scoring metric for CV
CV_SCORING = 'roc_auc'

# Target and time columns
# All other columns in the dataset will be treated as predictive features
TARGET_COLUMN = 'e.tdm'
TIME_COLUMN = 't.tdm'  # Not used in current phase

# Model hyperparameter search spaces
# For LogisticRegression with elastic net penalty, use C (inverse regularization strength)
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

# Stratified K-Fold settings (for balanced class distribution)
STRATIFIED_CV = True

# Performance metrics to report
METRICS = [
    'roc_auc',
    'accuracy',
    'precision',
    'recall',
    'f1'
]

# Logging settings
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
