"""
Utility functions for the ML pipeline.
Provides common helper functions for logging, validation, and model evaluation.
"""

import logging
import os
import shutil
import numpy as np
import pandas as pd
from datetime import datetime
from config import LOG_LEVEL, LOG_FORMAT, LOGS_DIR

# Module-level flag to ensure log archiving only happens once per session
_logs_archived = False

def archive_old_logs():
    """
    Archive old log files to a subdirectory.
    
    Creates an 'archive' subdirectory in LOGS_DIR and moves all existing log files
    to it, keeping only log files from the current run in the main logs directory.
    This function is called once at the start of each pipeline run.
    """
    global _logs_archived
    
    if _logs_archived:
        return
    
    _logs_archived = True
    
    # Create archive directory if it doesn't exist
    archive_dir = os.path.join(LOGS_DIR, 'archive')
    os.makedirs(archive_dir, exist_ok=True)
    
    # Get current timestamp (used to identify log files from this run)
    current_timestamp = datetime.now().strftime('%Y%m%d')
    
    # Move log files from previous runs to archive
    try:
        for filename in os.listdir(LOGS_DIR):
            filepath = os.path.join(LOGS_DIR, filename)
            
            # Skip if it's a directory
            if os.path.isdir(filepath):
                continue
            
            # Skip if it's a log file from today (current run)
            if current_timestamp in filename:
                continue
            
            # Skip if it's not a log file
            if not filename.endswith('.log'):
                continue
            
            # Move the file to archive
            dest_path = os.path.join(archive_dir, filename)
            shutil.move(filepath, dest_path)
    except Exception as e:
        # Log a warning but don't fail if archiving fails
        print(f"Warning: Could not archive old log files: {e}")


def setup_logging(script_name):
    """
    Configure logging for the pipeline scripts.
    
    Archives old log files on the first call, then creates a logger with
    file and console handlers.
    
    Parameters
    ----------
    script_name : str
        Name of the script being run (used in log file name)
    
    Returns
    -------
    logger : logging.Logger
        Configured logger instance
    """
    # Archive old logs on first setup call
    archive_old_logs()
    
    # Create logger
    logger = logging.getLogger(script_name)
    logger.setLevel(LOG_LEVEL)
    
    # Create formatters
    formatter = logging.Formatter(LOG_FORMAT)
    
    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(LOGS_DIR, f'{script_name}_{timestamp}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def validate_target_variable(df, target_column):
    """
    Validate that the target variable is binary and check for missing values.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_column : str
        Name of the target column
    
    Returns
    -------
    is_valid : bool
        True if validation passes
    report : dict
        Validation report with counts and warnings
    """
    report = {
        'column': target_column,
        'total_samples': len(df),
        'missing_count': df[target_column].isna().sum(),
        'unique_values': df[target_column].nunique(),
        'value_counts': df[target_column].value_counts().to_dict(),
        'warnings': []
    }
    
    # Check for missing values
    if report['missing_count'] > 0:
        report['warnings'].append(f"Missing values found: {report['missing_count']}")
    
    # Check for binary classification
    if report['unique_values'] != 2:
        report['warnings'].append(f"Expected 2 classes, found {report['unique_values']}")
    
    is_valid = len(report['warnings']) == 0
    return is_valid, report


def check_feature_coverage(df, expected_features, target_column):
    """
    Check that all expected features are present in the dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    expected_features : list
        List of expected feature names
    target_column : str
        Name of the target column
    
    Returns
    -------
    report : dict
        Coverage report
    """
    available_columns = set(df.columns)
    expected_cols = set(expected_features + [target_column])
    
    report = {
        'total_expected': len(expected_features),
        'total_available': len(df.columns) - 1,  # Exclude target
        'missing_features': list(expected_cols - available_columns),
        'extra_columns': list(available_columns - expected_cols)
    }
    
    return report


def assess_class_imbalance(y, threshold=0.3):
    """
    Assess whether the target variable has significant class imbalance.
    
    Parameters
    ----------
    y : pd.Series or np.ndarray
        Target variable
    threshold : float
        Acceptable minimum proportion for smaller class (default 0.3)
    
    Returns
    -------
    report : dict
        Imbalance assessment report
    """
    values, counts = np.unique(y, return_counts=True)
    proportions = counts / counts.sum()
    min_prop = proportions.min()
    max_prop = proportions.max()
    
    report = {
        'class_0_count': counts[0],
        'class_1_count': counts[1] if len(counts) > 1 else 0,
        'class_0_proportion': proportions[0],
        'class_1_proportion': proportions[1] if len(proportions) > 1 else 0,
        'imbalance_ratio': max_prop / min_prop,
        'is_imbalanced': min_prop < threshold,
        'recommendation': 'Use stratified CV' if min_prop < threshold else 'Standard CV acceptable'
    }
    
    return report


def get_feature_groups(df, gene_prefix='X', clinical_features=None):
    """
    Separate features into clinical and gene expression groups.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    gene_prefix : str
        Prefix for gene expression features
    clinical_features : list
        List of clinical feature names
    
    Returns
    -------
    feature_groups : dict
        Dictionary with 'clinical' and 'gene' lists
    """
    if clinical_features is None:
        clinical_features = []
    
    all_features = [col for col in df.columns if col not in ['e.tdm', 't.tdm']]
    gene_features = [col for col in all_features if col.startswith(gene_prefix)]
    other_clinical = [col for col in all_features if col not in gene_features]
    
    feature_groups = {
        'clinical': other_clinical,
        'gene_expression': gene_features,
        'all_features': all_features
    }
    
    return feature_groups


def format_cv_results(cv_scores, model_name):
    """
    Format cross-validation results for reporting.
    
    Parameters
    ----------
    cv_scores : np.ndarray
        Array of CV fold scores
    model_name : str
        Name of the model
    
    Returns
    -------
    report : dict
        Formatted results with mean, std, min, max
    """
    report = {
        'model': model_name,
        'n_folds': len(cv_scores),
        'mean_score': cv_scores.mean(),
        'std_score': cv_scores.std(),
        'min_score': cv_scores.min(),
        'max_score': cv_scores.max(),
        'all_fold_scores': cv_scores
    }
    
    return report
