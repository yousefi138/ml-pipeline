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
from config import LOG_LEVEL, LOG_FORMAT, LOGS_DIR, TARGET_COLUMN, TIME_COLUMN

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


def get_feature_groups(df, target_column=None, time_column=None):
    """
    Extract all predictive features from the dataframe.
    
    Automatically excludes target and time columns to get all features
    available for model training. This function is feature-agnostic and does not
    require predefined feature group definitions.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_column : str, optional
        Name of the target column. If None, uses config.TARGET_COLUMN
    time_column : str, optional
        Name of the time column. If None, uses config.TIME_COLUMN
    
    Returns
    -------
    feature_groups : dict
        Dictionary with 'all_features' containing all predictive features
    """
    # Use config values if not provided
    if target_column is None:
        target_column = TARGET_COLUMN
    if time_column is None:
        time_column = TIME_COLUMN
    
    # Exclude target and time columns from features
    exclude_cols = {target_column, time_column}
    all_features = [col for col in df.columns if col not in exclude_cols]
    
    feature_groups = {
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


class ShapleyCalculator:
    """
    Calculate and visualize SHAP (Shapley Additive exPlanations) values.
    Provides model-agnostic feature importance through SHAP value analysis.
    """
    
    def __init__(self, logger=None):
        """
        Initialize ShapleyCalculator.
        
        Parameters
        ----------
        logger : logging.Logger, optional
            Logger instance for reporting
        """
        self.logger = logger or logging.getLogger('shapley')
        try:
            import shap
            self.shap = shap
        except ImportError:
            self.logger.error("SHAP package not installed. Install with: pip install shap")
            raise
    
    def calculate_shapley_values(self, model, X, feature_names=None, max_samples=None):
        """
        Calculate SHAP values for a model and dataset.
        CRITICAL: Applies preprocessing to X so dimensions match SHAP values.
        """
        self.logger.info("Calculating SHAP values...")
        
        # Convert to DataFrame
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
            if feature_names:
                X_df.columns = feature_names
            else:
                feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
                X_df.columns = feature_names
        else:
            X_df = X.copy()
            if feature_names is None:
                feature_names = X_df.columns.tolist()
        
        # CRITICAL: Apply preprocessing FIRST to get transformed data
        # This ensures dimensions match what SHAP values will be computed on
        X_transformed = X_df
        transformed_feature_names = list(feature_names)
        
        if hasattr(model, 'named_steps') and 'preprocess' in model.named_steps:
            self.logger.debug("Applying preprocessing pipeline...")
            try:
                preprocess_pipe = model.named_steps['preprocess']
                X_transformed = preprocess_pipe.transform(X_df)
                
                # Get transformed feature names
                if hasattr(preprocess_pipe, 'get_feature_names_out'):
                    try:
                        transformed_feature_names = preprocess_pipe.get_feature_names_out().tolist()
                    except:
                        transformed_feature_names = [f"Feature_{i}" for i in range(X_transformed.shape[1])]
                else:
                    transformed_feature_names = [f"Feature_{i}" for i in range(X_transformed.shape[1])]
                    
                self.logger.debug(f"Transformed X from {X_df.shape} to {X_transformed.shape}")
            except Exception as e:
                self.logger.warning(f"Could not apply preprocessing: {str(e)}")
                X_transformed = X_df
        
        # Convert to numpy if needed
        if isinstance(X_transformed, pd.DataFrame):
            X_array = X_transformed.values
        else:
            X_array = X_transformed
        
        # Sample for efficiency
        if len(X_array) > (max_samples or 500):
            sample_size = max_samples or 500
            sample_indices = np.random.choice(len(X_array), size=sample_size, replace=False)
            X_sample = X_array[sample_indices]
            self.logger.info(
                f"Using {sample_size} samples for SHAP calculation "
                f"(total available: {len(X_array)})"
            )
        else:
            X_sample = X_array
        
        try:
            # Create background data
            background_size = min(50, max(5, len(X_sample) // 10))
            background_indices = np.random.choice(len(X_sample), size=min(background_size, len(X_sample)), replace=False)
            background_array = X_sample[background_indices]
            
            # Create explainer - KernelExplainer works with already-transformed (preprocessed) data
            explainer = self.shap.KernelExplainer(
                model.named_steps['clf'].predict_proba if (hasattr(model, 'named_steps') and 'clf' in model.named_steps)
                else model.predict_proba,
                background_array
            )
            
            # Calculate SHAP values
            self.logger.debug(f"Computing SHAP values for {len(X_sample)} samples...")
            shap_values = explainer.shap_values(X_sample)
            
            # Handle binary classification
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # Ensure numpy array
            if isinstance(shap_values, pd.DataFrame):
                shap_values = shap_values.values
            
            self.logger.info(f"✓ SHAP values calculated successfully")
            return shap_values, explainer, X_sample, transformed_feature_names
            
        except Exception as e:
            self.logger.error(f"Error calculating SHAP values: {str(e)}")
            raise
    
    def generate_summary_plot_base64(self, shap_values, X, feature_names=None,
                                     plot_type='bar', max_display=10, save_path=None):
        """
        Generate SHAP summary plot and return as base64-encoded image.
        
        Parameters
        ----------
        shap_values : np.ndarray
            SHAP values
        X : np.ndarray or pd.DataFrame
            Feature data
        feature_names : list, optional
            Feature names
        plot_type : str
            Type of plot: 'bar', 'beeswarm', or 'decisions'
        max_display : int
            Maximum number of features to display
        save_path : str, optional
            If provided, save plot to this file path as PNG
        
        Returns
        -------
        base64_string : str
            Base64-encoded PNG image
        """
        import matplotlib.pyplot as plt
        import io
        import base64
        
        self.logger.info(f"Generating SHAP {plot_type} plot...")
        
        # Convert to DataFrame for SHAP plotting
        if isinstance(X, np.ndarray):
            if feature_names is None:
                feature_names = [f"Feature {i}" for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=feature_names)
        else:
            X_df = X
            if feature_names is None:
                feature_names = X_df.columns.tolist()
        
        try:
            # Create figure
            fig = plt.figure(figsize=(10, 6))
            
            # Convert to numpy array
            if isinstance(X, np.ndarray):
                X_array = X
            else:
                X_array = X.values
            
            # Ensure feature_names match X dimensions
            if feature_names is None or len(feature_names) != X_array.shape[1]:
                feature_names = [f"Feature {i}" for i in range(X_array.shape[1])]
            
            self.logger.debug(f"Generating {plot_type} plot with X shape {X_array.shape}, {len(feature_names)} feature names")
            
            if plot_type == 'bar':
                # Bar plot: mean |SHAP| values per feature
                self.shap.summary_plot(
                    shap_values, X_array, feature_names=feature_names,
                    plot_type='bar',
                    max_display=max_display, show=False
                )
            elif plot_type == 'beeswarm':
                # Beeswarm/violin plot showing distribution of SHAP values
                self.shap.summary_plot(
                    shap_values, X_array, feature_names=feature_names,
                    plot_type='violin',
                    max_display=max_display, show=False
                )
            elif plot_type == 'decisions':
                # Decision plot
                self.shap.decision_plot(
                    shap_values.mean(),  # base_value
                    shap_values, X_array, feature_names=feature_names,
                    max_display=max_display, show=False
                )
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            base64_string = base64.b64encode(buffer.read()).decode()
            
            # Save to disk if path provided
            if save_path:
                try:
                    import os
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    with open(save_path, 'wb') as f:
                        buffer.seek(0)
                        f.write(buffer.read())
                    self.logger.info(f"✓ SHAP {plot_type} plot saved to {save_path}")
                except Exception as e:
                    self.logger.warning(f"Could not save plot to disk: {str(e)}")
            
            plt.close(fig)
            
            self.logger.info(f"✓ SHAP {plot_type} plot generated")
            return base64_string
        
        except Exception as e:
            self.logger.error(f"Error generating SHAP plot: {str(e)}")
            plt.close('all')
            return None
    
    def get_feature_importance_from_shap(self, shap_values, feature_names=None):
        """
        Extract mean absolute SHAP values as model-agnostic feature importance.
        
        Parameters
        ----------
        shap_values : np.ndarray
            SHAP values
        feature_names : list, optional
            Feature names
        
        Returns
        -------
        importance_df : pd.DataFrame
            DataFrame with features and their mean absolute SHAP values
        """
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(mean_abs_shap))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'shap_importance': mean_abs_shap
        }).sort_values('shap_importance', ascending=False)
        
        return importance_df
