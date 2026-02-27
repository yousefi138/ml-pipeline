"""
Data loading and preprocessing for the ML pipeline.
Handles reading, validation, and preparation of the breast cancer survival dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

from config import (
    DATA_FILE, TARGET_COLUMN, TIME_COLUMN, RANDOM_SEED, OUTCOME_TYPE
)
from outcome import get_outcome_handler
from utils import (
    setup_logging, validate_target_variable, check_feature_coverage,
    assess_class_imbalance, get_feature_groups
)

logger = setup_logging('data_prep')


def load_data(filepath=DATA_FILE):
    """
    Load the breast cancer survival dataset from CSV.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file
    
    Returns
    -------
    df : pd.DataFrame
        Loaded dataframe
    """
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Data loaded: {df.shape[0]} samples, {df.shape[1]} features")
    return df


def explore_data(df):
    """
    Generate exploratory statistics about the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    
    Returns
    -------
    exploration : dict
        Dictionary containing exploration results
    """
    logger.info("=" * 60)
    logger.info("DATA EXPLORATION")
    logger.info("=" * 60)
    
    exploration = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'basic_stats': df.describe().to_dict()
    }
    
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"\nData types:\n{df.dtypes}")
    logger.info(f"\nMissing values:\n{df.isnull().sum()}")
    logger.info(f"\nBasic statistics:\n{df.describe()}")
    
    return exploration


def validate_data(df, target_column=TARGET_COLUMN, time_column=TIME_COLUMN, outcome_type=OUTCOME_TYPE):
    """
    Perform comprehensive data validation using outcome-specific handlers.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_column : str
        Name of the target column
    time_column : str
        Name of the time column (used for survival analysis)
    outcome_type : str
        Type of outcome: 'binary' or 'survival'
    
    Returns
    -------
    validation_report : dict
        Comprehensive validation results
    """
    logger.info("=" * 60)
    logger.info("DATA VALIDATION")
    logger.info(f"Outcome type: {outcome_type.upper()}")
    logger.info("=" * 60)
    
    validation_report = {}
    
    # Get outcome handler and validate outcome
    outcome_handler = get_outcome_handler(outcome_type)
    is_valid_outcome, outcome_report = outcome_handler.validate_outcome(
        df, target_column, time_column
    )
    validation_report['outcome_variable'] = outcome_report
    logger.info(f"\nOutcome variable validation: {outcome_report}")
    
    if not is_valid_outcome:
        raise ValueError(f"Invalid outcome data: {outcome_report}")
    
    # Check feature coverage
    feature_groups = get_feature_groups(df)
    feature_report = check_feature_coverage(
        df, feature_groups['all_features'], target_column
    )
    validation_report['feature_coverage'] = feature_report
    logger.info(f"\nFeature coverage: {feature_report}")
    
    # Assess class imbalance (binary classification only)
    if outcome_type.lower() not in ['survival', 'survival_analysis']:
        y = df[target_column]
        imbalance_report = assess_class_imbalance(y)
        validation_report['class_imbalance'] = imbalance_report
        logger.info(f"\nClass imbalance assessment:\n{imbalance_report}")
    
    # Check for missing values in features
    feature_cols = feature_groups['all_features']
    missing_in_features = df[feature_cols].isnull().sum()
    if missing_in_features.sum() > 0:
        logger.warning(f"Missing values in features:\n{missing_in_features[missing_in_features > 0]}")
        validation_report['missing_in_features'] = missing_in_features.to_dict()
    
    logger.info("=" * 60)
    return validation_report


def preprocess_features(df, target_column=TARGET_COLUMN, time_column=TIME_COLUMN, scale_features=True, encode_categorical=True):
    """Preprocess features by separating target and features and optionally
    encoding categorical variables and scaling continuous features.

    This function is used in two modes:
      - For the main pipeline training, we disable both encoding and scaling
        (encode_categorical=False, scale_features=False) so that all
        preprocessing is handled inside scikit-learn Pipelines within the
        cross-validation loops (to avoid data leakage).
      - For ad-hoc exploration, the defaults preserve the previous behaviour
        of encoding categorical variables and scaling all features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_column : str
        Name of the target column
    scale_features : bool
        Whether to scale continuous features (default True)
    encode_categorical : bool
        Whether to label-encode categorical variables (default True)

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (preprocessed)
    y : pd.Series
        Target variable
    feature_groups : dict
        Dictionary describing feature groups
    encoders : dict or None
        Dictionary of fitted encoders for categorical variables if
        encode_categorical=True, else None
    scaler : StandardScaler or None
        Fitted scaler if scale_features=True, else None
    """
    logger.info("=" * 60)
    logger.info("FEATURE PREPROCESSING")
    logger.info("=" * 60)
    
    # Separate target and features
    y = df[target_column].copy()
    X = df.drop(columns=[target_column, time_column], errors='ignore').copy()
    
    logger.info(f"Features shape before preprocessing: {X.shape}")
    
    # Get feature groups
    feature_groups = get_feature_groups(X, target_column=target_column, time_column=time_column)
    logger.info(f"Predictive features: {len(feature_groups['all_features'])}")
    
    encoders = None
    # Identify and optionally encode categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    logger.info(f"Categorical columns identified: {categorical_cols}")

    if encode_categorical and categorical_cols:
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
            logger.info(f"Encoded {col}: {le.classes_}")
    
    # Scale continuous features if requested
    scaler = None
    if scale_features:
        # Identify continuous features (all numeric after optional encoding)
        continuous_cols = X.columns.tolist()
        scaler = StandardScaler()
        X[continuous_cols] = scaler.fit_transform(X[continuous_cols])
        logger.info(f"Scaled {len(continuous_cols)} continuous features")
    
    logger.info(f"Features shape after preprocessing: {X.shape}")
    logger.info("=" * 60)
    
    return X, y, feature_groups, encoders, scaler


def prepare_pipeline_data(filepath=DATA_FILE, target_column=None, time_column=None, outcome_type=None):
    """
    Complete data preparation pipeline: load, validate, and preprocess.
    Supports both binary classification and survival analysis outcomes.
    
    Parameters
    ----------
    filepath : str
        Path to the data file
    target_column : str, optional
        Name of the target column. If None, uses config.TARGET_COLUMN
    time_column : str, optional
        Name of the time column. If None, uses config.TIME_COLUMN
    outcome_type : str, optional
        Type of outcome: 'binary' or 'survival'. If None, uses config.OUTCOME_TYPE
    
    Returns
    -------
    results : dict
        Dictionary containing X, y (and T, E for survival), metadata, and preprocessing objects
    """
    # Use config values if not provided
    if target_column is None:
        target_column = TARGET_COLUMN
    if time_column is None:
        time_column = TIME_COLUMN
    if outcome_type is None:
        outcome_type = OUTCOME_TYPE
    
    logger.info("Starting data preparation pipeline...")
    logger.info(f"Outcome type: {outcome_type.upper()}")
    
    # Load data
    df = load_data(filepath)
    
    # Explore data
    exploration = explore_data(df)
    
    # Validate data
    validation_report = validate_data(df, target_column=target_column, 
                                      time_column=time_column, outcome_type=outcome_type)
    
    # Preprocess features
    # NOTE: For the main training pipeline, we disable encoding and scaling
    # here so that all preprocessing is performed inside scikit-learn
    # Pipelines within the nested cross-validation loops. This avoids
    # data leakage from fitting transformers on the full dataset.
    X, y, feature_groups, encoders, scaler = preprocess_features(
        df,
        target_column=target_column,
        time_column=time_column,
        scale_features=False,
        encode_categorical=False
    )
    
    # Extract outcome based on type
    outcome_handler = get_outcome_handler(outcome_type)
    outcome_data = outcome_handler.extract_outcome(df, target_column, time_column)
    
    results = {
        'X': X,
        'y': y,
        'outcome_type': outcome_type,
        'df_raw': df,
        'exploration': exploration,
        'validation_report': validation_report,
        'feature_groups': feature_groups,
        'encoders': encoders,
        'scaler': scaler,
        'target_column': target_column,
        'n_samples': X.shape[0],
        'n_features': X.shape[1]
    }
    
    # Add outcome-specific data
    if outcome_type.lower() in ['survival', 'survival_analysis']:
        results['T'] = outcome_data['T']  # Time to event
        results['E'] = outcome_data['E']  # Event indicator
    
    logger.info("Data preparation completed successfully!")
    logger.info(f"Final dataset: {results['n_samples']} samples, {results['n_features']} features")
    
    return results


if __name__ == '__main__':
    # Run data preparation pipeline
    data = prepare_pipeline_data()
    
    logger.info("\nData preparation script completed!")
    logger.info(f"X shape: {data['X'].shape}")
    logger.info(f"y shape: {data['y'].shape}")
