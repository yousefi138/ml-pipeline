"""
Data loading and preprocessing for the ML pipeline.
Handles reading, validation, and preparation of the breast cancer survival dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

from config import (
    DATA_FILE, TARGET_COLUMN, TIME_COLUMN, CLINICAL_FEATURES,
    GENE_FEATURES_PREFIX, RANDOM_SEED
)
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


def validate_data(df, target_column=TARGET_COLUMN):
    """
    Perform comprehensive data validation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_column : str
        Name of the target column
    
    Returns
    -------
    validation_report : dict
        Comprehensive validation results
    """
    logger.info("=" * 60)
    logger.info("DATA VALIDATION")
    logger.info("=" * 60)
    
    validation_report = {}
    
    # Validate target variable
    is_valid_target, target_report = validate_target_variable(df, target_column)
    validation_report['target_variable'] = target_report
    logger.info(f"\nTarget variable validation: {target_report}")
    
    # Check feature coverage
    feature_groups = get_feature_groups(df, GENE_FEATURES_PREFIX, CLINICAL_FEATURES)
    feature_report = check_feature_coverage(
        df, feature_groups['all_features'], target_column
    )
    validation_report['feature_coverage'] = feature_report
    logger.info(f"\nFeature coverage: {feature_report}")
    
    # Assess class imbalance
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


def preprocess_features(df, target_column=TARGET_COLUMN, scale_features=True):
    """
    Preprocess features: separate clinical and gene expression, encode categorical variables.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_column : str
        Name of the target column
    scale_features : bool
        Whether to scale continuous features (default True)
    
    Returns
    -------
    X : pd.DataFrame
        Feature matrix (preprocessed)
    y : pd.Series
        Target variable
    feature_groups : dict
        Dictionary describing feature groups
    encoders : dict
        Dictionary of fitted encoders for categorical variables
    scaler : StandardScaler or None
        Fitted scaler if scale_features=True
    """
    logger.info("=" * 60)
    logger.info("FEATURE PREPROCESSING")
    logger.info("=" * 60)
    
    # Separate target and features
    y = df[target_column].copy()
    X = df.drop(columns=[target_column, TIME_COLUMN], errors='ignore').copy()
    
    logger.info(f"Features shape before preprocessing: {X.shape}")
    
    # Get feature groups
    feature_groups = get_feature_groups(X, GENE_FEATURES_PREFIX, CLINICAL_FEATURES)
    logger.info(f"Clinical features: {len(feature_groups['clinical'])}")
    logger.info(f"Gene expression features: {len(feature_groups['gene_expression'])}")
    
    # Identify and encode categorical variables
    encoders = {}
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    logger.info(f"Categorical columns identified: {categorical_cols}")
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        logger.info(f"Encoded {col}: {le.classes_}")
    
    # Scale continuous features if requested
    scaler = None
    if scale_features:
        # Identify continuous features (all numeric after encoding)
        continuous_cols = X.columns.tolist()
        scaler = StandardScaler()
        X[continuous_cols] = scaler.fit_transform(X[continuous_cols])
        logger.info(f"Scaled {len(continuous_cols)} continuous features")
    
    logger.info(f"Features shape after preprocessing: {X.shape}")
    logger.info("=" * 60)
    
    return X, y, feature_groups, encoders, scaler


def prepare_pipeline_data(filepath=DATA_FILE):
    """
    Complete data preparation pipeline: load, validate, and preprocess.
    
    Parameters
    ----------
    filepath : str
        Path to the data file
    
    Returns
    -------
    results : dict
        Dictionary containing X, y, metadata, and preprocessing objects
    """
    logger.info("Starting data preparation pipeline...")
    
    # Load data
    df = load_data(filepath)
    
    # Explore data
    exploration = explore_data(df)
    
    # Validate data
    validation_report = validate_data(df)
    
    # Preprocess features
    X, y, feature_groups, encoders, scaler = preprocess_features(df)
    
    results = {
        'X': X,
        'y': y,
        'df_raw': df,
        'exploration': exploration,
        'validation_report': validation_report,
        'feature_groups': feature_groups,
        'encoders': encoders,
        'scaler': scaler,
        'target_column': TARGET_COLUMN,
        'n_samples': X.shape[0],
        'n_features': X.shape[1]
    }
    
    logger.info("Data preparation completed successfully!")
    logger.info(f"Final dataset: {results['n_samples']} samples, {results['n_features']} features")
    
    return results


if __name__ == '__main__':
    # Run data preparation pipeline
    data = prepare_pipeline_data()
    
    logger.info("\nData preparation script completed!")
    logger.info(f"X shape: {data['X'].shape}")
    logger.info(f"y shape: {data['y'].shape}")
