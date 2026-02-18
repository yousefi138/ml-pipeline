"""
Missingness imputation strategies for ML pipeline.
Implements multiple imputation approaches within the sklearn Pipeline framework
to avoid data leakage during cross-validation.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.base import BaseEstimator, TransformerMixin
from utils import setup_logging

logger = setup_logging('imputation')


class MissingnessAnalyzer:
    """
    Analyze missing data patterns in the dataset.
    """
    
    @staticmethod
    def assess_missingness(X):
        """
        Assess missing data patterns.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        
        Returns
        -------
        analysis : dict
            Dictionary containing missing data analysis
        """
        missing_summary = {
            'total_features': X.shape[1],
            'total_samples': X.shape[0],
            'features_with_missing': {},
            'percent_missing_by_feature': {},
            'total_missing_values': X.isnull().sum().sum(),
            'percent_total_missing': (X.isnull().sum().sum() / (X.shape[0] * X.shape[1])) * 100
        }
        
        missing_counts = X.isnull().sum()
        for feature in missing_counts[missing_counts > 0].index:
            count = missing_counts[feature]
            pct = (count / X.shape[0]) * 100
            missing_summary['features_with_missing'][feature] = count
            missing_summary['percent_missing_by_feature'][feature] = pct
        
        return missing_summary
    
    @staticmethod
    def report_missingness(X, prefix=""):
        """
        Log missingness report.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        prefix : str
            Prefix for logging messages
        """
        analysis = MissingnessAnalyzer.assess_missingness(X)
        
        logger.info(f"{prefix}Missing Data Analysis")
        logger.info(f"{prefix}─" * 40)
        logger.info(f"{prefix}Total features: {analysis['total_features']}")
        logger.info(f"{prefix}Total samples: {analysis['total_samples']}")
        logger.info(f"{prefix}Total missing values: {analysis['total_missing_values']}")
        logger.info(f"{prefix}Percent missing overall: {analysis['percent_total_missing']:.2f}%")
        
        if analysis['features_with_missing']:
            logger.info(f"{prefix}Features with missing values:")
            for feature, count in analysis['features_with_missing'].items():
                pct = analysis['percent_missing_by_feature'][feature]
                logger.info(f"{prefix}  • {feature}: {count} missing ({pct:.2f}%)")
        else:
            logger.info(f"{prefix}No missing values detected")
        
        return analysis


class MedianImputationTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for median imputation on numeric features and 
    most frequent imputation on categorical features.
    
    Wraps sklearn's SimpleImputer with appropriate strategies to handle
    mixed data types. Fits on training data during cross-validation,
    ensuring no data leakage.
    """
    
    def __init__(self):
        """Initialize median imputation transformer."""
        self.numeric_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.numeric_cols_ = None
        self.categorical_cols_ = None
        self.feature_names_ = None
    
    def fit(self, X, y=None):
        """
        Fit the imputers on training data.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Training features
        y : array-like, optional
            Target variable (not used)
        
        Returns
        -------
        self
        """
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        self.feature_names_ = X_df.columns.tolist() if hasattr(X_df, 'columns') else None
        
        # Identify numeric and categorical columns
        self.numeric_cols_ = X_df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols_ = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Fit numeric imputer if numeric columns exist
        if self.numeric_cols_:
            self.numeric_imputer.fit(X_df[self.numeric_cols_])
        
        # Fit categorical imputer if categorical columns exist
        if self.categorical_cols_:
            self.categorical_imputer.fit(X_df[self.categorical_cols_])
        
        return self
    
    def transform(self, X):
        """
        Transform by imputing missing values.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Features to transform
        
        Returns
        -------
        X_imputed : np.ndarray or pd.DataFrame
            Imputed features
        """
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        # Impute numeric columns
        if self.numeric_cols_:
            X_df[self.numeric_cols_] = self.numeric_imputer.transform(X_df[self.numeric_cols_])
        
        # Impute categorical columns
        if self.categorical_cols_:
            X_df[self.categorical_cols_] = self.categorical_imputer.transform(X_df[self.categorical_cols_])
        
        # Return as DataFrame if input was DataFrame
        if isinstance(X, pd.DataFrame):
            return X_df
        return X_df.values
    
    def get_feature_names_out(self, input_features=None):
        """Get feature names for pipeline compatibility."""
        if self.feature_names_ is not None:
            return np.array(self.feature_names_)
        if input_features is not None:
            return input_features
        return None


class KNNImputationTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for KNN-based imputation on numeric features and 
    most frequent imputation on categorical features.
    
    Uses sklearn's KNNImputer for numeric data (which naturally handles distance
    metrics) and SimpleImputer with most_frequent for categorical data.
    Ensures robustness to mixed data types during cross-validation.
    """
    
    def __init__(self, n_neighbors=5, weights='distance', metric='nan_euclidean'):
        """
        Initialize KNN imputation transformer.
        
        Parameters
        ----------
        n_neighbors : int
            Number of nearest neighbors to use (default 5)
        weights : str
            Weight function ('uniform' or 'distance', default 'distance')
        metric : str
            Distance metric ('nan_euclidean' recommended for missing values)
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.knn_imputer = KNNImputer(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric
        )
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.numeric_cols_ = None
        self.categorical_cols_ = None
        self.feature_names_ = None
    
    def fit(self, X, y=None):
        """
        Fit the imputers on training data.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Training features
        y : array-like, optional
            Target variable (not used)
        
        Returns
        -------
        self
        """
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        self.feature_names_ = X_df.columns.tolist() if hasattr(X_df, 'columns') else None
        
        # Identify numeric and categorical columns
        self.numeric_cols_ = X_df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols_ = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Fit KNN imputer on numeric columns
        if self.numeric_cols_:
            self.knn_imputer.fit(X_df[self.numeric_cols_])
        
        # Fit categorical imputer on categorical columns
        if self.categorical_cols_:
            self.categorical_imputer.fit(X_df[self.categorical_cols_])
        
        return self
    
    def transform(self, X):
        """
        Transform by imputing missing values using KNN for numeric features.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Features to transform
        
        Returns
        -------
        X_imputed : np.ndarray or pd.DataFrame
            KNN-imputed features
        """
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        # Impute numeric columns with KNN
        if self.numeric_cols_:
            X_df[self.numeric_cols_] = self.knn_imputer.transform(X_df[self.numeric_cols_])
        
        # Impute categorical columns with most frequent
        if self.categorical_cols_:
            X_df[self.categorical_cols_] = self.categorical_imputer.transform(X_df[self.categorical_cols_])
        
        # Return as DataFrame if input was DataFrame
        if isinstance(X, pd.DataFrame):
            return X_df
        return X_df.values
    
    def get_feature_names_out(self, input_features=None):
        """Get feature names for pipeline compatibility."""
        if self.feature_names_ is not None:
            return np.array(self.feature_names_)
        if input_features is not None:
            return input_features
        return None


def get_imputation_transformer(strategy='median', **kwargs):
    """
    Factory function to create an imputation transformer.
    
    Parameters
    ----------
    strategy : str
        Imputation strategy: 'median' or 'knn'
    **kwargs
        Additional arguments specific to the strategy
    
    Returns
    -------
    transformer : BaseEstimator
        Imputation transformer
    """
    if strategy == 'median':
        return MedianImputationTransformer()
    elif strategy == 'knn':
        n_neighbors = kwargs.get('n_neighbors', 5)
        weights = kwargs.get('weights', 'distance')
        metric = kwargs.get('metric', 'nan_euclidean')
        return KNNImputationTransformer(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric
        )
    else:
        raise ValueError(f"Unknown imputation strategy: {strategy}")


class NoImputation(BaseEstimator, TransformerMixin):
    """
    Pass-through transformer for testing without imputation.
    Useful as baseline and for data without missing values.
    """
    
    def fit(self, X, y=None):
        """Fit (no-op)."""
        return self
    
    def transform(self, X):
        """Transform (pass-through)."""
        return X
    
    def get_feature_names_out(self, input_features=None):
        """Get feature names for pipeline compatibility."""
        if input_features is not None:
            return input_features
        return None
