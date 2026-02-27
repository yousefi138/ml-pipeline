"""
Outcome type abstraction for supporting both binary classification and survival analysis.
This module provides a unified interface for different outcome types, enabling
the pipeline to switch between them with minimal code changes.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from utils import setup_logging

logger = setup_logging('outcome')


class OutcomeType(ABC):
    """
    Abstract base class for outcome types.
    Defines the interface that all outcome types must implement.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the outcome type."""
        pass
    
    @property
    @abstractmethod
    def cv_scoring_metric(self) -> str:
        """Return the primary scoring metric for cross-validation."""
        pass
    
    @abstractmethod
    def validate_outcome(self, df: pd.DataFrame, target_col: str, time_col: str = None) -> tuple:
        """
        Validate that the dataframe has valid outcome data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        target_col : str
            Name of the target column
        time_col : str, optional
            Name of the time column (used for survival analysis)
        
        Returns
        -------
        is_valid : bool
            Whether the outcome data is valid
        report : dict
            Validation report with details
        """
        pass
    
    @abstractmethod
    def extract_outcome(self, df: pd.DataFrame, target_col: str, time_col: str = None) -> dict:
        """
        Extract outcome variable(s) from dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        target_col : str
            Name of the target column
        time_col : str, optional
            Name of the time column (used for survival analysis)
        
        Returns
        -------
        outcome : dict
            Dictionary containing 'y' (and for survival: 'T', 'E')
        """
        pass
    
    @abstractmethod
    def get_cv_splitter(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        """
        Get the appropriate cross-validation splitter for this outcome type.
        
        Parameters
        ----------
        n_splits : int
            Number of CV folds
        shuffle : bool
            Whether to shuffle before splitting
        random_state : int
            Random seed
        
        Returns
        -------
        cv_splitter
            Configured CV splitter (StratifiedKFold or KFold)
        """
        pass
    
    @abstractmethod
    def get_evaluation_metrics(self) -> dict:
        """
        Get evaluation metrics appropriate for this outcome type.
        
        Returns
        -------
        metrics : dict
            Dictionary mapping metric name to sklearn scorer
        """
        pass


class BinaryOutcome(OutcomeType):
    """Outcome type for binary classification."""
    
    @property
    def name(self) -> str:
        return "binary_classification"
    
    @property
    def cv_scoring_metric(self) -> str:
        return "roc_auc"
    
    def validate_outcome(self, df: pd.DataFrame, target_col: str, time_col: str = None) -> tuple:
        """Validate binary target variable."""
        if target_col not in df.columns:
            return False, f"Target column '{target_col}' not found in dataframe"
        
        target = df[target_col]
        
        # Check for missing values
        if target.isnull().any():
            return False, f"Target column contains {target.isnull().sum()} missing values"
        
        # Check binary
        unique_vals = target.unique()
        if len(unique_vals) != 2:
            return False, f"Target is not binary. Found {len(unique_vals)} unique values: {unique_vals}"
        
        # Check values are 0/1 or can be interpreted as binary
        try:
            vals_numeric = pd.to_numeric(unique_vals)
            if not set(vals_numeric).issubset({0, 1}):
                # Allow other numeric pairs but warn
                logger.warning(f"Binary target contains values {unique_vals}, expected {{0, 1}}")
        except (ValueError, TypeError):
            return False, f"Target values {unique_vals} cannot be converted to numeric"
        
        # Count classes
        value_counts = target.value_counts()
        logger.info(f"Binary target distribution: {value_counts.to_dict()}")
        
        return True, {
            'valid': True,
            'n_classes': 2,
            'class_distribution': value_counts.to_dict(),
            'class_imbalance_ratio': value_counts.min() / value_counts.max()
        }
    
    def extract_outcome(self, df: pd.DataFrame, target_col: str, time_col: str = None) -> dict:
        """Extract binary target."""
        y = df[target_col].values.astype(int)
        return {'y': y}
    
    def get_cv_splitter(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        """Use stratified k-fold for balanced class distribution."""
        return StratifiedKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )
    
    def get_evaluation_metrics(self) -> dict:
        """Get classification metrics."""
        from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
        
        return {
            'roc_auc': make_scorer(roc_auc_score, needs_proba=True),
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, zero_division=0),
            'recall': make_scorer(recall_score, zero_division=0),
            'f1': make_scorer(f1_score, zero_division=0)
        }


class SurvivalOutcome(OutcomeType):
    """Outcome type for survival analysis."""
    
    @property
    def name(self) -> str:
        return "survival_analysis"
    
    @property
    def cv_scoring_metric(self) -> str:
        return "concordance_index"
    
    def validate_outcome(self, df: pd.DataFrame, target_col: str, time_col: str = None) -> tuple:
        """Validate survival outcome (time and event)."""
        from config import SURVIVAL_MINIMUM_TIME, SURVIVAL_TIME_EPSILON
        
        # Validate event column
        if target_col not in df.columns:
            return False, f"Event column '{target_col}' not found in dataframe"
        
        event = df[target_col]
        
        if event.isnull().any():
            return False, f"Event column contains {event.isnull().sum()} missing values"
        
        # Event should be binary (0/1)
        unique_events = event.unique()
        if len(unique_events) != 2:
            return False, f"Event column is not binary. Found values: {unique_events}"
        
        try:
            events_numeric = pd.to_numeric(unique_events)
            if not set(events_numeric).issubset({0, 1}):
                logger.warning(f"Event column contains values {unique_events}, expected {{0, 1}}")
        except (ValueError, TypeError):
            return False, f"Event values {unique_events} cannot be converted to numeric"
        
        # Validate time column
        if time_col is None:
            return False, "Time column is required for survival analysis"
        
        if time_col not in df.columns:
            return False, f"Time column '{time_col}' not found in dataframe"
        
        time = df[time_col]
        
        if time.isnull().any():
            return False, f"Time column contains {time.isnull().sum()} missing values"
        
        # Allow zero or near-zero times (standard in survival analysis)
        # If times <= SURVIVAL_MINIMUM_TIME are found, we'll add SURVIVAL_TIME_EPSILON in preprocessing
        n_zero_or_negative = (time <= 0).sum()
        if n_zero_or_negative > 0:
            logger.warning(
                f"Time column contains {n_zero_or_negative} zero or negative values. "
                f"These are valid in survival data (events at/near enrollment). "
                f"Will add epsilon={SURVIVAL_TIME_EPSILON} to ensure all times are positive."
            )
        
        # Get event counts
        event_counts = event.value_counts()
        n_events = int(event_counts.get(1, 0))
        n_censored = int(event_counts.get(0, 0))
        
        logger.info(f"Survival outcome: {n_events} events, {n_censored} censored observations")
        logger.info(f"Time range: [{time.min():.2f}, {time.max():.2f}]")
        
        return True, {
            'valid': True,
            'n_events': n_events,
            'n_censored': n_censored,
            'event_rate': n_events / len(event),
            'time_range': (float(time.min()), float(time.max())),
            'mean_follow_up': float(time.mean()),
            'n_zero_or_near_zero': int(n_zero_or_negative)
        }
    
    def extract_outcome(self, df: pd.DataFrame, target_col: str, time_col: str = None) -> dict:
        """Extract survival outcome (time and event), handling zero times."""
        from config import SURVIVAL_MINIMUM_TIME, SURVIVAL_TIME_EPSILON
        
        if time_col is None:
            raise ValueError("time_col must be specified for survival analysis")
        
        T = df[time_col].values.astype(float).copy()
        E = df[target_col].values.astype(int)
        
        # Handle zero or near-zero times by adding small epsilon
        # This is standard practice in survival analysis
        zero_time_mask = T <= SURVIVAL_MINIMUM_TIME
        n_adjusted = zero_time_mask.sum()
        if n_adjusted > 0:
            T[zero_time_mask] = T[zero_time_mask] + SURVIVAL_TIME_EPSILON
            logger.info(
                f"Added epsilon={SURVIVAL_TIME_EPSILON} to {n_adjusted} observations with "
                f"time <= {SURVIVAL_MINIMUM_TIME} to ensure positive times"
            )
        
        return {'T': T, 'E': E}
    
    def get_cv_splitter(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        """
        Use stratified k-fold based on event status to ensure balanced event distribution
        across folds.
        """
        return StratifiedKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )
    
    def get_evaluation_metrics(self) -> dict:
        """Get survival analysis metrics."""
        try:
            from sksurv.metrics import concordance_index_censored
        except ImportError:
            logger.warning("scikit-survival not installed; some metrics may be unavailable")
            return {}
        
        # Note: We'll use custom scoring functions that handle (T, E) tuples
        # These will be integrated into the training loop
        return {
            'concordance_index': 'concordance',  # String marker; will be handled specially
        }


def get_outcome_handler(outcome_type: str) -> OutcomeType:
    """
    Factory function to get the appropriate outcome handler.
    
    Parameters
    ----------
    outcome_type : str
        Type of outcome: 'binary' or 'survival'
    
    Returns
    -------
    handler : OutcomeType
        Configured outcome handler
    
    Raises
    ------
    ValueError
        If outcome_type is not recognized
    """
    outcome_types = {
        'binary': BinaryOutcome,
        'survival': SurvivalOutcome,
        'binary_classification': BinaryOutcome,
        'survival_analysis': SurvivalOutcome,
    }
    
    outcome_type_lower = outcome_type.lower().strip()
    
    if outcome_type_lower not in outcome_types:
        raise ValueError(
            f"Unknown outcome type '{outcome_type}'. "
            f"Available types: {list(outcome_types.keys())}"
        )
    
    handler_class = outcome_types[outcome_type_lower]
    logger.info(f"Using {handler_class().name} outcome handler")
    return handler_class()
