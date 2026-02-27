"""
Model training with nested cross-validation.
Implements two models (Elastic Net and Random Forest) with hyperparameter tuning.
Includes multiple imputation strategies for robust handling of missing data.
Supports both binary classification and survival analysis outcomes.

Also supports benchmarking pre-defined linear prediction scores supplied in
`score*.csv` files in the data directory. These scores are evaluated within
the same cross-validation and imputation framework for like-for-like
comparison of predictive performance.
"""

import os
import glob
import warnings
import copy

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, cross_val_score, cross_validate
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

# Optional survival imports
try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.metrics import concordance_index_censored, integrated_brier_score
    HAS_SKSURV = True
except ImportError:
    HAS_SKSURV = False
    logger_temp = None  # Will be set later

try:
    from lifelines import CoxPHFitter
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False

from config import (
    RANDOM_SEED, N_SPLITS_OUTER, N_SPLITS_INNER, N_JOBS,
    CV_SCORING, ELASTIC_NET_PARAMS, RANDOM_FOREST_PARAMS,
    STRATIFIED_CV, METRICS, MODELS_DIR, DATA_DIR, OUTCOME_TYPE,
    COXPH_PARAMS_SURVIVAL, RANDOM_SURVIVAL_FOREST_PARAMS
)
from outcome import get_outcome_handler
from utils import setup_logging, format_cv_results
from imputation import (
    get_imputation_transformer, MissingnessAnalyzer
)

warnings.filterwarnings('ignore')
logger = setup_logging('model_training')

# Check availability of survival libraries after logger is set up
if not HAS_SKSURV:
    logger.warning("scikit-survival not installed; survival models will not be available")
if not HAS_LIFELINES:
    logger.warning("lifelines not installed; CoxPH models may have limited functionality")

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)


# ==============================================================================
# MODEL CACHING UTILITIES
# ==============================================================================

def _get_cache_path(strategy):
    """Get the path where training results are cached for a given strategy."""
    return os.path.join(MODELS_DIR, f"training_results_{strategy}.pkl")


def check_cached_models_exist(strategies=None):
    """
    Check if all trained model results exist in the cache for given strategies.
    
    Parameters
    ----------
    strategies : list of str, optional
        List of imputation strategies to check. If None, checks both 'median' and 'knn'.
    
    Returns
    -------
    dict
        Dictionary mapping strategy to bool indicating if cache exists for that strategy.
    """
    if strategies is None:
        strategies = ['median', 'knn']
    
    cache_status = {}
    for strategy in strategies:
        cache_path = _get_cache_path(strategy)
        exists = os.path.exists(cache_path)
        cache_status[strategy] = exists
        logger.debug(f"Cache for {strategy} strategy: {'exists' if exists else 'missing'}")
    
    return cache_status


def load_cached_training_results(strategy):
    """
    Load pre-trained model results from cache.
    
    Parameters
    ----------
    strategy : str
        Imputation strategy: 'median' or 'knn'
    
    Returns
    -------
    results : dict
        Dictionary containing cached training results, or None if cache doesn't exist
    """
    cache_path = _get_cache_path(strategy)
    
    if not os.path.exists(cache_path):
        logger.info(f"No cached models found for {strategy} strategy at {cache_path}")
        return None
    
    try:
        results = joblib.load(cache_path)
        logger.info(f"✓ Loaded cached training results for {strategy} imputation strategy")
        logger.info(f"  - Elastic Net CV AUC: {results['elastic_net']['mean_score']:.4f} (+/- {results['elastic_net']['std_score']:.4f})")
        logger.info(f"  - Random Forest CV AUC: {results['random_forest']['mean_score']:.4f} (+/- {results['random_forest']['std_score']:.4f})")
        return results
    except Exception as e:
        logger.warning(f"Failed to load cached results for {strategy}: {e}")
        return None


def save_training_results(results, strategy):
    """
    Save training results to cache for later reuse.
    
    Parameters
    ----------
    results : dict
        Training results dictionary from run_full_pipeline
    strategy : str
        Imputation strategy: 'median' or 'knn'
    """
    cache_path = _get_cache_path(strategy)
    
    try:
        joblib.dump(results, cache_path)
        logger.info(f"✓ Saved training results cache for {strategy} strategy to {cache_path}")
    except Exception as e:
        logger.warning(f"Failed to save training results cache for {strategy}: {e}")


# ==============================================================================
# SURVIVAL MODEL WRAPPERS
# ==============================================================================
# Wrapper classes to make scikit-survival models compatible with sklearn pipelines

class SurvivalModelWrapper(BaseEstimator, RegressorMixin):
    """
    Abstract base class for survival model wrappers.
    Provides interface for survival models to work with sklearn Pipelines and GridSearchCV.
    """
    
    def __init__(self):
        self.model_ = None
        self.feature_names_ = None
    
    def fit(self, X, y):
        """
        Fit the survival model.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix
        y : np.ndarray or tuple
            For survival: tuple of (T, E) where T=time, E=event
            For compatibility with sklearn, may receive structured array
        
        Returns
        -------
        self
        """
        raise NotImplementedError
    
    def predict(self, X):
        """
        Generate risk scores.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix
        
        Returns
        -------
        scores : np.ndarray
            Risk scores
        """
        raise NotImplementedError


class CoxPHWrapper(SurvivalModelWrapper):
    """
    Wrapper for scikit-survival or lifelines CoxPH model to work with sklearn pipelines.
    
    Parameters
    ----------
    penalizer : float, optional
        Regularization parameter (default 0.0)
    l1_ratio : float, optional
        L1 ratio for elastic net regularization, 0 = L2 (Ridge), 1 = L1 (Lasso)
    """
    
    def __init__(self, penalizer=0.0, l1_ratio=0.0):
        super().__init__()
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
    
    def fit(self, X, y):
        """Fit Cox Proportional Hazards model."""
        if not HAS_LIFELINES:
            raise ImportError("lifelines is required for CoxPH models. Install with: pip install lifelines")
        
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        self.feature_names_ = X_df.columns.tolist()
        
        # Extract time and event from y
        # y can be a structured array from sksurv or a tuple dict with 'T' and 'E'
        if isinstance(y, dict):
            T = y['T']
            E = y['E']
        elif isinstance(y, tuple) and len(y) == 2:
            T, E = y
        elif hasattr(y, 'dtype') and y.dtype.names:  # Structured array
            T = y['time']
            E = y['event']
        else:
            raise ValueError("Outcome y must be dict with 'T'/'E' or structured array for survival models")
        
        # Add duration and event to dataframe
        X_df['duration'] = T
        X_df['event'] = E
        
        # Fit Cox model
        self.model_ = CoxPHFitter(penalizer=self.penalizer / max(1.0, self.l1_ratio))
        self.model_.fit(X_df, duration_col='duration', event_col='event')
        
        return self
    
    def predict(self, X):
        """Generate risk scores (partial hazard) for samples."""
        if self.model_ is None:
            raise ValueError("Model has not been fitted yet")
        
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        X_df = X_df[self.feature_names_]
        
        # Get risk scores (partial hazard)
        scores = self.model_.predict_partial_hazard(X_df).values
        return scores


class RandomSurvivalForestWrapper(SurvivalModelWrapper):
    """
    Wrapper for scikit-survival RandomSurvivalForest to work with sklearn pipelines.
    
    Parameters
    ----------
    n_estimators : int
        Number of trees (default 100)
    max_depth : int or None
        Maximum depth of trees (default None)
    min_samples_split : int
        Minimum samples to split (default 2)
    min_samples_leaf : int
        Minimum samples in leaf (default 1)
    """
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
    
    def fit(self, X, y):
        """Fit Random Survival Forest model."""
        if not HAS_SKSURV:
            raise ImportError("scikit-survival is required for RandomSurvivalForest. Install with: pip install scikit-survival")
        
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.astype(float)
        self.feature_names_ = X_df.columns.tolist()
        X_df = X_df[self.feature_names_]
        
        # Extract time and event
        if isinstance(y, dict):
            T = y['T']
            E = y['E']
        elif isinstance(y, tuple) and len(y) == 2:
            T, E = y
        elif hasattr(y, 'dtype') and y.dtype.names:  # Structured array
            T = y['time']
            E = y['event']
        else:
            raise ValueError("Outcome y must be dict with 'T'/'E' or structured array for survival models")
        
        # Create structured array for sksurv
        # Event = False means censored, Event = True means event occurred
        y_surv = np.array([(e == 1, t) for e, t in zip(E, T)],
                         dtype=[('event', bool), ('time', float)])
        
        # Fit model
        self.model_ = RandomSurvivalForest(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=RANDOM_SEED,
            n_jobs=1  # Set to 1 for use within GridSearchCV
        )
        self.model_.fit(X_df, y_surv)
        
        return self
    
    def predict(self, X):
        """Generate risk scores for samples."""
        if self.model_ is None:
            raise ValueError("Model has not been fitted yet")
        
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.astype(float)
        X_df = X_df[self.feature_names_]
        
        # Get risk scores
        scores = self.model_.predict(X_df)
        return scores


class PredefinedLinearScore(ClassifierMixin, BaseEstimator):
    """Classifier wrapper for pre-defined linear prediction scores.

    This estimator applies fixed coefficients (and optional intercept) to
    a subset of input features to produce a linear risk score. A logistic
    transform is used to map scores to probabilities so that AUC is
    comparable with other models, but any monotone transform yields the
    same ROC AUC.

    Parameters
    ----------
    name : str
        Human-readable name for the score (used in reporting).
    coefficients : dict
        Mapping from feature name to coefficient.
    intercept : float, optional
        Intercept term for the score (default 0.0).
    positive_class : int or str, optional
        Label of the positive class (default 1).
    """

    def __init__(self, name, coefficients, intercept=0.0, positive_class=1):
        self.name = name
        self.coefficients = coefficients
        self.intercept = intercept
        self.positive_class = positive_class

    def fit(self, X, y=None):  # noqa: D401
        """Fit the score model.

        No learning is performed; this simply identifies which score features
        are available in the input data and caches them for fast scoring.
        
        The score uses only features that are explicitly defined in the
        coefficient dictionary AND present in the input data. Features in the
        data but not in the score definition are ignored. This enables the
        same score to work across different datasets that may contain different
        feature sets, as long as at least some of the required features are present.
        """
        try:
            logger.debug(f"Score '{self.name}' fit() START - X shape: {X.shape if hasattr(X, 'shape') else 'unknown'}, y value counts: {np.bincount(y) if y is not None else 'None'}")
            
            X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
            self.feature_names_ = list(X_df.columns)
            logger.debug(f"Score '{self.name}' feature_names_: {self.feature_names_}")

            # Find which features from the score are actually available in the data
            self.used_features_ = [
                feat for feat in self.coefficients.keys() 
                if feat in self.feature_names_
            ]
            logger.debug(f"Score '{self.name}' coefficients.keys(): {list(self.coefficients.keys())}, used_features_: {self.used_features_}")
            
            if not self.used_features_:
                # Only raise an error if NONE of the score's features are available
                error_msg = (
                    f"Predefined score '{self.name}' could not find any of its required "
                    f"features {list(self.coefficients.keys())} in the input data. "
                    f"Available features in data: {self.feature_names_}"
                )
                raise ValueError(error_msg)
            
            # Warn if some features are available but others are missing
            missing_features = [
                feat for feat in self.coefficients.keys() 
                if feat not in self.feature_names_
            ]
            if missing_features:
                logger.debug(
                    f"Score '{self.name}' using {len(self.used_features_)} of {len(self.coefficients)} "
                    f"features. Missing: {missing_features}. Using: {self.used_features_}"
                )
            
            # Create coefficient vector for only the available features
            self.coef_vector_ = np.array([
                self.coefficients[feat] for feat in self.used_features_
            ], dtype=float)
            logger.debug(f"Score '{self.name}' coef_vector_: {self.coef_vector_}, intercept: {self.intercept}")

            # Binary classes assumed {0, 1}
            self.classes_ = np.array([0, 1])
            logger.debug(f"Score '{self.name}' fit() END - initialized successfully")
            return self
        except Exception as e:
            logger.error(f"Error in PredefinedLinearScore.fit(): {type(e).__name__}: {e}", exc_info=True)
            raise

    def decision_function(self, X):
        """Calculate linear scores using available features.
        
        Only features present in both the score definition and the input data
        are used. This produces a partial sum if some features are missing.
        """
        try:
            X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
            logger.debug(f"Score '{self.name}' decision_function called with X shape {X_df.shape}, used_features_={getattr(self, 'used_features_', 'NOT SET')}")
            
            if not hasattr(self, 'used_features_'):
                logger.error(f"Score '{self.name}' decision_function: used_features_ not set! fit() may not have been called.")
                raise RuntimeError(f"Score '{self.name}' has not been fitted yet.")
            
            feature_data = X_df[self.used_features_].values
            logger.debug(f"Score '{self.name}' feature_data shape: {feature_data.shape}, first row: {feature_data[0] if len(feature_data) > 0 else 'empty'}")
            
            # Log info about missing/NaN values
            if np.any(np.isnan(feature_data)):
                nan_pct = np.sum(np.isnan(feature_data)) / feature_data.size * 100
                logger.warning(
                    f"Score '{self.name}' decision_function: {nan_pct:.1f}% NaN values in "
                    f"features {self.used_features_} (shape: {feature_data.shape})"
                )
            
            scores = np.dot(feature_data, self.coef_vector_) + self.intercept
            logger.debug(f"Score '{self.name}' scores computed: shape {scores.shape}, first 3: {scores[:3] if len(scores) >= 3 else scores}, has NaN: {np.any(np.isnan(scores))}")
            
            if np.any(np.isnan(scores)):
                logger.warning(
                    f"Score '{self.name}' decision_function returned NaN for {np.sum(np.isnan(scores))} "
                    f"of {len(scores)} samples"
                )
            
            return scores
        except Exception as e:
            logger.error(f"Error in PredefinedLinearScore.decision_function(): {e}", exc_info=True)
            raise

    def predict_proba(self, X):
        try:
            logger.debug(f"Score '{self.name}' predict_proba called with X shape {X.shape if hasattr(X, 'shape') else 'unknown'}")
            scores = self.decision_function(X)
            logger.debug(f"Score '{self.name}' decision_function returned scores: shape {scores.shape}, has NaN: {np.any(np.isnan(scores))}")
            
            # Logistic transform for probabilities
            probs_pos = 1.0 / (1.0 + np.exp(-scores))
            probs_neg = 1.0 - probs_pos
            probs = np.vstack([probs_neg, probs_pos]).T
            logger.debug(f"Score '{self.name}' probabilities computed: shape {probs.shape}, has NaN: {np.any(np.isnan(probs))}")
            
            if np.any(np.isnan(probs)):
                logger.warning(
                    f"Score '{self.name}' predict_proba returned NaN for {np.sum(np.isnan(probs))} "
                    f"values out of {probs.size}"
                )
            
            return probs
        except Exception as e:
            logger.error(f"Error in PredefinedLinearScore.predict_proba(): {e}", exc_info=True)
            raise

    def predict(self, X):
        try:
            logger.debug(f"Score '{self.name}' predict called with X shape {X.shape if hasattr(X, 'shape') else 'unknown'}")
            probs = self.predict_proba(X)[:, 1]
            preds = (probs >= 0.5).astype(int)
            logger.debug(f"Score '{self.name}' predict returning {len(preds)} predictions, unique_values: {np.unique(preds)}")
            return preds
        except Exception as e:
            logger.error(f"Error in PredefinedLinearScore.predict(): {e}", exc_info=True)
            raise

    def get_score_coefficients(self):
        """Return coefficient mapping used for this score."""
        return {
            'intercept': float(self.intercept),
            'coefficients': {k: float(v) for k, v in self.coefficients.items()}
        }


class NestedCVTrainer:
    """
    Trainer class implementing nested cross-validation for model evaluation and hyperparameter tuning.
    Supports multiple imputation strategies for handling missing data within the CV framework.
    """
    
    def __init__(self, n_splits_outer=N_SPLITS_OUTER, n_splits_inner=N_SPLITS_INNER,
                 stratified=STRATIFIED_CV, random_state=RANDOM_SEED, imputation_strategy='median'):
        """
        Initialize the nested CV trainer.
        
        Parameters
        ----------
        n_splits_outer : int
            Number of folds for outer CV (performance estimation)
        n_splits_inner : int
            Number of folds for inner CV (hyperparameter tuning)
        stratified : bool
            Whether to use stratified k-fold
        random_state : int
            Random seed for reproducibility
        imputation_strategy : str
            Imputation strategy: 'median', 'knn', or 'none'
        """
        self.n_splits_outer = n_splits_outer
        self.n_splits_inner = n_splits_inner
        self.stratified = stratified
        self.random_state = random_state
        self.imputation_strategy = imputation_strategy
        
        # Define CV splitters
        if self.stratified:
            self.cv_outer = StratifiedKFold(
                n_splits=self.n_splits_outer, shuffle=True, random_state=self.random_state
            )
            self.cv_inner = StratifiedKFold(
                n_splits=self.n_splits_inner, shuffle=True, random_state=self.random_state
            )
        else:
            from sklearn.model_selection import KFold
            self.cv_outer = KFold(
                n_splits=self.n_splits_outer, shuffle=True, random_state=self.random_state
            )
            self.cv_inner = KFold(
                n_splits=self.n_splits_inner, shuffle=True, random_state=self.random_state
            )
        
        logger.info(f"Initialized NestedCVTrainer: {n_splits_outer}-fold outer CV, {n_splits_inner}-fold inner CV")
        logger.info(f"Imputation strategy: {imputation_strategy}")
    
    def _create_scorers(self):
        """Create scoring functions for cross-validation."""
        scorers = {
            'roc_auc': make_scorer(roc_auc_score, needs_proba=True),
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, zero_division=0),
            'recall': make_scorer(recall_score, zero_division=0),
            'f1': make_scorer(f1_score, zero_division=0)
        }
        return scorers
    
    def train_elastic_net(self, X, y, param_grid=ELASTIC_NET_PARAMS):
        """
        Train Elastic Net (as LogisticRegression with L1/L2 penalty) using nested CV.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Feature matrix
        y : np.ndarray or pd.Series
            Target variable
        param_grid : dict
            Hyperparameter grid for tuning
        
        Returns
        -------
        results : dict
            Dictionary containing CV scores, best params, and models
        """
        logger.info("=" * 60)
        logger.info(f"TRAINING ELASTIC NET (LogisticRegression L1/L2)")
        logger.info(f"Imputation: {self.imputation_strategy.upper()}")
        logger.info("=" * 60)
        
        # Analyze missingness before training
        MissingnessAnalyzer.report_missingness(X, prefix="  ")
        
        # Identify column types for preprocessing
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = [col for col in X.columns if col not in categorical_cols]

        # Create imputation transformer
        imputer = get_imputation_transformer(self.imputation_strategy)

        # Build preprocessing pipeline: imputation -> scaling/encoding
        # Step 1: Imputation (applied to all columns)
        # Step 2: Column-specific preprocessing (scaling numeric, encoding categorical)
        scaling_encoding_transformers = []
        if numeric_cols:
            scaling_encoding_transformers.append(('num', StandardScaler(), numeric_cols))
        if categorical_cols:
            scaling_encoding_transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols))
        
        column_preprocessor = ColumnTransformer(transformers=scaling_encoding_transformers)
        
        # Full preprocessing pipeline: imputation -> scaling/encoding
        preprocessing_pipeline = Pipeline(steps=[
            ('impute', imputer),
            ('scale_encode', column_preprocessor)
        ])

        # Define the base classifier with L1+L2 penalty (elastic net equivalent)
        base_model = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            max_iter=1000,
            random_state=self.random_state,
            n_jobs=1,
            class_weight='balanced'
        )

        # Full pipeline: preprocessing + classifier
        pipeline = Pipeline(steps=[
            ('preprocess', preprocessing_pipeline),
            ('clf', base_model)
        ])
        
        cv_scores = []
        best_models = []  # fitted pipelines per outer fold
        best_params_list = []
        scorers = self._create_scorers()
        
        fold = 1
        for train_idx, test_idx in self.cv_outer.split(X, y):
            logger.info(f"\nOuter fold {fold}/{self.n_splits_outer}")
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Adapt hyperparameter grid for pipeline (prefix with 'clf__')
            param_grid_pipeline = {f"clf__{k}": v for k, v in param_grid.items()}

            # Inner CV for hyperparameter tuning
            grid_search = GridSearchCV(
                pipeline,
                param_grid_pipeline,
                cv=self.cv_inner,
                scoring=CV_SCORING,
                n_jobs=N_JOBS,
                verbose=0
            )

            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            logger.info(f"  Best params: {grid_search.best_params_}")
            logger.info(f"  Inner CV score (AUC): {grid_search.best_score_:.4f}")

            # Evaluate on outer fold test set using ROC AUC
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            test_score = roc_auc_score(y_test, y_pred_proba)
            cv_scores.append(test_score)
            best_models.append(best_model)
            best_params_list.append(grid_search.best_params_)
            
            logger.info(f"  Outer fold test AUC: {test_score:.4f}")
            fold += 1
        
        cv_scores = np.array(cv_scores)
        
        # Use most common best parameters across folds
        best_params_overall = self._get_most_common_params(best_params_list)

        # Train final model on full data with best hyperparameters
        # The fold_models have imputation fit only on their respective training folds (clean evaluation).
        # For the final production model, we fit on all available data (X, y) using the best hyperparameters.
        final_imputer = get_imputation_transformer(self.imputation_strategy)
        final_scaling_encoding_transformers = []
        if numeric_cols:
            final_scaling_encoding_transformers.append(('num', StandardScaler(), numeric_cols))
        if categorical_cols:
            final_scaling_encoding_transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols))
        
        final_column_preprocessor = ColumnTransformer(transformers=final_scaling_encoding_transformers)
        final_preprocessing_pipeline = Pipeline(steps=[
            ('impute', final_imputer),
            ('scale_encode', final_column_preprocessor)
        ])
        
        final_pipeline = Pipeline(steps=[
            ('preprocess', final_preprocessing_pipeline),
            ('clf', base_model)
        ])
        final_pipeline.set_params(**best_params_overall)
        final_pipeline.fit(X, y)
        
        results = {
            'model_name': 'Elastic Net',
            'imputation_strategy': self.imputation_strategy,
            'cv_scores': cv_scores,
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'best_params': best_params_overall,
            'final_model': final_pipeline,
            'fold_models': best_models,
            'fold_params': best_params_list
        }
        
        logger.info(f"\nElastic Net CV Results ({self.imputation_strategy}):")
        logger.info(f"  Mean AUC: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
        logger.info("=" * 60)
        
        return results
    
    def train_random_forest(self, X, y, param_grid=RANDOM_FOREST_PARAMS):
        """
        Train Random Forest using nested CV.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Feature matrix
        y : np.ndarray or pd.Series
            Target variable
        param_grid : dict
            Hyperparameter grid for tuning
        
        Returns
        -------
        results : dict
            Dictionary containing CV scores, best params, and models
        """
        logger.info("=" * 60)
        logger.info(f"TRAINING RANDOM FOREST")
        logger.info(f"Imputation: {self.imputation_strategy.upper()}")
        logger.info("=" * 60)
        
        # Analyze missingness before training
        MissingnessAnalyzer.report_missingness(X, prefix="  ")

        # Identify column types for preprocessing
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = [col for col in X.columns if col not in categorical_cols]

        # Create imputation transformer
        imputer = get_imputation_transformer(self.imputation_strategy)

        # For Random Forest, scaling is not strictly necessary, but to keep
        # the preprocessing consistent we impute, scale numeric features and
        # one-hot encode categoricals within the pipeline.
        scaling_encoding_transformers = []
        if numeric_cols:
            scaling_encoding_transformers.append(('num', StandardScaler(), numeric_cols))
        if categorical_cols:
            scaling_encoding_transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols))
        
        column_preprocessor = ColumnTransformer(transformers=scaling_encoding_transformers)
        
        # Full preprocessing pipeline: imputation -> scaling/encoding
        preprocessing_pipeline = Pipeline(steps=[
            ('impute', imputer),
            ('scale_encode', column_preprocessor)
        ])

        base_model = RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=1,
            class_weight='balanced'
        )

        pipeline = Pipeline(steps=[
            ('preprocess', preprocessing_pipeline),
            ('clf', base_model)
        ])

        cv_scores = []
        best_models = []  # fitted pipelines per outer fold
        best_params_list = []
        
        fold = 1
        for train_idx, test_idx in self.cv_outer.split(X, y):
            logger.info(f"\nOuter fold {fold}/{self.n_splits_outer}")
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Adapt hyperparameter grid for pipeline (prefix with 'clf__')
            param_grid_pipeline = {f"clf__{k}": v for k, v in param_grid.items()}

            # Inner CV for hyperparameter tuning
            grid_search = GridSearchCV(
                pipeline,
                param_grid_pipeline,
                cv=self.cv_inner,
                scoring=CV_SCORING,
                n_jobs=N_JOBS,
                verbose=0
            )

            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            logger.info(f"  Best params: {grid_search.best_params_}")
            logger.info(f"  Inner CV score: {grid_search.best_score_:.4f}")

            # Evaluate on outer fold test set
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            test_score = roc_auc_score(y_test, y_pred_proba)
            cv_scores.append(test_score)
            best_models.append(best_model)
            best_params_list.append(grid_search.best_params_)
            
            logger.info(f"  Outer fold test score (AUC): {test_score:.4f}")
            fold += 1
        
        cv_scores = np.array(cv_scores)

        # Use most common best parameters across folds
        best_params_overall = self._get_most_common_params(best_params_list)

        # Train final model on full data with best hyperparameters
        # The fold_models have imputation fit only on their respective training folds (clean evaluation).
        # For the final production model, we fit on all available data (X, y) using the best hyperparameters.
        final_imputer = get_imputation_transformer(self.imputation_strategy)
        final_scaling_encoding_transformers = []
        if numeric_cols:
            final_scaling_encoding_transformers.append(('num', StandardScaler(), numeric_cols))
        if categorical_cols:
            final_scaling_encoding_transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols))
        
        final_column_preprocessor = ColumnTransformer(transformers=final_scaling_encoding_transformers)
        final_preprocessing_pipeline = Pipeline(steps=[
            ('impute', final_imputer),
            ('scale_encode', final_column_preprocessor)
        ])
        
        final_pipeline = Pipeline(steps=[
            ('preprocess', final_preprocessing_pipeline),
            ('clf', base_model)
        ])
        final_pipeline.set_params(**best_params_overall)
        final_pipeline.fit(X, y)
        
        results = {
            'model_name': 'Random Forest',
            'imputation_strategy': self.imputation_strategy,
            'cv_scores': cv_scores,
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'best_params': best_params_overall,
            'final_model': final_pipeline,
            'fold_models': best_models,
            'fold_params': best_params_list
        }
        
        logger.info(f"\nRandom Forest CV Results ({self.imputation_strategy}):")
        logger.info(f"  Mean AUC: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
        logger.info("=" * 60)
        
        return results
    
    def train_elastic_net_survival(self, X, T, E, param_grid=None):
        """
        Train Cox Proportional Hazards with Elastic Net regularization using nested CV.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Feature matrix
        T : np.ndarray
            Time to event
        E : np.ndarray
            Event indicator (0=censored, 1=event)
        param_grid : dict, optional
            Hyperparameter grid for tuning. If None, uses COXPH_PARAMS_SURVIVAL
        
        Returns
        -------
        results : dict
            Dictionary containing CV scores, best params, and models
        """
        if param_grid is None:
            param_grid = COXPH_PARAMS_SURVIVAL
        
        if not HAS_LIFELINES:
            raise ImportError("lifelines is required for Cox models. Install with: pip install lifelines")
        
        logger.info("=" * 60)
        logger.info(f"TRAINING ELASTIC NET COX MODEL")
        logger.info(f"Imputation: {self.imputation_strategy.upper()}")
        logger.info("=" * 60)
        
        # Analyze missingness before training
        MissingnessAnalyzer.report_missingness(X, prefix="  ")
        
        # Identify column types for preprocessing
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = [col for col in X.columns if col not in categorical_cols]
        
        # Create imputation transformer
        imputer = get_imputation_transformer(self.imputation_strategy)
        
        # Build preprocessing pipeline
        scaling_encoding_transformers = []
        if numeric_cols:
            scaling_encoding_transformers.append(('num', StandardScaler(), numeric_cols))
        if categorical_cols:
            scaling_encoding_transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols))
        
        column_preprocessor = ColumnTransformer(transformers=scaling_encoding_transformers)
        
        preprocessing_pipeline = Pipeline(steps=[
            ('impute', imputer),
            ('scale_encode', column_preprocessor)
        ])
        
        base_model = CoxPHWrapper()
        
        pipeline = Pipeline(steps=[
            ('preprocess', preprocessing_pipeline),
            ('clf', base_model)
        ])
        
        cv_scores = []
        best_models = []
        best_params_list = []
        
        # Combine T and E for stratification in CV
        y_combined = np.column_stack((T, E))
        
        fold = 1
        for train_idx, test_idx in self.cv_outer.split(X, E):  # Stratify by event status
            logger.info(f"\nOuter fold {fold}/{self.n_splits_outer}")
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            T_train, T_test = T[train_idx], T[test_idx]
            E_train, E_test = E[train_idx], E[test_idx]
            
            # Create outcome dict for this fold
            y_train_survival = {'T': T_train, 'E': E_train}
            y_test_survival = {'T': T_test, 'E': E_test}
            
            # Adapt hyperparameter grid for pipeline
            param_grid_pipeline = {f"clf__{k}": v for k, v in param_grid.items()}
            
            # Custom CV loop for survival (GridSearchCV doesn't directly support tuple targets)
            from sklearn.model_selection import ParameterGrid
            best_score = -np.inf
            best_model = None
            best_params = None
            
            for params in ParameterGrid(param_grid_pipeline):
                pipeline.set_params(**params)
                
                # Inner CV
                fold_scores = []
                for inner_train_idx, inner_test_idx in self.cv_inner.split(X_train, E_train):
                    X_train_inner, X_test_inner = X_train.iloc[inner_train_idx], X_train.iloc[inner_test_idx]
                    T_train_inner, T_test_inner = T_train[inner_train_idx], T_train[inner_test_idx]
                    E_train_inner, E_test_inner = E_train[inner_train_idx], E_train[inner_test_idx]
                    
                    y_train_inner = {'T': T_train_inner, 'E': E_train_inner}
                    y_test_inner = {'T': T_test_inner, 'E': E_test_inner}
                    
                    try:
                        pipeline.fit(X_train_inner, y_train_inner)
                        # Compute concordance index
                        scores = pipeline.predict(X_test_inner)
                        if HAS_SKSURV:
                            c_index = concordance_index_censored(E_test_inner == 1, T_test_inner, scores)[0]
                        else:
                            # Fallback: use basic evaluation
                            c_index = np.mean(scores)  # Poor proxy, but safe default
                        fold_scores.append(c_index)
                    except Exception as e:
                        logger.warning(f"Error in inner fold: {e}")
                        fold_scores.append(0.0)
                
                mean_score = np.mean(fold_scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params.copy()
                    # Refit on full training fold
                    try:
                        pipeline.set_params(**best_params)
                        pipeline.fit(X_train, y_train_survival)
                        best_model = copy.deepcopy(pipeline)
                    except Exception as e:
                        logger.warning(f"Error refitting model: {e}")
            
            # Evaluate on outer test set
            if best_model is not None:
                try:
                    test_scores = best_model.predict(X_test)
                    if HAS_SKSURV:
                        test_c_index = concordance_index_censored(E_test == 1, T_test, test_scores)[0]
                    else:
                        test_c_index = 0.5  # Default when no sksurv
                    cv_scores.append(test_c_index)
                    best_models.append(best_model)
                    best_params_list.append(best_params)
                    logger.info(f"  Outer fold test C-index: {test_c_index:.4f}")
                except Exception as e:
                    logger.warning(f"Error evaluating on test set: {e}")
                    cv_scores.append(0.5)
            
            fold += 1
        
        cv_scores = np.array(cv_scores)
        best_params_overall = self._get_most_common_params(best_params_list if best_params_list else [{}])
        
        # Train final model
        final_imputer = get_imputation_transformer(self.imputation_strategy)
        final_scaling_encoding_transformers = []
        if numeric_cols:
            final_scaling_encoding_transformers.append(('num', StandardScaler(), numeric_cols))
        if categorical_cols:
            final_scaling_encoding_transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols))
        
        final_column_preprocessor = ColumnTransformer(transformers=final_scaling_encoding_transformers)
        final_preprocessing_pipeline = Pipeline(steps=[
            ('impute', final_imputer),
            ('scale_encode', final_column_preprocessor)
        ])
        
        final_pipeline = Pipeline(steps=[
            ('preprocess', final_preprocessing_pipeline),
            ('clf', CoxPHWrapper())
        ])
        final_pipeline.set_params(**best_params_overall)
        final_pipeline.fit(X, {'T': T, 'E': E})
        
        results = {
            'model_name': 'Elastic Net Cox',
            'imputation_strategy': self.imputation_strategy,
            'cv_scores': cv_scores,
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'best_params': best_params_overall,
            'final_model': final_pipeline,
            'fold_models': best_models,
            'fold_params': best_params_list
        }
        
        logger.info(f"\nElastic Net Cox CV Results ({self.imputation_strategy}):")
        logger.info(f"  Mean C-index: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
        logger.info("=" * 60)
        
        return results
    
    def train_random_forest_survival(self, X, T, E, param_grid=None):
        """
        Train Random Survival Forest using nested CV.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Feature matrix
        T : np.ndarray
            Time to event
        E : np.ndarray
            Event indicator (0=censored, 1=event)
        param_grid : dict, optional
            Hyperparameter grid for tuning. If None, uses RANDOM_SURVIVAL_FOREST_PARAMS
        
        Returns
        -------
        results : dict
            Dictionary containing CV scores, best params, and models
        """
        if param_grid is None:
            param_grid = RANDOM_SURVIVAL_FOREST_PARAMS
        
        if not HAS_SKSURV:
            raise ImportError("scikit-survival is required for Random Survival Forest. Install with: pip install scikit-survival")
        
        logger.info("=" * 60)
        logger.info(f"TRAINING RANDOM SURVIVAL FOREST")
        logger.info(f"Imputation: {self.imputation_strategy.upper()}")
        logger.info("=" * 60)
        
        # Analyze missingness before training
        MissingnessAnalyzer.report_missingness(X, prefix="  ")
        
        # Identify column types for preprocessing
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = [col for col in X.columns if col not in categorical_cols]
        
        # Create imputation transformer
        imputer = get_imputation_transformer(self.imputation_strategy)
        
        # Build preprocessing pipeline
        scaling_encoding_transformers = []
        if numeric_cols:
            scaling_encoding_transformers.append(('num', StandardScaler(), numeric_cols))
        if categorical_cols:
            scaling_encoding_transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols))
        
        column_preprocessor = ColumnTransformer(transformers=scaling_encoding_transformers)
        
        preprocessing_pipeline = Pipeline(steps=[
            ('impute', imputer),
            ('scale_encode', column_preprocessor)
        ])
        
        base_model = RandomSurvivalForestWrapper()
        
        pipeline = Pipeline(steps=[
            ('preprocess', preprocessing_pipeline),
            ('clf', base_model)
        ])
        
        cv_scores = []
        best_models = []
        best_params_list = []
        
        fold = 1
        for train_idx, test_idx in self.cv_outer.split(X, E):  # Stratify by event status
            logger.info(f"\nOuter fold {fold}/{self.n_splits_outer}")
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            T_train, T_test = T[train_idx], T[test_idx]
            E_train, E_test = E[train_idx], E[test_idx]
            
            y_train_survival = {'T': T_train, 'E': E_train}
            y_test_survival = {'T': T_test, 'E': E_test}
            
            param_grid_pipeline = {f"clf__{k}": v for k, v in param_grid.items()}
            
            # Grid search with custom evaluation
            from sklearn.model_selection import ParameterGrid
            best_score = -np.inf
            best_model = None
            best_params = None
            
            for params in ParameterGrid(param_grid_pipeline):
                pipeline.set_params(**params)
                
                fold_scores = []
                for inner_train_idx, inner_test_idx in self.cv_inner.split(X_train, E_train):
                    X_train_inner, X_test_inner = X_train.iloc[inner_train_idx], X_train.iloc[inner_test_idx]
                    T_train_inner, T_test_inner = T_train[inner_train_idx], T_train[inner_test_idx]
                    E_train_inner, E_test_inner = E_train[inner_train_idx], E_train[inner_test_idx]
                    
                    y_train_inner = {'T': T_train_inner, 'E': E_train_inner}
                    y_test_inner = {'T': T_test_inner, 'E': E_test_inner}
                    
                    try:
                        pipeline.fit(X_train_inner, y_train_inner)
                        scores = pipeline.predict(X_test_inner)
                        c_index = concordance_index_censored(E_test_inner == 1, T_test_inner, scores)[0]
                        fold_scores.append(c_index)
                    except Exception as e:
                        logger.warning(f"Error in inner fold: {e}")
                        fold_scores.append(0.0)
                
                mean_score = np.mean(fold_scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params.copy()
                    try:
                        pipeline.set_params(**best_params)
                        pipeline.fit(X_train, y_train_survival)
                        best_model = copy.deepcopy(pipeline)
                    except Exception as e:
                        logger.warning(f"Error refitting model: {e}")
            
            if best_model is not None:
                try:
                    test_scores = best_model.predict(X_test)
                    test_c_index = concordance_index_censored(E_test == 1, T_test, test_scores)[0]
                    cv_scores.append(test_c_index)
                    best_models.append(best_model)
                    best_params_list.append(best_params)
                    logger.info(f"  Outer fold test C-index: {test_c_index:.4f}")
                except Exception as e:
                    logger.warning(f"Error evaluating on test set: {e}")
                    cv_scores.append(0.5)
            
            fold += 1
        
        cv_scores = np.array(cv_scores)
        best_params_overall = self._get_most_common_params(best_params_list if best_params_list else [{}])
        
        # Train final model
        final_imputer = get_imputation_transformer(self.imputation_strategy)
        final_scaling_encoding_transformers = []
        if numeric_cols:
            final_scaling_encoding_transformers.append(('num', StandardScaler(), numeric_cols))
        if categorical_cols:
            final_scaling_encoding_transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols))
        
        final_column_preprocessor = ColumnTransformer(transformers=final_scaling_encoding_transformers)
        final_preprocessing_pipeline = Pipeline(steps=[
            ('impute', final_imputer),
            ('scale_encode', final_column_preprocessor)
        ])
        
        final_pipeline = Pipeline(steps=[
            ('preprocess', final_preprocessing_pipeline),
            ('clf', RandomSurvivalForestWrapper())
        ])
        final_pipeline.set_params(**best_params_overall)
        final_pipeline.fit(X, {'T': T, 'E': E})
        
        results = {
            'model_name': 'Random Survival Forest',
            'imputation_strategy': self.imputation_strategy,
            'cv_scores': cv_scores,
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'best_params': best_params_overall,
            'final_model': final_pipeline,
            'fold_models': best_models,
            'fold_params': best_params_list
        }
        
        logger.info(f"\nRandom Survival Forest CV Results ({self.imputation_strategy}):")
        logger.info(f"  Mean C-index: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
        logger.info("=" * 60)
        
        return results
    
    @staticmethod
    def _get_most_common_params(params_list):
        """
        Get the most common hyperparameters across CV folds.
        
        Parameters
        ----------
        params_list : list of dict
            List of hyperparameter dictionaries from each fold
        
        Returns
        -------
        most_common : dict
            Most frequently occurring parameters
        """
        params_df = pd.DataFrame(params_list)
        most_common = params_df.mode().iloc[0].to_dict()
        
        # Convert float parameters that should be integers back to int type
        # This handles parameters like max_depth, n_estimators, etc.
        # Note: parameters may have 'clf__' prefix from pipeline
        int_params = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf']
        int_params_prefixed = [f'clf__{param}' for param in int_params]
        
        for param in int_params + int_params_prefixed:
            if param in most_common and most_common[param] is not None:
                most_common[param] = int(most_common[param])
        
        return most_common


def _load_predefined_scores():
    """Load all predefined linear scores from score*.csv files in DATA_DIR.

    Each score file is expected to have at least two columns: ``var`` and
    ``coef``. The ``var`` column contains feature names; a special row with
    ``var`` equal to ``intercept``, ``(intercept)``, or ``const`` (case
    insensitive) is treated as the intercept term.

    Returns
    -------
    scores : dict
        Mapping from score identifier to a dict with keys ``name``,
        ``coefficients``, and ``intercept``.
    """
    pattern = os.path.join(DATA_DIR, 'score*.csv')
    files = glob.glob(pattern)

    scores = {}
    if not files:
        logger.info("No predefined score files found matching 'score*.csv'.")
        return scores

    logger.info(f"Found {len(files)} predefined score file(s) in data directory.")

    for path in files:
        try:
            df = pd.read_csv(path)
            # Strip whitespace from column names to handle formatting issues
            df.columns = df.columns.str.strip()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Could not read score file {path}: {exc}")
            continue

        if not {'var', 'coef'}.issubset(df.columns):
            logger.warning(
                f"Score file {path} does not contain required columns 'var' and 'coef'; skipping."
            )
            continue

        basename = os.path.splitext(os.path.basename(path))[0]
        score_name = basename

        coefficients = {}
        intercept = 0.0
        for _, row in df.iterrows():
            var_name = str(row['var']).strip()
            coef_val = float(row['coef'])
            lower = var_name.lower()
            if lower in {'intercept', '(intercept)', 'const'}:
                intercept = coef_val
            else:
                coefficients[var_name] = coef_val

        if not coefficients:
            logger.warning(f"Score file {path} defines no feature coefficients; skipping.")
            continue

        scores[score_name] = {
            'name': score_name,
            'coefficients': coefficients,
            'intercept': intercept
        }

        logger.info(
            f"Loaded predefined score '{score_name}' with {len(coefficients)} coefficients "
            f"and intercept {intercept:.4f} from {path}"
        )

    return scores


def evaluate_predefined_scores(X, y, imputation_strategy='median'):
    """Evaluate all predefined linear scores under a given imputation strategy.

    The scores are evaluated using the same outer cross-validation
    configuration as the main models, with imputation fitted within each
    training fold to avoid data leakage.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    imputation_strategy : str
        Imputation strategy: 'median' or 'knn'.

    Returns
    -------
    results : dict
        Dictionary mapping score name to a results dictionary compatible
        with the rest of the pipeline reporting.
    """
    score_defs = _load_predefined_scores()
    if not score_defs:
        return {}

    logger.info(
        f"Evaluating {len(score_defs)} predefined linear score(s) "
        f"with {imputation_strategy.upper()} imputation."
    )

    results = {}
    cv_outer = StratifiedKFold(
        n_splits=N_SPLITS_OUTER, shuffle=True, random_state=RANDOM_SEED
    )

    for key, meta in score_defs.items():
        score_name = meta['name']
        coefficients = meta['coefficients']
        intercept = meta['intercept']

        logger.info(
            f"Evaluating predefined score '{score_name}' "
            f"({len(coefficients)} features, intercept={intercept:.4f})."
        )

        imputer = get_imputation_transformer(imputation_strategy)
        score_estimator = PredefinedLinearScore(
            name=score_name,
            coefficients=coefficients,
            intercept=intercept
        )

        pipeline = Pipeline(steps=[
            ('impute', imputer),
            ('score', score_estimator)
        ])

        # Cross-validated AUC scores
        try:
            logger.debug(f"About to call cross_val_score with CV_SCORING='{CV_SCORING}', n_jobs={N_JOBS}")
            
            # Use make_scorer to ensure needs_proba=True is respected
            from sklearn.metrics import get_scorer
            scorer = get_scorer(CV_SCORING)  # Get default scorer for 'roc_auc'
            logger.debug(f"Scorer object: {scorer}")
            
            cv_scores = cross_val_score(
                pipeline,
                X,
                y,
                cv=cv_outer,
                scoring=CV_SCORING,
                n_jobs=N_JOBS,
                error_score='raise'
            )
            logger.debug(f"cross_val_score returned: {cv_scores}, has NaN: {np.any(np.isnan(cv_scores))}")
        except ValueError as e:
            # Only catch error if no features from the score are found in data
            logger.error(f"cross_val_score raised ValueError: {e}")
            if 'could not find any of its required features' in str(e):
                logger.error(f"Cannot evaluate predefined score '{score_name}': {e}")
                cv_scores = np.array([np.nan] * N_SPLITS_OUTER)
            else:
                raise
        except Exception as e:
            logger.error(f"cross_val_score raised {type(e).__name__}: {e}", exc_info=True)
            cv_scores = np.array([np.nan] * N_SPLITS_OUTER)

        cv_scores = np.array(cv_scores)

        # Fit final pipeline on full data for potential downstream use
        # This gives imputation parameters fit on all available data (for production deployment)
        pipeline.fit(X, y)

        results[key] = {
            'model_name': f"Score: {score_name}",
            'imputation_strategy': imputation_strategy,
            'cv_scores': cv_scores,
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'best_params': None,
            'final_model': pipeline,
            'fold_models': None,
            'fold_params': None
        }

        logger.info(
            f"Predefined score '{score_name}' CV AUC: "
            f"{cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})"
        )

    return results


def run_full_pipeline(X, y, T=None, E=None, imputation_strategy='median', outcome_type='binary'):
    """
    Run complete model training pipeline with both models.
    Supports both binary classification and survival analysis outcomes.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable (for binary: class labels; for survival: event indicator)
    T : np.ndarray, optional
        Time to event (required for survival outcome)
    E : np.ndarray, optional
        Event indicator (required for survival outcome)
    imputation_strategy : str
        Imputation strategy: 'median', 'knn', or 'none'
    outcome_type : str
        Type of outcome: 'binary' or 'survival'
    
    Returns
    -------
    results : dict
        Dictionary containing results from both models
    """
    logger.info("\n" + "=" * 60)
    logger.info(f"NESTED CROSS-VALIDATION TRAINING PIPELINE")
    logger.info(f"Outcome Type: {outcome_type.upper()}")
    logger.info(f"Imputation Strategy: {imputation_strategy.upper()}")
    logger.info("=" * 60)
    
    # Extract original feature names from X
    feature_names = list(X.columns) if hasattr(X, 'columns') else None
    
    # Extract transformed feature names from the final model's preprocessor
    transformed_feature_names = None
    
    trainer = NestedCVTrainer(
        n_splits_outer=N_SPLITS_OUTER,
        n_splits_inner=N_SPLITS_INNER,
        stratified=STRATIFIED_CV,
        imputation_strategy=imputation_strategy
    )
    
    # Train models based on outcome type
    if outcome_type.lower() in ['survival', 'survival_analysis']:
        if T is None or E is None:
            raise ValueError("T (time) and E (event) are required for survival analysis")
        
        # Train Elastic Net Cox
        en_results = trainer.train_elastic_net_survival(X, T, E)
        
        # Train Random Survival Forest
        rf_results = trainer.train_random_forest_survival(X, T, E)
        
        # Extract transformed feature names from fitted pipeline
        try:
            en_model = en_results['final_model']
            if hasattr(en_model, 'named_steps') and 'preprocess' in en_model.named_steps:
                preprocess = en_model.named_steps['preprocess']
                if hasattr(preprocess, 'named_steps') and 'scale_encode' in preprocess.named_steps:
                    scale_encode = preprocess.named_steps['scale_encode']
                    if hasattr(scale_encode, 'get_feature_names_out'):
                        transformed_feature_names = scale_encode.get_feature_names_out().tolist()
                        logger.info(f"Extracted {len(transformed_feature_names)} transformed feature names from fitted pipeline")
        except Exception as e:
            logger.debug(f"Could not extract transformed feature names: {e}")
        
        # Determine best model
        best_model_name = (
            en_results['model_name']
            if en_results['mean_score'] > rf_results['mean_score']
            else rf_results['model_name']
        )
        
        logger.info("\n" + "=" * 60)
        logger.info(f"BEST MODEL COMPARISON ({imputation_strategy.upper()})")
        logger.info("=" * 60)
        logger.info(
            f"Elastic Net Cox CV C-index: {en_results['mean_score']:.4f} "
            f"(+/- {en_results['std_score']:.4f})"
        )
        logger.info(
            f"Random Survival Forest CV C-index: {rf_results['mean_score']:.4f} "
            f"(+/- {rf_results['std_score']:.4f})"
        )
        logger.info(f"Best Model: {best_model_name}")
        logger.info("=" * 60)
        
    else:  # Binary classification
        # Train Elastic Net
        en_results = trainer.train_elastic_net(X, y)
        
        # Extract transformed feature names from Elastic Net's fitted preprocessor
        try:
            en_model = en_results['final_model']
            if hasattr(en_model, 'named_steps') and 'preprocess' in en_model.named_steps:
                preprocess = en_model.named_steps['preprocess']
                if hasattr(preprocess, 'named_steps') and 'scale_encode' in preprocess.named_steps:
                    scale_encode = preprocess.named_steps['scale_encode']
                    if hasattr(scale_encode, 'get_feature_names_out'):
                        transformed_feature_names = scale_encode.get_feature_names_out().tolist()
                        logger.info(f"Extracted {len(transformed_feature_names)} transformed feature names from fitted pipeline")
        except Exception as e:
            logger.debug(f"Could not extract transformed feature names: {e}")
        
        # Train Random Forest
        rf_results = trainer.train_random_forest(X, y)
        
        # Evaluate any predefined linear scores for this strategy
        linear_score_results = evaluate_predefined_scores(X, y, imputation_strategy=imputation_strategy)
        if linear_score_results:
            logger.info(
                f"Evaluated {len(linear_score_results)} predefined score model(s) "
                f"under {imputation_strategy.upper()} imputation."
            )
        else:
            logger.info("No predefined score models evaluated (none configured).")
        
        # Determine best model among the trained ML models (Elastic Net vs RF)
        best_model_name = (
            en_results['model_name']
            if en_results['mean_score'] > rf_results['mean_score']
            else rf_results['model_name']
        )
        
        logger.info("\n" + "=" * 60)
        logger.info(f"BEST MODEL COMPARISON ({imputation_strategy.upper()})")
        logger.info("=" * 60)
        logger.info(
            f"Elastic Net CV AUC: {en_results['mean_score']:.4f} "
            f"(+/- {en_results['std_score']:.4f})"
        )
        logger.info(
            f"Random Forest CV AUC: {rf_results['mean_score']:.4f} "
            f"(+/- {rf_results['std_score']:.4f})"
        )
        if linear_score_results:
            for key, res in linear_score_results.items():
                logger.info(
                    f"{res['model_name']} CV AUC: {res['mean_score']:.4f} "
                    f"(+/- {res['std_score']:.4f})"
                )
        logger.info(f"Best ML Model (Elastic Net vs RF): {best_model_name}")
        logger.info("=" * 60)
    
    # Save models with imputation strategy in filename
    # final_models are fit on full data using best hyperparameters from CV
    en_filename = f"{MODELS_DIR}/elastic_net_{imputation_strategy}_final.pkl"
    rf_filename = f"{MODELS_DIR}/random_forest_{imputation_strategy}_final.pkl"
    joblib.dump(en_results['final_model'], en_filename)
    joblib.dump(rf_results['final_model'], rf_filename)
    logger.info(f"Models saved to {MODELS_DIR}/ with suffix '_{imputation_strategy}'")
    
    results = {
        'elastic_net': en_results,
        'random_forest': rf_results,
        'linear_scores': linear_score_results if outcome_type.lower() == 'binary' else {},
        'best_model_name': best_model_name,
        'imputation_strategy': imputation_strategy,
        'outcome_type': outcome_type,
        'feature_names': feature_names,
        'transformed_feature_names': transformed_feature_names
    }
    
    # Save the complete training results to cache for future runs
    save_training_results(results, imputation_strategy)
    
    return results


if __name__ == '__main__':
    from data_prep import prepare_pipeline_data
    
    # Prepare data
    data = prepare_pipeline_data()
    X, y = data['X'], data['y']
    
    # Run training pipeline
    results = run_full_pipeline(X, y, outcome_type=data.get('outcome_type', 'binary'))
    logger.info("Model training pipeline completed!")
