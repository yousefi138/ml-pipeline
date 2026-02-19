"""
Model training with nested cross-validation.
Implements two models (Elastic Net and Random Forest) with hyperparameter tuning.
Includes multiple imputation strategies for robust handling of missing data.

Also supports benchmarking pre-defined linear prediction scores supplied in
`score*.csv` files in the data directory. These scores are evaluated within
the same cross-validation and imputation framework for like-for-like
comparison of predictive performance.
"""

import os
import glob
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
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

from config import (
    RANDOM_SEED, N_SPLITS_OUTER, N_SPLITS_INNER, N_JOBS,
    CV_SCORING, ELASTIC_NET_PARAMS, RANDOM_FOREST_PARAMS,
    STRATIFIED_CV, METRICS, MODELS_DIR, DATA_DIR
)
from utils import setup_logging, format_cv_results
from imputation import (
    get_imputation_transformer, MissingnessAnalyzer
)

warnings.filterwarnings('ignore')
logger = setup_logging('model_training')

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)


class PredefinedLinearScore(BaseEstimator, ClassifierMixin):
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

        No learning is performed; this simply validates that all required
        features are present and caches their order for fast scoring.
        """
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        self.feature_names_ = list(X_df.columns)

        missing_features = [
            feat for feat in self.coefficients.keys() if feat not in self.feature_names_
        ]
        if missing_features:
            raise ValueError(
                f"Predefined score '{self.name}' requires missing features: {missing_features}"
            )

        self.used_features_ = [feat for feat in self.feature_names_ if feat in self.coefficients]
        self.coef_vector_ = np.array([
            self.coefficients[feat] for feat in self.used_features_
        ], dtype=float)

        # Binary classes assumed {0, 1}
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        scores = np.dot(X_df[self.used_features_].values, self.coef_vector_) + self.intercept
        return scores

    def predict_proba(self, X):
        scores = self.decision_function(X)
        # Logistic transform for probabilities
        probs_pos = 1.0 / (1.0 + np.exp(-scores))
        probs_neg = 1.0 - probs_pos
        return np.vstack([probs_neg, probs_pos]).T

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)

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
            scaling_encoding_transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols))
        
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
        
        # Train final pipeline on full data with best hyperparameters
        # (Use most common best parameters across folds)
        best_params_overall = self._get_most_common_params(best_params_list)

        # Rebuild pipeline to ensure a fresh, unfitted preprocessor
        final_imputer = get_imputation_transformer(self.imputation_strategy)
        final_scaling_encoding_transformers = []
        if numeric_cols:
            final_scaling_encoding_transformers.append(('num', StandardScaler(), numeric_cols))
        if categorical_cols:
            final_scaling_encoding_transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols))
        
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
            scaling_encoding_transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols))
        
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

        # Train final pipeline on full data with best hyperparameters
        best_params_overall = self._get_most_common_params(best_params_list)

        final_imputer = get_imputation_transformer(self.imputation_strategy)
        final_scaling_encoding_transformers = []
        if numeric_cols:
            final_scaling_encoding_transformers.append(('num', StandardScaler(), numeric_cols))
        if categorical_cols:
            final_scaling_encoding_transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols))
        
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
        cv_scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv_outer,
            scoring=CV_SCORING,
            n_jobs=N_JOBS
        )

        cv_scores = np.array(cv_scores)

        # Fit final pipeline on full data for potential downstream use
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


def run_full_pipeline(X, y, imputation_strategy='median'):
    """
    Run complete model training pipeline with both models.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    imputation_strategy : str
        Imputation strategy: 'median', 'knn', or 'none'
    
    Returns
    -------
    results : dict
        Dictionary containing results from both models
    """
    logger.info("\n" + "=" * 60)
    logger.info(f"NESTED CROSS-VALIDATION TRAINING PIPELINE")
    logger.info(f"Imputation Strategy: {imputation_strategy.upper()}")
    logger.info("=" * 60)
    
    trainer = NestedCVTrainer(
        n_splits_outer=N_SPLITS_OUTER,
        n_splits_inner=N_SPLITS_INNER,
        stratified=STRATIFIED_CV,
        imputation_strategy=imputation_strategy
    )
    
    # Train Elastic Net
    en_results = trainer.train_elastic_net(X, y)
    
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
    en_filename = f"{MODELS_DIR}/elastic_net_{imputation_strategy}_final.pkl"
    rf_filename = f"{MODELS_DIR}/random_forest_{imputation_strategy}_final.pkl"
    joblib.dump(en_results['final_model'], en_filename)
    joblib.dump(rf_results['final_model'], rf_filename)
    logger.info(f"Models saved to {MODELS_DIR}/ with suffix '_{imputation_strategy}'")
    
    results = {
        'elastic_net': en_results,
        'random_forest': rf_results,
        'linear_scores': linear_score_results,
        'best_model_name': best_model_name,
        'imputation_strategy': imputation_strategy
    }
    
    return results


if __name__ == '__main__':
    from data_prep import prepare_pipeline_data
    
    # Prepare data
    data = prepare_pipeline_data()
    X, y = data['X'], data['y']
    
    # Run training pipeline
    results = run_full_pipeline(X, y)
    logger.info("Model training pipeline completed!")
