"""
Model training with nested cross-validation.
Implements two models (Elastic Net and Random Forest) with hyperparameter tuning.
Includes multiple imputation strategies for robust handling of missing data.
"""

import numpy as np
import pandas as pd
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
import warnings

from config import (
    RANDOM_SEED, N_SPLITS_OUTER, N_SPLITS_INNER, N_JOBS,
    CV_SCORING, ELASTIC_NET_PARAMS, RANDOM_FOREST_PARAMS,
    STRATIFIED_CV, METRICS, MODELS_DIR
)
from utils import setup_logging, format_cv_results
from imputation import (
    get_imputation_transformer, MissingnessAnalyzer
)

warnings.filterwarnings('ignore')
logger = setup_logging('model_training')

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)


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
    
    # Determine best model
    best_model_name = (en_results['model_name'] if en_results['mean_score'] > rf_results['mean_score']
                       else rf_results['model_name'])
    
    logger.info("\n" + "=" * 60)
    logger.info(f"BEST MODEL COMPARISON ({imputation_strategy.upper()})")
    logger.info("=" * 60)
    logger.info(f"Elastic Net CV AUC: {en_results['mean_score']:.4f} (+/- {en_results['std_score']:.4f})")
    logger.info(f"Random Forest CV AUC: {rf_results['mean_score']:.4f} (+/- {rf_results['std_score']:.4f})")
    logger.info(f"Best Model: {best_model_name}")
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
