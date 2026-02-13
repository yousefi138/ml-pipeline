"""
Model training with nested cross-validation.
Implements two models (Elastic Net and Random Forest) with hyperparameter tuning.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, cross_val_score, cross_validate
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import joblib
import warnings

from config import (
    RANDOM_SEED, N_SPLITS_OUTER, N_SPLITS_INNER, N_JOBS,
    CV_SCORING, ELASTIC_NET_PARAMS, RANDOM_FOREST_PARAMS,
    STRATIFIED_CV, METRICS, MODELS_DIR
)
from utils import setup_logging, format_cv_results

warnings.filterwarnings('ignore')
logger = setup_logging('model_training')

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)


class NestedCVTrainer:
    """
    Trainer class implementing nested cross-validation for model evaluation and hyperparameter tuning.
    """
    
    def __init__(self, n_splits_outer=N_SPLITS_OUTER, n_splits_inner=N_SPLITS_INNER,
                 stratified=STRATIFIED_CV, random_state=RANDOM_SEED):
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
        """
        self.n_splits_outer = n_splits_outer
        self.n_splits_inner = n_splits_inner
        self.stratified = stratified
        self.random_state = random_state
        
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
        logger.info("TRAINING ELASTIC NET (LogisticRegression L1/L2)")
        logger.info("=" * 60)
        
        # Define the base model with L1+L2 penalty (elastic net equivalent)
        base_model = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            max_iter=1000,
            random_state=self.random_state,
            n_jobs=1,
            class_weight='balanced'
        )
        
        cv_scores = []
        best_models = []
        best_params_list = []
        scorers = self._create_scorers()
        
        fold = 1
        for train_idx, test_idx in self.cv_outer.split(X, y):
            logger.info(f"\nOuter fold {fold}/{self.n_splits_outer}")
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Inner CV for hyperparameter tuning
            grid_search = GridSearchCV(
                base_model,
                param_grid,
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
        
        # Train final model on full data with best hyperparameters
        # (Use most common best parameters across folds)
        best_params_overall = self._get_most_common_params(best_params_list)
        final_model = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            max_iter=1000,
            random_state=self.random_state,
            class_weight='balanced',
            **best_params_overall
        )
        final_model.fit(X, y)
        
        results = {
            'model_name': 'Elastic Net',
            'cv_scores': cv_scores,
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'best_params': best_params_overall,
            'final_model': final_model,
            'fold_models': best_models,
            'fold_params': best_params_list
        }
        
        logger.info(f"\nElastic Net CV Results:")
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
        logger.info("TRAINING RANDOM FOREST")
        logger.info("=" * 60)
        
        base_model = RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=1,
            class_weight='balanced'
        )
        
        cv_scores = []
        best_models = []
        best_params_list = []
        
        fold = 1
        for train_idx, test_idx in self.cv_outer.split(X, y):
            logger.info(f"\nOuter fold {fold}/{self.n_splits_outer}")
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Inner CV for hyperparameter tuning
            grid_search = GridSearchCV(
                base_model,
                param_grid,
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
        
        # Train final model on full data with best hyperparameters
        best_params_overall = self._get_most_common_params(best_params_list)
        final_model = RandomForestClassifier(
            random_state=self.random_state,
            class_weight='balanced',
            **best_params_overall
        )
        final_model.fit(X, y)
        
        results = {
            'model_name': 'Random Forest',
            'cv_scores': cv_scores,
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'best_params': best_params_overall,
            'final_model': final_model,
            'fold_models': best_models,
            'fold_params': best_params_list
        }
        
        logger.info(f"\nRandom Forest CV Results:")
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
        return most_common


def run_full_pipeline(X, y):
    """
    Run complete model training pipeline with both models.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    
    Returns
    -------
    results : dict
        Dictionary containing results from both models
    """
    logger.info("\n" + "=" * 60)
    logger.info("NESTED CROSS-VALIDATION TRAINING PIPELINE")
    logger.info("=" * 60)
    
    trainer = NestedCVTrainer(
        n_splits_outer=N_SPLITS_OUTER,
        n_splits_inner=N_SPLITS_INNER,
        stratified=STRATIFIED_CV
    )
    
    # Train Elastic Net
    en_results = trainer.train_elastic_net(X, y)
    
    # Train Random Forest
    rf_results = trainer.train_random_forest(X, y)
    
    # Determine best model
    best_model_name = (en_results['model_name'] if en_results['mean_score'] > rf_results['mean_score']
                       else rf_results['model_name'])
    
    logger.info("\n" + "=" * 60)
    logger.info("BEST MODEL COMPARISON")
    logger.info("=" * 60)
    logger.info(f"Elastic Net CV AUC: {en_results['mean_score']:.4f} (+/- {en_results['std_score']:.4f})")
    logger.info(f"Random Forest CV AUC: {rf_results['mean_score']:.4f} (+/- {rf_results['std_score']:.4f})")
    logger.info(f"Best Model: {best_model_name}")
    logger.info("=" * 60)
    
    # Save models
    joblib.dump(en_results['final_model'], f"{MODELS_DIR}/elastic_net_final.pkl")
    joblib.dump(rf_results['final_model'], f"{MODELS_DIR}/random_forest_final.pkl")
    logger.info(f"Models saved to {MODELS_DIR}/")
    
    results = {
        'elastic_net': en_results,
        'random_forest': rf_results,
        'best_model_name': best_model_name
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
