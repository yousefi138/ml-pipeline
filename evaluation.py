"""
Model evaluation and comprehensive performance reporting.
Calculates metrics, generates confusion matrices, and produces final reports.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, auc, accuracy_score, precision_score, recall_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
import json

from config import (
    REPORTS_DIR, MODELS_DIR, RANDOM_SEED, TARGET_COLUMN
)
from utils import setup_logging

logger = setup_logging('evaluation')


class ModelEvaluator:
    """
    Comprehensive evaluator for classification models with performance metrics and reporting.
    """
    
    def __init__(self, model_results_dict):
        """
        Initialize evaluator with nested CV results from both models.
        
        Parameters
        ----------
        model_results_dict : dict
            Dictionary containing 'elastic_net' and 'random_forest' results
        """
        self.results = model_results_dict
        self.en_results = model_results_dict['elastic_net']
        self.rf_results = model_results_dict['random_forest']
        self.best_model_name = model_results_dict['best_model_name']
    
    def generate_cv_summary(self):
        """
        Generate summary statistics from cross-validation results.
        
        Returns
        -------
        summary : pd.DataFrame
            Summary statistics for both models
        """
        logger.info("=" * 60)
        logger.info("CROSS-VALIDATION SUMMARY")
        logger.info("=" * 60)
        
        summary_data = {
            'Model': [self.en_results['model_name'], self.rf_results['model_name']],
            'Mean CV AUC': [self.en_results['mean_score'], self.rf_results['mean_score']],
            'Std CV AUC': [self.en_results['std_score'], self.rf_results['std_score']],
            'Min CV AUC': [self.en_results['cv_scores'].min(), self.rf_results['cv_scores'].min()],
            'Max CV AUC': [self.en_results['cv_scores'].max(), self.rf_results['cv_scores'].max()],
            'Best Hyperparameters': [str(self.en_results['best_params']), 
                                     str(self.rf_results['best_params'])]
        }
        
        summary_df = pd.DataFrame(summary_data)
        logger.info(f"\n{summary_df.to_string()}")
        logger.info("=" * 60)
        
        return summary_df
    
    def get_final_model_parameters(self):
        """
        Get and format final model parameters for both models refitted on full data.
        
        Returns
        -------
        params_report : dict
            Final model parameters and relevant information
        """
        logger.info("=" * 60)
        logger.info("FINAL MODEL PARAMETERS (Refitted on Full Data)")
        logger.info("=" * 60)
        
        params_report = {}
        
        # Elastic Net parameters (unwrap from Pipeline if necessary)
        en_model = self.en_results['final_model']
        if hasattr(en_model, 'named_steps') and 'clf' in en_model.named_steps:
            en_base = en_model.named_steps['clf']
        else:
            en_base = en_model
        en_params = {
            'model_type': 'LogisticRegression (Elastic Net)',
            'hyperparameters': self.en_results['best_params'],
            'cv_mean_auc': float(self.en_results['mean_score']),
            'cv_std_auc': float(self.en_results['std_score']),
            'coefficients': en_base.coef_[0].tolist() if hasattr(en_base, 'coef_') else None,
            'intercept': float(en_base.intercept_[0]) if hasattr(en_base, 'intercept_') else None,
            'classes': en_base.classes_.tolist() if hasattr(en_base, 'classes_') else None
        }
        params_report['elastic_net'] = en_params
        
        # Random Forest parameters (unwrap from Pipeline if necessary)
        rf_model = self.rf_results['final_model']
        if hasattr(rf_model, 'named_steps') and 'clf' in rf_model.named_steps:
            rf_base = rf_model.named_steps['clf']
        else:
            rf_base = rf_model
        rf_params = {
            'model_type': 'RandomForestClassifier',
            'hyperparameters': self.rf_results['best_params'],
            'cv_mean_auc': float(self.rf_results['mean_score']),
            'cv_std_auc': float(self.rf_results['std_score']),
            'n_trees': rf_base.n_estimators,
            'feature_importances': rf_base.feature_importances_.tolist() if hasattr(rf_base, 'feature_importances_') else None,
            'classes': rf_base.classes_.tolist() if hasattr(rf_base, 'classes_') else None
        }
        params_report['random_forest'] = rf_params
        
        # Log summary
        logger.info("\nElastic Net Parameters:")
        logger.info(f"  Hyperparameters: {en_params['hyperparameters']}")
        logger.info(f"  CV Mean AUC: {en_params['cv_mean_auc']:.4f} (+/- {en_params['cv_std_auc']:.4f})")
        
        logger.info("\nRandom Forest Parameters:")
        logger.info(f"  Hyperparameters: {rf_params['hyperparameters']}")
        logger.info(f"  CV Mean AUC: {rf_params['cv_mean_auc']:.4f} (+/- {rf_params['cv_std_auc']:.4f})")
        logger.info("=" * 60)
        
        return params_report
    
    def generate_fold_scores_report(self):
        """
        Generate detailed report of per-fold cross-validation scores.
        
        Returns
        -------
        fold_report : pd.DataFrame
            Per-fold scores for both models
        """
        logger.info("=" * 60)
        logger.info("PER-FOLD CROSS-VALIDATION SCORES")
        logger.info("=" * 60)
        
        n_folds = len(self.en_results['cv_scores'])
        fold_data = {
            'Fold': list(range(1, n_folds + 1)),
            'Elastic Net AUC': self.en_results['cv_scores'],
            'Random Forest AUC': self.rf_results['cv_scores']
        }
        
        fold_report = pd.DataFrame(fold_data)
        logger.info(f"\n{fold_report.to_string()}")
        logger.info("=" * 60)
        
        return fold_report
    
    def create_comprehensive_report(self):
        """
        Create comprehensive evaluation report combining all metrics and summaries.
        
        Returns
        -------
        full_report : dict
            Complete evaluation report
        """
        logger.info("\n" + "=" * 60)
        logger.info("GENERATING COMPREHENSIVE EVALUATION REPORT")
        logger.info("=" * 60)
        
        full_report = {
            'timestamp': datetime.now().isoformat(),
            'best_model': self.best_model_name,
            'cv_summary': self.generate_cv_summary().to_dict(orient='list'),
            'fold_scores': self.generate_fold_scores_report().to_dict(orient='list'),
            'final_model_parameters': self.get_final_model_parameters()
        }
        
        return full_report
    
    def save_report_to_json(self, filename='model_evaluation_report.json'):
        """Save full report to JSON file."""
        report = self.create_comprehensive_report()
        filepath = f"{REPORTS_DIR}/{filename}"
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {filepath}")
        return filepath
    
    def save_summary_to_csv(self, filename='cv_summary.csv'):
        """Save CV summary to CSV file."""
        summary = self.generate_cv_summary()
        filepath = f"{REPORTS_DIR}/{filename}"
        summary.to_csv(filepath, index=False)
        logger.info(f"Summary saved to {filepath}")
        return filepath
    
    def plot_cv_scores(self, filename='cv_scores_comparison.png'):
        """
        Create visualization comparing CV scores across folds.
        
        Parameters
        ----------
        filename : str
            Output filename for the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        folds = np.arange(1, len(self.en_results['cv_scores']) + 1)
        
        # Elastic Net
        axes[0].plot(folds, self.en_results['cv_scores'], 'o-', label='CV Scores', linewidth=2, markersize=8)
        axes[0].axhline(self.en_results['mean_score'], color='r', linestyle='--', label=f"Mean: {self.en_results['mean_score']:.4f}")
        axes[0].fill_between(folds, 
                             self.en_results['mean_score'] - self.en_results['std_score'],
                             self.en_results['mean_score'] + self.en_results['std_score'],
                             alpha=0.2)
        axes[0].set_title('Elastic Net CV Scores')
        axes[0].set_xlabel('Fold')
        axes[0].set_ylabel('AUC Score')
        axes[0].set_ylim([0, 1])
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Random Forest
        axes[1].plot(folds, self.rf_results['cv_scores'], 'o-', label='CV Scores', linewidth=2, markersize=8)
        axes[1].axhline(self.rf_results['mean_score'], color='r', linestyle='--', label=f"Mean: {self.rf_results['mean_score']:.4f}")
        axes[1].fill_between(folds,
                             self.rf_results['mean_score'] - self.rf_results['std_score'],
                             self.rf_results['mean_score'] + self.rf_results['std_score'],
                             alpha=0.2)
        axes[1].set_title('Random Forest CV Scores')
        axes[1].set_xlabel('Fold')
        axes[1].set_ylabel('AUC Score')
        axes[1].set_ylim([0, 1])
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = f"{REPORTS_DIR}/{filename}"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"CV scores plot saved to {filepath}")
        plt.close()
    
    def plot_model_comparison(self, filename='model_comparison.png'):
        """
        Create bar plot comparing mean CV AUCs with error bars.
        
        Parameters
        ----------
        filename : str
            Output filename for the plot
        """
        models = [self.en_results['model_name'], self.rf_results['model_name']]
        means = [self.en_results['mean_score'], self.rf_results['mean_score']]
        stds = [self.en_results['std_score'], self.rf_results['std_score']]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#1f77b4', '#ff7f0e']
        bars = ax.bar(models, means, yerr=stds, capsize=10, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Mean CV AUC', fontsize=12)
        ax.set_title('Model Comparison: Mean CV AUC with Standard Deviation', fontsize=13)
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                   f'{mean:.4f}\n±{std:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        filepath = f"{REPORTS_DIR}/{filename}"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Model comparison plot saved to {filepath}")
        plt.close()


def generate_full_evaluation_report(model_results):
    """
    Generate complete evaluation and reporting workflow.
    
    Parameters
    ----------
    model_results : dict
        Results from model_training.run_full_pipeline()
    """
    logger.info("\n" + "=" * 80)
    logger.info("FULL EVALUATION AND REPORTING WORKFLOW")
    logger.info("=" * 80)
    
    evaluator = ModelEvaluator(model_results)
    
    # Generate all reports and visualizations
    evaluator.save_report_to_json()
    evaluator.save_summary_to_csv()
    evaluator.plot_cv_scores()
    evaluator.plot_model_comparison()
    
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION COMPLETE")
    logger.info("All reports and visualizations saved to:")
    logger.info(f"  - {REPORTS_DIR}/")
    logger.info(f"  - {MODELS_DIR}/")
    logger.info("=" * 80)


if __name__ == '__main__':
    from model_training import run_full_pipeline
    from data_prep import prepare_pipeline_data
    
    # Prepare data and train models
    data = prepare_pipeline_data()
    X, y = data['X'], data['y']
    
    results = run_full_pipeline(X, y)
    
    # Generate evaluation report
    generate_full_evaluation_report(results)
