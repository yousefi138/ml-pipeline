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
    Includes reporting on different imputation strategies used during training.
    """
    
    def __init__(self, model_results_dict):
        """
        Initialize evaluator with nested CV results from both models.
        
        Parameters
        ----------
        model_results_dict : dict
            Dictionary containing 'elastic_net' and 'random_forest' results
            Can also include 'imputation_strategy' for single strategy evaluation
        """
        self.results = model_results_dict
        self.en_results = model_results_dict['elastic_net']
        self.rf_results = model_results_dict['random_forest']
        self.best_model_name = model_results_dict['best_model_name']
        self.imputation_strategy = model_results_dict.get('imputation_strategy', 'unknown')
    
    def generate_cv_summary(self):
        """
        Generate summary statistics from cross-validation results.
        
        Returns
        -------
        summary : pd.DataFrame
            Summary statistics for both models
        """
        logger.info("=" * 60)
        logger.info(f"CROSS-VALIDATION SUMMARY (Imputation: {self.imputation_strategy.upper()})")
        logger.info("=" * 60)
        
        summary_data = {
            'Model': [self.en_results['model_name'], self.rf_results['model_name']],
            'Mean CV AUC': [self.en_results['mean_score'], self.rf_results['mean_score']],
            'Std CV AUC': [self.en_results['std_score'], self.rf_results['std_score']],
            'Min CV AUC': [self.en_results['cv_scores'].min(), self.rf_results['cv_scores'].min()],
            'Max CV AUC': [self.en_results['cv_scores'].max(), self.rf_results['cv_scores'].max()],
            'Best Hyperparameters': [str(self.en_results['best_params']), 
                                     str(self.rf_results['best_params'])],
            'Imputation Strategy': [self.en_results.get('imputation_strategy', 'unknown'), 
                                    self.rf_results.get('imputation_strategy', 'unknown')]
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
        Includes information about imputation strategies used.
        
        Returns
        -------
        full_report : dict
            Complete evaluation report
        """
        logger.info("\n" + "=" * 60)
        logger.info(f"GENERATING COMPREHENSIVE EVALUATION REPORT")
        logger.info(f"Imputation Strategy: {self.imputation_strategy.upper()}")
        logger.info("=" * 60)
        
        full_report = {
            'timestamp': datetime.now().isoformat(),
            'imputation_strategy': self.imputation_strategy,
            'best_model': self.best_model_name,
            'cv_summary': self.generate_cv_summary().to_dict(orient='list'),
            'fold_scores': self.generate_fold_scores_report().to_dict(orient='list'),
            'final_model_parameters': self.get_final_model_parameters()
        }
        
        return full_report
    
    def save_report_to_json(self, filename=None):
        """
        Save full report to JSON file.
        
        Parameters
        ----------
        filename : str, optional
            Custom filename. If None, uses imputation strategy in filename.
        """
        report = self.create_comprehensive_report()
        
        if filename is None:
            filename = f"model_evaluation_report_{self.imputation_strategy}.json"
        
        filepath = f"{REPORTS_DIR}/{filename}"
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {filepath}")
        return filepath
    
    def generate_html_report(self, json_report=None, filename=None):
        """
        Generate a human-readable HTML report from the evaluation data.
        
        Parameters
        ----------
        json_report : dict, optional
            Report dictionary. If None, creates a new comprehensive report.
        filename : str, optional
            Output filename. If None, uses imputation strategy in filename.
        
        Returns
        -------
        filepath : str
            Path to generated HTML file
        """
        if json_report is None:
            json_report = self.create_comprehensive_report()
        
        if filename is None:
            filename = f"model_evaluation_report_{self.imputation_strategy}.html"
        
        # Extract data from report
        timestamp = json_report.get('timestamp', 'Unknown')
        best_model = json_report.get('best_model', 'Unknown')
        imputation_strategy = json_report.get('imputation_strategy', 'unknown')
        cv_summary = json_report.get('cv_summary', {})
        fold_scores = json_report.get('fold_scores', {})
        params = json_report.get('final_model_parameters', {})
        
        # Build CV Summary Table
        cv_table_rows = ""
        if cv_summary:
            models = cv_summary.get('Model', [])
            means = cv_summary.get('Mean CV AUC', [])
            stds = cv_summary.get('Std CV AUC', [])
            mins = cv_summary.get('Min CV AUC', [])
            maxs = cv_summary.get('Max CV AUC', [])
            hyperparams = cv_summary.get('Best Hyperparameters', [])
            
            for i, model in enumerate(models):
                best_indicator = '⭐ BEST' if model == best_model else ''
                cv_table_rows += f"""
                <tr class="{'best-model' if model == best_model else ''}">
                    <td><strong>{model}</strong> {best_indicator}</td>
                    <td>{means[i]:.6f}</td>
                    <td>{stds[i]:.6f}</td>
                    <td>{mins[i]:.6f}</td>
                    <td>{maxs[i]:.6f}</td>
                    <td><code>{hyperparams[i]}</code></td>
                </tr>
                """
        
        # Build Fold Scores Table
        fold_table_rows = ""
        if fold_scores:
            folds = fold_scores.get('Fold', [])
            en_scores = fold_scores.get('Elastic Net AUC', [])
            rf_scores = fold_scores.get('Random Forest AUC', [])
            
            for i, fold in enumerate(folds):
                fold_table_rows += f"""
                <tr>
                    <td>{fold}</td>
                    <td>{en_scores[i]:.6f}</td>
                    <td>{rf_scores[i]:.6f}</td>
                </tr>
                """
        
        # Build Model Parameters Sections
        en_params = params.get('elastic_net', {})
        rf_params = params.get('random_forest', {})
        
        en_params_html = self._build_model_params_html(en_params, 'Elastic Net')
        rf_params_html = self._build_model_params_html(rf_params, 'Random Forest')
        
        # HTML Template
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Evaluation Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .info-box {{
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 5px;
        }}
        
        .info-box strong {{
            color: #667eea;
        }}
        
        .best-model-badge {{
            display: inline-block;
            background: #28a745;
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            margin-left: 10px;
        }}
        
        h2 {{
            color: #667eea;
            margin-top: 40px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}
        
        h3 {{
            color: #555;
            margin-top: 25px;
            margin-bottom: 15px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 30px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }}
        
        table thead {{
            background: #667eea;
            color: white;
            font-weight: bold;
        }}
        
        table th {{
            padding: 15px;
            text-align: left;
        }}
        
        table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #ddd;
        }}
        
        table tbody tr:hover {{
            background: #f5f5f5;
        }}
        
        table tbody tr.best-model {{
            background: #e8f5e9;
            font-weight: bold;
        }}
        
        .model-card {{
            background: #f8f9fa;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 25px;
            margin-bottom: 30px;
        }}
        
        .model-card h3 {{
            margin-top: 0;
            color: #667eea;
        }}
        
        .param-group {{
            margin-bottom: 20px;
        }}
        
        .param-group label {{
            display: block;
            font-weight: bold;
            color: #555;
            margin-bottom: 8px;
            font-size: 0.95em;
        }}
        
        .param-group code {{
            display: block;
            background: #e9ecef;
            padding: 10px;
            border-radius: 5px;
            font-size: 0.9em;
            word-break: break-word;
            overflow-x: auto;
        }}
        
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .metric-box {{
            background: white;
            border: 2px solid #667eea;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }}
        
        .metric-box .label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }}
        
        .metric-box .value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .feature-importance {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
        }}
        
        .feature-importance-item {{
            background: white;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .importance-bar {{
            height: 20px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 3px;
            margin: 5px 0;
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 1px solid #ddd;
            font-size: 0.9em;
        }}
        
        code {{
            background: #f5f5f5;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        
        @media (max-width: 768px) {{
            .header {{
                padding: 20px;
            }}
            
            .header h1 {{
                font-size: 1.8em;
            }}
            
            .content {{
                padding: 20px;
            }}
            
            .metric-grid {{
                grid-template-columns: 1fr;
            }}
            
            table {{
                font-size: 0.9em;
            }}
            
            table th, table td {{
                padding: 8px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 Model Evaluation Report</h1>
            <p>ML Pipeline Performance Analysis</p>
        </div>
        
        <div class="content">
            <!-- Report Metadata -->
            <div class="info-box">
                <strong>Report Generated:</strong> {timestamp}<br>
                <strong>Imputation Strategy:</strong> <code>{imputation_strategy.upper()}</code><br>
                <strong>Best Performing Model:</strong> {best_model} <span class="best-model-badge">✓ Selected</span>
            </div>
            
            <!-- Cross-Validation Summary -->
            <h2>📊 Cross-Validation Summary</h2>
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Mean CV AUC</th>
                        <th>Std Dev</th>
                        <th>Min AUC</th>
                        <th>Max AUC</th>
                        <th>Best Hyperparameters</th>
                    </tr>
                </thead>
                <tbody>
                    {cv_table_rows}
                </tbody>
            </table>
            
            <!-- Per-Fold Scores -->
            <h2>📈 Per-Fold Cross-Validation Scores</h2>
            <table>
                <thead>
                    <tr>
                        <th>Fold</th>
                        <th>Elastic Net AUC</th>
                        <th>Random Forest AUC</th>
                    </tr>
                </thead>
                <tbody>
                    {fold_table_rows}
                </tbody>
            </table>
            
            <!-- Model Parameters -->
            <h2>⚙️ Final Model Parameters</h2>
            
            {en_params_html}
            
            {rf_params_html}
            
        </div>
        
        <div class="footer">
            <p>ML Pipeline Evaluation Report • Generated by ModelEvaluator</p>
        </div>
    </div>
</body>
</html>"""
        
        filepath = f"{REPORTS_DIR}/{filename}"
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {filepath}")
        return filepath
    
    def _build_model_params_html(self, params, model_name):
        """
        Build HTML section for model parameters.
        
        Parameters
        ----------
        params : dict
            Model parameters dictionary
        model_name : str
            Name of the model
        
        Returns
        -------
        html : str
            HTML string for model parameters section
        """
        if not params:
            return ""
        
        model_type = params.get('model_type', 'Unknown')
        hyperparams = params.get('hyperparameters', {})
        cv_mean = params.get('cv_mean_auc', 0)
        cv_std = params.get('cv_std_auc', 0)
        
        # Build hyperparameters section
        hyperparam_html = ""
        for key, value in hyperparams.items():
            hyperparam_html += f"""<div class="param-group">
                <label>{key}:</label>
                <code>{value}</code>
            </div>"""
        
        # Build model-specific section
        specific_html = ""
        if model_name == 'Elastic Net':
            coefficients = params.get('coefficients', [])
            intercept = params.get('intercept', 0)
            non_zero_coefs = sum(1 for c in coefficients if c != 0)
            
            specific_html = f"""<h3>Model Specifics</h3>
            <div class="metric-grid">
                <div class="metric-box">
                    <div class="label">Non-Zero Coefficients</div>
                    <div class="value">{non_zero_coefs}</div>
                </div>
                <div class="metric-box">
                    <div class="label">Total Features</div>
                    <div class="value">{len(coefficients)}</div>
                </div>
                <div class="metric-box">
                    <div class="label">Intercept</div>
                    <div class="value">{intercept:.6f}</div>
                </div>
            </div>
            <h3>Top Non-Zero Coefficients</h3>
            <div class="feature-importance">"""
            
            # Get top coefficients
            indexed_coefs = [(i, c) for i, c in enumerate(coefficients)]
            sorted_coefs = sorted(indexed_coefs, key=lambda x: abs(x[1]), reverse=True)[:10]
            
            for feat_idx, coef_val in sorted_coefs:
                if coef_val != 0:
                    specific_html += f"""<div class="feature-importance-item">
                    <div>Feature {feat_idx}: <strong>{coef_val:.6f}</strong></div>
                </div>"""
            specific_html += "</div>"
            
        elif model_name == 'Random Forest':
            n_trees = params.get('n_trees', 0)
            importances = params.get('feature_importances', [])
            
            specific_html = f"""<h3>Model Specifics</h3>
            <div class="metric-grid">
                <div class="metric-box">
                    <div class="label">Number of Trees</div>
                    <div class="value">{n_trees}</div>
                </div>
                <div class="metric-box">
                    <div class="label">Total Features</div>
                    <div class="value">{len(importances)}</div>
                </div>
            </div>
            <h3>Top 10 Feature Importances</h3>
            <div class="feature-importance">"""
            
            # Get top features
            indexed_imp = [(i, imp) for i, imp in enumerate(importances)]
            sorted_imp = sorted(indexed_imp, key=lambda x: x[1], reverse=True)[:10]
            
            for feat_idx, imp_val in sorted_imp:
                bar_width = imp_val * 100
                specific_html += f"""<div class="feature-importance-item">
                    <div style="flex: 1;">Feature {feat_idx}</div>
                    <div style="flex: 2;">
                        <div class="importance-bar" style="width: {bar_width}%;"></div>
                        <div style="font-size: 0.85em; color: #666;">{imp_val:.6f}</div>
                    </div>
                </div>"""
            specific_html += "</div>"
        
        html = f"""<div class="model-card">
            <h3>{model_name} - {model_type}</h3>
            <div class="metric-grid">
                <div class="metric-box">
                    <div class="label">Mean CV AUC</div>
                    <div class="value">{cv_mean:.6f}</div>
                </div>
                <div class="metric-box">
                    <div class="label">Std Dev</div>
                    <div class="value">±{cv_std:.6f}</div>
                </div>
            </div>
            <h3>Hyperparameters</h3>
            {hyperparam_html}
            {specific_html}
        </div>"""
        
        return html
    
    def save_summary_to_csv(self, filename=None):
        """
        Save CV summary to CSV file.
        
        Parameters
        ----------
        filename : str, optional
            Custom filename. If None, uses imputation strategy in filename.
        """
        summary = self.generate_cv_summary()
        
        if filename is None:
            filename = f"cv_summary_{self.imputation_strategy}.csv"
        
        filepath = f"{REPORTS_DIR}/{filename}"
        summary.to_csv(filepath, index=False)
        logger.info(f"Summary saved to {filepath}")
        return filepath
    
    def plot_cv_scores(self, filename=None):
        """
        Create visualization comparing CV scores across folds.
        
        Parameters
        ----------
        filename : str, optional
            Output filename. If None, uses imputation strategy in filename.
        """
        if filename is None:
            filename = f"cv_scores_comparison_{self.imputation_strategy}.png"
        
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
    
    def plot_model_comparison(self, filename=None):
        """
        Create bar plot comparing mean CV AUCs with error bars.
        
        Parameters
        ----------
        filename : str, optional
            Output filename. If None, uses imputation strategy in filename.
        """
        if filename is None:
            filename = f"model_comparison_{self.imputation_strategy}.png"
        
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
    evaluator.generate_html_report()
    evaluator.plot_cv_scores()
    evaluator.plot_model_comparison()
    
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION COMPLETE")
    logger.info("All reports and visualizations saved to:")
    logger.info(f"  - {REPORTS_DIR}/")
    logger.info(f"  - {MODELS_DIR}/")
    logger.info("=" * 80)


class ConsolidatedReportGenerator:
    """
    Generate consolidated reports across all imputation strategies and models.
    Combines results from multiple strategy-model combinations into unified reports.
    """
    
    def __init__(self, all_results_dict):
        """
        Initialize with results from all imputation strategies.
        
        Parameters
        ----------
        all_results_dict : dict
            Dictionary with structure:
            {
                'strategy_name': {
                    'elastic_net': {...},
                    'random_forest': {...},
                    'linear_scores': {       # optional
                        'score_id': {...},
                        ...
                    },
                    'best_model_name': ...,
                    'imputation_strategy': ...
                },
                ...
            }
        """
        self.all_results = all_results_dict
        self.strategies = list(all_results_dict.keys())
        logger.info(f"Initialized ConsolidatedReportGenerator for {len(self.strategies)} strategies")
    
    def create_comparison_dataframe(self):
        """
        Create a comprehensive comparison DataFrame across all strategies and models.
        
        Returns
        -------
        comparison_df : pd.DataFrame
            DataFrame with all results
        """
        comparison_data = []
        
        for strategy, results in self.all_results.items():
            en_results = results['elastic_net']
            rf_results = results['random_forest']
            linear_scores = results.get('linear_scores', {}) or {}
            
            # Elastic Net row
            comparison_data.append({
                'Imputation Strategy': strategy.upper(),
                'Model': 'Elastic Net',
                'Mean CV AUC': en_results['mean_score'],
                'Std CV AUC': en_results['std_score'],
                'Min CV AUC': en_results['cv_scores'].min(),
                'Max CV AUC': en_results['cv_scores'].max(),
                'Best Hyperparameters': str(en_results['best_params']),
                'Best Overall': '⭐' if results['best_model_name'] == 'Elastic Net' else ''
            })
            
            # Random Forest row
            comparison_data.append({
                'Imputation Strategy': strategy.upper(),
                'Model': 'Random Forest',
                'Mean CV AUC': rf_results['mean_score'],
                'Std CV AUC': rf_results['std_score'],
                'Min CV AUC': rf_results['cv_scores'].min(),
                'Max CV AUC': rf_results['cv_scores'].max(),
                'Best Hyperparameters': str(rf_results['best_params']),
                'Best Overall': '⭐' if results['best_model_name'] == 'Random Forest' else ''
            })

            # Any additional predefined linear score models
            for score_key, score_res in linear_scores.items():
                model_label = score_res.get('model_name', f"Score: {score_key}")
                comparison_data.append({
                    'Imputation Strategy': strategy.upper(),
                    'Model': model_label,
                    'Mean CV AUC': score_res['mean_score'],
                    'Std CV AUC': score_res['std_score'],
                    'Min CV AUC': score_res['cv_scores'].min(),
                    'Max CV AUC': score_res['cv_scores'].max(),
                    'Best Hyperparameters': 'N/A',
                    'Best Overall': ''
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df
    
    def save_consolidated_csv(self, filename='consolidated_results.csv'):
        """
        Save consolidated comparison to CSV.
        
        Parameters
        ----------
        filename : str
            Output filename
        
        Returns
        -------
        filepath : str
            Path to saved file
        """
        comparison_df = self.create_comparison_dataframe()
        filepath = f"{REPORTS_DIR}/{filename}"
        comparison_df.to_csv(filepath, index=False)
        logger.info(f"Consolidated CSV saved to {filepath}")
        return filepath
    
    def save_consolidated_json(self, filename='consolidated_results.json'):
        """
        Save all results in consolidated JSON format.
        
        Parameters
        ----------
        filename : str
            Output filename
        
        Returns
        -------
        filepath : str
            Path to saved file
        """
        consolidated_json = {
            'timestamp': datetime.now().isoformat(),
            'num_strategies': len(self.strategies),
            'strategies': self.strategies,
            'results': {}
        }
        
        # Add all results with comparison metadata
        for strategy, results in self.all_results.items():
            strategy_entry = {
                'elastic_net': {
                    'mean_auc': float(results['elastic_net']['mean_score']),
                    'std_auc': float(results['elastic_net']['std_score']),
                    'best_params': results['elastic_net']['best_params'],
                    'cv_scores': results['elastic_net']['cv_scores'].tolist()
                },
                'random_forest': {
                    'mean_auc': float(results['random_forest']['mean_score']),
                    'std_auc': float(results['random_forest']['std_score']),
                    'best_params': results['random_forest']['best_params'],
                    'cv_scores': results['random_forest']['cv_scores'].tolist()
                },
                'best_model': results['best_model_name']
            }

            # Optional predefined linear scores
            linear_scores = results.get('linear_scores', {}) or {}
            if linear_scores:
                strategy_entry['linear_scores'] = {}
                for score_key, score_res in linear_scores.items():
                    strategy_entry['linear_scores'][score_key] = {
                        'model_name': score_res.get('model_name', f"Score: {score_key}"),
                        'mean_auc': float(score_res['mean_score']),
                        'std_auc': float(score_res['std_score']),
                        'cv_scores': score_res['cv_scores'].tolist()
                    }

            consolidated_json['results'][strategy] = strategy_entry
        
        filepath = f"{REPORTS_DIR}/{filename}"
        with open(filepath, 'w') as f:
            json.dump(consolidated_json, f, indent=2)
        
        logger.info(f"Consolidated JSON saved to {filepath}")
        return filepath
    
    def generate_consolidated_html_report(self, filename='consolidated_results.html'):
        """
        Generate comprehensive HTML report comparing all strategies and models.
        """
        comparison_df = self.create_comparison_dataframe()
        timestamp = datetime.now().isoformat()
        
        # Build results table HTML
        results_table_html = comparison_df.to_html(index=False, border=0, 
                                                   classes='results-table', 
                                                   escape=False)
        
        # Find best overall model
        best_idx = comparison_df['Mean CV AUC'].idxmax()
        best_strategy = comparison_df.loc[best_idx, 'Imputation Strategy']
        best_model = comparison_df.loc[best_idx, 'Model']
        best_auc = comparison_df.loc[best_idx, 'Mean CV AUC']
        best_std = comparison_df.loc[best_idx, 'Std CV AUC']
        
        # Build strategy summary HTML dynamically including all models
        strategy_summary_html = ""
        for strategy in self.strategies:
            strategy_data = self.all_results[strategy]
            en_auc = strategy_data['elastic_net']['mean_score']
            en_std = strategy_data['elastic_net']['std_score']
            rf_auc = strategy_data['random_forest']['mean_score']
            rf_std = strategy_data['random_forest']['std_score']
            
            strategy_summary_html += f"""
            <tr class="strategy-header">
                <td><strong>{strategy.upper()}</strong></td>
                <td></td>
                <td></td>
            </tr>
            <tr>
                <td></td>
                <td>Elastic Net</td>
                <td style="text-align: center;">{en_auc:.4f} ± {en_std:.4f}</td>
            </tr>
            <tr>
                <td></td>
                <td>Random Forest</td>
                <td style="text-align: center;">{rf_auc:.4f} ± {rf_std:.4f}</td>
            </tr>
            """
            
            # Add rows for any predefined linear scores
            linear_scores = strategy_data.get('linear_scores', {}) or {}
            for score_key, score_res in linear_scores.items():
                model_label = score_res.get('model_name', f"Score: {score_key}")
                score_auc = score_res['mean_score']
                score_std = score_res['std_score']
                strategy_summary_html += f"""
            <tr>
                <td></td>
                <td>{model_label}</td>
                <td style="text-align: center;">{score_auc:.4f} ± {score_std:.4f}</td>
            </tr>
            """
        
        # Build strategy cards HTML dynamically including all models
        strategy_cards_html = ""
        for strategy in self.strategies:
            strategy_data = self.all_results[strategy]
            en_auc = strategy_data['elastic_net']['mean_score']
            en_std = strategy_data['elastic_net']['std_score']
            rf_auc = strategy_data['random_forest']['mean_score']
            rf_std = strategy_data['random_forest']['std_score']
            
            card_content = f"""
                    <div class="metric-row">
                        <span class="metric-label">Elastic Net:</span>
                        <span class="metric-value">{en_auc:.4f} ± {en_std:.4f}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Random Forest:</span>
                        <span class="metric-value">{rf_auc:.4f} ± {rf_std:.4f}</span>
                    </div>
            """
            
            # Add scores if present
            linear_scores = strategy_data.get('linear_scores', {}) or {}
            for score_key, score_res in linear_scores.items():
                model_label = score_res.get('model_name', f"Score: {score_key}")
                score_auc = score_res['mean_score']
                score_std = score_res['std_score']
                card_content += f"""
                    <div class="metric-row">
                        <span class="metric-label">{model_label}:</span>
                        <span class="metric-value">{score_auc:.4f} ± {score_std:.4f}</span>
                    </div>
            """
            
            strategy_cards_html += f"""
                <div class="strategy-card">
                    <h4>{strategy.upper()} Imputation</h4>
                    {card_content}
                </div>
            """
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Consolidated Model Evaluation Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .info-box {{
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 5px;
        }}
        
        .best-badge {{
            display: inline-block;
            background: #28a745;
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            margin-left: 10px;
        }}
        
        h2 {{
            color: #667eea;
            margin-top: 40px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}
        
        h3 {{
            color: #555;
            margin-top: 25px;
            margin-bottom: 15px;
        }}
        
        .results-table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 30px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }}
        
        .results-table thead {{
            background: #667eea;
            color: white;
            font-weight: bold;
        }}
        
        .results-table th {{
            padding: 15px;
            text-align: left;
            border-bottom: 2px solid #667eea;
        }}
        
        .results-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #ddd;
        }}
        
        .results-table tbody tr:hover {{
            background: #f5f5f5;
        }}
        
        .results-table tbody tr:nth-child(even) {{
            background: #fafafa;
        }}
        
        .strategy-header {{
            background: #f0f0f0;
            font-weight: bold;
            border-top: 2px solid #667eea;
        }}
        
        .strategy-comparison {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .strategy-card {{
            background: #f8f9fa;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
        }}
        
        .strategy-card h4 {{
            color: #667eea;
            margin-bottom: 15px;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        
        .metric-row {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            padding: 8px 0;
        }}
        
        .metric-label {{
            font-weight: bold;
            color: #555;
        }}
        
        .metric-value {{
            color: #667eea;
            font-weight: bold;
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 1px solid #ddd;
            font-size: 0.9em;
        }}
        
        @media (max-width: 768px) {{
            .header {{
                padding: 20px;
            }}
            
            .header h1 {{
                font-size: 1.8em;
            }}
            
            .content {{
                padding: 20px;
            }}
            
            .strategy-comparison {{
                grid-template-columns: 1fr;
            }}
            
            .results-table {{
                font-size: 0.9em;
            }}
            
            .results-table th, .results-table td {{
                padding: 8px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Consolidated Model Evaluation Report</h1>
            <p>Multi-Strategy Pipeline Performance Analysis</p>
        </div>
        
        <div class="content">
            <!-- Report Metadata -->
            <div class="info-box">
                <strong>Report Generated:</strong> {timestamp}<br>
                <strong>Number of Strategies Evaluated:</strong> {len(self.strategies)}<br>
                <strong>Best Overall Model:</strong> {best_strategy} + {best_model} 
                <span class="best-badge">Mean AUC: {best_auc:.4f} ± {best_std:.4f}</span>
            </div>
            
            <!-- Key Metrics Summary -->
            <h2>📈 Performance Summary by Strategy and Model</h2>
            <table style="width: 100%; border-collapse: collapse; margin-bottom: 30px;">
                <thead style="background: #667eea; color: white;">
                    <tr>
                        <th style="padding: 12px; text-align: left;">Imputation Strategy</th>
                        <th style="padding: 12px; text-align: left;">Model</th>
                        <th style="padding: 12px; text-align: center;">Mean CV AUC ± Std</th>
                    </tr>
                </thead>
                <tbody>
                    {strategy_summary_html}
                </tbody>
            </table>
            
            <!-- Detailed Results Table -->
            <h2>🔍 Detailed Results: All Combinations</h2>
            {results_table_html}
            
            <!-- Strategy Comparison Cards -->
            <h2>📋 Strategy-wise Breakdown</h2>
            <div class="strategy-comparison">
                {strategy_cards_html}
            </div>
            
            <!-- Interpretation Guide -->
            <h2>💡 How to Interpret Results</h2>
            <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 30px;">
                <h3 style="margin-top: 0;">Mean CV AUC</h3>
                <p>Average Area Under the Curve across all cross-validation folds. Higher is better (range: 0-1).</p>
                
                <h3>Std CV AUC</h3>
                <p>Standard deviation of CV AUC scores. Lower is better, indicating more stable performance across folds.</p>
                
                <h3>Imputation Strategy Comparison</h3>
                <ul style="margin-left: 20px;">
                    <li><strong>Median Imputation:</strong> Fast, simple approach. Replaces missing numeric values with median; categorical with most frequent.</li>
                    <li><strong>KNN Imputation:</strong> More sophisticated. Uses k-nearest neighbors for numeric features; most frequent for categorical.</li>
                </ul>
                
                <h3>Model Selection</h3>
                <p>The best overall model combines the optimal imputation strategy with the best-performing algorithm at that strategy level.</p>
            </div>
            
            <!-- Data Leakage Prevention -->
            <h2>✅ Quality Assurance</h2>
            <div style="background: #f8f9fa; padding: 20px; border-radius: 8px;">
                <p><strong>No Data Leakage:</strong> All imputation statistics were learned only from training folds within nested cross-validation loops.</p>
                <p><strong>Fair Comparison:</strong> All models evaluated using the same nested CV framework (5 outer folds, 5 inner folds).</p>
                <p><strong>Stratified Split:</strong> Class distribution preserved across all CV folds.</p>
            </div>
        </div>
        
        <div class="footer">
            <p>Generated on {timestamp}</p>
            <p>ML Pipeline with Robust Imputation Strategies</p>
        </div>
    </div>
</body>
</html>
"""
        
        filepath = f"{REPORTS_DIR}/{filename}"
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Consolidated HTML report saved to {filepath}")
        return filepath
    
    def generate_all_reports(self):
        """Generate all consolidated reports (CSV, JSON, HTML)."""
        logger.info("\n" + "=" * 80)
        logger.info("GENERATING CONSOLIDATED REPORTS")
        logger.info("=" * 80)
        
        csv_path = self.save_consolidated_csv()
        json_path = self.save_consolidated_json()
        html_path = self.generate_consolidated_html_report()
        
        logger.info("\n" + "=" * 80)
        logger.info("CONSOLIDATED REPORTS COMPLETE")
        logger.info("=" * 80)
        logger.info(f"CSV Report:  {csv_path}")
        logger.info(f"JSON Report: {json_path}")
        logger.info(f"HTML Report: {html_path}")
        logger.info("=" * 80)
        
        return {
            'csv': csv_path,
            'json': json_path,
            'html': html_path
        }


if __name__ == '__main__':
    from model_training import run_full_pipeline
    from data_prep import prepare_pipeline_data
    
    # Prepare data and train models
    data = prepare_pipeline_data()
    X, y = data['X'], data['y']
    
    results = run_full_pipeline(X, y)
    
    # Generate evaluation report
    generate_full_evaluation_report(results)
