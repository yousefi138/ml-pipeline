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
    
    def generate_html_report(self, json_report=None, filename='model_evaluation_report.html'):
        """
        Generate a human-readable HTML report from the evaluation data.
        
        Parameters
        ----------
        json_report : dict, optional
            Report dictionary. If None, creates a new comprehensive report.
        filename : str
            Output filename for the HTML report
        
        Returns
        -------
        filepath : str
            Path to generated HTML file
        """
        if json_report is None:
            json_report = self.create_comprehensive_report()
        
        # Extract data from report
        timestamp = json_report.get('timestamp', 'Unknown')
        best_model = json_report.get('best_model', 'Unknown')
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
    evaluator.generate_html_report()
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
