"""
Main ML Pipeline Orchestration Script
Orchestrates the complete workflow: data preparation, model training, and evaluation.
Supports multiple imputation strategies for robust handling of missing data.
"""

import sys
import argparse
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Set seed for reproducibility
np.random.seed(42)

import config
from config import (
    RANDOM_SEED, PROJECT_PATH, DATA_FILE, DATA_DIR, RESULTS_DIR, MODELS_DIR, REPORTS_DIR
)
from data_prep import prepare_pipeline_data
from model_training import run_full_pipeline
from evaluation import generate_full_evaluation_report, ConsolidatedReportGenerator
from imputation import MissingnessAnalyzer
from utils import setup_logging

logger = setup_logging('ml_pipeline_main')


def print_header(title):
    """Print formatted header for pipeline sections."""
    print("\n" + "=" * 80)
    print(f"  {title.upper()}")
    print("=" * 80)


def main():
    """
    Execute the complete ML pipeline workflow with multiple imputation strategies.
    """
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='ML Pipeline with Robust Missingness Handling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python ml_pipeline.py
  python ml_pipeline.py --dataset breast_cancer_survival.csv
  python ml_pipeline.py --dataset /full/path/to/breast_cancer_survival.csv
  python ml_pipeline.py --dataset flchain_survival.csv --target-column death --time-column futime
        '''
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Path to the input dataset (CSV file). If not provided, uses default from config. '
             'Can be a filename (assumed to be in data/ directory) or full path.'
    )
    parser.add_argument(
        '--target-column',
        type=str,
        default=None,
        help='Name of the target column in the dataset. If not provided, uses default from config.'
    )
    parser.add_argument(
        '--time-column',
        type=str,
        default=None,
        help='Name of the time column in the dataset. If not provided, uses default from config.'
    )
    
    args = parser.parse_args()
    
    # Update config with command line arguments if provided
    if args.target_column:
        config.TARGET_COLUMN = args.target_column
        logger.info(f"Using target column from command line: {args.target_column}")
    
    if args.time_column:
        config.TIME_COLUMN = args.time_column
        logger.info(f"Using time column from command line: {args.time_column}")
    
    # Also update in utils module since it imports these from config
    import utils
    utils.TARGET_COLUMN = config.TARGET_COLUMN
    utils.TIME_COLUMN = config.TIME_COLUMN
    
    # Determine the dataset path to use
    if args.dataset:
        dataset_path = args.dataset
        # If it's just a filename (no path separators), assume it's in the data directory
        if os.sep not in dataset_path and '/' not in dataset_path:
            dataset_path = os.path.join(DATA_DIR, dataset_path)
    else:
        dataset_path = DATA_FILE
    
    # Validate that the dataset exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}", file=sys.stderr)
        sys.exit(1)
    
    print_header("ML Pipeline - Robust to Missingness with Multiple Imputation Strategies")
    logger.info(f"Pipeline started at {datetime.now().isoformat()}")
    logger.info(f"Project path: {PROJECT_PATH}")
    logger.info(f"Data file: {dataset_path}")
    logger.info(f"Target column: {config.TARGET_COLUMN}")
    logger.info(f"Time column: {config.TIME_COLUMN}")
    
    try:
        # ====== STEP 1: DATA PREPARATION ======
        print_header("Step 1: Data Loading and Preparation")
        logger.info("Initiating data preparation...")
        
        data = prepare_pipeline_data(
            dataset_path,
            target_column=config.TARGET_COLUMN,
            time_column=config.TIME_COLUMN
        )
        X = data['X']
        y = data['y']
        
        logger.info(f"✓ Data preparation completed")
        logger.info(f"  - Samples: {X.shape[0]}")
        logger.info(f"  - Features: {X.shape[1]}")
        logger.info(f"  - Target classes: {y.nunique()}")
        
        # ====== STEP 1.5: ASSESS MISSINGNESS ======
        print_header("Step 1.5: Missingness Assessment")
        logger.info("Analyzing missing data patterns...")
        
        missingness_analysis = MissingnessAnalyzer.assess_missingness(X)
        MissingnessAnalyzer.report_missingness(X, prefix="  ")
        
        if missingness_analysis['total_missing_values'] == 0:
            logger.warning("No missing values detected in the dataset.")
            logger.warning("Running pipeline with median imputation as baseline (no-op for this data).")
        
        # ====== STEP 2: MODEL TRAINING WITH MULTIPLE IMPUTATION STRATEGIES ======
        print_header("Step 2: Model Training with Multiple Imputation Strategies")
        logger.info("Training models with different imputation approaches...")
        
        # Define imputation strategies to test
        imputation_strategies = ['median', 'knn']
        all_results = {}
        strategy_summary = []
        
        for strategy in imputation_strategies:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training pipeline with {strategy.upper()} imputation")
            logger.info(f"{'='*60}")
            
            results = run_full_pipeline(X, y, imputation_strategy=strategy)
            all_results[strategy] = results
            
            # Store summary for later comparison
            strategy_summary.append({
                'Imputation Strategy': strategy.upper(),
                'Elastic Net Mean AUC': results['elastic_net']['mean_score'],
                'Elastic Net Std AUC': results['elastic_net']['std_score'],
                'Random Forest Mean AUC': results['random_forest']['mean_score'],
                'Random Forest Std AUC': results['random_forest']['std_score'],
                'Best Model': results['best_model_name']
            })
            
            logger.info(f"✓ {strategy.upper()} imputation training completed")
            logger.info(f"  - Best model: {results['best_model_name']}")
            logger.info(f"  - Elastic Net CV AUC: {results['elastic_net']['mean_score']:.4f} (+/- {results['elastic_net']['std_score']:.4f})")
            logger.info(f"  - Random Forest CV AUC: {results['random_forest']['mean_score']:.4f} (+/- {results['random_forest']['std_score']:.4f})")
        
        # ====== STEP 3: EVALUATION AND CONSOLIDATED REPORTING ======
        print_header("Step 3: Evaluation and Consolidated Reporting")
        logger.info("Generating consolidated evaluation report across all strategies...")
        
        # Generate unified consolidated reports for all strategy-model combinations
        consolidated_reporter = ConsolidatedReportGenerator(all_results)
        report_paths = consolidated_reporter.generate_all_reports()
        
        logger.info(f"✓ Consolidated reports generated successfully")
        logger.info(f"  - HTML Report: {report_paths['html']}")
        logger.info(f"  - JSON Report: {report_paths['json']}")
        logger.info(f"  - CSV Report:  {report_paths['csv']}")
        
        # Also generate per-strategy reports for detailed analysis
        logger.info("\nGenerating detailed per-strategy reports...")
        for strategy, results in all_results.items():
            logger.info(f"  - Generating detailed reports for {strategy.upper()} imputation...")
            generate_full_evaluation_report(results)
        
        # ====== STEP 4: STRATEGY COMPARISON SUMMARY ======
        print_header("Step 4: Strategy Comparison Summary")
        logger.info("Comparing performance across imputation strategies...")
        
        strategy_comparison_df = pd.DataFrame(strategy_summary)
        logger.info("\nImputation Strategy Performance Comparison:")
        logger.info(f"\n{strategy_comparison_df.to_string(index=False)}")
        
        # Save strategy comparison to CSV
        strategy_comparison_path = f"{REPORTS_DIR}/imputation_strategy_comparison.csv"
        strategy_comparison_df.to_csv(strategy_comparison_path, index=False)
        logger.info(f"\nStrategy comparison saved to {strategy_comparison_path}")
        
        # ====== PIPELINE SUMMARY ======
        print_header("Pipeline Summary: Missingness Imputation Analysis with Consolidated Reporting")
        
        summary_text = f"""
        ML Pipeline Execution Summary: Robust Missingness Handling
        ──────────────────────────────────────────────────────────────
        
        DATA
          • Total samples: {X.shape[0]}
          • Total features: {X.shape[1]}
          • Target classes: {y.nunique()}
          • Class distribution: {dict(y.value_counts())}
        
        MISSINGNESS ANALYSIS
          • Total missing values: {missingness_analysis['total_missing_values']}
          • Percent missing: {missingness_analysis['percent_total_missing']:.2f}%
          • Features with missing values: {len(missingness_analysis['features_with_missing'])}
          {format_missing_features(missingness_analysis)}
        
        IMPUTATION STRATEGIES TESTED
        {format_strategy_results(all_results)}
        
        STRATEGY COMPARISON
        {strategy_comparison_df.to_string(index=False)}
        
        OUTPUTS
          • Consolidated Reports: {REPORTS_DIR}/
            - consolidated_results.html (Single unified report of all combinations)
            - consolidated_results.json (Detailed metadata in JSON)
            - consolidated_results.csv (Tabular summary)
          • Per-Strategy Detail Reports: {REPORTS_DIR}/
            - model_evaluation_report_*.html, *.json, *.csv (One set per strategy)
            - CV scores and model comparison plots
          • Strategy Comparison: {REPORTS_DIR}/imputation_strategy_comparison.csv
          • Models: {MODELS_DIR}/
            - Best models for each imputation strategy
        
        KEY INSIGHTS
          • Multiple imputation strategies tested: {', '.join(s.upper() for s in imputation_strategies)}
          • Consolidated reporting: All strategy-model combinations in single unified HTML report
          • Best overall model selection: Based on cross-validation AUC across all combinations
          • Data leakage prevention: Imputation fitted only on training folds within nested CV
          • Per-strategy details: Additional detailed reports generated for deeper analysis
        
        ──────────────────────────────────────────────────────────────
        """
        
        logger.info(summary_text)
        print(summary_text)
        
        logger.info(f"Pipeline completed successfully at {datetime.now().isoformat()}")
        print("\n" + "=" * 80)
        print("  ✓ ML PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        print("  ✓ Multiple imputation strategies evaluated and compared")
        print("=" * 80 + "\n")
        
        return all_results
    
    except Exception as e:
        error_msg = f"Pipeline failed with error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        print(f"\n✗ ERROR: {error_msg}")
        sys.exit(1)


def format_missing_features(analysis):
    """Format missing features summary for display."""
    if not analysis['features_with_missing']:
        return "          • No missing values detected"
    
    lines = ["          • Missing by feature:"]
    for feature, count in sorted(analysis['features_with_missing'].items()):
        pct = analysis['percent_missing_by_feature'][feature]
        lines.append(f"            - {feature}: {count} ({pct:.2f}%)")
    return "\n".join(lines)


def format_strategy_results(all_results):
    """Format strategy results for display."""
    lines = []
    for strategy, results in all_results.items():
        lines.append(f"\n        {strategy.upper()} IMPUTATION")
        lines.append(f"          • Elastic Net CV AUC: {results['elastic_net']['mean_score']:.4f} (+/- {results['elastic_net']['std_score']:.4f})")
        lines.append(f"          • Random Forest CV AUC: {results['random_forest']['mean_score']:.4f} (+/- {results['random_forest']['std_score']:.4f})")
        # Any predefined linear scores evaluated under this strategy
        linear_scores = results.get('linear_scores', {}) or {}
        if linear_scores:
            lines.append("          • Predefined Linear Scores:")
            for key, res in linear_scores.items():
                lines.append(
                    f"            - {res['model_name']}: CV AUC "
                    f"{res['mean_score']:.4f} (+/- {res['std_score']:.4f})"
                )
        lines.append(f"          • Best ML Model (Elastic Net vs RF): {results['best_model_name']}")
    return "\n".join(lines)


def main_single_strategy(imputation_strategy='median'):
    """
    Execute the ML pipeline with a single imputation strategy.
    Useful for focused analysis on one approach.
    
    Parameters
    ----------
    imputation_strategy : str
        Imputation strategy: 'median' or 'knn'
    """
    
    print_header(f"ML Pipeline - Single Strategy ({imputation_strategy.upper()})")
    logger.info(f"Pipeline started at {datetime.now().isoformat()}")
    logger.info(f"Imputation strategy: {imputation_strategy.upper()}")
    
    try:
        # Data preparation
        print_header("Step 1: Data Loading and Preparation")
        data = prepare_pipeline_data()
        X = data['X']
        y = data['y']
        
        logger.info(f"✓ Data preparation completed")
        logger.info(f"  - Samples: {X.shape[0]}")
        logger.info(f"  - Features: {X.shape[1]}")
        
        # Model training
        print_header(f"Step 2: Model Training ({imputation_strategy.upper()} Imputation)")
        results = run_full_pipeline(X, y, imputation_strategy=imputation_strategy)
        
        # Evaluation and reporting
        print_header("Step 3: Evaluation and Reporting")
        generate_full_evaluation_report(results)
        
        logger.info(f"✓ Pipeline completed successfully")
        logger.info(f"  - Best model: {results['best_model_name']}")
        
        print("\n" + "=" * 80)
        print(f"  ✓ ML PIPELINE COMPLETED ({imputation_strategy.upper()} IMPUTATION)")
        print("=" * 80 + "\n")
        
        return results
    
    except Exception as e:
        error_msg = f"Pipeline failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        print(f"\n✗ ERROR: {error_msg}")
        sys.exit(1)


if __name__ == '__main__':
    results = main()
