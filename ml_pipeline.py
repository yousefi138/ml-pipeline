"""
Main ML Pipeline Orchestration Script
Orchestrates the complete workflow: data preparation, model training, and evaluation.
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Set seed for reproducibility
np.random.seed(42)

from config import (
    RANDOM_SEED, PROJECT_PATH, DATA_FILE, RESULTS_DIR, MODELS_DIR, REPORTS_DIR
)
from data_prep import prepare_pipeline_data
from model_training import run_full_pipeline
from evaluation import generate_full_evaluation_report
from utils import setup_logging

logger = setup_logging('ml_pipeline_main')


def print_header(title):
    """Print formatted header for pipeline sections."""
    print("\n" + "=" * 80)
    print(f"  {title.upper()}")
    print("=" * 80)


def main():
    """
    Execute the complete ML pipeline workflow.
    """
    
    print_header("ML Pipeline - Scalable & Reproducible Supervised Learning")
    logger.info(f"Pipeline started at {datetime.now().isoformat()}")
    logger.info(f"Project path: {PROJECT_PATH}")
    logger.info(f"Data file: {DATA_FILE}")
    
    try:
        # ====== STEP 1: DATA PREPARATION ======
        print_header("Step 1: Data Loading and Preparation")
        logger.info("Initiating data preparation...")
        
        data = prepare_pipeline_data()
        X = data['X']
        y = data['y']
        
        logger.info(f"✓ Data preparation completed")
        logger.info(f"  - Samples: {X.shape[0]}")
        logger.info(f"  - Features: {X.shape[1]}")
        logger.info(f"  - Target classes: {y.nunique()}")
        
        # ====== STEP 2: MODEL TRAINING WITH NESTED CV ======
        print_header("Step 2: Model Training with Nested Cross-Validation")
        logger.info("Initiating model training with nested 5-fold CV...")
        
        results = run_full_pipeline(X, y)
        
        logger.info(f"✓ Model training completed")
        logger.info(f"  - Best model: {results['best_model_name']}")
        logger.info(f"  - Elastic Net CV AUC: {results['elastic_net']['mean_score']:.4f} (+/- {results['elastic_net']['std_score']:.4f})")
        logger.info(f"  - Random Forest CV AUC: {results['random_forest']['mean_score']:.4f} (+/- {results['random_forest']['std_score']:.4f})")
        
        # ====== STEP 3: EVALUATION AND REPORTING ======
        print_header("Step 3: Evaluation and Comprehensive Reporting")
        logger.info("Generating evaluation reports and visualizations...")
        
        generate_full_evaluation_report(results)
        
        logger.info(f"✓ Evaluation completed")
        logger.info(f"  - Reports saved to: {REPORTS_DIR}/")
        logger.info(f"  - Models saved to: {MODELS_DIR}/")
        
        # ====== PIPELINE SUMMARY ======
        print_header("Pipeline Summary")
        
        summary_text = f"""
        Pipeline Execution Summary
        ──────────────────────────────────────────────────
        
        DATA
          • Total samples: {X.shape[0]}
          • Total features: {X.shape[1]}
          • Target classes: {y.nunique()}
          • Class distribution: {dict(y.value_counts())}
        
        MODELS TRAINED
          • Elastic Net (LogisticRegression with L1/L2)
            - Mean CV AUC: {results['elastic_net']['mean_score']:.4f} (+/- {results['elastic_net']['std_score']:.4f})
            - Best params: {results['elastic_net']['best_params']}
          
          • Random Forest
            - Mean CV AUC: {results['random_forest']['mean_score']:.4f} (+/- {results['random_forest']['std_score']:.4f})
            - Best params: {results['random_forest']['best_params']}
        
        BEST MODEL
          • Model: {results['best_model_name']}
          • CV AUC: {max(results['elastic_net']['mean_score'], results['random_forest']['mean_score']):.4f}
        
        OUTPUTS
          • Reports: {REPORTS_DIR}/
          • Models: {MODELS_DIR}/
          • Visualizations: CV scores and model comparison plots
        
        ──────────────────────────────────────────────────
        """
        
        logger.info(summary_text)
        print(summary_text)
        
        logger.info(f"Pipeline completed successfully at {datetime.now().isoformat()}")
        print("\n" + "=" * 80)
        print("  ✓ ML PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        print("=" * 80 + "\n")
        
        return results
    
    except Exception as e:
        error_msg = f"Pipeline failed with error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        print(f"\n✗ ERROR: {error_msg}")
        sys.exit(1)


if __name__ == '__main__':
    results = main()
