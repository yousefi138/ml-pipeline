# ML Pipeline - Environment Setup Guide

This guide covers how to set up your compute environment to run the ML Pipeline using Mamba (or Conda).

## Prerequisites

You need to have mamba (or conda) installed on your system. If you don't have it:

- **Mamba Installation**: https://github.com/conda-forge/miniforge
- **Conda Installation**: https://www.anaconda.com/

(Mamba is recommended as it's faster than conda)

## Quick Setup (5 minutes)

### 1. Navigate to the Pipeline Directory

```bash
cd /path/to/ml-pipeline/working/scripts/ml-pipeline
```

Replace `/path/to/ml-pipeline` with the actual path where you have the pipeline code.

### 2. Create and Activate the Mamba Environment

```bash
# Create environment with all dependencies
mamba create -n ml-pipeline python=3.10 pandas numpy scikit-learn scikit-survival pyyaml matplotlib seaborn joblib shap -y

# Activate the environment
mamba activate ml-pipeline
```

This creates a new environment called `ml-pipeline` with Python 3.10 and all required packages.

### 3. Verify Installation

Test that the environment is set up correctly:

```bash
python -c "import pandas, numpy, sklearn, matplotlib, seaborn, joblib, shap; print('✓ All dependencies installed successfully')"
```

You should see: `✓ All dependencies installed successfully`

### 4. (Optional) Create Requirements File

If you want to document the exact versions used, generate a requirements file:

```bash
mamba list --export > environment.yml
```

This creates an `environment.yml` file that can be used to recreate the exact environment later.

## Environment Details

### Python Version
- **Recommended**: Python 3.10 or 3.11
- **Minimum**: Python 3.8

### Required Packages

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | ≥1.3.0 | Data manipulation and analysis |
| numpy | ≥1.21.0 | Numerical computing |
| scikit-learn | ≥1.0.0 | Machine learning models and CV |
| scikit-survival | ≥0.17.0 | Survival analysis support |
| PyYAML | ≥5.4.0 | Configuration file parsing |
| matplotlib | ≥3.4.0 | Plotting and visualization |
| seaborn | ≥0.11.0 | Statistical visualizations |
| joblib | ≥1.1.0 | Model caching and parallelization |
| shap | ≥0.41.0 | SHAP explainability values |

## Working with Environments

### Activating Your Environment

```bash
mamba activate ml-pipeline
```

### Deactivating Your Environment

```bash
mamba deactivate
```

### Listing All Environments

```bash
mamba env list
```

### Removing an Environment

If you need to remove the environment:

```bash
mamba remove --name ml-pipeline --all
```

## Running the Pipeline

Once your environment is set up and activated:

```bash
mamba activate ml-pipeline
cd /path/to/ml-pipeline/working/scripts/ml-pipeline
python ml_pipeline.py
```

See [QUICK_GUIDE.md](QUICK_GUIDE.md) for pipeline usage and options.

## Troubleshooting

### Issue: `mamba: command not found`

**Solution**: Mamba is not installed or not in your PATH. 
- Install miniforge from https://github.com/conda-forge/miniforge
- Restart your terminal after installation

### Issue: `ModuleNotFoundError` when running pipeline

**Solution**: Make sure your mamba environment is activated:
```bash
mamba activate ml-pipeline
```

You should see `(ml-pipeline)` in your terminal prompt.

### Issue: Package installation fails

**Solution**: Update mamba and retry:
```bash
mamba update -n base conda-forge::mamba -y
mamba create -n ml-pipeline python=3.10 pandas numpy scikit-learn scikit-survival pyyaml matplotlib seaborn joblib shap -y
```

### Issue: Slow performance or memory issues

**Solution**: Adjust the number of parallel jobs in `config.py`:
```python
N_JOBS = 4  # Instead of -1 (all cores)
```

This limits the pipeline to use 4 CPU cores instead of all available cores.

## Alternative: Using Conda Instead of Mamba

If you prefer conda (slightly slower but more widely available):

```bash
# Create environment with conda
conda create -n ml-pipeline python=3.10 pandas numpy scikit-learn scikit-survival pyyaml matplotlib seaborn joblib shap -y

# Activate
conda activate ml-pipeline

# Rest of the workflow is identical
```

## Environment Verification Checklist

Before running the pipeline, verify:

- [ ] Mamba/Conda is installed: `mamba --version`
- [ ] Environment created: `mamba env list`
- [ ] Environment activated: Terminal shows `(ml-pipeline)`
- [ ] Pipeline code available: `ls ml_pipeline.py`
- [ ] Config file present: `ls config.yml`
- [ ] Data file present: `ls data/` (should have CSV files)
- [ ] Dependencies installed: `python -c "import pandas, sklearn, etc"`

## Next Steps

Once your environment is set up:

1. Read [QUICK_GUIDE.md](QUICK_GUIDE.md) for a 5-minute overview
2. Review [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) for detailed documentation
3. Run the pipeline: `python ml_pipeline.py`

For more help, check the logs in `results/logs/` after running the pipeline.
