import pandas as pd
import yaml
import os
from sksurv.datasets import load_breast_cancer

# Read config.yml to get project path
config_path = "config.yml"
with open(config_path, 'r') as config_file:
    config = yaml.safe_load(config_file)

project_path = config['default']['project']

# Load data
X, y = load_breast_cancer()

# Convert y (structured array) into a DataFrame
y_df = pd.DataFrame(y)

# Combine X and y into one DataFrame
df = pd.concat([y_df, pd.DataFrame(X, columns=X.columns)], axis=1)

# Save to CSV using project path
output_path = os.path.join(project_path, "data", "breast_cancer_survival.csv")
df.to_csv(output_path, index=False)
