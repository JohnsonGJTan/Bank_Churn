"""
Model Training Pipeline Script

This script trains a machine learning pipeline for bank churn prediction
using XGBoost with grid search for hyperparameter optimization.

Usage:
    python bin/build_pipeline.py --clean-data-path data/clean/train.csv --pipeline-save-path pipelines/

The script performs the following steps:
1. Loads preprocessed training data
2. Configures feature preprocessing (one-hot encoding for categorical features)
3. Configures XGBoost classifier with initial parameters
4. Performs grid search with stratified k-fold cross-validation
5. Saves the best estimator to the specified path

The final model is a scikit-learn Pipeline that includes:
- ColumnTransformer for feature preprocessing
- XGBClassifier for binary classification
"""

import argparse

import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train and save model")
parser.add_argument("--clean-data-path", type=str, help="Path to cleaned training data.")
parser.add_argument("--pipeline-save-path", type=str, help="Path to model save directory.")

args = parser.parse_args()

# Load training data
df_train = pd.read_csv(args.clean_data_path)
X, y = df_train.drop(['Exited'], axis = 1), df_train['Exited']

# Configure preprocessing: One-hot encode categorical features
# - Geography: France, Spain, Germany
# - Gender: Male, Female
# - drop='first' to avoid multicollinearity
# - remainder='passthrough' keeps other features unchanged
processor = ColumnTransformer(
    [
        (
            'OHEnc', 
            OneHotEncoder(dtype='int', sparse_output=False, handle_unknown='ignore'),
            ['Geography', 'Gender']
        )
    ],
    remainder='passthrough', verbose_feature_names_out=False
)

# Configure XGBoost classifier with initial hyperparameters
# - scale_pos_weight=4: Handles class imbalance (typically ~20% churn rate)
# - subsample/colsample: Prevent overfitting through row/column sampling
# - n_jobs=-1: Use all available CPU cores
classifier = XGBClassifier(
    n_estimators = 1000,
    max_depth = 4,
    learning_rate = 0.05,
    scale_pos_weight=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1, 
)

# Build model pipeline
model_pipeline = Pipeline([
    ('processor', processor),
    ('classifier', classifier)
])

# Define hyperparameter grid for optimization
# Searching over learning rate, number of trees, max depth, sampling, and class weight
param_grid = {
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [3, 5, 7],
    'classifier__subsample': [0.8, 1.0],
    'classifier__scale_pos_weight': [1, 2, 3, 4]
}

# Use stratified k-fold to preserve class distribution across folds
shared_cv = StratifiedKFold(shuffle=True, random_state=42)

# Perform grid search with average precision as scoring metric
# (appropriate for imbalanced classification)
model_grid = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    cv=shared_cv,
    scoring='average_precision',
    n_jobs=-1
)

# Train model with grid search
model_grid.fit(X, y)

# Save the best model pipeline to disk using joblib
joblib.dump(model_grid.best_estimator_, args.pipeline_save_path + "final_model.joblib")