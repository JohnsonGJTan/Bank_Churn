"""
Custom Feature Transformers for Bank Churn Prediction

This module provides custom scikit-learn compatible transformers for handling
the specific distributional characteristics of bank churn features:

- CreditScoreTransformer: Handles right-censored credit scores (max=850)
- BalanceTransformer: Handles zero-inflated account balance distribution
- AgeTransformer: Handles right-skewed age distribution with rounding spikes
- TenureTransformer: Adds polynomial features for tenure
- RatioTransformer: Creates ratio features between continuous variables

All transformers inherit from BaseEstimator and TransformerMixin to ensure
compatibility with scikit-learn pipelines.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, StandardScaler

# CreditScore transformer for censored feature
class CreditScoreTransformer(BaseEstimator, TransformerMixin):
    """
    Transform credit scores with right-censoring at 850.
    
    Credit scores in the dataset are censored at 850 (maximum possible value),
    creating a non-normal distribution. This transformer:
    1. Scales the credit scores using RobustScaler (resistant to outliers)
    2. Creates a binary flag indicating whether the score equals 850
    
    This approach allows the model to:
    - Handle the censored distribution appropriately
    - Identify customers at the maximum credit score as a special group
    
    Attributes:
        scaler (RobustScaler): Scaler fitted on credit score values
        
    Output Features:
        - CreditScore: Robust-scaled credit score
        - credit_score_850: Binary flag (1 if score=850, 0 otherwise)
    """
    def __init__(self):
        self.scaler = RobustScaler()
    
    def fit(self, X, y=None):
        X = np.asarray(X).flatten()
        self.scaler.fit(X.reshape(-1, 1))
        return self
    
    def transform(self, X):
        X = np.asarray(X).flatten()
        flag = (X == 850).astype(int)
        scaled = self.scaler.transform(X.reshape(-1, 1)).flatten()
        
        return np.column_stack([scaled, flag])
    
    def get_feature_names_out(self, input_features=None):
        return ['CreditScore', 'credit_score_850']

# Balance transformer for zero-inflated distribution
class BalanceTransformer(BaseEstimator, TransformerMixin):
    """
    Transform account balance with zero-inflation.
    
    Many customers have zero balance, creating a zero-inflated distribution.
    This transformer:
    1. Fits StandardScaler only on non-zero balances
    2. Scales non-zero balances, keeps zeros as 0
    3. Creates a binary flag indicating zero balance
    
    This approach allows the model to:
    - Handle the spike at zero appropriately
    - Identify zero-balance customers as a distinct segment
    - Scale non-zero balances normally
    
    Attributes:
        scaler (StandardScaler): Scaler fitted on non-zero balance values
        
    Output Features:
        - Balance: Scaled account balance (zeros remain 0)
        - balance_0: Binary flag (1 if balance=0, 0 otherwise)
    """
    def __init__(self):
        self.masked_scaler = StandardScaler()
    
    def fit(self, X, y=None):
        X = np.asarray(X).flatten()
        # Fit scaler on non-zero values
        non_zero = X[X != 0]
        self.masked_scaler.fit(non_zero.reshape(-1, 1))
        return self
    
    def transform(self, X):
        X = np.asarray(X).flatten()
        flag = (X == 0).astype(int)
        scaled = self.masked_scaler.transform(X.reshape(-1, 1)).flatten()
        scaled[X == 0] = 0  # Keep zeros as 0 after scaling
        
        return np.column_stack([scaled, flag])
    
    def get_feature_names_out(self, input_features=None):
        return ['Balance', 'balance_0']

# Age transformer for right skewed normal with spikes caused by rounding
class AgeTransformer(BaseEstimator, TransformerMixin):
    """
    Transform age with log transformation and rounding flag.
    
    Age distribution is right-skewed with spikes at round numbers (30, 40, 50, 60, 70)
    due to age rounding. This transformer:
    1. Applies log transformation to reduce right skew
    2. Scales the log-transformed ages
    3. Creates a binary flag for round-number ages
    
    This approach allows the model to:
    - Handle the skewed distribution through log transformation
    - Identify potential age rounding/estimation patterns
    
    Attributes:
        scaler (StandardScaler): Scaler fitted on log-transformed age values
        
    Output Features:
        - Age: Scaled log-transformed age
        - age_round_flag: Binary flag (1 if age in [30,40,50,60,70], 0 otherwise)
    """
    def __init__(self):
        self.scaler = StandardScaler()
    
    def fit(self, X, y=None):
        X = np.asarray(X).flatten()
        log_age = np.log(X)
        self.scaler.fit(log_age.reshape(-1, 1))
        return self
    
    def transform(self, X):
        X = np.asarray(X).flatten()
        round_flag = np.isin(X, [30, 40, 50, 60, 70]).astype(int)
        log_age = np.log(X)
        scaled = self.scaler.transform(log_age.reshape(-1, 1)).flatten()
        
        return np.column_stack([scaled, round_flag])
    
    def get_feature_names_out(self, input_features=None):
        return ['Age', 'age_round_flag']

# Tenure transformer
class TenureTransformer(BaseEstimator, TransformerMixin):
    """
    Add polynomial features for customer tenure.
    
    Creates squared tenure to capture potential non-linear relationships
    between customer tenure and churn probability.
    
    Output Features:
        - Tenure: Original tenure in years
        - Tenure_sq: Squared tenure (Tenure²)
        
    Note:
        No fitting required; transformation is deterministic.
    """
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = np.asarray(X).flatten()
        tenure_sq = X ** 2
        
        return np.column_stack([X, tenure_sq])
    
    def get_feature_names_out(self, input_features=None):
        return ['Tenure', 'Tenure_sq']


# Ratio transformer for CreditScore/Age, EstimatedSalary/Age, Balance/Age
class RatioTransformer(BaseEstimator, TransformerMixin):
    """
    Create ratio features between financial metrics and age.
    
    Generates ratio features that may capture life-stage adjusted financial metrics:
    - CreditScore/Age: Credit score relative to age
    - EstimatedSalary/Age: Annual income relative to age  
    - Balance/Age: Savings relative to age
    
    These ratios can help identify customers who are over/under-performing
    financially relative to their age cohort.
    
    Input Features Expected (in order):
        X[:, 0]: CreditScore
        X[:, 1]: Age
        X[:, 2]: EstimatedSalary
        X[:, 3]: Balance
        
    Output Features:
        - CreditScore, Age, EstimatedSalary, Balance: Original features
        - CreditScore_Age_ratio: CreditScore / Age
        - EstimatedSalary_Age_ratio: EstimatedSalary / Age
        - Balance_Age_ratio: Balance / Age
        
    Attributes:
        feature_names_in_ (list): Names of input features
    """

    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        self.feature_names_in_ = ['CreditScore', 'Age', 'EstimatedSalary', 'Balance']
        return self
    
    def transform(self, X):
        X = np.asarray(X)
        # Assume X has columns: CreditScore, Age, EstimatedSalary, Balance
        credit_age = X[:, 0] / X[:, 1]
        salary_age = X[:, 2] / X[:, 1]
        balance_age = X[:, 3] / X[:, 1]
        return np.column_stack([X, credit_age, salary_age, balance_age])
    
    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_ + ['CreditScore_Age_ratio', 'EstimatedSalary_Age_ratio', 'Balance_Age_ratio']