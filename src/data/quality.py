"""
Data Quality and Exploratory Analysis Module

This module provides functions for data quality checks and exploratory
data analysis, including duplicate detection, column summarization,
multicollinearity assessment, and missing value visualization.

Functions:
    check_duplicates: Check for and visualize duplicate values in a column
    summarize_cols: Summarize all columns with value counts or statistics
    check_multicolinearity: Assess multicollinearity using pairplots and VIF
    check_missing: Check for missing values and visualize with missingno
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import missingno as msno
from dython.nominal import associations

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def check_duplicates(df, col):
    """
    Check for duplicate values in a column and visualize their distribution.
    
    If duplicates are found, creates a histogram showing the count of how many
    times each value appears.
    
    Parameters:
        df (pd.DataFrame): Input dataframe
        col (str): Column name to check for duplicates
        
    Returns:
        None: Prints summary and displays histogram if duplicates exist
    """
    freq = df[col].value_counts()
    if freq.iloc[0] == 1:
        print(f'No duplicates in {col}')
    else:
        num_dupes = len(freq[freq > 1])
        print(f'Number of values in {col} with duplicates: {num_dupes}')

        dup_counts = freq[freq > 1]
        plt.figure(figsize=(8,6))
        plt.hist(dup_counts, bins=range(2, int(dup_counts.max())+2), edgecolor='black')
        plt.xlabel('Number of Duplicates')
        plt.ylabel('Count of CustomerId')
        plt.title('Histogram of Duplicate Counts for CustomerId')
        plt.show()

def summarize_cols(df, many=25):
    """
    Summarize all columns in a dataframe with appropriate statistics.
    
    For categorical/string columns: displays value counts
    For numeric columns with few unique values: displays value counts sorted by value
    For numeric columns with many unique values: displays descriptive statistics
    
    Parameters:
        df (pd.DataFrame): Input dataframe to summarize
        many (int, optional): Threshold for "many" unique values. Default is 25.
                             Columns with more unique values than this are summarized
                             with describe() instead of value_counts()
                             
    Returns:
        None: Displays summaries using IPython.display
    """
    cols_many = []
    for col in df.columns:
        unique_count = df[col].nunique()
        if df[col].dtypes.name in ['object', 'string']:
            display(df[col].value_counts().sort_values(ascending=False))
        elif unique_count < many:
            display(df[col].value_counts().sort_index())
        else:
            cols_many.append(col)

    if cols_many:
        print(f"Combined describe for columns with >{many} unique values: {cols_many}")
        display(df[cols_many].describe())

def check_multicolinearity(df, num_features, cat_features, target):
    """
    Assess multicollinearity among features using visualizations and VIF.
    
    Creates a pairplot to visualize relationships between features colored by target,
    and calculates Variance Inflation Factor (VIF) for numerical features.
    
    Parameters:
        df (pd.DataFrame): Input dataframe
        num_features (list): List of numerical feature column names
        cat_features (list): List of categorical feature column names
        target (str): Name of the target column for coloring the pairplot
        
    Returns:
        pd.DataFrame: VIF scores for each numerical feature and constant term.
                     Higher VIF (>5-10) indicates potential multicollinearity.
                     
    Note:
        VIF measures how much the variance of a coefficient is inflated due to
        linear correlations with other features. VIF = 1/(1 - R²) where R² is
        from regressing that feature on all other features.
    """

    df_features = df[cat_features + num_features]

    associations(df_features, nominal_columns=cat_features, numerical_columns=num_features)

    sns.pairplot(df[num_features + cat_features + [target]], hue=target)
    plt.show()

    df_encoded = pd.get_dummies(df_features, columns=cat_features, drop_first=True)
    df_encoded = add_constant(df_encoded.astype(float).dropna())

    vif_data = pd.DataFrame()
    vif_data["Feature"] = df_encoded.columns
    vif_data["VIF"] = [variance_inflation_factor(df_encoded.values, i) for i in range(len(df_encoded.columns))]
    
    return vif_data.sort_values(by="VIF", ascending=False)

def check_missing(df):
    """
    Check for missing values in a DataFrame.
    
    If no missing values are found, prints a confirmation message.
    If missing values are present, displays a missingno matrix visualization.
    
    Parameters:
        df (pd.DataFrame): Input dataframe to check for missing values
        
    Returns:
        None: Either prints a message or displays a plot
    """
    if df.isnull().sum().sum() == 0:
        print("No missing values found in the dataset.")
    else:
        msno.matrix(df)