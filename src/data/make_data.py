"""
Data Loading and Preprocessing Module

This module provides functions for downloading the bank churn dataset from Kaggle,
cleaning and preprocessing the data, and creating binned versions of continuous features
for exploratory data analysis.

Functions:
    get_dataset: Download and load the raw dataset from Kaggle
    clean_dataset: Clean and preprocess the raw dataset
    create_bins: Create binned versions of continuous features
"""

import io
import os
import requests
import zipfile
import hashlib
import sqlite3

import pandas as pd

def get_dataset() -> pd.DataFrame:
    """
    Download and load the bank churn dataset from Kaggle.
    
    This function authenticates with the Kaggle API using environment variables
    (KAGGLE_USERNAME and KAGGLE_KEY) and downloads the 'churn.csv' file from
    the 'mathchi/churn-for-bank-customers' dataset.
    
    Returns:
        pd.DataFrame: The raw bank churn dataset
        
    Raises:
        ValueError: If Kaggle credentials are not found in environment variables
                   or if the expected file is not in the downloaded archive
        RuntimeError: If the download fails for any reason
        
    Environment Variables Required:
        KAGGLE_USERNAME: Kaggle account username
        KAGGLE_KEY: Kaggle API key
    """
    
    KAGGLE_USERNAME = os.environ.get('KAGGLE_USERNAME')
    KAGGLE_KEY = os.environ.get('KAGGLE_KEY')

    if not KAGGLE_USERNAME or not KAGGLE_KEY:
        raise ValueError("Credentials not found in environment variables")
    
    dataset_slug = "mathchi/churn-for-bank-customers"
    file_name = "churn.csv"
    url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset_slug}"
    
    response = requests.get(url, auth=(KAGGLE_USERNAME, KAGGLE_KEY), timeout=30)

    # Check to see if file was downloaded correctly
    try:
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            if file_name not in z.namelist():
                raise ValueError(f"File '{file_name}' not found in the downloaded ZIP archive.")
            with z.open(file_name) as f:
                df = pd.read_csv(f)
        return df

    except Exception as e:
        raise RuntimeError(f"Failed to download dataset: {e}")

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the raw bank churn dataset.
    
    This function performs the following preprocessing steps:
    1. Validates the input dataframe using MD5 checksum
    2. Drops unnecessary columns (CustomerId, RowNumber, Surname)
    3. Encodes categorical variables:
       - Gender: Male=0, Female=1
       - Geography: France=0, Spain=1, Germany=2
    
    Parameters:
        df (pd.DataFrame): Raw bank churn dataset
        
    Returns:
        pd.DataFrame: Cleaned dataset with encoded categorical variables
        
    Raises:
        ValueError: If the input dataframe's checksum doesn't match expected value
                   (ensures correct dataset is being processed)
    """

    # Check that we got the correct df
    checksum = hashlib.md5(df.to_csv().encode('utf-8')).hexdigest()
    if checksum != 'b64fead4dbbd83ac0b2276be7d210bbf':
        raise ValueError("Checksum for df is incorrect.")
    
    df_clean = df.drop(['CustomerId', 'RowNumber', 'Surname'], axis=1)

    # Convert gender to numeric
    gender_dict = {'Male': 0, 'Female': 1}
    df_clean['Gender'] = df_clean['Gender'].map(gender_dict).fillna(-1)

    # Convert countries to numeric
    country_dict = {'France': 0, 'Spain': 1, 'Germany': 2}
    df_clean['Geography'] = df_clean['Geography'].map(country_dict).fillna(-1)

    return df_clean

def create_bins(df: pd.DataFrame, bins_dict: dict, labels_dict: dict) -> pd.DataFrame:
    """
    Create binned versions of continuous features for exploratory data analysis.
    
    This function creates new columns with '_binned' suffix containing categorical
    versions of continuous features, useful for segmentation and visualization.
    
    Parameters:
        df (pd.DataFrame): Input dataframe with continuous features
        bins_dict (dict): Dictionary mapping feature names to bin edges
                         e.g., {'Age': [0, 30, 40, 50, 60, 100]}
        labels_dict (dict): Dictionary mapping feature names to bin labels
                           e.g., {'Age': ['<30', '30-40', '40-50', '50-60', '60+']}
                           
    Returns:
        pd.DataFrame: Copy of input dataframe with additional binned feature columns
        
    Note:
        - Original features are preserved
        - New columns are named '{feature}_binned'
        - Bins include lowest value (include_lowest=True)
        - Bins are ordered categories
    """
    
    df_binned = df.copy()
    for feature, bins in bins_dict.items():
        df_binned[feature + '_binned'] = pd.cut(df_binned[feature], bins=bins, labels=labels_dict[feature], include_lowest=True, ordered=True)

    return df_binned