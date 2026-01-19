"""
Data Pipeline Script

This script downloads the raw bank churn dataset from Kaggle, cleans it,
and splits it into training and test sets for model development.

Usage:
    python bin/build_data.py --raw-dir-path data/raw/ --clean-dir-path data/clean/

The script performs the following steps:
1. Loads Kaggle API credentials from environment variables
2. Downloads the raw dataset from Kaggle
3. Cleans and preprocesses the data
4. Creates a stratified train/test split (80/20)
5. Saves all datasets to specified directories
"""

import argparse

import pandas as pd
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv, find_dotenv

from src.data.make_data import get_dataset, clean_dataset

parser = argparse.ArgumentParser(description="Download and clean bank churn dataset.")
parser.add_argument("--raw-dir-path", type=str, help="Path to save the raw dataset CSV.")
parser.add_argument("--clean-dir-path", type=str, help="Path to save the clean datasets CSV.")
args = parser.parse_args()

if args.raw_dir_path:
    # Load Kaggle API credentials from .env file
    load_dotenv(find_dotenv())
    
    # Download raw dataset from Kaggle
    df_raw = get_dataset()
    df_raw.to_csv(args.raw_dir_path + 'raw_data.csv', index=False)

    print(f"Raw data saved to {args.raw_dir_path}")
    
    if args.clean_dir_path:
        # Clean the dataset (remove unnecessary columns, encode categorical variables)
        df_clean = clean_dataset(df_raw)
        
        # Create stratified train/test split to preserve class distribution
        train, test = train_test_split(
            df_clean,
            test_size=0.2,
            shuffle=True,
            random_state=42,
            stratify=df_clean['Exited']
        )
        
        train.to_csv(args.clean_dir_path + 'train.csv', index=False)
        test.to_csv(args.clean_dir_path + 'test.csv', index=False)

        print(f"Data cleaned successfully and saved to {args.clean_dir_path}")
    
    else:
        
        print("cleaned data ready (not saved)")

else:
    
    print("Raw data loaded (not saved)")
