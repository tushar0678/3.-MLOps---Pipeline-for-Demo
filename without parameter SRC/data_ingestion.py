"""
data_ingestion.py

This module handles the ingestion of data for an ML pipeline:
- Loads data from a given URL or file path.
- Preprocesses the data (renaming and dropping columns).
- Splits into train/test sets.
- Saves the datasets for downstream ML processes.

Author: Tushar Shukla
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging
import yaml  # Currently unused, but could be used for config management.

# ----------------------------
# Logging Configuration Setup
# ----------------------------

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Create a logger object with the name 'data_ingestion'
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)  # Capture all logs at DEBUG level and above

# Console handler to display logs on the terminal
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# File handler to save logs to a file
log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

# Log format structure
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(data_url: str) -> pd.DataFrame:
    """
    Load data from a CSV file hosted at a URL or file path.

    Args:
        data_url (str): Path or URL to the CSV file.

    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame.

    Raises:
        pd.errors.ParserError: If CSV parsing fails.
        Exception: For other unexpected errors.
    """
    try:
        df = pd.read_csv(data_url)  # Method from pandas to read CSV into DataFrame
        logger.debug('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the raw data.

    Operations:
    - Drops unnamed junk columns.
    - Renames 'v1' to 'target' and 'v2' to 'text'.

    Args:
        df (pd.DataFrame): Raw data.

    Returns:
        pd.DataFrame: Preprocessed data.

    Raises:
        KeyError: If expected columns are missing.
        Exception: For other processing errors.
    """
    try:
        # Drop irrelevant unnamed columns
        df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
        # Rename columns to meaningful names
        df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
        logger.debug('Data preprocessing completed')
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """
    Save the train and test datasets as CSV files.

    Args:
        train_data (pd.DataFrame): Training dataset.
        test_data (pd.DataFrame): Testing dataset.
        data_path (str): Base directory where data should be saved.

    Raises:
        Exception: If writing the files fails.
    """
    try:
        # Define path for raw data storage
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)  # Ensure directory exists

        # Save train and test datasets to CSV
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)

        logger.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise


def main():
    """
    Main function that drives the entire data ingestion pipeline:
    - Downloads the dataset.
    - Preprocesses the dataset.
    - Splits it into train/test sets.
    - Saves the results to disk.
    """
    try:
        test_size = 0.2  # 20% of data used for testing
        data_path = 'https://raw.githubusercontent.com/tushar0678/4.-MLOps--Dataset/refs/heads/main/spam.csv'

        # Load the dataset
        df = load_data(data_url=data_path)

        # Clean and prepare the data
        final_df = preprocess_data(df)

        # Split into training and testing sets using sklearn utility
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=2)

        # Save the split datasets locally
        save_data(train_data, test_data, data_path='./data')
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")


# This ensures the script runs only when executed directly, not when imported
if __name__ == '__main__':
    main()
