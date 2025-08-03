"""
feature_engineering.py

This script applies TF-IDF vectorization to text data in train/test datasets.
Steps:
- Loads YAML config for parameters
- Loads processed CSV datasets
- Transforms text using TfidfVectorizer
- Saves the output feature matrices as CSV

Author: Tushar Shukla
"""

import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import yaml

# ----------------------------
# Logging Configuration Setup
# ----------------------------

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('feature_engineering')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(file_path: str) -> pd.DataFrame:

    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)  # Fill missing text values with empty string
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    """
    Apply TF-IDF vectorization to train and test text data.

    Args:
        train_data (pd.DataFrame): Preprocessed training dataset.
        test_data (pd.DataFrame): Preprocessed testing dataset.
        max_features (int): Maximum number of TF-IDF features.

    Returns:
        tuple: (train_df, test_df) with transformed features and labels.

    Raises:
        Exception: For any TF-IDF related errors.
    """
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)

        # Extract raw text and labels
        X_train = train_data['text'].values
        y_train = train_data['target'].values
        X_test = test_data['text'].values
        y_test = test_data['target'].values

        # Fit and transform training data
        X_train_tfidf = vectorizer.fit_transform(X_train)

        # Transform test data (only transform, not fit)
        X_test_tfidf = vectorizer.transform(X_test)

        # Convert sparse matrix to DataFrame
        train_df = pd.DataFrame(X_train_tfidf.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_tfidf.toarray())
        test_df['label'] = y_test

        logger.debug('TF-IDF applied and data transformed')
        return train_df, test_df
    except Exception as e:
        logger.error('Error during TF-IDF transformation: %s', e)
        raise


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """
    Save DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): Data to save.
        file_path (str): Output file path.

    Raises:
        Exception: For file writing errors.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug('Data saved to %s', file_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise


def main():
    """
    Main function to execute the TF-IDF feature engineering pipeline:
    - Load config params
    - Load preprocessed train/test data
    - Apply TF-IDF transformation
    - Save results to disk
    """
    try:
        # Load parameters from params.yaml
        max_features = 40

        # Load preprocessed data
        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')

        # Apply TF-IDF vectorization
        train_df, test_df = apply_tfidf(train_data, test_data, max_features)

        # Save vectorized feature matrices
        save_data(train_df, os.path.join("./data", "processed", "train_tfidf.csv"))
        save_data(test_df, os.path.join("./data", "processed", "test_tfidf.csv"))

    except Exception as e:
        logger.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")


# Ensure script runs only when executed directly
if __name__ == '__main__':
    main()
