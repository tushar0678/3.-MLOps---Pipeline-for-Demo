"""
model_building.py

This script:
- Loads processed training data
- Loads model hyperparameters from YAML
- Trains a RandomForestClassifier
- Saves the trained model as a .pkl file

Author: Tushar Shukla
"""

import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
import yaml

# ----------------------------
# Logging Configuration Setup
# ----------------------------

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'model_building.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)



def load_data(file_path: str) -> pd.DataFrame:
    """
    Load CSV dataset into a pandas DataFrame.

    Args:
        file_path (str): Path to the input CSV.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError, ParserError, Exception
    """
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s with shape %s', file_path, df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> RandomForestClassifier:
    """
    Train a RandomForestClassifier using training data and parameters.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        params (dict): Hyperparameters like n_estimators and random_state.

    Returns:
        RandomForestClassifier: Trained model instance.

    Raises:
        ValueError: If X and y sizes mismatch.
        Exception: For other training errors.
    """
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("Mismatch: X_train rows != y_train length.")

        # Initialize classifier with hyperparameters
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])


        logger.debug('Model initialized with %d samples', X_train.shape[0])
        clf.fit(X_train, y_train)  # Train the model
        logger.debug('Model training completed')

        return clf
    except ValueError as e:
        logger.error('ValueError during model training: %s', e)
        raise
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise


def save_model(model, file_path: str) -> None:
    """
    Serialize and save the trained model using pickle.

    Args:
        model: Trained ML model object.
        file_path (str): Path where model is to be saved.

    Raises:
        Exception: If writing fails.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as file:
            pickle.dump(model, file)

        logger.debug('Model saved to %s', file_path)
    except FileNotFoundError as e:
        logger.error('File path not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise


def main():
    """
    Main driver function:
    - Loads model params
    - Loads training data
    - Trains model
    - Saves trained model to disk
    """
    try:
        params = {'n_estimators':25, 'random_state' : 2}

        # Load processed training data
        train_data = load_data('./data/processed/train_tfidf.csv')

        # Split into features and label
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        # Train model
        clf = train_model(X_train, y_train, params)

        # Save trained model
        model_save_path = 'models/model.pkl'
        save_model(clf, model_save_path)

    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")


# Run only when executed directly
if __name__ == '__main__':
    main()
