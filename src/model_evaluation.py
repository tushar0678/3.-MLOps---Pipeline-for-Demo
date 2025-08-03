"""
model_evaluation.py

This script:
- Loads a trained model
- Loads processed test data
- Evaluates the model using standard classification metrics
- Logs metrics with dvclive for experiment tracking
- Saves evaluation results to a JSON file

Author: Tushar Shukla
"""

import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import yaml
#from dvclive import Live

# ----------------------------
# Logging Configuration Setup
# ----------------------------

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# def load_params(params_path: str) -> dict:
#     """
#     Load parameters from a YAML file.

#     Args:
#         params_path (str): Path to YAML config file.

#     Returns:
#         dict: Parameters dictionary.
#     """
#     try:
#         with open(params_path, 'r') as file:
#             params = yaml.safe_load(file)
#         logger.debug('Parameters retrieved from %s', params_path)
#         return params
#     except FileNotFoundError:
#         logger.error('File not found: %s', params_path)
#         raise
#     except yaml.YAMLError as e:
#         logger.error('YAML error: %s', e)
#         raise
#     except Exception as e:
#         logger.error('Unexpected error: %s', e)
#         raise


def load_model(file_path: str):
    """
    Load a trained ML model from a pickle (.pkl) file.

    Args:
        file_path (str): Path to the model file.

    Returns:
        model: Trained model object.
    """
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate a trained classifier on test data using standard metrics.

    Args:
        clf: Trained classifier.
        X_test (np.ndarray): Feature matrix for test data.
        y_test (np.ndarray): True labels.

    Returns:
        dict: Dictionary with evaluation metrics.
    """
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]  # Probabilities for ROC AUC

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }

        logger.debug('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise


def save_metrics(metrics: dict, file_path: str) -> None:
    """
    Save evaluation metrics to a JSON file.

    Args:
        metrics (dict): Evaluation metrics.
        file_path (str): Output JSON file path.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)

        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise


def main():
    """
    Main driver function:
    - Loads test data & model
    - Evaluates the model
    - Logs metrics via DVC Live
    - Saves metrics to reports/metrics.json
    """
    try:
        # Load configuration parameters
        # params = load_params(params_path='params.yaml')

        # Load the trained model
        clf = load_model('./models/model.pkl')

        # Load processed test data
        test_data = load_data('./data/processed/test_tfidf.csv')
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        # Evaluate model
        metrics = evaluate_model(clf, X_test, y_test)



        # Use dvclive for experiment tracking
        # with Live(save_dvc_exp=True) as live:
        #     for metric_name, value in metrics.items():
        #         live.log_metric(metric_name, value)  # Log each metric correctly

        #     live.log_params(params)  # Logs all params for reproducibility




        # Save metrics to reports/
        save_metrics(metrics, 'reports/metrics.json')

    except Exception as e:
        logger.error('Failed to complete the model evaluation process: %s', e)
        print(f"Error: {e}")


# Run only when this script is executed directly
if __name__ == '__main__':
    main()
