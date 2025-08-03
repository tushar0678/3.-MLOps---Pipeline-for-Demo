"""
data_preprocessing.py

This module performs preprocessing tasks on the training and testing datasets.
Steps:
- Label encodes the target column.
- Removes duplicate records.
- Cleans and normalizes the text using NLP techniques:
  - Lowercasing
  - Tokenization
  - Stopword & punctuation removal
  - Stemming

Author: Tushar Shukla
"""

import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

import string
import nltk


# Downloads required NLTK corpora on first run
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')


# ----------------------------
# Logging Configuration Setup
# ----------------------------

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def transform_text(text):
    """
    Cleans and transforms input text into a normalized form.
    
    Steps:
    - Converts text to lowercase.
    - Tokenizes text into words using nltk.
    - Removes non-alphanumeric characters.
    - Filters out stopwords and punctuation.
    - Applies stemming using PorterStemmer.
    
    Args:
        text (str): Raw text to clean.

    Returns:
        str: Cleaned and normalized text.
    """
    ps = PorterStemmer()

    # Step 1: Convert to lowercase
    text = text.lower()

    # Step 2: Tokenize into words
    text = nltk.word_tokenize(text)

    # Step 3: Remove non-alphanumeric tokens
    text = [word for word in text if word.isalnum()]

    # Step 4: Remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]

    # Step 5: Stem words (e.g., "running" -> "run")
    text = [ps.stem(word) for word in text]

    # Step 6: Re-join tokens into a cleaned string
    return " ".join(text)


def preprocess_df(df, text_column='text', target_column='target'):
    """
    Preprocess the DataFrame:
    - Label-encode the target column.
    - Remove duplicate rows.
    - Apply text normalization to the text column.
    
    Args:
        df (pd.DataFrame): Input dataset.
        text_column (str): Name of the column containing text data.
        target_column (str): Name of the column with labels.

    Returns:
        pd.DataFrame: Cleaned and preprocessed DataFrame.

    Raises:
        KeyError: If specified columns don't exist.
        Exception: For general failures.
    """
    try:
        logger.debug('Starting preprocessing for DataFrame')

        # Encode the target column: converts text labels (e.g., spam/ham) into integers (e.g., 0/1)
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('Target column encoded')

        # Drop duplicate rows, keeping the first occurrence
        df = df.drop_duplicates(keep='first')
        logger.debug('Duplicates removed')

        # Apply text cleaning to every row in the text column
        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug('Text column transformed')

        return df

    except KeyError as e:
        logger.error('Column not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error during text normalization: %s', e)
        raise


def main(text_column='text', target_column='target'):
    """
    Main function:
    - Loads raw train/test data.
    - Applies preprocessing.
    - Saves the cleaned datasets in `data/interim/` directory.

    Args:
        text_column (str): Text column name.
        target_column (str): Target/label column name.
    """
    try:
        # Load raw data files
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded properly')

        # Preprocess both datasets
        train_processed_data = preprocess_df(train_data, text_column, target_column)
        test_processed_data = preprocess_df(test_data, text_column, target_column)

        # Create output directory for processed data
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)

        # Save processed datasets
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)

        logger.debug('Processed data saved to %s', data_path)

    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")


# Only run this script if executed directly
if __name__ == '__main__':
    main()
