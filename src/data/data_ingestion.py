import os
import yaml
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Logging configuration
logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("data_ingestion.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def read_yaml(params_path: str) -> dict:
    """Load parameters from YAML file.
    
    Args:
        params_path (str): Path to YAML parameter file
        
    Returns:
        dict: Dictionary containing parameters
    """
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug(f"Parameters retrieved from {params_path}")
        return params
    except FileNotFoundError as e:
        logger.error(f"Error reading YAML file {params_path}: {e}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML: {e}")
        raise
    except Exception as e:
        logger.error(f'Unexpected error: {e}')
        raise
    
def ingest_data(data_url: str) -> pd.DataFrame:
    """Load data from CSV file or URL.
    
    Args:
        data_url (str): Path or URL to CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    try:
        df = pd.read_csv(data_url)
        logger.debug(f"Data ingested from {data_url}")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file from {data_url}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error ingesting data from {data_url}: {e}")
        raise
    
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess data.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    try:
        # Removing missing values
        df.dropna(inplace=True)
        # Removing duplicates
        df.drop_duplicates(inplace=True)
        # Removing rows with empty strings
        df = df[df['clean_comment'].str.strip() != '']
        
        logger.debug("Data preprocessing completed: Missing values, duplicates, and empty strings removed")
        return df
    
    except KeyError as e:
        logger.error(f"Error preprocessing data: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during preprocessing: {e}")
        raise
    
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save train and test data to CSV files.
    
    Args:
        train_data (pd.DataFrame): Training data
        test_data (pd.DataFrame): Testing data
        data_path (str): Base directory path to save data
    """
    try:
        raw_data_path = os.path.join(data_path, "raw")
        
        os.makedirs(raw_data_path, exist_ok=True)
        
        # Save the train and test datasets to CSV files
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        
        logger.debug(f"Data split into train and test sets and saved to {raw_data_path}")
    except Exception as e:
        logger.error(f"Error saving split data: {e}")
        raise

# ✅ CORRECT: main() at module level (no indentation)
def main():
    """Main execution function for data ingestion pipeline."""
    try: 
        # Load parameters from YAML
        params_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../params.yaml')
        params = read_yaml(params_path=params_path)  # ✅ Fixed: Use read_yaml, not ingest_data
        
        # Get parameters
        test_size = params['split_data']['test_size']
        random_state = params['split_data']['random_state']  # ✅ Fixed: Define random_state
        
        # Ingest data
        df = ingest_data(data_url="https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv")
        
        # Preprocess data
        final_df = preprocess_data(df=df)
        
        # Split data
        train_data, test_data = train_test_split(
            final_df, 
            test_size=test_size, 
            random_state=random_state
        )
        
        # Save data
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/')
        save_data(train_data=train_data, test_data=test_data, data_path=data_path)
        
        logger.info("Data ingestion pipeline completed successfully!")
    
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()