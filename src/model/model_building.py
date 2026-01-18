import os
import yaml
import pickle
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer


# ─── LOGGING SETUP ──────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_building.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load model parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug("Parameters retrieved from %s", params_path)
        return params
    except FileNotFoundError:
        logger.error("File not found: %s", params_path)
        raise
    except yaml.YAMLError as e:
        logger.error("YAML error: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise
        
def load_data(data_path: str) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    try:
        df = pd.read_csv(data_path)
        df.fillna('', inplace=True)
        logger.debug("Data loaded from %s", data_path)
        return df
    except pd.errors.ParserError as e:
        logger.error("Parsing error: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise
        
def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int, ngram_range: tuple) -> tuple:
    """Apply TF-IDF vectorization to text data."""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        
        X_train = train_data['clean_comment'].values  # Fixed column name
        y_train = train_data['category'].values
        
        X_test = test_data['clean_comment'].values  # Fixed: was incomplete
        y_test = test_data['category'].values
        
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        logger.debug(f"TF-IDF transformation completed. Train shape: {X_train_tfidf.shape}")
        
        with open(os.path.join(get_root_directory(), 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f)
            logger.debug("TF-IDF vectorizer saved to tfidf_vectorizer.pkl")
            
        return X_train_tfidf, y_train, X_test_tfidf, y_test
            
    except Exception as e:
        logger.error("Error in TF-IDF vectorization: %s", e)
        raise
    
def train_lgbm(X_train: np.ndarray, y_train: np.ndarray, learning_rate: float,
               max_depth: int, n_estimators: int) -> lgb.LGBMClassifier:
    """Train a LightGBM model."""
    try:
        best_model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=3,
            metric='multi_logloss',
            class_weight='balanced',
            is_unbalance=True,
            reg_alpha=0.1,
            reg_lambda=1.0,
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators
        )
        
        best_model.fit(X_train, y_train)        
        logger.debug("LightGBM model training completed.")
        return best_model
        
    except Exception as e:
        logger.error("Error in LightGBM training: %s", e)
        raise
        
def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug("Model saved to %s", file_path)
    except Exception as e:
        logger.error("Error saving model: %s", e)
        raise
        
# ─── HELPER FUNCTION ──────────────────────────────────────────────
def get_root_directory() -> str:
    """Get the root directory of the project."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '..', '..'))
    
def main():
    try:
        root_dir = get_root_directory()
        
        # Load parameters
        params = load_params(os.path.join(root_dir, 'params.yaml'))
        max_features = params['model_building']['max_features']
        ngram_range = tuple(params['model_building']['ngrams_range'])    # ✅ Added 's'
        learning_rate = params['model_building']['learning_rate']        # ✅ Fixed brackets
        max_depth = params['model_building']['max_depth']                # ✅ Fixed brackets
        n_estimators = params['model_building']['n_estimators']
        
        # Load data
        train_data = load_data(os.path.join(root_dir, 'data/interim/train_processed.csv'))
        test_data = load_data(os.path.join(root_dir, 'data/interim/test_processed.csv'))
        
        # Apply TF-IDF
        X_train_tfidf, y_train, X_test_tfidf, y_test = apply_tfidf(
            train_data, test_data, max_features, ngram_range
        )
        
        # Train model
        best_model = train_lgbm(X_train_tfidf, y_train, learning_rate, max_depth, n_estimators)
        
        # Save model
        save_model(best_model, os.path.join(root_dir, 'lgbm_model.pkl'))
        
        logger.info("Model building pipeline completed successfully!")
    
    except Exception as e:
        logger.error("Error in model building pipeline: %s", e)
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()