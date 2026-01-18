import os
import json
import yaml
import pickle
import dagshub
import logging

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import seaborn as sns
import matplotlib.pyplot as plt

from mlflow.models import infer_signature
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ─── LOGGING SETUP ──────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)  
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_evaluation.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter) 
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug("Data loaded from %s", file_path)
        return df
    except Exception as e:
        logger.error("Error loading data from %s: %s", file_path, e)
        raise
    
def load_model(model_path: str):
    """Load trained model from pickle file."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', model_path)
        return model
    except Exception as e:
        logger.error("Error loading model from %s: %s", model_path, e)
        raise
    
def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    """Load TF-IDF vectorizer from pickle file."""
    try:
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        logger.debug('TF-IDF Vectorizer loaded from %s', vectorizer_path)
        return vectorizer
    except Exception as e:
        logger.error("Error loading vectorizer from %s: %s", vectorizer_path, e)
        raise
    
def load_params(params_path: str) -> dict:
    """Load model parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug("Parameters retrieved from %s", params_path)
        return params
    except Exception as e:
        logger.error("Unexpected error %s: %s", params_path, e)
        raise

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """Evaluate the model and generate evaluation metrics and plots."""
    try:
        # Predictions
        y_pred = model.predict(X_test)
        
        # Classification Report
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        logger.debug("Model evaluation completed.")
        
        return report, cm
    except Exception as e:
        logger.error("Error during model evaluation: %s", e)
        raise

def log_confusion_matrix(cm, dataset_name, class_labels=['negative', 'neutral', 'positive']):
    """Log confusion matrix as an artifact."""
    try:
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap='Blues')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix for ' + dataset_name)
        
        cm_path = f'confusion_matrix_{dataset_name.replace(" ", "_")}.png'
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()
        logger.debug("Confusion matrix saved to %s", cm_path)
    except Exception as e:
        logger.error("Error logging confusion matrix: %s", e)
        raise
        
def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save model information including parameters and metrics to a JSON file."""
    try:
        model_info = {
            'run_id': run_id,
            'model_path': model_path
        }
        
        with open(file_path, 'w') as f:
            json.dump(model_info, f, indent=4)
        
        logger.debug("Model information saved to %s", file_path)
        
    except Exception as e:
        logger.error("Error saving model information to %s: %s", file_path, e)
        raise
    
def main():
    """Main function to evaluate the model."""
    
    # 1. Initialize DagsHub FIRST
    # This sets the tracking URI and authentication headers for the remote server
    dagshub.init(repo_owner='raijiwan275',
                repo_name='youtubanalysis_mlflow',
                mlflow=True)
    
    # 2. Set the Experiment name AFTER DagsHub is initialized
    # This ensures the experiment is created/found on DagsHub, not locally
    mlflow.set_experiment("youtube_sentiment_analysis")
    
    # 3. Start the run
    with mlflow.start_run() as run:
        try:
            # Use relative paths based on project root
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            params = load_params(os.path.join(root_dir, 'params.yaml'))
            
            # Log parameters
            for key, value in params.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        mlflow.log_param(f"{key}.{sub_key}", sub_value)
                else:
                    mlflow.log_param(key, value)
                
            # Load model and vectorizer
            model = load_model(os.path.join(root_dir, 'lgbm_model.pkl'))
            vectorizer = load_vectorizer(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))
            
            # Load test data
            test_data = load_data(os.path.join(root_dir, 'data/interim/test_processed.csv'))
            
            # Transform test data
            X_test_tfidf = vectorizer.transform(test_data['clean_comment'].values)
            y_test = test_data['category'].values
            
            # Create input example for MLflow model signature
            input_example = pd.DataFrame(
                X_test_tfidf[:5].toarray(), 
                columns=vectorizer.get_feature_names_out()
            )
            
            # Infer signature
            signature = infer_signature(input_example, model.predict(X_test_tfidf[:5]))
            
            # Log model to DagsHub MLflow
            mlflow.sklearn.log_model(
                model, 
                "model", 
                signature=signature, 
                input_example=input_example
            )
            
            # Save run info locally for DVC or other tools
            artifact_path = mlflow.get_artifact_uri()
            model_path = f'{artifact_path}/model'
            save_model_info(run.info.run_id, model_path, 'experiment_info.json')
            
            # Log vectorizer as artifact
            mlflow.log_artifact(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))
            
            # Evaluate model
            report, cm = evaluate_model(model, X_test_tfidf, y_test)
            
            # Log metrics
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    mlflow.log_metric(f"test_{label}_precision", metrics['precision'])
                    mlflow.log_metric(f"test_{label}_recall", metrics['recall'])
                    mlflow.log_metric(f"test_{label}_f1-score", metrics['f1-score'])
                    
            # Log confusion matrix plot
            log_confusion_matrix(cm, 'test_data')  
            
            # Set tags
            mlflow.set_tag("model_type", "LightGBM")
            mlflow.set_tag("dataset", "YouTube Sentiment Analysis")
            
            logger.info("Model evaluation pipeline completed successfully!")
            
        except Exception as e:
            logger.error("Error in model evaluation pipeline: %s", e)
            raise # Raise the error so DVC knows the stage failed

if __name__ == "__main__":
    main()