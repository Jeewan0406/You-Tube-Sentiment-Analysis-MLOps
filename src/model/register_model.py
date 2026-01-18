import os
import json
import mlflow
import logging
import dagshub

# 1. ENSURE LOGS DIRECTORY EXISTS
os.makedirs('logs', exist_ok=True)

# 2. CORRECT DAGSHUB MLFLOW URI (Remove the /#/experiments)
dagshub.init(repo_owner='raijiwan275',
                repo_name='youtubanalysis_mlflow',
                mlflow=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  

filter_handler = logging.FileHandler('logs/register_model.log')
filter_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter) 
filter_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(filter_handler)

def load_model_info(file_path: str) -> dict:
    """Load model information from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug(f"Model information loaded successfully from {file_path}")
        return model_info
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading model information from {file_path}: {e}")
        raise

def register_model(model_name: str, model_info: dict) -> None:
    try:
        # Note: In your previous script you saved it as 'model_path' in JSON
        # Ensure the key matches what you saved in model_evaluation.py
        model_uri = f"runs:/{model_info['run_id']}/model" 
        
        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        
        # Move to Staging
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        logger.info(f"Model {model_name} registered successfully with version {model_version.version}")
    except Exception as e:
        logger.error(f"Error registering model {model_name}: {e}")
        raise

def main():
    try:
        # NOTE: Verify if your file is 'experiment_info.json' (from previous step) 
        # or 'experiments_info.json'
        model_info_path = 'experiment_info.json' 
        model_info = load_model_info(model_info_path)
        
        model_name = "YouTube_Analysis_Model"
        register_model(model_name, model_info)
    
    except Exception as e:
        logger.error(f"Error in main execution: {e}")   
        print(f'Error: {e} ')
        
if __name__ == "__main__":
    main()