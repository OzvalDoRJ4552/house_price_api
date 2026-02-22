import joblib
import numpy as np
import json
from typing import Dict, Any
from pathlib import Path 


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.json"

# Global variable to store the loaded model
model = None
metadata = None

def load_model_and_metadata() -> bool:
    global model, metadata

    try:
        model = joblib.load(MODEL_PATH)  

        with open(METADATA_PATH, 'r', encoding="utf-8") as f:
            metadata = json.load(f)
            
        print("Model and metadata loaded successfully!")
        return True

    except Exception as e:
        print(f"Error loading model or metadata: {e}")
        model = None
        metadata = None
        return False

def make_prediction(house_features: Dict[str, Any]) -> float:
    """
    Make a price prediction for a single house.

    Args:
        house_features: Dictionary containing all 13 features

    Returns:
        Predicted price as a float

    This function should:
    1. Extract the features in the correct order (same order as training)
    2. Convert to numpy array with shape (1, 13)
    3. Use model.predict() to get prediction
    4. Return the prediction as a float
    """
    global model, metadata

    if model is None:
        raise ValueError("Model not loaded")
    if metadata is None:
        raise ValueError("Model metadata not loaded")

    
    feature_values = [house_features[feature] for feature in metadata['features']]  # Replace with list comprehension
    X = np.array([feature_values], dtype=float)

    
    prediction = model.predict(X)[0]
    return round(float(prediction), 2)

def get_model_info() -> Dict[str, Any]:
    """
    Get information about the loaded model.

    Returns:
        Dictionary containing model metadata

    This function should simply return the loaded metadata dictionary.
    If metadata is not loaded, it should raise an error.
    """
    global metadata
    if metadata is None:
        raise ValueError("Model metadata not loaded")
    return metadata


def check_health() -> Dict[str, Any]:
    """
    Check the health status of the service.

    Returns:
        Dictionary with health status information

    This function should:
    1. Check if model is loaded (model is not None)
    2. Check if metadata is loaded (metadata is not None)
    3. Return a dictionary with status information
    """
    global model, metadata
    
    model_loaded = model is not None
    metadata_loaded = metadata is not None
    is_healthy = model_loaded and metadata_loaded

    return {
        "status": "healthy" if is_healthy else "unhealthy",  
        "model_loaded": model_loaded,  
        "message": "Service is ready" if is_healthy else "Model or metadata not loaded"  
    }

    
