"""CNN V3 DenseNet201 detection tool with blob storage caching."""

from outputModel.image_detection_output_model import IndividualModelResult
import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import List
import time
import os
import logging
import tempfile
import hashlib
from pathlib import Path
import threading

logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Global model cache and thread-safe lock
_cnn_v3_model = None
_model_lock = threading.Lock()

def get_cached_model_path(model_key: str, blob_config: dict) -> str:
    """
    Get cached model path or download from blob storage.
    Priority: Cache → Blob download to cache
    
    Args:
        model_key: Model key identifier
        blob_config: Blob storage configuration from MODEL_REGISTRY
        
    Returns:
        Local file path to model in cache
    """
    # Check for path in blob config
    if 'path' in blob_config:
        try:
            from utils.blob_storage_helper import get_blob_loader
            blob_loader = get_blob_loader()
            
            if blob_loader and blob_loader.blob_service_client:
                # Try .keras format (primary)
                filename = 'cnn_v3_finetuned_DenseNet_model.keras'
                blob_path = f"{blob_config['path']}/{filename}"
                
                downloaded_path = blob_loader.download_model(
                    container_name=blob_config['container'],
                    blob_name=blob_path
                )
                
                if downloaded_path:
                    logger.info(f"Downloaded {filename} from blob storage")
                    return downloaded_path
                
                logger.warning(f"Failed to download {model_key} from blob storage")
            else:
                logger.warning("Blob storage not configured")
                
        except Exception as e:
            logger.error(f"Error getting {model_key}: {e}")
        
        # If failed, return expected cache path (will fail gracefully in caller)
        cache_dir = Path(tempfile.gettempdir()) / "deepfake_models_cache"
        blob_path = f"{blob_config['container']}/{blob_config['path']}/cnn_v3_finetuned_DenseNet_model.keras"
        path_hash = hashlib.md5(blob_path.encode()).hexdigest()
        cache_path = cache_dir / f"{path_hash}_cnn_v3_finetuned_DenseNet_model.keras"
        logger.error(f"Failed to get model {model_key} - returning fallback path")
        return str(cache_path)
    
    # No blob config provided
    logger.error(f"No blob configuration provided for {model_key}")
    return ""

def load_cnn_v3_model():
    """
    Load CNN V3 DenseNet model (thread-safe).
    Tries multiple locations (local, cache, blob).
    """
    global _cnn_v3_model
    
    with _model_lock:
        if _cnn_v3_model is not None:
            return _cnn_v3_model
        
        try:
            logger.info("Loading CNN V3 FineTuned model...")
            
            # Try local paths first
            script_dir = Path(__file__).parent.parent.parent
            local_paths = [
                script_dir / "Models" / "CNN_V3_FineTuned" / "cnn_v3_finetuned_DenseNet_model.keras",
            ]
            
            model_path = None
            for path in local_paths:
                if path.exists():
                    model_path = str(path)
                    logger.info(f"Found local model at: {model_path}")
                    break
            
            # If not found locally, try blob storage
            if model_path is None:
                logger.info("Model not found locally, trying blob storage...")
                blob_config = {
                    'container': 'models',
                    'path': 'CNN_V3_FineTuned'
                }
                model_path = get_cached_model_path('cnn_v3_finetuned', blob_config)
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"CNN V3 FineTuned model not found at: {model_path}\n"
                    f"Please ensure the model file exists in Models/CNN_V3_FineTuned/ folder.\n"
                    f"Expected file: cnn_v3_finetuned_DenseNet_model.keras"
                )
            
            logger.info(f"Loading model from: {model_path}")
            
            # Load model - supports .keras format
            model = keras.models.load_model(model_path, compile=False)
            
            # Compile model
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            _cnn_v3_model = model
            logger.info(f"✓ Loaded CNN V3 DenseNet model successfully")
            logger.info(f"Model input shape: {model.input_shape}")
            logger.info(f"Model output shape: {model.output_shape}")
            
            return _cnn_v3_model
            
        except Exception as e:
            logger.error(f"Failed to load CNN V3 FineTuned model: {e}", exc_info=True)
            return None

def preprocess_image_for_densenet(image_path: str) -> np.ndarray:
    """
    Preprocess image for DenseNet model using TensorFlow operations.
    Matches the exact preprocessing used during training.
    
    Args:
        image_path: File path to image
        
    Returns:
        Preprocessed image tensor ready for model inference [1, 224, 224, 3], float32, range [0, 1]
    """
    # Use TensorFlow operations (same as training)
    img_bytes = tf.io.read_file(image_path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.ensure_shape(img, [None, None, 3])
    img = tf.image.resize(img, (224, 224))
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img, axis=0)
    
    return img.numpy()


def detect_with_cnn_v3_tool(image_path: str) -> dict:
    """
    Agent tool: Detect deepfakes using the CNN V3 FineTuned model.
    
    Args:
        image_path: File path to image to analyze
        
    Returns:
        Dictionary with detection results
    """
    logger.info("Running detect_with_cnn_v3_tool...")
    start_time = time.time()
    
    try:
        logger.info("Loading CNN V3 DenseNet model...")
        # Load model (uses cache if available, thread-safe)
        model = load_cnn_v3_model()
        
        if model is None:
            error_msg = "Failed to load CNN V3 DenseNet model - model file may be missing"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        logger.info("Preprocessing image...")
        # Preprocess using TensorFlow (matches training preprocessing exactly)
        img_batch = preprocess_image_for_densenet(image_path)
        
        logger.info("Running inference...")
        # Run inference (thread-safe with prediction lock)
        with _model_lock:
            prediction = model.predict(img_batch, verbose=0)[0]
        
        # Handle output format:
        # DenseNet model outputs single sigmoid value: >0.5 = REAL, <=0.5 = FAKE
        # raw_score represents the probability of being REAL (matches training evaluation)
        THRESHOLD = 0.5
        raw_score = float(prediction[0]) if len(prediction.shape) > 0 else float(prediction)
        
        # Determine prediction based on threshold
        is_real = raw_score > THRESHOLD
        prediction_label = 'REAL' if is_real else 'FAKE'
        
        # Calculate confidence and probabilities
        if is_real:
            confidence = raw_score
            real_prob = raw_score
            fake_prob = 1.0 - raw_score
        else:
            confidence = 1.0 - raw_score
            real_prob = raw_score
            fake_prob = 1.0 - raw_score
        
        processing_time = time.time() - start_time
        
        logger.info(f"CNN V3 DenseNet prediction: {prediction_label} (confidence: {confidence:.2f}, raw_score: {raw_score:.4f}, threshold: {THRESHOLD})")
        
        # Return as dictionary
        result = IndividualModelResult(
            model='cnn_v3_finetuned',
            prediction=prediction_label,
            confidence=confidence,
            fakeScore=fake_prob,
            realScore=real_prob
        )
        return result.model_dump()
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"CNN V3 detection error: {e}", exc_info=True)
        
        # Return error result as dictionary
        error_result = IndividualModelResult(
            model='cnn_v3_finetuned',
            prediction='REAL',
            confidence=0.0,
            fakeScore=0.0,
            realScore=0.0
        )
        logger.warning(f"Returning error result: {error_result.model_dump()}")
        return error_result.model_dump()


def detect_batch_with_cnn_v3_tool(image_paths: List[str]) -> List[dict]:
    """
    Agent tool: Batch detect deepfakes using the CNN V3 FineTuned model.
    
    Efficient batch processing for multiple images/frames.
    
    Args:
        image_paths: List of file paths to images
        
    Returns:
        List of dictionaries with detection results for each image
    """
    results = []
    
    # Process each image path
    for image_path in image_paths:
        result = detect_with_cnn_v3_tool(image_path)
        results.append(result)
    
    return results
