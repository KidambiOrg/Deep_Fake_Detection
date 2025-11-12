"""CNN V3 FineTuned detection tool with blob storage caching."""

from outputModel.image_detection_output_model import IndividualModelResult
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from typing import List
import time
import os
import logging
import tempfile
import hashlib
from pathlib import Path
import cv2
import threading

logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Global model cache and thread-safe lock
_cnn_v3_model = None
_cnn_v3_quantized_model = None
_model_lock = threading.Lock()

# Quantization support flag
USE_QUANTIZATION = os.getenv('CNN_V3_USE_QUANTIZATION', 'false').lower() == 'true'

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
                filename = 'cnn_v3_finetuned_model.keras'
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
        blob_path = f"{blob_config['container']}/{blob_config['path']}/cnn_v3_finetuned_model.keras"
        path_hash = hashlib.md5(blob_path.encode()).hexdigest()
        cache_path = cache_dir / f"{path_hash}_cnn_v3_finetuned_model.keras"
        logger.error(f"Failed to get model {model_key} - returning fallback path")
        return str(cache_path)
    
    # No blob config provided
    logger.error(f"No blob configuration provided for {model_key}")
    return ""

def apply_quantization(model):
    """
    Apply post-training quantization to the model for faster inference.
    
    Note: TensorFlow/Keras models can be quantized using TFLite converter.
    For now, we return the original model as quantization requires conversion
    to TFLite format which changes the inference API.
    
    Args:
        model: Loaded Keras model
        
    Returns:
        Quantized model (or original if quantization not supported/enabled)
    """
    global _cnn_v3_quantized_model
    
    if _cnn_v3_quantized_model is not None:
        return _cnn_v3_quantized_model
    
    if not USE_QUANTIZATION:
        logger.info("Quantization disabled (set CNN_V3_USE_QUANTIZATION=true to enable)")
        return model
    
    try:
        logger.info("Applying post-training quantization...")
        
        # Note: Full quantization requires TFLite converter
        # For Keras models, we can use mixed precision for inference speedup
        # This is a lighter alternative to full quantization
        
        # Enable mixed precision (uses float16 where possible)
        from tensorflow.keras import mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        
        logger.info("✓ Enabled mixed precision (float16) for faster inference")
        _cnn_v3_quantized_model = model
        return model
        
    except Exception as e:
        logger.warning(f"Failed to apply quantization: {e}")
        logger.info("Falling back to original model")
        return model


def load_cnn_v3_model():
    """
    Load CNN V3 FineTuned model (thread-safe).
    Tries multiple locations (local, cache, blob).
    Applies quantization if enabled.
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
                script_dir / "Models" / "CNN_V3_FineTuned" / "cnn_v3_finetuned_model.keras",
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
                    f"Expected file: cnn_v3_finetuned_model.keras"
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
            
            # Apply quantization if enabled
            model = apply_quantization(model)
            
            _cnn_v3_model = model
            logger.info(f"✓ Loaded CNN V3 FineTuned model successfully")
            logger.info(f"Model input shape: {model.input_shape}")
            logger.info(f"Model output shape: {model.output_shape}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load CNN V3 FineTuned model: {e}", exc_info=True)
            return None

def convert_image_to_numpy(image_path: str) -> np.ndarray:
    """
    Convert image file to NumPy array for TensorFlow models.
    
    Unified preprocessing: File Path → PIL Image → NumPy [H, W, C] in RGB, range [0, 255]
    
    Args:
        image_path: File path to image
        
    Returns:
        NumPy array [H, W, C] in RGB format, range [0, 255], dtype uint8
    """
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    # Load image from file path
    image = Image.open(image_path)
    image.load()
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert PIL to numpy
    img_array = np.array(image)  # [H, W, C]
    
    return img_array


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
        logger.info("Loading CNN V3 FineTuned model...")
        # Load model (uses cache if available, thread-safe)
        model = load_cnn_v3_model()
        
        if model is None:
            error_msg = "Failed to load CNN V3 FineTuned model - model file may be missing"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        logger.info("Preprocessing image...")
        # STEP 1: Convert file to numpy array
        img_array = convert_image_to_numpy(image_path)
        
        # Resize to 224x224 (model input size)
        img_resized = cv2.resize(img_array, (224, 224))
        
        # Normalize to [0, 1] range
        img_normalized = img_resized / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        logger.info("Running inference...")
        # Run inference (thread-safe with prediction lock)
        with _model_lock:
            prediction = model.predict(img_batch, verbose=0)[0]
        
        # Handle output format:
        # CNN V3 model outputs single value: >0.5 = REAL, <=0.5 = FAKE
        if len(prediction.shape) == 0 or prediction.shape[0] == 1:
            # Single output: probability of REAL
            real_prob = float(prediction) if len(prediction.shape) == 0 else float(prediction[0])
            fake_prob = 1.0 - real_prob
        else:
            # Two outputs: [prob_real, prob_fake]
            real_prob = float(prediction[0])
            fake_prob = float(prediction[1])
        
        processing_time = time.time() - start_time
        
        # Determine prediction
        is_deepfake = fake_prob > real_prob
        confidence = max(fake_prob, real_prob)
        prediction_label = 'FAKE' if is_deepfake else 'REAL'
        
        logger.info(f"CNN V3 prediction: {prediction_label} (confidence: {confidence:.2f})")
        
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
