"""MobileNetV2 detection tool with blob storage caching."""

from outputModel.image_detection_output_model import IndividualModelResult
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from typing import Union, List
import time
import os
import logging
import tempfile
import hashlib
from pathlib import Path
import cv2

logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Global model cache
_mobilenetv2_model = None

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
                # Try both .keras and .h5 formats
                for filename in ['mobilenetv2_deepfake_model.keras', 'mobilenetv2_deepfake.h5', 'mobilenetv2_finetuned.h5']:
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
        blob_path = f"{blob_config['container']}/{blob_config['path']}/mobilenetv2_deepfake_model.keras"
        path_hash = hashlib.md5(blob_path.encode()).hexdigest()
        cache_path = cache_dir / f"{path_hash}_mobilenetv2_deepfake_model.keras"
        logger.error(f"Failed to get model {model_key} - returning fallback path")
        return str(cache_path)
    
    # No blob config provided
    logger.error(f"No blob configuration provided for {model_key}")
    return ""

def load_mobilenetv2_model():
    """
    Load MobileNetV2 model.
    Tries multiple file formats (.keras, .h5) and multiple locations (local, cache, blob).
    """
    global _mobilenetv2_model
    
    if _mobilenetv2_model is not None:
        return _mobilenetv2_model
    
    try:
        logger.info("Loading MobileNetV2 model...")
        
        # Try local paths first
        script_dir = Path(__file__).parent.parent.parent
        local_paths = [
            script_dir / "Models" / "MobileNetV2_FineTuned" / "mobilenetv2_deepfake_model.keras",
            script_dir / "Models" / "MobileNetV2_FineTuned" / "mobilenetv2_deepfake.h5",
            script_dir / "Models" / "MobileNetV2_FineTuned" / "mobilenetv2_finetuned.h5",
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
                'path': 'MobileNetV2_FineTuned'
            }
            model_path = get_cached_model_path('mobilenetv2', blob_config)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"MobileNetV2 model not found at: {model_path}\n"
                f"Please ensure the model file exists in Models/MobileNetV2_FineTuned/ folder.\n"
                f"Expected files: mobilenetv2_deepfake_model.keras or mobilenetv2_deepfake.h5"
            )
        
        logger.info(f"Loading model from: {model_path}")
        
        # Load model - supports both .h5 and .keras formats
        model = keras.models.load_model(model_path, compile=False)
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        _mobilenetv2_model = model
        logger.info(f"✓ Loaded MobileNetV2 model successfully")
        logger.info(f"Model input shape: {model.input_shape}")
        logger.info(f"Model output shape: {model.output_shape}")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to load MobileNetV2 model: {e}", exc_info=True)
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


def detect_with_mobilenetv2_tool(image_path: str) -> dict:
    """
    Agent tool: Detect deepfakes using the MobileNetV2 model.
    
    Args:
        image_path: File path to image to analyze
        
    Returns:
        Dictionary with detection results
    """
    logger.info("Running detect_with_mobilenetv2_tool...")
    start_time = time.time()
    
    try:
        logger.info("Loading MobileNetV2 model...")
        # Load model (uses cache if available)
        model = load_mobilenetv2_model()
        
        if model is None:
            error_msg = "Failed to load MobileNetV2 model - model file may be missing"
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
        # Run inference
        prediction = model.predict(img_batch, verbose=0)[0]
        
        # Handle both output formats:
        # IMPORTANT: MobileNetV2 model was trained with [0=FAKE, 1=REAL] convention
        # So output closer to 1.0 means REAL, closer to 0.0 means FAKE (REVERSED from typical)
        if len(prediction.shape) == 0 or prediction.shape[0] == 1:
            # Single output: probability of REAL (reversed from typical convention)
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
        
        logger.info(f"MobileNetV2 prediction: {prediction_label} (confidence: {confidence:.2f})")
        
        # Return as dictionary
        result = IndividualModelResult(
            model='mobilenetv2',
            prediction=prediction_label,
            confidence=confidence,
            fakeScore=fake_prob,
            realScore=real_prob
        )
        return result.model_dump()
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"MobileNetV2 detection error: {e}", exc_info=True)
        
        # Return error result as dictionary
        error_result = IndividualModelResult(
            model='mobilenetv2',
            prediction='REAL',
            confidence=0.0,
            fakeScore=0.0,
            realScore=0.0
        )
        logger.warning(f"Returning error result: {error_result.model_dump()}")
        return error_result.model_dump()


def detect_batch_with_mobilenetv2_tool(image_paths: List[str]) -> List[dict]:
    """
    Agent tool: Batch detect deepfakes using the MobileNetV2 model.
    
    Efficient batch processing for multiple images/frames.
    
    Args:
        image_paths: List of file paths to images
        
    Returns:
        List of dictionaries with detection results for each image
    """
    results = []
    
    # Process each image path
    for image_path in image_paths:
        result = detect_with_mobilenetv2_tool(image_path)
        results.append(result)
    
    return results
