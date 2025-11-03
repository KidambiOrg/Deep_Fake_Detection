"""NoEscape detection tool with blob storage caching."""

from outputModel.image_detection_output_model import IndividualModelResult
import torch
import torch.nn as nn
from safetensors.torch import load_file
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import base64
import io
import time
import os
import logging
import tempfile
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)

# Import the actual Nonescape model from imagedetectionlib
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from imagedetectionlib.detect_deepfake_nonescape import NonescapeClassifier

# Global model cache
_nonescape_model = None
_device = None

def get_cached_model_path(model_key: str, blob_config: dict) -> str:
    """
    Get cached model path or download from blob storage.
    Priority: Cache → Blob download to cache
    
    Note: blob_storage_helper now handles cache checking internally,
    so we just delegate to it directly (no double checking).
    
    Args:
        model_key: Model key identifier
        blob_config: Blob storage configuration from MODEL_REGISTRY
        
    Returns:
        Local file path to model in cache
    """
    # Check for single file blob
    if 'blob' in blob_config:
        try:
            from utils.blob_storage_helper import get_blob_loader
            blob_loader = get_blob_loader()
            
            if blob_loader and blob_loader.blob_service_client:
                # download_model now checks cache first internally
                downloaded_path = blob_loader.download_model(
                    container_name=blob_config['container'],
                    blob_name=blob_config['blob']
                )
                
                if downloaded_path:
                    return downloaded_path
                else:
                    logger.warning(f"Failed to get {model_key} from cache or blob storage")
            else:
                logger.warning("Blob storage not configured")
                
        except Exception as e:
            logger.error(f"Error getting {model_key}: {e}")
        
        # If failed, return expected cache path (will fail gracefully in caller)
        cache_dir = Path(tempfile.gettempdir()) / "deepfake_models_cache"
        blob_path = f"{blob_config['container']}/{blob_config['blob']}"
        path_hash = hashlib.md5(blob_path.encode()).hexdigest()
        filename = Path(blob_config['blob']).name
        cache_path = cache_dir / f"{path_hash}_{filename}"
        logger.error(f"Failed to get model {model_key} - returning fallback path")
        return str(cache_path)
    
    # No blob config provided
    logger.error(f"No blob configuration provided for {model_key}")
    return ""

def load_nonescape_model():
    """
    Load NoEscape model from cache or blob storage.
    Follows caching pattern from app_detect_image.py
    """
    global _nonescape_model, _device
    
    if _nonescape_model is not None:
        return _nonescape_model, _device
    
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NonescapeClassifier()
    
    # Get model path with caching logic
    blob_config = {
        'container': 'models',
        'blob': 'nonescape/nonescape-v0.safetensors'
    }
    model_path = get_cached_model_path('nonescape-full', blob_config)
    
    # Load model weights
    if os.path.exists(model_path):
        try:
            state_dict = load_file(model_path)
            logger.info(f"Loading NoEscape Full model from: {model_path}")
            logger.info(f"State dict has {len(state_dict)} keys, first 5: {list(state_dict.keys())[:5]}")
            
            model.load_state_dict(state_dict)
            logger.info("Successfully loaded NoEscape Full model weights")
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            logger.warning("Continuing with randomly initialized weights")
    else:
        logger.warning(f"Model file not found: {model_path}")
        logger.warning("Using randomly initialized weights")
    
    model = model.to(_device)
    model.eval()
    
    _nonescape_model = model
    return model, _device

def preprocess_image(image_path: str):
    """
    Preprocess image from file path for NoEscape model.
    Loads image from disk and applies necessary transformations.
    MUST MATCH imagedetectionlib/detect_deepfake_nonescape.py preprocessing exactly!
    """
    from PIL import ImageFile
    from torchvision.transforms import v2 as T
    
    # Allow loading truncated images
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    logger.info(f"Loading image from: {image_path}")
    
    # Load image from file path
    image = Image.open(image_path)
    
    # Load the image data completely (detect truncation early)
    image.load()
    
    # Ensure RGB format
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # EXACT transforms from imagedetectionlib (Resize 256 -> CenterCrop 224 -> JPEG compression)
    transform = T.Compose([
        T.ToImage(),
        T.Resize(256),
        T.CenterCrop(224),
        T.JPEG(quality=100),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return transform(image).unsqueeze(0)

def detect_with_nonescape_tool(image_path: str) -> dict:
    """
    Agent tool: Detect deepfakes using the NoEscape model.
    
    Returns dictionary representation of IndividualModelResult for JSON serialization.
    
    Args:
        image_path: File path to image to analyze
        
    Returns:
        Dictionary with detection results (model, prediction, confidence, fakeScore, realScore)
    """
    logger.info("Running detect_with_nonescape_tool...")
    
    start_time = time.time()
    
    try:
        logger.info("Loading NoEscape model...")
        # Load model (uses cache if available)
        model, device = load_nonescape_model()
        
        logger.info("Preprocessing image...")
        # Preprocess image from file path
        image_tensor = preprocess_image(image_path)
        image_tensor = image_tensor.to(device)
        
        logger.info("Running inference...")
        # Run inference (matching imagedetectionlib EXACTLY)
        # Index 0 = authentic (real), Index 1 = AI-generated (fake)
        with torch.no_grad():
            probs = model(image_tensor)
            real_prob = float(probs[0][0].item())  # Index 0 = authentic/real
            fake_prob = float(probs[0][1].item())  # Index 1 = AI-generated/fake
        
        processing_time = time.time() - start_time
        
        # Determine prediction based on which score is higher
        is_deepfake = fake_prob > real_prob
        confidence = max(fake_prob, real_prob)
        prediction = 'FAKE' if is_deepfake else 'REAL'
        
        # Return as dictionary (JSON serializable for Agent Framework)
        result = IndividualModelResult(
            model='nonescape-full',
            prediction=prediction,
            confidence=confidence,
            fakeScore=fake_prob,
            realScore=real_prob
        )
        return result.model_dump()
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"NoEscape detection error: {e}")
        
        # Return error result as dictionary
        error_result = IndividualModelResult(
            model='nonescape-full',
            prediction='REAL',
            confidence=0.0,
            fakeScore=0.0,
            realScore=0.0
        )
        return error_result.model_dump()


 