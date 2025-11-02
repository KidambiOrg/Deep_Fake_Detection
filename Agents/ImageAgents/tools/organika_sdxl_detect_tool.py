"""Organika SDXL detection tool with blob storage caching."""

from outputModel.image_detection_output_model import IndividualModelResult
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import base64
import io
import time
import os
import logging
import tempfile
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)

# Global model cache
_organika_model = None
_organika_processor = None
_device = None

def get_cached_model_directory(model_key: str, blob_config: dict) -> str:
    """
    Get cached model directory or download from blob storage.
    Priority: Cache → Blob download to cache
    
    Note: blob_storage_helper now handles cache checking internally,
    so we just delegate to it directly (no double checking).
    
    Args:
        model_key: Model key identifier
        blob_config: Blob storage configuration with blob_prefix
        
    Returns:
        Local directory path to model in cache
    """
    # Check for directory blob
    if 'blob_prefix' in blob_config:
        try:
            from utils.blob_storage_helper import get_blob_loader
            blob_loader = get_blob_loader()
            
            if blob_loader and blob_loader.blob_service_client:
                # download_directory now checks cache first internally
                downloaded_path = blob_loader.download_directory(
                    container_name=blob_config['container'],
                    blob_prefix=blob_config['blob_prefix']
                )
                
                if downloaded_path:
                    return str(downloaded_path)
                else:
                    logger.warning(f"Failed to get {model_key} from cache or blob storage")
            else:
                logger.warning("Blob storage not configured")
                
        except Exception as e:
            logger.error(f"Error getting {model_key}: {e}")
        
        # If failed, return expected cache path (will fail gracefully in caller)
        cache_dir = Path(tempfile.gettempdir()) / "deepfake_models_cache"
        cache_path = cache_dir / blob_config['container'] / blob_config['blob_prefix']
        logger.error(f"Failed to get model {model_key} - returning fallback path")
        return str(cache_path)
    
    # No blob config provided
    logger.error(f"No blob configuration provided for {model_key}")
    return ""

def load_organika_model():
    """
    Load Organika SDXL model from cache or blob storage.
    Follows caching pattern from app_detect_image.py
    """
    global _organika_model, _organika_processor, _device
    
    if _organika_model is not None:
        return _organika_model, _organika_processor, _device
    
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Download from blob storage or use local path
    blob_config = {
        'container': 'models',
        'blob_prefix': 'huggingface/Organika/sdxl-detector'
    }
    
    model_path = get_cached_model_directory('organika-sdxl', blob_config)
    
    if model_path is None:
        # Fallback to local path
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        model_path = os.path.join(project_root, 'Models', 'huggingface', 'Organika', 'sdxl-detector')
        if not os.path.exists(model_path):
            logger.error(f"Model not found at fallback path: {model_path}")
            return None, None, _device
    
    # Load model and processor
    try:
        logger.info(f"Loading Organika SDXL model from: {model_path}")
        processor = AutoImageProcessor.from_pretrained(model_path)
        model = AutoModelForImageClassification.from_pretrained(model_path)
        model = model.to(_device)
        model.eval()
        
        _organika_model = model
        _organika_processor = processor
        
        logger.info(f"✓ Loaded Organika SDXL model from: {model_path}")
        return model, processor, _device
        
    except Exception as e:
        logger.error(f"Failed to load Organika SDXL model: {e}")
        return None, None, _device

def preprocess_image(image_path: str, processor=None):
    """Preprocess image from file path for Organika SDXL model."""
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    # Load image from file path
    image = Image.open(image_path)
    image.load()
    
    # Ensure RGB format
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Use processor to prepare inputs
    if processor:
        inputs = processor(images=image, return_tensors="pt")
        return inputs
    else:
        raise ValueError("Processor is required for preprocessing")

def detect_with_organika_sdxl_tool(image_path: str) -> dict:
    """
    Agent tool: Detect deepfakes using the Organika SDXL model.
    
    Returns dictionary representation of IndividualModelResult for JSON serialization.
    
    Args:
        image_path: File path to image to analyze
        
    Returns:
        Dictionary with detection results
    """
    start_time = time.time()
    
    try:
        # Load model (uses cache if available)
        model, processor, device = load_organika_model()
        
        if model is None or processor is None:
            raise Exception("Failed to load Organika SDXL model")
        
        # Preprocess image from file path
        inputs = preprocess_image(image_path, processor)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            
            # Model labels: {0: 'Fake', 1: 'Real'} (HuggingFace standard)
            # probabilities[0][0] = probability of FAKE class (AI-generated)
            # probabilities[0][1] = probability of REAL class
            fake_prob = probabilities[0][0].item()
            real_prob = probabilities[0][1].item()
        
        processing_time = time.time() - start_time
        
        # Determine prediction
        is_deepfake = fake_prob > real_prob
        confidence = max(fake_prob, real_prob)
        prediction = 'FAKE' if is_deepfake else 'REAL'
        
        # Return as dictionary
        result = IndividualModelResult(
            model='organika-sdxl',
            prediction=prediction,
            confidence=confidence,
            fakeScore=fake_prob,
            realScore=real_prob
        )
        return result.model_dump()
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Organika SDXL detection error: {e}")
        
        # Return error result as dictionary
        error_result = IndividualModelResult(
            model='organika-sdxl',
            prediction='REAL',
            confidence=0.0,
            fakeScore=0.0,
            realScore=0.0
        )
        return error_result.model_dump()
