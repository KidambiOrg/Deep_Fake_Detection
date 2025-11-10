"""DimaVit detection tool with blob storage caching and 8-bit quantization."""

from outputModel.image_detection_output_model import IndividualModelResult
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor, BitsAndBytesConfig
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
_dima_model = None
_dima_processor = None
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

def load_dima_model():
    """
    Load DimaVit model from cache or blob storage.
    Follows caching pattern from app_detect_image.py
    """
    global _dima_model, _dima_processor, _device
    
    if _dima_model is not None:
        return _dima_model, _dima_processor, _device
    
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get model directory with caching logic
    blob_config = {
        'container': 'models',
        'blob_prefix': 'huggingface/dima806/deepfake_vs_real_image_detection'
    }
    model_path = get_cached_model_directory('dima-vit', blob_config)
    
    # Load model and processor
    try:
        processor = AutoImageProcessor.from_pretrained(model_path)
        
        # Configure 8-bit quantization for memory efficiency
        # Only use quantization on CUDA devices (not CPU)
        if _device.type == 'cuda':
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
            model = AutoModelForImageClassification.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto"
            )
            logger.info("✓ Using 8-bit quantization (50% memory reduction)")
        else:
            model = AutoModelForImageClassification.from_pretrained(model_path)
            model = model.to(_device)
        
        model.eval()
        
        _dima_model = model
        _dima_processor = processor
        
        logger.info(f"Loaded DimaVit model from: {model_path}")
        return model, processor, _device
        
    except Exception as e:
        logger.error(f"Failed to load DimaVit model: {e}")
        return None, None, _device

def detect_with_dima_vit_tool(image_path: str) -> dict:
    """
    Agent tool: Detect deepfakes using the DimaVit model.
    
    Returns dictionary representation of IndividualModelResult for JSON serialization.
    
    Args:
        image_path: File path to image to analyze
        
    Returns:
        Dictionary with detection results
    """
    start_time = time.time()
    
    try:
        # Load model (uses cache if available)
        model, processor, device = load_dima_model()
        
        if model is None or processor is None:
            raise Exception("Failed to load DimaVit model")
        
        # Load and preprocess image from file path
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        
        image = Image.open(image_path)
        image.load()
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            
            # IMPORTANT: Read labels dynamically from model config (matches imagedetectionlib)
            # Different models may have different label orders!
            labels = model.config.id2label
            logger.info(f"DimaVit model labels: {labels}")
            
            # Build scores dictionary
            scores = {labels[i]: float(probabilities[0][i].cpu()) for i in range(len(labels))}
            logger.info(f"DimaVit scores: {scores}")
            
            # Find fake and real scores by searching for keywords (matches imagedetectionlib logic)
            fake_prob = 0.0
            real_prob = 0.0
            fake_label_found = None
            real_label_found = None
            deepfake_indicators = ['fake', 'deepfake', 'artificial', 'generated', 'synthetic', 'ai']
            real_indicators = ['real', 'authentic', 'genuine', 'human', 'realism']
            
            for label, score in scores.items():
                label_lower = label.lower()
                # Check for fake indicators
                if not fake_label_found:
                    for indicator in deepfake_indicators:
                        if indicator in label_lower:
                            fake_prob = score
                            fake_label_found = label
                            break
                # Check for real indicators
                if not real_label_found:
                    for indicator in real_indicators:
                        if indicator in label_lower:
                            real_prob = score
                            real_label_found = label
                            break
            
            logger.info(f"DimaVit found - Fake: {fake_label_found}={fake_prob:.4f}, Real: {real_label_found}={real_prob:.4f}")
        
        processing_time = time.time() - start_time
        
        # Determine prediction based on which score is higher
        is_deepfake = fake_prob > real_prob
        confidence = max(fake_prob, real_prob)
        prediction = 'FAKE' if is_deepfake else 'REAL'
        
        # Return as dictionary
        result = IndividualModelResult(
            model='dima-vit',
            prediction=prediction,
            confidence=confidence,
            fakeScore=fake_prob,
            realScore=real_prob
        )
        return result.model_dump()
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"DimaVit detection error: {e}")
        
        # Return error result as dictionary
        error_result = IndividualModelResult(
            model='dima-vit',
            prediction='REAL',
            confidence=0.0,
            fakeScore=0.0,
            realScore=0.0
        )
        return error_result.model_dump()

