"""DimaVit detection tool with blob storage caching."""

from outputModel.image_detection_output_model import IndividualModelResult
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
from typing import Optional
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

def get_cached_model_directory(model_key: str, blob_config: dict, required_files: Optional[list] = None) -> str:
    """
    Get cached model directory or download from blob storage.
    Priority: Cache (validated) → Blob download to cache
    
    Args:
        model_key: Model key identifier
        blob_config: Blob storage configuration with blob_prefix
        required_files: List of required files to validate cache (e.g., ['config.json', 'preprocessor_config.json'])
        
    Returns:
        Local directory path to model in cache
    """
    # Check for directory blob
    if 'blob_prefix' in blob_config:
        try:
            from utils.blob_storage_helper import get_blob_loader
            blob_loader = get_blob_loader()
            
            if blob_loader and blob_loader.blob_service_client:
                # First check if cache exists and is valid
                cache_dir = Path(tempfile.gettempdir()) / "deepfake_models_cache"
                blob_prefix_normalized = blob_config['blob_prefix'].replace('/', os.sep)
                local_dir = cache_dir / blob_config['container'] / blob_prefix_normalized
                
                # Validate cache - check if all required files exist
                force_download = False
                if local_dir.exists() and required_files:
                    missing_files = [f for f in required_files if not (local_dir / f).exists()]
                    if missing_files:
                        logger.warning(f"Cache incomplete for {model_key}, missing: {missing_files}. Re-downloading...")
                        force_download = True
                
                # Download with validation (force if cache is incomplete)
                downloaded_path = blob_loader.download_directory(
                    container_name=blob_config['container'],
                    blob_prefix=blob_config['blob_prefix'],
                    force_download=force_download
                )
                
                if downloaded_path:
                    # Final validation after download
                    if required_files:
                        missing_files = [f for f in required_files if not (Path(downloaded_path) / f).exists()]
                        if missing_files:
                            logger.error(f"Downloaded model still missing files: {missing_files}")
                            return ""
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
    
    # Required files for HuggingFace model to load properly
    required_files = ['config.json', 'preprocessor_config.json', 'model.safetensors']
    
    model_path = get_cached_model_directory('dima-vit', blob_config, required_files)
    
    # Load model and processor
    try:
        processor = AutoImageProcessor.from_pretrained(model_path)
        
        # Load model with standard float32 precision
        model = AutoModelForImageClassification.from_pretrained(
            model_path,
            torch_dtype=torch.float32
        )
        model = model.to(_device)
        model.eval()
        
        _dima_model = model
        _dima_processor = processor
        
        logger.info(f"Loaded DimaVit model from: {model_path}")
        return model, processor, _device
        
    except Exception as e:
        logger.error(f"Failed to load DimaVit model: {e}")
        return None, None, _device


def convert_image_to_tensor(image_path: str) -> torch.Tensor:
    """
    Convert image file to PyTorch tensor.
    
    Unified preprocessing: File Path → PIL Image → Tensor [C, H, W] in RGB, range [0, 255]
    
    Args:
        image_path: File path to image
        
    Returns:
        PyTorch tensor [C, H, W] in RGB format, range [0, 255], dtype uint8
    """
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    # Load image from file path
    image = Image.open(image_path)
    image.load()
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert PIL to tensor
    import torchvision.transforms.functional as F
    tensor = F.to_tensor(image)  # [C, H, W], float32 [0, 1]
    
    # Convert to uint8 [0, 255]
    tensor = (tensor * 255).to(torch.uint8)
    
    return tensor


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
        
        # STEP 1: Convert file to tensor
        tensor = convert_image_to_tensor(image_path)
        
        # STEP 2: Convert tensor to PIL for processor
        import torchvision.transforms.functional as F
        tensor_float = tensor.to(torch.float32) / 255.0  # [C, H, W] float32 [0, 1]
        image = F.to_pil_image(tensor_float)
        
        # STEP 3: Process through model processor
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            
            # CRITICAL FIX: DimaVit model has SWAPPED labels!
            # Model config says: 0="Real", 1="Fake"
            # But actual behavior: high score on 0 means FAKE, high score on 1 means REAL
            # This has been verified with test images and official HuggingFace model behavior
            
            # Get raw probabilities
            prob_0 = float(probabilities[0][0].cpu())  # Config says "Real" but actually means FAKE
            prob_1 = float(probabilities[0][1].cpu())  # Config says "Fake" but actually means REAL
            
            # Swap the labels to match actual behavior
            fake_prob = prob_0  # Index 0 actually detects FAKE despite label saying "Real"
            real_prob = prob_1  # Index 1 actually detects REAL despite label saying "Fake"
            
            logger.info(f"DimaVit scores (CORRECTED for label swap) - Fake: {fake_prob:.4f}, Real: {real_prob:.4f}")
        
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

