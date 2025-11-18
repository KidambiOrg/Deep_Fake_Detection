"""DimaVit V2 Fine-tuned detection tool with blob storage caching."""

from outputModel.image_detection_output_model import IndividualModelResult
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from typing import Optional
import time
import os
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Global model cache
_dimavit_v2_model = None
_dimavit_v2_processor = None
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

def load_dimavit_v2_model():
    """
    Load DimaVit V2 Fine-tuned model from cache or blob storage.
    Follows caching pattern from app_detect_image.py
    """
    global _dimavit_v2_model, _dimavit_v2_processor, _device
    
    if _dimavit_v2_model is not None:
        return _dimavit_v2_model, _dimavit_v2_processor, _device
    
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get model directory with caching logic
    blob_config = {
        'container': 'models',
        'blob_prefix': 'dimavit_v2_FineTuned'
    }
    
    # Required files for HuggingFace model to load properly
    required_files = ['config.json', 'preprocessor_config.json', 'model.safetensors']
    
    model_path = get_cached_model_directory('dimavit-v2-finetuned', blob_config, required_files)
    
    # Load model and processor
    try:
        processor = ViTImageProcessor.from_pretrained(model_path)
        
        # Load model with standard float32 precision
        model = ViTForImageClassification.from_pretrained(
            model_path,
            torch_dtype=torch.float32
        )
        model = model.to(_device)
        model.eval()
        
        _dimavit_v2_model = model
        _dimavit_v2_processor = processor
        
        logger.info(f"Loaded DimaVit V2 Fine-tuned model from: {model_path}")
        return model, processor, _device
        
    except Exception as e:
        logger.error(f"Failed to load DimaVit V2 Fine-tuned model: {e}")
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


def detect_with_dimavit_v2_finetuned_tool(image_path: str) -> dict:
    """
    Agent tool: Detect deepfakes using the DimaVit V2 Fine-tuned model.
    
    Returns dictionary representation of IndividualModelResult for JSON serialization.
    
    Args:
        image_path: File path to image to analyze
        
    Returns:
        Dictionary with detection results
    """
    start_time = time.time()
    
    try:
        # Load model (uses cache if available)
        model, processor, device = load_dimavit_v2_model()
        
        if model is None or processor is None:
            raise Exception("Failed to load DimaVit V2 Fine-tuned model")
        
        # STEP 1: Convert file to tensor
        tensor = convert_image_to_tensor(image_path)
        
        # STEP 2: Convert tensor to PIL for processor
        import torchvision.transforms.functional as F
        tensor_float = tensor.to(torch.float32) / 255.0  # [C, H, W] float32 [0, 1]
        image = F.to_pil_image(tensor_float)
        
        # STEP 3: Use processor transforms (from training script)
        image_mean, image_std = processor.image_mean, processor.image_std
        size = processor.size["height"]
        
        transforms = Compose([
            Resize((size, size)),
            ToTensor(),
            Normalize(mean=image_mean, std=image_std)
        ])
        
        pixel_values = transforms(image).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(pixel_values)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            predicted_class = logits.argmax(dim=1).item()
        
        # Get probabilities - model uses standard labels: 0=Real, 1=Deepfake
        real_prob = float(probs[0][0].cpu())
        fake_prob = float(probs[0][1].cpu())
        
        logger.info(f"DimaVit V2 Fine-tuned scores - Real: {real_prob:.4f}, Fake: {fake_prob:.4f}")
        
        processing_time = time.time() - start_time
        
        # Determine prediction based on which score is higher
        is_deepfake = fake_prob > real_prob
        confidence = max(fake_prob, real_prob)
        prediction = 'FAKE' if is_deepfake else 'REAL'
        
        # Return as dictionary
        result = IndividualModelResult(
            model='dimavit-v2-finetuned',
            prediction=prediction,
            confidence=confidence,
            fakeScore=fake_prob,
            realScore=real_prob
        )
        return result.model_dump()
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"DimaVit V2 Fine-tuned detection error: {e}")
        
        # Return error result as dictionary
        error_result = IndividualModelResult(
            model='dimavit-v2-finetuned',
            prediction='REAL',
            confidence=0.0,
            fakeScore=0.0,
            realScore=0.0
        )
        return error_result.model_dump()
