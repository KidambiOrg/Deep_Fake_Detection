"""Organika SDXL detection tool with blob storage caching."""

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
_organika_model = None
_organika_processor = None
_device = None

def load_organika_model():
    """
    Load Organika SDXL model from local Models folder.
    """
    global _organika_model, _organika_processor, _device
    
    if _organika_model is not None:
        return _organika_model, _organika_processor, _device
    
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load from local Models folder only
    script_dir = Path(__file__).parent.parent.parent.parent
    model_path = script_dir / "Models" / "huggingface" / "Organika" / "sdxl-detector"
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Organika SDXL model not found at: {model_path}\n"
            f"Please ensure the model directory exists in Models/huggingface/Organika/sdxl-detector folder.\n"
            f"Expected files: config.json, preprocessor_config.json, model.safetensors"
        )
    
    model_path = str(model_path)
    
    # Load model and processor
    try:
        logger.info(f"Loading Organika SDXL model from: {model_path}")
        processor = AutoImageProcessor.from_pretrained(model_path)
        
        # Load model with standard float32 precision
        model = AutoModelForImageClassification.from_pretrained(
            model_path,
            torch_dtype=torch.float32
        )
        model = model.to(_device)
        model.eval()
        
        _organika_model = model
        _organika_processor = processor
        
        logger.info(f"✓ Loaded Organika SDXL model from: {model_path}")
        return model, processor, _device
        
    except Exception as e:
        logger.error(f"Failed to load Organika SDXL model: {e}")
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
        # BUG FIX: Azure Agent Framework sometimes duplicates the path on retries
        from utils.path_fix_helper import fix_duplicated_path
        image_path = fix_duplicated_path(image_path)
        
        # Load model (uses cache if available)
        model, processor, device = load_organika_model()
        
        if model is None or processor is None:
            raise Exception("Failed to load Organika SDXL model")
        
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
            
            # IMPORTANT: Read labels dynamically from model config (matches imagedetectionlib)
            # Different models may have different label orders!
            labels = model.config.id2label
            
            # Build scores dictionary
            scores = {labels[i]: float(probabilities[0][i].cpu()) for i in range(len(labels))}
            
            # Find fake and real scores by searching for keywords (matches imagedetectionlib logic)
            fake_prob = 0.0
            real_prob = 0.0
            deepfake_indicators = ['fake', 'deepfake', 'artificial', 'generated', 'synthetic', 'ai']
            real_indicators = ['real', 'authentic', 'genuine', 'human', 'realism']
            
            for label, score in scores.items():
                label_lower = label.lower()
                # Check for fake indicators
                for indicator in deepfake_indicators:
                    if indicator in label_lower:
                        fake_prob = score
                        break
                # Check for real indicators
                for indicator in real_indicators:
                    if indicator in label_lower:
                        real_prob = score
                        break
        
        processing_time = time.time() - start_time
        
        # Determine prediction based on which score is higher
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
