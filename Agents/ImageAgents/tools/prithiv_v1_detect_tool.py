"""Prithiv V1 detection tool with blob storage caching."""

from outputModel.image_detection_output_model import IndividualModelResult
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import numpy as np
from typing import Union, List, Optional
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
_prithiv_model = None
_prithiv_processor = None
_device = None

def load_prithiv_model():
    """
    Load Prithiv V1 model from local Models folder.
    """
    global _prithiv_model, _prithiv_processor, _device
    
    if _prithiv_model is not None:
        return _prithiv_model, _prithiv_processor, _device
    
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load from local Models folder only
    script_dir = Path(__file__).parent.parent.parent.parent
    model_path = script_dir / "Models" / "huggingface" / "prithivMLMods" / "deepfake-detector-model-v1"
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Prithiv V1 model not found at: {model_path}\n"
            f"Please ensure the model directory exists in Models/huggingface/prithivMLMods/deepfake-detector-model-v1 folder.\n"
            f"Expected files: config.json, preprocessor_config.json, model.safetensors"
        )
    
    model_path = str(model_path)
    
    # Load model and processor
    try:
        logger.info(f"Loading Prithiv V1 model from: {model_path}")
        processor = AutoImageProcessor.from_pretrained(model_path)
        
        # Try SigLIP architecture first (more accurate for this model)
        model_loaded = False
        try:
            model = SiglipForImageClassification.from_pretrained(
                model_path,
                torch_dtype=torch.float32
            )
            model = model.to(_device)
            logger.info("✓ Using SiglipForImageClassification (optimal for Prithiv)")
            model_loaded = True
        except Exception as e:
            logger.warning(f"Failed to load with SiglipForImageClassification: {e}")
        
        # Fallback to AutoModel
        if not model_loaded:
            model = AutoModelForImageClassification.from_pretrained(
                model_path,
                torch_dtype=torch.float32
            )
            model = model.to(_device)
            logger.info("Using AutoModelForImageClassification (fallback)")
        
        model.eval()
        
        _prithiv_model = model
        _prithiv_processor = processor
        
        logger.info(f"✓ Loaded Prithiv V1 model from: {model_path}")
        return model, processor, _device
        
    except Exception as e:
        logger.error(f"Failed to load Prithiv V1 model: {e}")
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
    img_array = np.array(image)  # [H, W, C]
    tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # [C, H, W]
    
    return tensor


def detect_with_prithiv_v1_tool(image_path: str) -> dict:
    """
    Agent tool: Detect deepfakes using the Prithiv V1 model.
    
    Internally converts image to tensor for unified processing pipeline.
    All images (file paths, base64, binary) → Tensor → Model
    
    Returns dictionary representation of IndividualModelResult for JSON serialization.
    
    Args:
        image_path: File path to image, or base64 string, or data URI
        
    Returns:
        Dictionary with detection results
    """
    start_time = time.time()
    
    try:
        # BUG FIX: Azure Agent Framework sometimes duplicates the path on retries
        from utils.path_fix_helper import fix_duplicated_path
        image_path = fix_duplicated_path(image_path)
        
        # Load model (uses cache if available)
        model, processor, device = load_prithiv_model()
        
        if model is None or processor is None:
            raise Exception("Failed to load Prithiv V1 model")
        
        # STEP 1: Convert file to tensor [C, H, W] - unified preprocessing
        tensor = convert_image_to_tensor(image_path)
        
        # STEP 2: Convert tensor to PIL Image for processor compatibility
        if tensor.device.type != 'cpu':
            tensor = tensor.cpu()
        
        img_array = tensor.numpy().transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
        image = Image.fromarray(img_array, mode='RGB')
        
        # STEP 3: Process through model using tensor pipeline
        inputs = processor(images=image, return_tensors="pt")
        
        # Move inputs to device
        # Note: For 8-bit quantized models, bitsandbytes handles dtype conversions automatically
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
            model='prithiv-v1',
            prediction=prediction,
            confidence=confidence,
            fakeScore=fake_prob,
            realScore=real_prob
        )
        return result.model_dump()
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Prithiv V1 detection error: {e}")
        
        # Return error result as dictionary
        error_result = IndividualModelResult(
            model='prithiv-v1',
            prediction='REAL',
            confidence=0.0,
            fakeScore=0.0,
            realScore=0.0
        )
        return error_result.model_dump()


def detect_batch_with_prithiv_v1_tool(image_paths: List[str]) -> List[dict]:
    """
    Agent tool: Batch detect deepfakes using the Prithiv V1 model.
    
    Efficient batch processing for multiple images/frames.
    Internally converts all images to tensors for unified processing.
    
    Args:
        image_paths: List of file paths to images
        
    Returns:
        List of dictionaries with detection results for each image
    """
    results = []
    
    # Process each image path
    for image_path in image_paths:
        result = detect_with_prithiv_v1_tool(image_path=image_path)
        results.append(result)
    
    return results



