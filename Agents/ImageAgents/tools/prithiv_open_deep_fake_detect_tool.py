"""Prithiv Open Deepfake detection tool with blob storage caching."""

from outputModel.image_detection_output_model import IndividualModelResult
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor, SiglipForImageClassification
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
_prithiv_open_model = None
_prithiv_open_processor = None
_device = None

def load_prithiv_open_model():
    """
    Load Prithiv Open Deepfake model from local Models folder.
    """
    global _prithiv_open_model, _prithiv_open_processor, _device
    
    if _prithiv_open_model is not None:
        return _prithiv_open_model, _prithiv_open_processor, _device
    
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load from local Models folder only
    script_dir = Path(__file__).parent.parent.parent.parent
    model_path = script_dir / "Models" / "huggingface" / "prithiv_open_deep_fake"
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Prithiv Open Deepfake model not found at: {model_path}\n"
            f"Please ensure the model directory exists in Models/huggingface/prithiv_open_deep_fake folder.\n"
            f"Expected files: config.json, preprocessor_config.json, model.safetensors"
        )
    
    model_path = str(model_path)
    
    # Load model and processor
    try:
        logger.info(f"Loading Prithiv Open Deepfake model from: {model_path}")
        
        # Try SigLIP processor first (this model uses SigLIP architecture)
        processor = None
        try:
            from transformers import SiglipImageProcessor
            processor = SiglipImageProcessor.from_pretrained(model_path)
            logger.info("✓ Using SiglipImageProcessor")
        except Exception as e:
            logger.warning(f"Failed to load SigLIP processor, trying Auto: {e}")
            try:
                processor = AutoImageProcessor.from_pretrained(model_path)
                logger.info("✓ Using AutoImageProcessor")
            except Exception as e2:
                logger.error(f"Failed to load processor: {e2}")
                return None, None, _device
        
        # Try SigLIP architecture first (more accurate for this model)
        model_loaded = False
        try:
            model = SiglipForImageClassification.from_pretrained(
                model_path,
                torch_dtype=torch.float32
            )
            model = model.to(_device)
            logger.info("✓ Using SiglipForImageClassification (optimal for Prithiv Open)")
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
        
        _prithiv_open_model = model
        _prithiv_open_processor = processor
        
        logger.info(f"✓ Loaded Prithiv Open Deepfake model from: {model_path}")
        return model, processor, _device
        
    except Exception as e:
        logger.error(f"Failed to load Prithiv Open Deepfake model: {e}")
        return None, None, _device

def detect_with_prithiv_open_deep_fake_tool(image_path: str) -> dict:
    """
    Agent tool: Detect deepfakes using the Prithiv Open Deepfake model.
    
    OPTIMIZED: Accepts preprocessed PIL image directly (no file I/O).
    
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
        model, processor, device = load_prithiv_open_model()
        
        if model is None or processor is None:
            raise Exception("Failed to load Prithiv Open Deepfake model")
        
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
            model='prithiv-open-deep-fake',
            prediction=prediction,
            confidence=confidence,
            fakeScore=fake_prob,
            realScore=real_prob
        )
        return result.model_dump()
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Prithiv Open Deepfake detection error: {e}")
        
        # Return error result as dictionary
        error_result = IndividualModelResult(
            model='prithiv-open-deep-fake',
            prediction='REAL',
            confidence=0.0,
            fakeScore=0.0,
            realScore=0.0
        )
        return error_result.model_dump()
