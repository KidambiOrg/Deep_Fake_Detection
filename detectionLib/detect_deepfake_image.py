"""
Deepfake Image Detection using multiple pre-trained models

This module implements deepfake image detection using various pre-trained models
from Hugging Face that can classify images as real or deepfake/AI-generated.

Supported Models:
1. prithivMLMods/deepfake-detector-model-v1 (SigLIP-based)
2. dima806/deepfake_vs_real_image_detection (Vision Transformer)
3. umm-maybe/AI-image-detector (EfficientNet-based)
4. Organika/sdxl-detector (Stable Diffusion detector)
5. EfficientNet-B4/B7 (torchvision/timm)
6. Xception (timm)
7. ResNet34/50/101 (torchvision/timm)
8. Vision Transformer (torchvision)
9. DeiT (timm)
10. Swin Transformer (torchvision/timm)
"""

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, SiglipForImageClassification, AutoModelForImageClassification
from torchvision import models, transforms
from PIL import Image
import numpy as np
import logging
import warnings
from typing import List, Tuple, Dict, Optional, Union, Any, BinaryIO
import os
import time
import base64
import io
from dataclasses import dataclass
from pathlib import Path

# Try to import timm, but don't fail if not available
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    logging.warning("timm library not available. Some models will not be accessible.")

# Suppress warnings
warnings.filterwarnings("ignore", message=".*torch.load.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Model Registry - Available deepfake detection models
MODEL_REGISTRY = {
    # HuggingFace Models
    'prithiv-v1': {
        'name': 'prithivMLMods/deepfake-detector-model-v1',
        'description': 'SigLIP-based deepfake detector (default)',
        'architecture': 'SigLIP',
        'source': 'huggingface',
        'labels': {0: 'Fake', 1: 'Real'},
        'recommended_threshold': 0.5,
    },
    'dima-vit': {
        'name': 'dima806/deepfake_vs_real_image_detection',
        'description': 'Vision Transformer based detector',
        'architecture': 'ViT',
        'source': 'huggingface',
        'labels': None,  # Will be loaded from model
        'recommended_threshold': 0.5,
    },
    'umm-efficientnet': {
        'name': 'umm-maybe/AI-image-detector',
        'description': 'EfficientNet-based AI image detector',
        'architecture': 'EfficientNet',
        'source': 'huggingface',
        'labels': None,
        'recommended_threshold': 0.5,
    },
    'organika-sdxl': {
        'name': 'Organika/sdxl-detector',
        'description': 'Specialized for Stable Diffusion generated images',
        'architecture': 'Custom',
        'source': 'huggingface',
        'labels': None,
        'recommended_threshold': 0.5,
    },
    'npr-ensemble': {
        'name': 'npr-perturbation-rank',
        'description': 'NPR (Normalized Perturbation Rank) method using image perturbations for deepfake detection',
        'architecture': 'NPR',
        'source': 'npr',
        'labels': {0: 'Real', 1: 'Fake'},
        'recommended_threshold': 0.5,  # Standard threshold for balanced detection
        'num_perturbations': 15,  # More perturbations for better statistical reliability
        'perturbation_strength': 0.03,  # Subtle perturbations reveal AI artifacts better
    },
    
    # TorchVision/Timm Models
    'efficientnet_b4': {
        'name': 'efficientnet_b4',
        'description': 'EfficientNet-B4 from torchvision (pretrained on ImageNet)',
        'architecture': 'EfficientNet',
        'source': 'torchvision',
        'model_func': lambda: models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT),
        'input_size': 380,
        'modify_classifier': True,
        'classifier_attr': 'classifier',
        'classifier_in_features': 1792,
        'recommended_threshold': 0.5,
    },
    'efficientnet_b7': {
        'name': 'efficientnet_b7',
        'description': 'EfficientNet-B7 from torchvision (high accuracy, larger model)',
        'architecture': 'EfficientNet',
        'source': 'torchvision',
        'model_func': lambda: models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.DEFAULT),
        'input_size': 600,
        'modify_classifier': True,
        'classifier_attr': 'classifier',
        'classifier_in_features': 2560,
        'recommended_threshold': 0.5,
    },
    'efficientnet_b4_timm': {
        'name': 'efficientnet_b4',
        'description': 'EfficientNet-B4 from timm (pretrained on ImageNet)',
        'architecture': 'EfficientNet',
        'source': 'timm',
        'model_name': 'efficientnet_b4',
        'input_size': 380,
        'modify_classifier': True,
        'recommended_threshold': 0.5,
    },
    'efficientnet_b7_timm': {
        'name': 'efficientnet_b7',
        'description': 'EfficientNet-B7 from timm (high accuracy, larger model)',
        'architecture': 'EfficientNet',
        'source': 'timm',
        'model_name': 'efficientnet_b7',
        'input_size': 600,
        'modify_classifier': True,
        'recommended_threshold': 0.5,
    },
    'xception': {
        'name': 'xception',
        'description': 'Xception architecture (excellent for deepfake detection)',
        'architecture': 'Xception',
        'source': 'timm',
        'model_name': 'xception',
        'input_size': 299,
        'modify_classifier': True,
        'recommended_threshold': 0.5,
    },
    'xception41': {
        'name': 'xception41',
        'description': 'Xception-41 variant (balanced performance)',
        'architecture': 'Xception',
        'source': 'timm',
        'model_name': 'xception41',
        'input_size': 299,
        'modify_classifier': True,
        'recommended_threshold': 0.5,
    },
    'xception65': {
        'name': 'xception65',
        'description': 'Xception-65 variant (higher capacity)',
        'architecture': 'Xception',
        'source': 'timm',
        'model_name': 'xception65',
        'input_size': 299,
        'modify_classifier': True,
        'recommended_threshold': 0.5,
    },
    'xception71': {
        'name': 'xception71',
        'description': 'Xception-71 deeper variant (enhanced for deepfake detection)',
        'architecture': 'Xception',
        'source': 'timm',
        'model_name': 'xception71',
        'input_size': 299,
        'modify_classifier': True,
        'recommended_threshold': 0.5,
    },
    'resnet34': {
        'name': 'resnet34',
        'description': 'ResNet-34 from torchvision (balanced speed and accuracy)',
        'architecture': 'ResNet',
        'source': 'torchvision',
        'model_func': lambda: models.resnet34(weights=models.ResNet34_Weights.DEFAULT),
        'input_size': 224,
        'modify_classifier': True,
        'classifier_attr': 'fc',
        'classifier_in_features': 512,
        'recommended_threshold': 0.5,
    },
    'resnet50': {
        'name': 'resnet50',
        'description': 'ResNet-50 from torchvision (pretrained on ImageNet)',
        'architecture': 'ResNet',
        'source': 'torchvision',
        'model_func': lambda: models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
        'input_size': 224,
        'modify_classifier': True,
        'classifier_attr': 'fc',
        'classifier_in_features': 2048,
        'recommended_threshold': 0.5,
    },
    'resnet101': {
        'name': 'resnet101',
        'description': 'ResNet-101 from torchvision (pretrained on ImageNet)',
        'architecture': 'ResNet',
        'source': 'torchvision',
        'model_func': lambda: models.resnet101(weights=models.ResNet101_Weights.DEFAULT),
        'input_size': 224,
        'modify_classifier': True,
        'classifier_attr': 'fc',
        'classifier_in_features': 2048,
        'recommended_threshold': 0.5,
    },
    'vit_b_16': {
        'name': 'vit_b_16',
        'description': 'Vision Transformer B/16 from torchvision',
        'architecture': 'ViT',
        'source': 'torchvision',
        'model_func': lambda: models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT),
        'input_size': 224,
        'modify_classifier': True,
        'classifier_attr': 'heads',
        'classifier_in_features': 768,
        'recommended_threshold': 0.5,
    },
    'deit_base': {
        'name': 'deit_base_patch16_224',
        'description': 'DeiT Base (Data-efficient Image Transformer) - efficient training',
        'architecture': 'DeiT',
        'source': 'timm',
        'model_name': 'deit_base_patch16_224',
        'input_size': 224,
        'modify_classifier': True,
        'recommended_threshold': 0.5,
    },
    'deit_small': {
        'name': 'deit_small_patch16_224',
        'description': 'DeiT Small (Data-efficient Image Transformer) - faster variant',
        'architecture': 'DeiT',
        'source': 'timm',
        'model_name': 'deit_small_patch16_224',
        'input_size': 224,
        'modify_classifier': True,
        'recommended_threshold': 0.5,
    },
    'deit_tiny': {
        'name': 'deit_tiny_patch16_224',
        'description': 'DeiT Tiny (Data-efficient Image Transformer) - lightweight',
        'architecture': 'DeiT',
        'source': 'timm',
        'model_name': 'deit_tiny_patch16_224',
        'input_size': 224,
        'modify_classifier': True,
        'recommended_threshold': 0.5,
    },
    'swin_b': {
        'name': 'swin_b',
        'description': 'Swin Transformer Base from torchvision (hierarchical vision transformer)',
        'architecture': 'Swin Transformer',
        'source': 'torchvision',
        'model_func': lambda: models.swin_b(weights=models.Swin_B_Weights.DEFAULT),
        'input_size': 224,
        'modify_classifier': True,
        'classifier_attr': 'head',
        'classifier_in_features': 1024,
        'recommended_threshold': 0.5,
    },
    'swin_t': {
        'name': 'swin_t',
        'description': 'Swin Transformer Tiny from torchvision (faster, lighter version)',
        'architecture': 'Swin Transformer',
        'source': 'torchvision',
        'model_func': lambda: models.swin_t(weights=models.Swin_T_Weights.DEFAULT),
        'input_size': 224,
        'modify_classifier': True,
        'classifier_attr': 'head',
        'classifier_in_features': 768,
        'recommended_threshold': 0.5,
    },
    'swin_v2_b': {
        'name': 'swin_v2_b',
        'description': 'Swin Transformer V2 Base from torchvision (improved version)',
        'architecture': 'Swin Transformer',
        'source': 'torchvision',
        'model_func': lambda: models.swin_v2_b(weights=models.Swin_V2_B_Weights.DEFAULT),
        'input_size': 256,
        'modify_classifier': True,
        'classifier_attr': 'head',
        'classifier_in_features': 1024,
        'recommended_threshold': 0.5,
    },
    'swin_base_timm': {
        'name': 'swin_base_patch4_window7_224',
        'description': 'Swin Transformer Base from timm (hierarchical vision transformer)',
        'architecture': 'Swin Transformer',
        'source': 'timm',
        'model_name': 'swin_base_patch4_window7_224',
        'input_size': 224,
        'modify_classifier': True,
        'recommended_threshold': 0.5,
    },
    'swin_large_timm': {
        'name': 'swin_large_patch4_window7_224',
        'description': 'Swin Transformer Large from timm (high capacity model)',
        'architecture': 'Swin Transformer',
        'source': 'timm',
        'model_name': 'swin_large_patch4_window7_224',
        'input_size': 224,
        'modify_classifier': True,
        'recommended_threshold': 0.5,
    },
}


def list_available_models() -> Dict[str, Dict[str, Any]]:
    """
    Get a list of all available deepfake detection models
    
    Returns:
        Dictionary of available models with their details
    """
    return MODEL_REGISTRY.copy()


@dataclass
class DeepfakeDetectionResult:
    """Result of deepfake detection analysis"""
    is_deepfake: bool
    confidence: float
    scores: Dict[str, float]
    processing_time: float
    image_path: Optional[str] = None
    model_name: Optional[str] = None
    
    def get_recommendation(self, threshold: float = 0.5) -> str:
        """
        Get a recommendation based on the detection results
        
        Args:
            threshold: The threshold used for detection
            
        Returns:
            A string with recommendations
        """
        fake_score = self.scores.get('Fake', self.scores.get('fake', 0.0))
        real_score = self.scores.get('Real', self.scores.get('real', 0.0))
        
        if self.is_deepfake:
            if fake_score > 0.8:
                return "High confidence deepfake detection. Very likely to be fake."
            elif fake_score > 0.6:
                return "Moderate confidence deepfake detection. Likely to be fake."
            else:
                return "Low confidence deepfake detection. Consider manual review or using ensemble voting with multiple models."
        else:
            if real_score > 0.8:
                return "High confidence real image. Very likely to be authentic."
            elif real_score > 0.6:
                return "Moderate confidence real image. Likely to be authentic."
            elif fake_score < 0.3:
                return f"Low fake score ({fake_score:.2%}). Image appears real, but consider:\n" \
                       f"  - Lowering threshold (current: {threshold})\n" \
                       f"  - Trying a different model (use list_available_models())\n" \
                       f"  - Using ensemble voting with multiple models"
            else:
                return "Borderline case. Consider:\n" \
                       f"  - Manual review\n" \
                       f"  - Using ensemble voting with multiple models\n" \
                       f"  - Adjusting threshold"


@dataclass
class EnsembleResult:
    """Result from ensemble voting of multiple models"""
    is_deepfake: bool
    confidence: float
    individual_results: List[DeepfakeDetectionResult]
    voting_summary: Dict[str, int]
    average_fake_score: float
    average_real_score: float


class DeepfakeImageDetector:
    """
    Deepfake Image Detector supporting multiple pre-trained models
    
    This class provides functionality to detect whether an image is real or deepfake/AI-generated
    using various pre-trained transformer models.
    """
    
    def __init__(self, 
                 model_name: str = "prithivMLMods/deepfake-detector-model-v1",
                 device: Optional[str] = None):
        """
        Initialize the deepfake detector
        
        Args:
            model_name: Name of the pre-trained model to use. Can be:
                - Full Hugging Face model name (e.g., 'prithivMLMods/deepfake-detector-model-v1')
                - Short name from MODEL_REGISTRY (e.g., 'prithiv-v1', 'dima-vit')
            device: Device to run inference on ('cpu', 'cuda', or None for auto-detection)
        """
        # Check if model_name is a short name in registry
        if model_name in MODEL_REGISTRY:
            self.model_key = model_name
            self.model_name = MODEL_REGISTRY[model_name]['name']
            self.model_info = MODEL_REGISTRY[model_name]
        else:
            self.model_key = None
            self.model_name = model_name
            self.model_info = None
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.processor = None
        self.model = None
        self.labels = None
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the pre-trained model and processor"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load processor and model
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            
            # Try different model architectures
            model_loaded = False
            
            # Try SigLIP first (for prithiv model)
            if self.model_info and self.model_info['architecture'] == 'SigLIP':
                try:
                    self.model = SiglipForImageClassification.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float32 if self.device.type == 'cpu' else torch.float16
                    )
                    logger.info("Using SiglipForImageClassification")
                    model_loaded = True
                except Exception as e:
                    logger.warning(f"Failed to load with SiglipForImageClassification: {e}")
            
            # Fallback to AutoModel
            if not model_loaded:
                try:
                    self.model = AutoModelForImageClassification.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float32 if self.device.type == 'cpu' else torch.float16
                    )
                    logger.info("Using AutoModelForImageClassification")
                    model_loaded = True
                except Exception as e:
                    logger.error(f"Failed to load model: {e}")
                    raise
            
            # Move model to device
            if self.model is not None:
                self.model = self.model.to(self.device)
                self.model.eval()
                
                # Get label mappings
                self.labels = self.model.config.id2label
            else:
                raise RuntimeError("Failed to load model")
            
            logger.info(f"Model loaded successfully. Labels: {self.labels}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_image(self, image_input: Union[str, Image.Image, np.ndarray, BinaryIO, bytes]) -> Image.Image:
        """
        Preprocess image input to PIL Image format
        
        Args:
            image_input: Image as:
                - File path (str)
                - PIL Image object
                - Numpy array
                - Base64 encoded string (detected by 'data:image' prefix)
                - Binary stream/file object
                - Raw bytes
            
        Returns:
            PIL Image object
        """
        if isinstance(image_input, str):
            # Check if it's base64 encoded image
            if image_input.startswith('data:image'):
                # Extract base64 data from data URI
                if ',' in image_input:
                    base64_data = image_input.split(',')[1]
                else:
                    base64_data = image_input
                
                # Decode base64
                try:
                    image_bytes = base64.b64decode(base64_data)
                    image = Image.open(io.BytesIO(image_bytes))
                except Exception as e:
                    raise ValueError(f"Invalid base64 image data: {e}")
                    
            elif self._is_base64_string(image_input):
                # Raw base64 string without data URI prefix
                try:
                    image_bytes = base64.b64decode(image_input)
                    image = Image.open(io.BytesIO(image_bytes))
                except Exception as e:
                    raise ValueError(f"Invalid base64 image data: {e}")
            else:
                # File path
                if not os.path.exists(image_input):
                    raise FileNotFoundError(f"Image file not found: {image_input}")
                image = Image.open(image_input)
                
        elif isinstance(image_input, np.ndarray):
            # Numpy array
            image = Image.fromarray(image_input)
            
        elif isinstance(image_input, Image.Image):
            # Already PIL Image
            image = image_input
            
        elif isinstance(image_input, bytes):
            # Raw bytes
            try:
                image = Image.open(io.BytesIO(image_input))
            except Exception as e:
                raise ValueError(f"Invalid image bytes: {e}")
                
        elif hasattr(image_input, 'read'):
            # Stream/file-like object
            try:
                image = Image.open(image_input)
            except Exception as e:
                raise ValueError(f"Invalid image stream: {e}")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
        
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    
    def _is_base64_string(self, s: str) -> bool:
        """
        Check if a string is a valid base64 encoded string
        
        Args:
            s: String to check
            
        Returns:
            True if valid base64, False otherwise
        """
        try:
            # Check if string length is valid for base64
            if len(s) < 4 or len(s) % 4 != 0:
                return False
            
            # Try to decode and check if it results in valid image data
            decoded = base64.b64decode(s, validate=True)
            
            # Check if decoded data starts with common image file signatures
            image_signatures = [
                b'\xff\xd8\xff',  # JPEG
                b'\x89PNG',       # PNG
                b'GIF8',          # GIF
                b'BM',            # BMP
                b'RIFF',          # WebP (RIFF container)
            ]
            
            return any(decoded.startswith(sig) for sig in image_signatures)
            
        except Exception:
            return False
    
    def detect(self, 
               image_input: Union[str, Image.Image, np.ndarray, BinaryIO, bytes],
               threshold: float = 0.6) -> DeepfakeDetectionResult:
        """
        Detect if an image is deepfake/AI-generated
        
        Args:
            image_input: Image as:
                - File path (str)
                - PIL Image object
                - Numpy array
                - Base64 encoded string (with or without data URI prefix)
                - Binary stream/file object
                - Raw bytes
            threshold: Threshold for deepfake classification (0.0 to 1.0)
            
        Returns:
            DeepfakeDetectionResult object with detection results
        """
        start_time = time.time()
        
        try:
            # Preprocess image
            image = self.preprocess_image(image_input)
            
            # Get image path if available
            image_path = image_input if isinstance(image_input, str) else None
            
            # Check if processor and model are loaded
            if self.processor is None:
                raise RuntimeError("Processor not initialized")
            if self.model is None:
                raise RuntimeError("Model not initialized")
            
            # Process image
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Get probabilities
                probabilities = F.softmax(logits, dim=-1)
                
                # Build scores dictionary with null check
                if self.labels is None:
                    raise RuntimeError("Model labels not initialized")
                
                scores = {self.labels[i]: float(probabilities[0][i].cpu()) 
                         for i in range(len(self.labels))}
                
                # Find the fake/deepfake score directly
                fake_score = 0.0
                fake_label = None
                deepfake_indicators = ['fake', 'deepfake', 'artificial', 'generated', 'synthetic']
                
                for label, score in scores.items():
                    label_lower = label.lower()
                    for indicator in deepfake_indicators:
                        if indicator in label_lower:
                            fake_score = score
                            fake_label = label
                            break
                    if fake_label:
                        break
                
                # If we found a fake label, use its score directly
                if fake_label:
                    is_deepfake = fake_score >= threshold
                    confidence = fake_score
                else:
                    # Fallback to old logic if no fake label found
                    max_label = max(scores.keys(), key=lambda k: scores[k])
                    confidence = scores[max_label]
                    is_deepfake = self._is_deepfake_label(max_label, confidence, threshold)
                
                processing_time = time.time() - start_time
                
                return DeepfakeDetectionResult(
                    is_deepfake=is_deepfake,
                    confidence=confidence,
                    scores=scores,
                    processing_time=processing_time,
                    image_path=image_path,
                    model_name=self.model_name
                )
                
        except Exception as e:
            logger.error(f"Error during detection: {str(e)}")
            raise
    
    def _is_deepfake_label(self, label: str, confidence: float, threshold: float) -> bool:
        """
        Determine if a label indicates deepfake based on common label patterns
        
        Args:
            label: Predicted label
            confidence: Confidence score
            threshold: Classification threshold
            
        Returns:
            True if deepfake, False if real
        """
        # Common patterns for deepfake labels
        deepfake_indicators = ['fake', 'deepfake', 'artificial', 'generated', 'synthetic']
        real_indicators = ['real', 'authentic', 'genuine', 'human']
        
        label_lower = label.lower()
        
        # Check for deepfake indicators
        for indicator in deepfake_indicators:
            if indicator in label_lower:
                # If model says "Fake" with high confidence, it IS a deepfake
                return confidence >= threshold
        
        # Check for real indicators
        for indicator in real_indicators:
            if indicator in label_lower:
                # If model says "Real" with high confidence, it is NOT a deepfake
                # So return True (is_deepfake) only if confidence is LOW (meaning model is uncertain)
                return confidence < (1.0 - threshold)
        
        # Default: use threshold on confidence
        # If we can't determine from label, assume higher confidence means more likely to be the predicted class
        return confidence >= threshold
    
    def detect_batch(self, 
                     image_inputs: List[Any],
                     threshold: float = 0.5) -> List[DeepfakeDetectionResult]:
        """
        Detect deepfakes in a batch of images
        
        Args:
            image_inputs: List of images supporting multiple formats:
                - File paths (str)
                - PIL Image objects
                - Numpy arrays
                - Base64 encoded strings (with or without data URI prefix)
                - Binary streams/file objects
                - Raw bytes
            threshold: Threshold for deepfake classification
            
        Returns:
            List of DeepfakeDetectionResult objects
        """
        results = []
        
        logger.info(f"Processing batch of {len(image_inputs)} images")
        
        for i, image_input in enumerate(image_inputs):
            try:
                logger.info(f"Processing image {i+1}/{len(image_inputs)}")
                result = self.detect(image_input, threshold)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing image {i+1}: {str(e)}")
                # Add error result
                results.append(DeepfakeDetectionResult(
                    is_deepfake=False,
                    confidence=0.0,
                    scores={},
                    processing_time=0.0,
                    image_path=image_input if isinstance(image_input, str) else None
                ))
        
        return results
    
    def analyze_directory(self, 
                         directory_path: str,
                         extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp', '.gif'],
                         threshold: float = 0.6) -> List[DeepfakeDetectionResult]:
        """
        Analyze all images in a directory for deepfakes
        
        Args:
            directory_path: Path to directory containing images
            extensions: List of image file extensions to process
            threshold: Threshold for deepfake classification
            
        Returns:
            List of DeepfakeDetectionResult objects
        """
        directory = Path(directory_path)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Find all image files
        image_files = []
        for ext in extensions:
            image_files.extend(directory.glob(f"*{ext}"))
            # image_files.extend(directory.glob(f"*{ext.upper()}"))
        
        logger.info(f"Found {len(image_files)} image files in {directory_path}")
        
        # Convert to strings
        image_paths = [str(img) for img in image_files]
        
        return self.detect_batch(image_paths, threshold)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if self.model is None:
            raise RuntimeError("Model not initialized")
            
        return {
            'model_name': self.model_name,
            'model_key': self.model_key,
            'device': str(self.device),
            'labels': self.labels,
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'model_size_mb': sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024 / 1024
        }


class EnsembleDetector:
    """
    Enhanced ensemble detector that uses multiple models for improved accuracy.
    Supports HuggingFace, torchvision, and timm models.
    """
    
    def __init__(self, 
                 model_keys: Optional[List[str]] = None,
                 device: Optional[str] = None):
        """
        Initialize ensemble detector with multiple models
        
        Args:
            model_keys: List of model keys from MODEL_REGISTRY to use. 
                       If None, uses default HuggingFace models.
            device: Device to run inference on ('cpu', 'cuda', or None for auto-detection)
        
        Example:
            # Use HuggingFace models only
            ensemble = EnsembleDetector(model_keys=['prithiv-v1', 'dima-vit'])
            
            # Mix HuggingFace and torchvision models
            ensemble = EnsembleDetector(model_keys=[
                'prithiv-v1', 'efficientnet_b4', 'xception', 'resnet50'
            ])
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        if model_keys is None:
            # Use default HuggingFace models that are most reliable
            model_keys = [ 'prithiv-v1', 'dima-vit', 'organika-sdxl']
        
        # Validate model keys
        invalid_keys = [key for key in model_keys if key not in MODEL_REGISTRY]
        if invalid_keys:
            raise ValueError(
                f"Invalid model keys: {invalid_keys}. "
                f"Available models: {list(MODEL_REGISTRY.keys())}"
            )
        
        self.model_keys = model_keys
        self.detectors = {}
        logger.info(f"Initializing ensemble with models: {model_keys}")
        
        # Load all models
        self._load_models()
        
        if not self.detectors:
            raise RuntimeError("No models could be loaded for ensemble")
        
        logger.info(f"Ensemble initialized with {len(self.detectors)} models")
    
    def _load_models(self):
        """Load all specified models"""
        for key in self.model_keys:
            try:
                config = MODEL_REGISTRY[key]
                
                if config['source'] == 'huggingface':
                    # Use existing HuggingFace loading logic
                    detector = DeepfakeImageDetector(model_name=key, device=str(self.device))
                    self.detectors[key] = detector
                    logger.info(f"✓ Loaded HuggingFace model: {key}")
                
                elif config['source'] == 'npr':
                    # Load NPR detector
                    npr_detector = self._create_npr_detector(config)
                    self.detectors[key] = npr_detector
                    logger.info(f"✓ Loaded NPR model: {key}")
                    
                elif config['source'] == 'torchvision':
                    # Load torchvision model
                    model = self._load_torchvision_model(config)
                    self.detectors[key] = self._wrap_pytorch_model(model, key, config)
                    logger.info(f"✓ Loaded torchvision model: {key}")
                    
                elif config['source'] == 'timm':
                    # Load timm model
                    if not TIMM_AVAILABLE:
                        logger.warning(f"⚠ Skipping {key}: timm library not installed")
                        continue
                    model = self._load_timm_model(config)
                    self.detectors[key] = self._wrap_pytorch_model(model, key, config)
                    logger.info(f"✓ Loaded timm model: {key}")
                
            except Exception as e:
                logger.warning(f"⚠ Failed to load model {key}: {e}")
    
    def _create_npr_detector(self, config: Dict):
        """Create NPR detector with improved scoring algorithm"""
        try:
            from scipy.ndimage import gaussian_filter
        except ImportError:
            logger.warning("scipy not installed. NPR model requires scipy for image perturbations.")
            raise
        
        class NPRDetectorWrapper:
            """NPR detector with optimized parameters for accurate real/fake classification"""
            
            def __init__(self, device, config):
                self.device = device
                self.model_name = config['name']
                self.model_key = 'npr-ensemble'
                
                # Initialize base detector for NPR scoring
                base_detector = DeepfakeImageDetector(model_name='prithiv-v1', device=str(device))
                
                # NPR configuration with optimized parameters
                self.num_perturbations = config.get('num_perturbations', 15)
                self.perturbation_strength = config.get('perturbation_strength', 0.03)
                self.base_detector = base_detector
                
                logger.info(f"NPR detector initialized with {self.num_perturbations} perturbations, strength={self.perturbation_strength}")
            
            def _generate_perturbations(self, image: Image.Image) -> List[Image.Image]:
                """Generate perturbed versions of the image"""
                perturbations = []
                img_array = np.array(image).astype(np.float32) / 255.0
                
                for i in range(self.num_perturbations):
                    perturbed = img_array.copy()
                    
                    # Apply different perturbation types based on index
                    perturbation_type = i % 5
                    
                    if perturbation_type == 0:
                        # Gaussian noise
                        noise = np.random.normal(0, self.perturbation_strength, perturbed.shape)
                        perturbed = np.clip(perturbed + noise, 0, 1)
                    
                    elif perturbation_type == 1:
                        # Gaussian blur
                        sigma = self.perturbation_strength * 2
                        for c in range(3):
                            perturbed[:, :, c] = gaussian_filter(perturbed[:, :, c], sigma=sigma)
                    
                    elif perturbation_type == 2:
                        # Brightness adjustment
                        brightness_factor = 1.0 + (np.random.rand() - 0.5) * self.perturbation_strength * 2
                        perturbed = np.clip(perturbed * brightness_factor, 0, 1)
                    
                    elif perturbation_type == 3:
                        # Contrast adjustment
                        mean_color = perturbed.mean(axis=(0, 1), keepdims=True)
                        contrast_factor = 1.0 + (np.random.rand() - 0.5) * self.perturbation_strength * 2
                        perturbed = np.clip(mean_color + (perturbed - mean_color) * contrast_factor, 0, 1)
                    
                    elif perturbation_type == 4:
                        # Saturation adjustment
                        gray = perturbed.mean(axis=2, keepdims=True)
                        saturation_factor = 1.0 + (np.random.rand() - 0.5) * self.perturbation_strength * 2
                        perturbed = np.clip(gray + (perturbed - gray) * saturation_factor, 0, 1)
                    
                    # Convert back to PIL Image
                    perturbed_uint8 = (perturbed * 255).astype(np.uint8)
                    perturbed_img = Image.fromarray(perturbed_uint8)
                    perturbations.append(perturbed_img)
                
                return perturbations
            
            def detect(self, image_input: Union[str, Image.Image, np.ndarray, BinaryIO, bytes],
                      threshold: float = 0.5) -> DeepfakeDetectionResult:
                """Detect deepfake using simplified NPR method with robust scoring"""
                start_time = time.time()
                
                # Preprocess image
                image = self.base_detector.preprocess_image(image_input)
                
                # Get original prediction
                original_result = self.base_detector.detect(image, threshold=0.5)
                original_fake_score = original_result.scores.get('Fake', original_result.scores.get('fake', 0.0))
                
                # Generate perturbations
                perturbations = self._generate_perturbations(image)
                
                # Get predictions for perturbations
                perturbed_scores = []
                for i, perturbed_img in enumerate(perturbations):
                    try:
                        result = self.base_detector.detect(perturbed_img, threshold=0.5)
                        fake_score = result.scores.get('Fake', result.scores.get('fake', 0.0))
                        perturbed_scores.append(fake_score)
                    except Exception as e:
                        logger.warning(f"Failed to process perturbation {i}: {e}")
                
                if not perturbed_scores:
                    raise RuntimeError("Failed to process any perturbations")
                
                # Calculate statistics
                perturbed_mean = np.mean(perturbed_scores)
                perturbed_std = np.std(perturbed_scores)
                
                # Coefficient of variation (measures relative variability)
                cv = perturbed_std / (abs(perturbed_mean) + 1e-6) if abs(perturbed_mean) > 1e-6 else 0
                
                # SIMPLIFIED NPR SCORING ALGORITHM
                # Core principle: Use original fake score as base, adjust based on perturbation stability
                # - AI-generated images are MORE STABLE (low CV, low std) → keep or boost score
                # - Real images are LESS STABLE (high CV, high std) → reduce score
                
                final_fake_score = original_fake_score
                
                # Adjust based on stability metrics
                if cv < 0.10:
                    # Very stable (low variance) → likely AI-generated
                    # Boost the fake score
                    stability_boost = 1.0 + (0.10 - cv) * 1.5  # Up to 1.15x boost
                    final_fake_score = min(final_fake_score * stability_boost, 1.0)
                elif cv > 0.18:
                    # Very unstable (high variance) → likely real
                    # Reduce the fake score
                    instability_penalty = 0.8 - (cv - 0.18) * 1.0  # Down to 0.6x
                    instability_penalty = max(instability_penalty, 0.5)
                    final_fake_score = final_fake_score * instability_penalty
                
                # Additional adjustment based on score drop consistency
                # If perturbations consistently have lower scores, that's another AI indicator
                score_drop = original_fake_score - perturbed_mean
                if score_drop > 0.05 and cv < 0.15:
                    # Consistent drop with low variance → strong AI indicator
                    final_fake_score = min(final_fake_score * 1.1, 1.0)
                elif score_drop < -0.05:
                    # Scores increased under perturbation → unusual, might be real
                    final_fake_score = final_fake_score * 0.9
                
                # Classification using the threshold
                is_deepfake = final_fake_score >= threshold
                
                # Confidence is based on distance from threshold
                distance_from_threshold = abs(final_fake_score - threshold)
                confidence = min(final_fake_score if is_deepfake else (1.0 - final_fake_score), 1.0)
                
                processing_time = time.time() - start_time
                
                # Build detailed scores dictionary
                scores = {
                    'Fake': final_fake_score,
                    'Real': 1.0 - final_fake_score,
                    'Original Fake Score': original_fake_score,
                    'Perturbed Mean': perturbed_mean,
                    'Perturbed Std': perturbed_std,
                    'Coefficient of Variation': cv,
                    'Score Drop': score_drop
                }
                
                image_path = image_input if isinstance(image_input, str) else None
                
                logger.info(f"NPR Analysis: original_fake={original_fake_score:.3f}, "
                           f"perturbed_mean={perturbed_mean:.3f}, "
                           f"std={perturbed_std:.3f}, cv={cv:.3f}, "
                           f"score_drop={score_drop:.3f}")
                logger.info(f"Final Fake Score: {final_fake_score:.3f} (threshold={threshold})")
                logger.info(f"Classification: {'FAKE' if is_deepfake else 'REAL'} (confidence={confidence:.3f})")
                
                return DeepfakeDetectionResult(
                    is_deepfake=is_deepfake,
                    confidence=confidence,
                    scores=scores,
                    processing_time=processing_time,
                    image_path=image_path,
                    model_name='NPR-Perturbation-Rank'
                )
        
        return NPRDetectorWrapper(device=self.device, config=config)
    
    def _load_torchvision_model(self, config: Dict) -> torch.nn.Module:
        """Load model from torchvision"""
        import torch.nn as nn
        
        model = config['model_func']()
        
        if config.get('modify_classifier', False):
            # Modify final layer for binary classification
            classifier_attr = config['classifier_attr']
            in_features = config['classifier_in_features']
            
            if classifier_attr == 'classifier':
                # For EfficientNet
                model.classifier = nn.Sequential(
                    nn.Dropout(p=0.3, inplace=True),
                    nn.Linear(in_features, 2)
                )
            elif classifier_attr == 'fc':
                # For ResNet
                model.fc = nn.Linear(in_features, 2)
            elif classifier_attr == 'heads':
                # For ViT
                model.heads = nn.Linear(in_features, 2)
            elif classifier_attr == 'head':
                # For Swin Transformer
                model.head = nn.Linear(in_features, 2)
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def _load_timm_model(self, config: Dict) -> torch.nn.Module:
        """Load model from timm library"""
        model = timm.create_model(
            config['model_name'],
            pretrained=True,
            num_classes=2  # Binary classification
        )
        model = model.to(self.device)
        model.eval()
        return model
    
    def _wrap_pytorch_model(self, model: torch.nn.Module, model_key: str, config: Dict):
        """Wrap PyTorch model to match DeepfakeImageDetector interface"""
        class PyTorchModelWrapper:
            """Wrapper to make PyTorch models compatible with DeepfakeImageDetector interface"""
            
            def __init__(self, model, device, input_size, model_key, model_name):
                self.model = model
                self.device = device
                self.model_key = model_key
                self.model_name = model_name
                self.transform = transforms.Compose([
                    transforms.Resize((input_size, input_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
            
            def preprocess_image(self, image_input: Union[str, Image.Image, np.ndarray, BinaryIO, bytes]) -> Image.Image:
                """Preprocess image input to PIL Image format"""
                if isinstance(image_input, str):
                    if not os.path.exists(image_input):
                        raise FileNotFoundError(f"Image file not found: {image_input}")
                    image = Image.open(image_input)
                elif isinstance(image_input, np.ndarray):
                    image = Image.fromarray(image_input)
                elif isinstance(image_input, Image.Image):
                    image = image_input
                elif isinstance(image_input, bytes):
                    image = Image.open(io.BytesIO(image_input))
                elif hasattr(image_input, 'read'):
                    image = Image.open(image_input)
                else:
                    raise ValueError(f"Unsupported image input type: {type(image_input)}")
                
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                return image
            
            def detect(self, image_input: Union[str, Image.Image, np.ndarray, BinaryIO, bytes],
                      threshold: float = 0.5) -> DeepfakeDetectionResult:
                """Predict if image is deepfake"""
                start_time = time.time()
                
                # Preprocess image
                image = self.preprocess_image(image_input)
                image_path = image_input if isinstance(image_input, str) else None
                
                # Transform and prepare input
                input_tensor = self.transform(image).unsqueeze(0).to(self.device)
                
                # Run inference
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    real_prob = float(probabilities[0][0].cpu())
                    fake_prob = float(probabilities[0][1].cpu())
                
                processing_time = time.time() - start_time
                
                return DeepfakeDetectionResult(
                    is_deepfake=fake_prob >= threshold,
                    confidence=fake_prob,
                    scores={'Real': real_prob, 'Fake': fake_prob},
                    processing_time=processing_time,
                    image_path=image_path,
                    model_name=self.model_name
                )
        
        return PyTorchModelWrapper(
            model=model,
            device=self.device,
            input_size=config['input_size'],
            model_key=model_key,
            model_name=config['name']
        )
    
    def detect(self, 
               image_input: Union[str, Image.Image, np.ndarray, BinaryIO, bytes],
               threshold: float = 0.5,
               voting_method: str = 'majority') -> EnsembleResult:
        """
        Detect deepfake using ensemble of models
        
        Args:
            image_input: Image input (same formats as DeepfakeImageDetector)
            threshold: Threshold for each individual model
            voting_method: Voting method to use:
                - 'majority': Simple majority vote
                - 'weighted': Weighted by confidence
                - 'unanimous': All models must agree
                - 'average': Average of fake scores
                - 'maximum': Use prediction from model with highest confidence
            
        Returns:
            EnsembleResult with combined prediction
        """
        # Get predictions from all models
        individual_results = []
        for key, detector in self.detectors.items():
            try:
                result = detector.detect(image_input, threshold)
                individual_results.append(result)
            except Exception as e:
                logger.error(f"Error in model {key}: {e}")
        
        if not individual_results:
            raise RuntimeError("All models failed to process the image")
        
        # Calculate voting
        fake_votes = sum(1 for r in individual_results if r.is_deepfake)
        real_votes = len(individual_results) - fake_votes
        
        voting_summary = {
            'fake': fake_votes,
            'real': real_votes,
            'total': len(individual_results)
        }
        
        # Calculate average scores
        fake_scores = []
        real_scores = []
        for result in individual_results:
            fake_score = result.scores.get('Fake', result.scores.get('fake', 0.0))
            real_score = result.scores.get('Real', result.scores.get('real', 0.0))
            fake_scores.append(fake_score)
            real_scores.append(real_score)
        
        avg_fake_score = sum(fake_scores) / len(fake_scores) if fake_scores else 0.0
        avg_real_score = sum(real_scores) / len(real_scores) if real_scores else 0.0
        
        # Determine final prediction based on voting method
        if voting_method == 'majority':
            is_deepfake = fake_votes > real_votes
            confidence = max(avg_fake_score, avg_real_score)
        
        elif voting_method == 'weighted':
            # Weight votes by confidence
            weighted_fake = sum(r.confidence for r in individual_results if r.is_deepfake)
            weighted_real = sum(r.confidence for r in individual_results if not r.is_deepfake)
            is_deepfake = weighted_fake > weighted_real
            confidence = max(weighted_fake, weighted_real) / len(individual_results)
        
        elif voting_method == 'unanimous':
            # All models must agree
            is_deepfake = fake_votes == len(individual_results)
            confidence = avg_fake_score if is_deepfake else avg_real_score
        
        elif voting_method == 'average':
            # Use average fake score
            is_deepfake = avg_fake_score >= threshold
            confidence = avg_fake_score
        
        elif voting_method == 'maximum':
            # Use prediction from model with highest confidence
            max_confidence_result = max(individual_results, key=lambda r: r.confidence)
            is_deepfake = max_confidence_result.is_deepfake
            confidence = max_confidence_result.confidence
        
        else:
            raise ValueError(f"Unknown voting method: {voting_method}")
        
        return EnsembleResult(
            is_deepfake=is_deepfake,
            confidence=confidence,
            individual_results=individual_results,
            voting_summary=voting_summary,
            average_fake_score=avg_fake_score,
            average_real_score=avg_real_score
        )
    
    def compare_models(self, 
                      image_input: Union[str, Image.Image, np.ndarray, BinaryIO, bytes],
                      threshold: float = 0.5) -> Dict[str, Any]:
        """
        Compare predictions from all models
        
        Args:
            image_input: Image input
            threshold: Threshold for detection
            
        Returns:
            Dictionary with comparison results
        """
        results = []
        for key, detector in self.detectors.items():
            try:
                result = detector.detect(image_input, threshold)
                results.append({
                    'model': result.model_name,
                    'model_key': key,
                    'is_deepfake': result.is_deepfake,
                    'confidence': result.confidence,
                    'scores': result.scores,
                    'processing_time': result.processing_time
                })
            except Exception as e:
                logger.error(f"Error in model {key}: {e}")
        
        return {
            'image': image_input if isinstance(image_input, str) else 'N/A',
            'threshold': threshold,
            'model_results': results,
            'agreement': len(set(r['is_deepfake'] for r in results)) == 1
        }
    
    def analyze_directory(self,
                         directory_path: str,
                         extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp', '.gif'],
                         threshold: float = 0.5,
                         voting_method: str = 'majority') -> List[EnsembleResult]:
        """
        Analyze all images in a directory using ensemble
        
        Args:
            directory_path: Path to directory containing images
            extensions: List of image file extensions to process
            threshold: Threshold for deepfake classification
            voting_method: Voting method to use
            
        Returns:
            List of EnsembleResult objects
        """
        directory = Path(directory_path)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Find all image files
        image_files = []
        for ext in extensions:
            image_files.extend(directory.glob(f"*{ext}"))
        
        logger.info(f"Found {len(image_files)} image files in {directory_path}")
        
        results = []
        for i, image_file in enumerate(image_files):
            try:
                logger.info(f"Processing {i+1}/{len(image_files)}: {image_file.name}")
                result = self.detect(str(image_file), threshold, voting_method)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {image_file.name}: {e}")
        
        return results


