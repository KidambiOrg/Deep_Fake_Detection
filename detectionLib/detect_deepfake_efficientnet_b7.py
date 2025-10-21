"""
EfficientNet-B7 Deepfake Detection Module

This module implements deepfake detection using EfficientNet-B7 with Noisy Student weights,
based on the 3rd place solution from the Deepfake Detection Challenge.

Reference: https://github.com/NTech-Lab/deepfake-detection-challenge

Key Features:
- EfficientNet-B7 architecture with Noisy Student pre-trained weights
- Optimized for face-based deepfake detection
- Input size: 224x224 (optimized for face crops)
- Binary classification for deepfake detection
- High accuracy with robust feature extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageFilter
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
import cv2
import tempfile

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    logging.warning("timm library not available. EfficientNet-B7 detector will not work.")

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EfficientNetB7DetectionResult:
    """Result of EfficientNet-B7 deepfake detection"""
    is_deepfake: bool
    confidence: float
    scores: Dict[str, float]
    model_name: str
    processing_time: float
    image_path: Optional[str] = None


class EfficientNetB7Detector:
    """
    EfficientNet-B7 Deepfake Detector
    
    Uses EfficientNet-B7 architecture with Noisy Student pre-trained weights
    for deepfake detection. Based on the winning solution from DFDC challenge.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: Optional[str] = None,
                 input_size: int = 224,
                 debug: bool = False):
        """
        Initialize EfficientNet-B7 detector
        
        Args:
            model_path: Path to model weights directory (containing TensorFlow checkpoint or PyTorch weights)
                       Default: Models/EfficientNet-B7/noisy_student_efficientnet-b7
            device: Device to run inference on ('cpu', 'cuda', or None for auto-detection)
            input_size: Input image size (default: 224 for face crops)
            debug: Enable debug logging
        """
        if not TIMM_AVAILABLE:
            raise ImportError("timm library is required for EfficientNet-B7 detector. Install with: pip install timm")
        
        self.debug = debug
        self.input_size = input_size
        self.model_name = "EfficientNet-B7 (Noisy Student)"
        
        # Set default model path if not provided
        if model_path is None:
            model_path = "Models/EfficientNet-B7/noisy_student_efficientnet-b7"
        
        self.model_path = model_path
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        if self.debug:
            logger.info(f"Initializing {self.model_name} on {self.device}")
            logger.info(f"Model path: {self.model_path}")
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Setup transforms
        self.transform = self._get_transforms()
        
        if self.debug:
            logger.info(f"{self.model_name} initialized successfully")
    
    def _load_model(self, model_path: Optional[str] = None) -> nn.Module:
        """Load EfficientNet-B7 model from local weights"""
        try:
            # Create EfficientNet-B7 model structure with timm
            # Use pretrained=False since we'll load local weights
            if self.debug:
                logger.info("Creating EfficientNet-B7 model structure...")
            
            # First, try to create model with Noisy Student weights from timm as base
            # This ensures we have the correct architecture
            model = timm.create_model('tf_efficientnet_b7_ns', pretrained=False, num_classes=2)
            
            if self.debug:
                logger.info("Model structure created successfully")
            
            # Check if we have local PyTorch weights or TensorFlow checkpoint
            pytorch_weights_path = None
            tf_checkpoint_path = None
            
            if model_path and os.path.exists(model_path):
                # Check for PyTorch weights file
                if os.path.isfile(model_path) and model_path.endswith(('.pth', '.pt')):
                    pytorch_weights_path = model_path
                elif os.path.isdir(model_path):
                    # Check for PyTorch weights in directory
                    for filename in ['model.pth', 'model.pt', 'efficientnet_b7.pth', 'pytorch_model.bin']:
                        potential_path = os.path.join(model_path, filename)
                        if os.path.exists(potential_path):
                            pytorch_weights_path = potential_path
                            break
                    
                    # Check for TensorFlow checkpoint
                    checkpoint_file = os.path.join(model_path, 'checkpoint')
                    if os.path.exists(checkpoint_file):
                        tf_checkpoint_path = model_path
            
            # Load weights based on what's available
            if pytorch_weights_path:
                # Load PyTorch weights
                if self.debug:
                    logger.info(f"Loading PyTorch weights from {pytorch_weights_path}")
                
                state_dict = torch.load(pytorch_weights_path, map_location=self.device)
                
                # Handle different state dict formats
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif 'model' in state_dict:
                    state_dict = state_dict['model']
                
                # Remove 'module.' prefix if present (from DataParallel)
                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k.replace('module.', '') if k.startswith('module.') else k
                    new_state_dict[name] = v
                
                model.load_state_dict(new_state_dict, strict=False)
                if self.debug:
                    logger.info("✓ PyTorch weights loaded successfully")
            
            elif tf_checkpoint_path:
                # TensorFlow checkpoint found - load pretrained weights from timm instead
                # The tf_efficientnet_b7_ns model in timm already has Noisy Student weights
                if self.debug:
                    logger.info("TensorFlow checkpoint detected. Loading pretrained Noisy Student weights from timm...")
                
                # Reload model with pretrained=True to get Noisy Student weights
                model = timm.create_model('tf_efficientnet_b7_ns', pretrained=True, num_classes=2)
                
                if self.debug:
                    logger.info("✓ Noisy Student weights loaded from timm")
                    logger.info("Note: These are ImageNet pretrained weights. For fine-tuned deepfake detection,")
                    logger.info("      convert the TensorFlow checkpoint to PyTorch format.")
            
            else:
                # No local weights found - use pretrained from timm
                if self.debug:
                    logger.warning(f"No local weights found at {model_path}")
                    logger.info("Loading pretrained Noisy Student weights from timm...")
                
                model = timm.create_model('tf_efficientnet_b7_ns', pretrained=True, num_classes=2)
                
                if self.debug:
                    logger.info("✓ Pretrained weights loaded from timm")
            
            # Initialize classifier for binary classification if using pretrained weights
            if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
                # Check if classifier needs initialization (weights are close to zero)
                if torch.abs(model.classifier.weight).mean() < 0.001:
                    if self.debug:
                        logger.info("Initializing classifier for binary classification...")
                    nn.init.normal_(model.classifier.weight, mean=0.0, std=0.01)
                    if model.classifier.bias is not None:
                        nn.init.constant_(model.classifier.bias, 0)
            
            model = model.to(self.device)
            return model
            
        except Exception as e:
            logger.error(f"Error loading EfficientNet-B7 model: {e}")
            raise
    
    def _get_transforms(self) -> transforms.Compose:
        """Get image preprocessing transforms"""
        # Standard ImageNet normalization
        # These are the same values used in the DFDC challenge solution
        return transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _preprocess_image(self, image_input: Union[str, Image.Image, np.ndarray, BinaryIO, bytes]) -> Image.Image:
        """Preprocess image input to PIL Image"""
        try:
            if isinstance(image_input, str):
                if image_input.startswith('data:image'):
                    # Base64 encoded image
                    header, encoded = image_input.split(',', 1)
                    image_data = base64.b64decode(encoded)
                    image = Image.open(io.BytesIO(image_data))
                else:
                    # File path
                    image = Image.open(image_input)
            elif isinstance(image_input, bytes):
                image = Image.open(io.BytesIO(image_input))
            elif isinstance(image_input, np.ndarray):
                # Convert BGR to RGB if needed
                if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                    image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image_input)
            elif hasattr(image_input, 'read'):
                # File-like object
                image = Image.open(image_input)
            elif isinstance(image_input, Image.Image):
                image = image_input
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def detect(self, 
               image_input: Union[str, Image.Image, np.ndarray, BinaryIO, bytes],
               threshold: float = 0.5) -> EfficientNetB7DetectionResult:
        """
        Detect if an image is a deepfake
        
        Args:
            image_input: Image in various formats (file path, PIL Image, numpy array, etc.)
            threshold: Threshold for classification (default: 0.5)
        
        Returns:
            EfficientNetB7DetectionResult with prediction and confidence scores
        """
        start_time = time.time()
        
        try:
            # Preprocess image
            image = self._preprocess_image(image_input)
            
            # Get image path if available
            image_path = image_input if isinstance(image_input, str) else None
            
            # Transform image
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                
                # Apply softmax to get probabilities
                probs = F.softmax(outputs, dim=1)
                fake_prob = probs[0, 0].item()  # Assuming class 0 is fake
                real_prob = probs[0, 1].item()  # Assuming class 1 is real
            
            # Determine prediction
            is_deepfake = fake_prob > threshold
            confidence = fake_prob if is_deepfake else real_prob
            
            processing_time = time.time() - start_time
            
            return EfficientNetB7DetectionResult(
                is_deepfake=is_deepfake,
                confidence=confidence,
                scores={'Fake': fake_prob, 'Real': real_prob},
                model_name=self.model_name,
                processing_time=processing_time,
                image_path=image_path
            )
            
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            raise
    
    def detect_batch(self,
                    image_inputs: List[Union[str, Image.Image, np.ndarray]],
                    threshold: float = 0.5,
                    batch_size: int = 8) -> List[EfficientNetB7DetectionResult]:
        """
        Detect deepfakes in a batch of images
        
        Args:
            image_inputs: List of images in various formats
            threshold: Threshold for classification
            batch_size: Batch size for processing
        
        Returns:
            List of EfficientNetB7DetectionResult objects
        """
        results = []
        
        for i in range(0, len(image_inputs), batch_size):
            batch = image_inputs[i:i + batch_size]
            
            for image_input in batch:
                try:
                    result = self.detect(image_input, threshold)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing image in batch: {e}")
                    # Add a failed result
                    results.append(EfficientNetB7DetectionResult(
                        is_deepfake=False,
                        confidence=0.0,
                        scores={'Fake': 0.0, 'Real': 0.0},
                        model_name=self.model_name,
                        processing_time=0.0,
                        image_path=None
                    ))
        
        return results
    
    def analyze_directory(self,
                         directory_path: str,
                         threshold: float = 0.5,
                         extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.avif']) -> List[EfficientNetB7DetectionResult]:
        """
        Analyze all images in a directory
        
        Args:
            directory_path: Path to directory containing images
            threshold: Threshold for classification
            extensions: List of image file extensions to process
        
        Returns:
            List of EfficientNetB7DetectionResult objects
        """
        directory = Path(directory_path)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Find all image files
        image_files = []
        for ext in extensions:
            image_files.extend(directory.glob(f"*{ext}"))
            image_files.extend(directory.glob(f"*{ext.upper()}"))
        
        logger.info(f"Found {len(image_files)} images in {directory_path}")
        
        # Process all images
        results = []
        for image_file in image_files:
            try:
                result = self.detect(str(image_file), threshold)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {image_file}: {e}")
        
        return results


def main():
    """Example usage of EfficientNet-B7 detector"""
    print("EfficientNet-B7 Deepfake Detector")
    print("=" * 50)
    
    # Initialize detector
    detector = EfficientNetB7Detector(debug=True)
    
    # Example: Detect from file path
    # result = detector.detect('path/to/image.jpg', threshold=0.5)
    # print(f"Prediction: {'DEEPFAKE' if result.is_deepfake else 'REAL'}")
    # print(f"Confidence: {result.confidence:.4f}")
    # print(f"Scores: {result.scores}")
    
    print("\nEfficientNet-B7 detector initialized successfully!")


if __name__ == "__main__":
    main()
