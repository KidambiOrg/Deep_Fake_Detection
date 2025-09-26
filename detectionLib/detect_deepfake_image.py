"""
Deepfake Image Detection using prithivMLMods/deepfake-detector-model-v1

This module implements deepfake image detection using a pre-trained model
from Hugging Face that can classify images as real or deepfake/AI-generated.

Model: prithivMLMods/deepfake-detector-model-v1
"""

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, SiglipForImageClassification, AutoModelForImageClassification
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

# Suppress warnings
warnings.filterwarnings("ignore", message=".*torch.load.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DeepfakeDetectionResult:
    """Result of deepfake detection analysis"""
    is_deepfake: bool
    confidence: float
    scores: Dict[str, float]
    processing_time: float
    image_path: Optional[str] = None


class DeepfakeImageDetector:
    """
    Deepfake Image Detector using prithivMLMods/deepfake-detector-model-v1
    
    This class provides functionality to detect whether an image is real or deepfake/AI-generated
    using a pre-trained transformer model.
    """
    
    def __init__(self, 
                 model_name: str = "prithivMLMods/deepfake-detector-model-v1",
                 device: Optional[str] = None):
        """
        Initialize the deepfake detector
        
        Args:
            model_name: Name of the pre-trained model to use
            device: Device to run inference on ('cpu', 'cuda', or None for auto-detection)
        """
        self.model_name = model_name
        
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
            
            # Use SiglipForImageClassification since we know this model uses SigLIP architecture
            try:
                self.model = SiglipForImageClassification.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32 if self.device.type == 'cpu' else torch.float16
                )
                logger.info("Using SiglipForImageClassification")
            except Exception as e:
                logger.warning(f"Failed to load with SiglipForImageClassification: {e}")
                logger.info("Falling back to AutoModelForImageClassification")
                self.model = AutoModelForImageClassification.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32 if self.device.type == 'cpu' else torch.float16
                )
            
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
                
                # Determine if deepfake
                # Assuming the model has labels like 'REAL' and 'FAKE' or similar
                # We'll use the highest probability score
                max_label = max(scores.keys(), key=lambda k: scores[k])
                confidence = scores[max_label]
                
                # Determine if deepfake based on label
                is_deepfake = self._is_deepfake_label(max_label, confidence, threshold)
                
                processing_time = time.time() - start_time
                
                return DeepfakeDetectionResult(
                    is_deepfake=is_deepfake,
                    confidence=confidence,
                    scores=scores,
                    processing_time=processing_time,
                    image_path=image_path
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
                return confidence >= threshold
        
        # Check for real indicators
        for indicator in real_indicators:
            if indicator in label_lower:
                return confidence < threshold
        
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
            'device': str(self.device),
            'labels': self.labels,
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'model_size_mb': sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024 / 1024
        }


