"""
UniversalFakeDetectV2 Detector

This module implements deepfake detection using the UniversalFakeDetectV2 model from TrueMedia.
Based on: https://github.com/truemediaorg/ml-models/blob/main/UniversalFakeDetectV2/

The model uses CLIP image embeddings and a custom detection head trained for deepfake detection.
"""

import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPModel
from PIL import Image
import numpy as np
import logging
import time
from typing import Union, BinaryIO, Optional
from dataclasses import dataclass
import os
import io

# Set up logging
logger = logging.getLogger(__name__)


class UniversalFakeDetectv2(nn.Module):
    """
    Universal Fake Detect V2 model architecture
    Simple MLP classifier on top of CLIP embeddings
    """
    def __init__(self, num_classes=1, clip_feature_dim=768):
        super(UniversalFakeDetectv2, self).__init__()
        self.fc1 = nn.Linear(clip_feature_dim, 768)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(768, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


@dataclass
class UniversalFakeDetectResult:
    """Result of UniversalFakeDetectV2 detection analysis"""
    is_fake: bool
    fake_probability: float
    real_probability: float
    confidence: float
    processing_time: float
    image_path: Optional[str] = None
    model_name: str = "UniversalFakeDetectV2"


class UniversalFakeDetectV2Detector:
    """
    UniversalFakeDetectV2 Detector
    
    This detector uses CLIP image embeddings combined with a trained MLP classifier
    to detect deepfakes and AI-generated images.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Initialize the UniversalFakeDetectV2 detector
        
        Args:
            model_path: Path to the pretrained model weights (.pth file)
                       Default: "pretrained_weights/continued_ft_from_concat_20240930.pth"
            device: Device to run inference on ('cpu', 'cuda', or None for auto-detection)
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Set default model path if not provided
        if model_path is None:
            model_path = "pretrained_weights/continued_ft_from_concat_20240930.pth"
        
        self.model_path = model_path
        self.model_name = "UniversalFakeDetectV2"
        
        # Initialize components
        self.model = None
        self.backbone = None
        self.processor = None
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the pre-trained model and CLIP backbone"""
        try:
            logger.info(f"Loading UniversalFakeDetectV2 model")
            
            # Initialize the detection model
            self.model = UniversalFakeDetectv2()
            
            # Load pretrained weights if available
            if os.path.exists(self.model_path):
                logger.info(f"Loading pretrained weights from: {self.model_path}")
                ckpt = torch.load(self.model_path, map_location=self.device, weights_only=True)
                
                if "model" in ckpt.keys():
                    self.model.load_state_dict(ckpt['model'])
                else:
                    self.model.load_state_dict(ckpt)
                
                logger.info("Pretrained weights loaded successfully")
            else:
                logger.warning(f"Pretrained weights not found at {self.model_path}")
                logger.warning("Using model with random initialization - results may be unreliable")
                logger.info("To download pretrained weights, visit:")
                logger.info("https://github.com/truemediaorg/ml-models/tree/main/UniversalFakeDetectV2")
            
            self.model.eval()
            self.model.to(self.device)
            
            # Load CLIP backbone
            clip_backbone = "openai/clip-vit-large-patch14"
            logger.info(f"Loading CLIP backbone: {clip_backbone}")
            self.backbone = CLIPModel.from_pretrained(clip_backbone).to(self.device)
            self.processor = CLIPImageProcessor.from_pretrained(clip_backbone)
            
            logger.info("UniversalFakeDetectV2 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _is_valid_image(self, file_path_or_image: Union[str, Image.Image]) -> Optional[Image.Image]:
        """
        Validate and load image
        
        Args:
            file_path_or_image: File path or PIL Image
            
        Returns:
            PIL Image in RGB format, or None if invalid
        """
        try:
            if isinstance(file_path_or_image, str):
                if not os.path.exists(file_path_or_image):
                    logger.error(f"Image file not found: {file_path_or_image}")
                    return None
                img = Image.open(file_path_or_image)
            elif isinstance(file_path_or_image, Image.Image):
                img = file_path_or_image
            else:
                return None
            
            return img.convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return None
    
    def preprocess_image(self, 
                        image_input: Union[str, Image.Image, np.ndarray, BinaryIO, bytes]) -> Image.Image:
        """
        Preprocess image input to PIL Image format
        
        Args:
            image_input: Image as:
                - File path (str)
                - PIL Image object
                - Numpy array
                - Binary stream/file object
                - Raw bytes
            
        Returns:
            PIL Image object in RGB format
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image file not found: {image_input}")
            image = Image.open(image_input)
                
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input)
            
        elif isinstance(image_input, Image.Image):
            image = image_input
            
        elif isinstance(image_input, bytes):
            try:
                image = Image.open(io.BytesIO(image_input))
            except Exception as e:
                raise ValueError(f"Invalid image bytes: {e}")
                
        elif hasattr(image_input, 'read'):
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
    
    def detect(self, 
               image_input: Union[str, Image.Image, np.ndarray, BinaryIO, bytes],
               threshold: float = 0.5) -> UniversalFakeDetectResult:
        """
        Detect if an image is fake/AI-generated
        
        Args:
            image_input: Image in various formats
            threshold: Threshold for fake classification (0.0 to 1.0)
            
        Returns:
            UniversalFakeDetectResult object with detection results
        """
        start_time = time.time()
        
        try:
            # Preprocess image
            image = self.preprocess_image(image_input)
            
            # Get image path if available
            image_path = image_input if isinstance(image_input, str) else None
            
            # Validate image
            img = self._is_valid_image(image)
            if not img:
                raise ValueError("Invalid image")
            
            # Process image with CLIP processor
            img_processed = self.processor(
                images=img, 
                return_tensors="pt", 
                size=(224, 224), 
                do_rescale=True
            )["pixel_values"].to(self.device)
            
            # Run inference
            with torch.no_grad():
                # Get CLIP embeddings
                img_embeds = self.backbone.get_image_features(img_processed)
                
                # Get fake probability
                logits = self.model(img_embeds)
                fake_prob = torch.sigmoid(logits).item()
            
            real_prob = 1.0 - fake_prob
            is_fake = fake_prob >= threshold
            confidence = fake_prob if is_fake else real_prob
            
            processing_time = time.time() - start_time
            
            return UniversalFakeDetectResult(
                is_fake=is_fake,
                fake_probability=fake_prob,
                real_probability=real_prob,
                confidence=confidence,
                processing_time=processing_time,
                image_path=image_path,
                model_name=self.model_name
            )
            
        except Exception as e:
            logger.error(f"Error during detection: {str(e)}")
            raise
    
    def get_model_info(self):
        """Get information about the loaded model"""
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'model_path': self.model_path,
            'backbone': 'openai/clip-vit-large-patch14',
            'num_parameters': sum(p.numel() for p in self.model.parameters()) if self.model else 0,
        }


def real_or_fake_threshold(probability: float, threshold: float = 0.5) -> str:
    """
    Classify image as REAL or FAKE based on probability
    
    Args:
        probability: Fake probability (0.0 to 1.0)
        threshold: Classification threshold
        
    Returns:
        "FAKE" or "REAL"
    """
    return "FAKE" if probability >= threshold else "REAL"
