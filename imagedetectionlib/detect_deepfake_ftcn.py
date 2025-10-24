"""
FTCN (Fully Temporal Convolutional Network) Deepfake Detection Module

This module implements deepfake detection using FTCN architecture,
which uses temporal features for enhanced fake detection capabilities.

Key Features:
- FTCN architecture for deepfake detection
- Optimized for face-based deepfake detection
- Input size: 224x224
- Binary classification for deepfake detection
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

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FTCNDetectionResult:
    """Result of FTCN deepfake detection"""
    is_deepfake: bool
    confidence: float
    scores: Dict[str, float]
    model_name: str
    processing_time: float
    image_path: Optional[str] = None


class FTCNDetector:
    """
    FTCN Deepfake Detector
    
    Uses FTCN (Fully Temporal Convolutional Network) architecture for deepfake detection.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: Optional[str] = None,
                 input_size: int = 224,
                 debug: bool = False):
        """
        Initialize FTCN detector
        
        Args:
            model_path: Path to model weights (default: Models/FTCN/ftcn_tt.pth)
            device: Device to run inference on ('cpu', 'cuda', or None for auto-detection)
            input_size: Input image size (default: 224)
            debug: Enable debug logging
        """
        self.debug = debug
        self.input_size = input_size
        self.model_name = "FTCN"
        
        # Set default model path if not provided
        if model_path is None:
            script_dir = Path(__file__).parent.parent
            model_path = str(script_dir / "Models" / "FTCN" / "ftcn_tt.pth")
        
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
    
    def _create_ftcn_model(self, num_classes: int = 2) -> nn.Module:
        """
        Create FTCN model architecture
        
        Args:
            num_classes: Number of output classes (default: 2 for binary classification)
            
        Returns:
            FTCN model
        """
        class FTCN(nn.Module):
            """FTCN architecture for deepfake detection"""
            
            def __init__(self, num_classes=2):
                super(FTCN, self).__init__()
                
                # Convolutional layers
                self.conv1 = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
                
                self.conv2 = nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
                
                self.conv3 = nn.Sequential(
                    nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
                
                self.conv4 = nn.Sequential(
                    nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
                
                # Global average pooling
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                
                # Fully connected layers
                self.fc = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(512, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(256, num_classes)
                )
            
            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.conv3(x)
                x = self.conv4(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.fc(x)
                return x
        
        return FTCN(num_classes=num_classes)
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load FTCN model from weights"""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"FTCN model not found at: {model_path}\n"
                    f"Please ensure the file 'ftcn_tt.pth' exists in the Models/FTCN folder."
                )
            
            if self.debug:
                logger.info(f"Loading FTCN model from {model_path}")
            
            # Create model architecture
            model = self._create_ftcn_model(num_classes=2)
            
            # Load weights
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                # Check for different keys that might contain the state dict
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    # Assume checkpoint is the state dict itself
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if present (from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace('module.', '') if k.startswith('module.') else k
                new_state_dict[name] = v
            
            # Load the state dict
            try:
                model.load_state_dict(new_state_dict, strict=True)
            except RuntimeError as e:
                if self.debug:
                    logger.warning(f"Strict loading failed: {e}")
                    logger.info("Attempting to load with strict=False...")
                model.load_state_dict(new_state_dict, strict=False)
            
            model = model.to(self.device)
            
            if self.debug:
                logger.info("[OK] FTCN model loaded successfully")
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading FTCN model: {e}")
            raise
    
    def _get_transforms(self) -> transforms.Compose:
        """Get image preprocessing transforms"""
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
               threshold: float = 0.5) -> FTCNDetectionResult:
        """
        Detect if an image is a deepfake
        
        Args:
            image_input: Image in various formats (file path, PIL Image, numpy array, etc.)
            threshold: Threshold for classification (default: 0.5)
        
        Returns:
            FTCNDetectionResult with prediction and confidence scores
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
                
                # IMPORTANT: Based on the FTCN model training, the class labels are:
                # Class 0 = Fake, Class 1 = Real (inverted from typical convention)
                fake_prob = probs[0, 0].item()
                real_prob = probs[0, 1].item()
            
            # Determine prediction
            is_deepfake = fake_prob > threshold
            confidence = fake_prob if is_deepfake else real_prob
            
            processing_time = time.time() - start_time
            
            return FTCNDetectionResult(
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
                    batch_size: int = 8) -> List[FTCNDetectionResult]:
        """
        Detect deepfakes in a batch of images
        
        Args:
            image_inputs: List of images in various formats
            threshold: Threshold for classification
            batch_size: Batch size for processing
        
        Returns:
            List of FTCNDetectionResult objects
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
                    results.append(FTCNDetectionResult(
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
                         extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.avif']) -> List[FTCNDetectionResult]:
        """
        Analyze all images in a directory
        
        Args:
            directory_path: Path to directory containing images
            threshold: Threshold for classification
            extensions: List of image file extensions to process
        
        Returns:
            List of FTCNDetectionResult objects
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
    """Example usage of FTCN detector"""
    print("FTCN Deepfake Detector")
    print("=" * 50)
    
    # Initialize detector
    detector = FTCNDetector(debug=True)
    
    # Example: Detect from file path
    # result = detector.detect('path/to/image.jpg', threshold=0.5)
    # print(f"Prediction: {'DEEPFAKE' if result.is_deepfake else 'REAL'}")
    # print(f"Confidence: {result.confidence:.4f}")
    # print(f"Scores: {result.scores}")
    
    print("\nFTCN detector initialized successfully!")


if __name__ == "__main__":
    main()
