"""
MesoNet Deepfake Detection Module

This module implements deepfake detection using MesoNet, a lightweight CNN architecture
designed to detect subtle facial forgery at the mesoscopic level.

Based on the paper:
"MesoNet: a Compact Facial Video Forgery Detection Network"
https://arxiv.org/abs/1809.00888

Key Features:
- Lightweight CNN architecture (compact model)
- Designed for 256x256 facial frames
- Detects subtle inconsistencies in deepfakes
- Binary classification (Real vs Fake)
- Fast inference suitable for real-time detection
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


class MesoNet(nn.Module):
    """
    MesoNet Architecture for Deepfake Detection
    
    A compact CNN that analyzes images at the mesoscopic level to detect
    subtle artifacts introduced by deepfake generation algorithms.
    """
    
    def __init__(self, num_classes=1):
        """
        Initialize MesoNet model
        
        Args:
            num_classes: Number of output classes (1 for binary classification with sigmoid)
        """
        super(MesoNet, self).__init__()
        
        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        
        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(8)
        
        self.conv3 = nn.Conv2d(8, 16, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(16)
        
        self.conv4 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm2d(16)
        
        # Max pooling layers (matching Keras architecture)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256->128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128->64
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64->32
        self.pool4 = nn.MaxPool2d(kernel_size=4, stride=4)  # 32->8 (matches Keras!)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        # After pooling: 256->128->64->32->8, with 16 channels: 8*8*16 = 1024
        self.fc1 = nn.Linear(16 * 8 * 8, 16)
        self.fc2 = nn.Linear(16, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network"""
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)  # 256->128
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)  # 128->64
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)  # 64->32
        
        # Conv block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)  # 32->8
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


@dataclass
class MesoNetDetectionResult:
    """Result of MesoNet deepfake detection analysis"""
    is_fake: bool
    fake_probability: float
    real_probability: float
    confidence: float
    processing_time: float
    image_path: Optional[str] = None
    model_name: str = "MesoNet"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            'is_fake': self.is_fake,
            'fake_probability': self.fake_probability,
            'real_probability': self.real_probability,
            'confidence': self.confidence,
            'processing_time': self.processing_time,
            'image_path': self.image_path,
            'model_name': self.model_name
        }


class MesoNetDetector:
    """
    MesoNet-based deepfake detector
    
    This detector uses the MesoNet architecture for lightweight and efficient
    deepfake detection on facial images.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: Optional[str] = None,
                 threshold: float = 0.5,
                 input_size: int = 256):
        """
        Initialize MesoNet detector
        
        Args:
            model_path: Path to pretrained MesoNet model (.h5 or .pth file)
            device: Device to run inference on ('cpu', 'cuda', or None for auto-detection)
            threshold: Threshold for fake classification (0.0 to 1.0)
            input_size: Input image size (default: 256x256 as per MesoNet paper)
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.threshold = threshold
        self.input_size = input_size
        self.model_name = "MesoNet"
        
        logger.info(f"Initializing MesoNet detector on device: {self.device}")
        logger.info(f"Input size: {input_size}x{input_size}, Threshold: {threshold}")
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _convert_keras_to_pytorch(self, keras_model, pytorch_model):
        """
        Convert Keras H5 model weights to PyTorch model.
        Maps layer weights from Keras to corresponding PyTorch layers.
        
        Args:
            keras_model: Loaded Keras model
            pytorch_model: PyTorch MesoNet model to load weights into
        """
        import numpy as np
        
        logger.info("Starting Keras to PyTorch weight conversion...")
        
        # Get all layers from Keras model
        keras_layers = {layer.name: layer for layer in keras_model.layers}
        
        # MesoNet architecture mapping:
        # Conv2d layers: conv1, conv2, conv3, conv4
        # BatchNorm2d layers: bn1, bn2, bn3, bn4
        # Fully connected: fc1, fc2
        
        def copy_conv_weights(keras_layer_name, pytorch_layer):
            """Copy convolutional layer weights"""
            if keras_layer_name in keras_layers:
                keras_layer = keras_layers[keras_layer_name]
                weights = keras_layer.get_weights()
                if len(weights) >= 1:
                    # Keras: (height, width, in_channels, out_channels)
                    # PyTorch: (out_channels, in_channels, height, width)
                    kernel = np.transpose(weights[0], (3, 2, 0, 1))
                    pytorch_layer.weight.data = torch.from_numpy(kernel).float()
                    if len(weights) > 1 and pytorch_layer.bias is not None:
                        pytorch_layer.bias.data = torch.from_numpy(weights[1]).float()
                    logger.info(f"  ✓ Copied {keras_layer_name} -> {pytorch_layer}")
                    return True
            return False
        
        def copy_bn_weights(keras_layer_name, pytorch_layer):
            """Copy batch normalization layer weights"""
            if keras_layer_name in keras_layers:
                keras_layer = keras_layers[keras_layer_name]
                weights = keras_layer.get_weights()
                if len(weights) >= 4:
                    # Keras BN: [gamma, beta, moving_mean, moving_var]
                    # PyTorch BN: weight, bias, running_mean, running_var
                    pytorch_layer.weight.data = torch.from_numpy(weights[0]).float()
                    pytorch_layer.bias.data = torch.from_numpy(weights[1]).float()
                    pytorch_layer.running_mean.data = torch.from_numpy(weights[2]).float()
                    pytorch_layer.running_var.data = torch.from_numpy(weights[3]).float()
                    logger.info(f"  ✓ Copied {keras_layer_name} -> {pytorch_layer}")
                    return True
            return False
        
        def copy_fc_weights(keras_layer_name, pytorch_layer):
            """Copy fully connected layer weights"""
            if keras_layer_name in keras_layers:
                keras_layer = keras_layers[keras_layer_name]
                weights = keras_layer.get_weights()
                if len(weights) >= 1:
                    # Keras: (in_features, out_features)
                    # PyTorch: (out_features, in_features)
                    kernel = np.transpose(weights[0], (1, 0))
                    pytorch_layer.weight.data = torch.from_numpy(kernel).float()
                    if len(weights) > 1 and pytorch_layer.bias is not None:
                        pytorch_layer.bias.data = torch.from_numpy(weights[1]).float()
                    logger.info(f"  ✓ Copied {keras_layer_name} -> {pytorch_layer}")
                    return True
            return False
        
        # Try different naming conventions
        conv_names = [
            ['conv2d', 'conv2d_1', 'conv2d_2', 'conv2d_3'],  # Standard Keras naming
            ['conv1', 'conv2', 'conv3', 'conv4'],  # Alternative naming
        ]
        
        bn_names = [
            ['batch_normalization', 'batch_normalization_1', 'batch_normalization_2', 'batch_normalization_3'],
            ['bn1', 'bn2', 'bn3', 'bn4'],
        ]
        
        fc_names = [
            ['dense', 'dense_1'],  # Standard Keras naming
            ['fc1', 'fc2'],  # Alternative naming
        ]
        
        # Try to copy conv layers
        logger.info("Converting convolutional layers...")
        pytorch_convs = [pytorch_model.conv1, pytorch_model.conv2, pytorch_model.conv3, pytorch_model.conv4]
        for i, pytorch_conv in enumerate(pytorch_convs):
            copied = False
            for name_set in conv_names:
                if i < len(name_set):
                    if copy_conv_weights(name_set[i], pytorch_conv):
                        copied = True
                        break
            if not copied:
                logger.warning(f"  ⚠ Could not find Keras layer for conv{i+1}")
        
        # Try to copy batch norm layers
        logger.info("Converting batch normalization layers...")
        pytorch_bns = [pytorch_model.bn1, pytorch_model.bn2, pytorch_model.bn3, pytorch_model.bn4]
        for i, pytorch_bn in enumerate(pytorch_bns):
            copied = False
            for name_set in bn_names:
                if i < len(name_set):
                    if copy_bn_weights(name_set[i], pytorch_bn):
                        copied = True
                        break
            if not copied:
                logger.warning(f"  ⚠ Could not find Keras layer for bn{i+1}")
        
        # Try to copy fully connected layers
        logger.info("Converting fully connected layers...")
        pytorch_fcs = [pytorch_model.fc1, pytorch_model.fc2]
        for i, pytorch_fc in enumerate(pytorch_fcs):
            copied = False
            for name_set in fc_names:
                if i < len(name_set):
                    if copy_fc_weights(name_set[i], pytorch_fc):
                        copied = True
                        break
            if not copied:
                logger.warning(f"  ⚠ Could not find Keras layer for fc{i+1}")
        
        logger.info("Weight conversion completed!")
    
    def _load_model(self, model_path: Optional[str] = None) -> nn.Module:
        """
        Load MesoNet model. If given a .h5 file, checks for a converted .pth version first.
        
        Args:
            model_path: Path to pretrained model file (.h5 or .pth)
            
        Returns:
            Loaded PyTorch model
        """
        try:
            model = MesoNet(num_classes=1)
            
            if model_path and os.path.exists(model_path):
                logger.info(f"Loading pretrained MesoNet model from: {model_path}")
                
                # Check file extension
                if model_path.endswith('.h5'):
                    # Check if converted PyTorch version exists
                    pth_path = model_path.replace('.h5', '_converted.pth')
                    
                    if os.path.exists(pth_path):
                        logger.info(f"Found converted PyTorch model: {pth_path}")
                        logger.info("Loading converted model (skipping conversion)")
                        state_dict = torch.load(pth_path, map_location='cpu')
                        model.load_state_dict(state_dict)
                        logger.info("✓ Converted MesoNet model loaded successfully")
                    else:
                        # Load Keras model and convert to PyTorch
                        logger.info("No converted model found. Converting from Keras .h5 file...")
                        try:
                            # Try to import tensorflow/keras
                            import tensorflow as tf
                            from tensorflow import keras
                            
                            # Load Keras model
                            keras_model = keras.models.load_model(model_path)
                            logger.info("Keras model loaded successfully")
                            
                            # Convert Keras weights to PyTorch
                            self._convert_keras_to_pytorch(keras_model, model)
                            logger.info("✓ Successfully converted Keras weights to PyTorch")
                            
                            # Save converted model for future use
                            logger.info(f"Saving converted model to: {pth_path}")
                            torch.save(model.state_dict(), pth_path)
                            logger.info("✓ Converted model saved successfully")
                            
                        except ImportError:
                            logger.warning("TensorFlow/Keras not installed. Cannot load .h5 file.")
                            logger.warning("Install with: pip install tensorflow")
                            logger.warning("Using randomly initialized weights")
                        except Exception as e:
                            logger.error(f"Error converting Keras model to PyTorch: {str(e)}")
                            logger.warning("Using randomly initialized weights")
                
                elif model_path.endswith('.pth') or model_path.endswith('.pt'):
                    # Load PyTorch model
                    logger.info("Loading PyTorch model weights")
                    state_dict = torch.load(model_path, map_location='cpu')
                    model.load_state_dict(state_dict)
                    logger.info("✓ MesoNet model loaded successfully")
                
                else:
                    logger.warning(f"Unsupported model file format: {model_path}")
                    logger.warning("Supported formats: .h5 (Keras), .pth/.pt (PyTorch)")
                    logger.warning("Using randomly initialized MesoNet model.")
            
            else:
                if model_path:
                    logger.warning(f"Model file not found: {model_path}")
                else:
                    logger.info("No model path provided")
                logger.info("Using randomly initialized MesoNet model")
                logger.info("Download the pretrained model from:")
                logger.info("https://github.com/muffin-123/DeepFake-Detection-mesonet_model.h5-")
            
            # Move model to device and set to eval mode
            model = model.to(self.device)
            model.eval()
            
            # Log model info
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"MesoNet model loaded: {total_params:,} parameters")
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading MesoNet model: {e}")
            raise
    
    def _preprocess_image(self, image: Union[Image.Image, np.ndarray, str, BinaryIO]) -> Image.Image:
        """
        Preprocess image for inference
        
        Args:
            image: Input image (PIL Image, numpy array, file path, or binary stream)
            
        Returns:
            Preprocessed PIL Image
        """
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            pil_image = Image.open(image)
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, bytes):
            pil_image = Image.open(io.BytesIO(image))
        elif hasattr(image, 'read'):
            pil_image = Image.open(image)
        else:
            raise ValueError(f"Unsupported image input type: {type(image)}")
        
        # Convert to RGB
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        return pil_image
    
    def detect_single(self, 
                     image: Union[Image.Image, np.ndarray, str, BinaryIO],
                     threshold: Optional[float] = None) -> MesoNetDetectionResult:
        """
        Detect deepfake in a single image
        
        Args:
            image: Input image
            threshold: Optional threshold override (uses default if None)
            
        Returns:
            MesoNetDetectionResult with detection results
        """
        start_time = time.time()
        
        if threshold is None:
            threshold = self.threshold
        
        try:
            # Preprocess image
            pil_image = self._preprocess_image(image)
            image_path = image if isinstance(image, str) else None
            
            # Transform image
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                output = self.model(input_tensor)
                # Apply sigmoid for binary classification
                fake_prob = torch.sigmoid(output).item()
                real_prob = 1.0 - fake_prob
            
            # Determine if fake
            is_fake = fake_prob >= threshold
            confidence = fake_prob if is_fake else real_prob
            
            processing_time = time.time() - start_time
            
            return MesoNetDetectionResult(
                is_fake=is_fake,
                fake_probability=fake_prob,
                real_probability=real_prob,
                confidence=confidence,
                processing_time=processing_time,
                image_path=image_path,
                model_name=self.model_name
            )
            
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            raise
    
    def detect_batch(self, 
                    images: List[Union[Image.Image, np.ndarray, str, BinaryIO]],
                    threshold: Optional[float] = None) -> List[MesoNetDetectionResult]:
        """
        Detect deepfakes in a batch of images
        
        Args:
            images: List of input images
            threshold: Optional threshold override
            
        Returns:
            List of MesoNetDetectionResult objects
        """
        results = []
        
        logger.info(f"Processing batch of {len(images)} images")
        
        for i, image in enumerate(images):
            try:
                logger.info(f"Processing image {i+1}/{len(images)}")
                result = self.detect_single(image, threshold)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing image {i+1}: {e}")
                # Add error result
                results.append(MesoNetDetectionResult(
                    is_fake=False,
                    fake_probability=0.0,
                    real_probability=0.0,
                    confidence=0.0,
                    processing_time=0.0,
                    image_path=image if isinstance(image, str) else None,
                    model_name=self.model_name
                ))
        
        return results
    
    def analyze_directory(self,
                         directory_path: str,
                         extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.avif', '.webp'],
                         threshold: Optional[float] = None) -> List[MesoNetDetectionResult]:
        """
        Analyze all images in a directory
        
        Args:
            directory_path: Path to directory containing images
            extensions: List of image file extensions to process
            threshold: Optional threshold override
            
        Returns:
            List of MesoNetDetectionResult objects
        """
        directory = Path(directory_path)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Find all image files
        image_files = []
        for ext in extensions:
            image_files.extend(directory.glob(f"*{ext}"))
            image_files.extend(directory.glob(f"*{ext.upper()}"))
        
        logger.info(f"Found {len(image_files)} image files in {directory_path}")
        
        # Convert to strings
        image_paths = [str(img) for img in image_files]
        
        return self.detect_batch(image_paths, threshold)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'input_size': self.input_size,
            'threshold': self.threshold,
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'model_size_mb': sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024 / 1024
        }


# Factory function for easy initialization
def create_mesonet_detector(model_path: Optional[str] = None,
                            device: Optional[str] = None,
                            threshold: float = 0.5,
                            input_size: int = 256) -> MesoNetDetector:
    """
    Factory function to create MesoNet detector
    
    Args:
        model_path: Path to pretrained model
        device: Device to use
        threshold: Detection threshold
        input_size: Input image size
        
    Returns:
        Initialized MesoNetDetector
    """
    return MesoNetDetector(
        model_path=model_path,
        device=device,
        threshold=threshold,
        input_size=input_size
    )
