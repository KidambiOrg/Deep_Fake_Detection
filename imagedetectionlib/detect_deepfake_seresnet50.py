"""
SEResNet-50 Deepfake Detection Module

This module implements deepfake detection using SEResNet-50 (Squeeze-and-Excitation ResNet-50),
a state-of-the-art architecture that enhances traditional ResNet with channel attention mechanisms.

The model uses the pretrained SEResNet-50 from moskomule/senet.pytorch and fine-tunes it
for binary classification (real vs fake) in deepfake detection tasks.

Key Features:
- SEResNet-50 architecture with Squeeze-and-Excitation blocks
- Pretrained on ImageNet for robust feature extraction
- Channel attention mechanism for improved discrimination
- Binary classification for deepfake detection
- Support for various image preprocessing techniques
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
import sys

# Create a compatibility module for senet if it doesn't exist
# This allows loading old .pkl files that reference the senet module
if 'senet' not in sys.modules:
    import types
    senet_module = types.ModuleType('senet')
    senet_se_resnet_module = types.ModuleType('senet.se_resnet')
    senet_se_module_module = types.ModuleType('senet.se_module')
    
    # Register all modules in sys.modules
    sys.modules['senet'] = senet_module
    sys.modules['senet.se_resnet'] = senet_se_resnet_module
    sys.modules['senet.se_module'] = senet_se_module_module
    
    # Set up package structure - se_resnet and se_module are submodules of senet
    senet_module.se_resnet = senet_se_resnet_module
    senet_module.se_module = senet_se_module_module
    
    # Mark senet as a package
    senet_module.__package__ = 'senet'
    senet_module.__path__ = []
    
    # We'll add the actual classes after they're defined below
    # This forward reference will be filled in after the class definitions

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SELayer(nn.Module):
    """Squeeze-and-Excitation Layer"""
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class SEBasicBlock(nn.Module):
    """SE-ResNet Basic Block"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class SEBottleneck(nn.Module):
    """SE-ResNet Bottleneck Block"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class SEResNet(nn.Module):
    """SE-ResNet Architecture"""
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(SEResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, SEBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, SEBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# Populate the senet modules with the necessary classes for pickle compatibility
# This allows unpickling of .pkl weight files that reference senet module classes
sys.modules['senet.se_resnet'].SEBottleneck = SEBottleneck
sys.modules['senet.se_resnet'].SEBasicBlock = SEBasicBlock
sys.modules['senet.se_resnet'].SEResNet = SEResNet
sys.modules['senet.se_resnet'].SELayer = SELayer

# Also populate senet.se_module in case pickle references it
sys.modules['senet.se_module'].SELayer = SELayer

def se_resnet50(num_classes=1000, pretrained=False, weights_path=None):
    """Constructs a SE-ResNet-50 model - matches reference implementation"""
    # Use the custom SEResNet implementation for compatibility
    model = SEResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    
    if pretrained:
        try:
            # First try to load from local weights file if provided
            if weights_path and os.path.exists(weights_path):
                logger.info(f"Loading SEResNet-50 weights from local file: {weights_path}")
                # Add safe globals for torchvision models
                from torchvision.models import resnet
                torch.serialization.add_safe_globals([resnet.ResNet])
                pretrained_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
                if num_classes != 1000:
                    # Copy all layers except the final classifier
                    model_dict = model.state_dict()
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'fc' not in k}
                    model_dict.update(pretrained_dict)
                    model.load_state_dict(model_dict)
                else:
                    model.load_state_dict(pretrained_dict)
            else:
                # Load from default weights path in Models folder (no online download)
                script_dir = Path(__file__).parent.parent
                
                # Try .pth files first (safer, no senet module dependency)
                default_weights_path = script_dir / "Models" / "SERESNet_50" / "seresnet50_pretrained.pth"
                
                # Fallback to .pkl file if .pth not found
                if not default_weights_path.exists():
                    default_weights_path = script_dir / "Models" / "SERESNet_50" / "seresnet50-60a8950a85b2b.pkl"
                
                if not default_weights_path.exists():
                    raise FileNotFoundError(
                        f"SEResNet-50 pretrained weights not found at: {default_weights_path}\n"
                        f"Please ensure either 'seresnet50_pretrained.pth' or 'seresnet50-60a8950a85b2b.pkl' exists in the Models/SERESNet_50 folder.\n"
                        f"You can download the .pkl file from: https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl"
                    )
                
                logger.info(f"Loading SEResNet-50 weights from: {default_weights_path}")
                
                # Try loading with weights_only=True first (safer), fall back to False if needed
                try:
                    # Add safe globals for torchvision models to allow loading
                    from torchvision.models import resnet
                    torch.serialization.add_safe_globals([resnet.ResNet])
                    loaded_data = torch.load(default_weights_path, map_location='cpu', weights_only=True)
                    logger.info("[OK] Loaded weights successfully with weights_only=True")
                except Exception as e:
                    logger.warning(f"Failed to load with weights_only=True: {e}")
                    logger.info("Retrying with weights_only=False (you trust this file from Models folder)")
                    loaded_data = torch.load(default_weights_path, map_location='cpu', weights_only=False)
                    logger.info("[OK] Loaded weights successfully with weights_only=False")
                
                # Handle both cases: loaded_data could be a state_dict or a full model
                if isinstance(loaded_data, dict):
                    pretrained_dict = loaded_data
                elif isinstance(loaded_data, nn.Module):
                    # Extract state_dict from the model object
                    pretrained_dict = loaded_data.state_dict()
                    logger.info("[OK] Extracted state_dict from loaded model object")
                else:
                    raise TypeError(f"Expected state_dict (dict) or model (nn.Module), got {type(loaded_data)}")
                
                if num_classes != 1000:
                    # Copy all layers except the final classifier for binary classification
                    model_dict = model.state_dict()
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'fc' not in k}
                    model_dict.update(pretrained_dict)
                    model.load_state_dict(model_dict)
                    logger.info(f"[OK] Adapted model for {num_classes} classes (binary classification)")
                else:
                    model.load_state_dict(pretrained_dict)
                    logger.info(f"[OK] Loaded model with {num_classes} classes")
        except Exception as e:
            logger.warning(f"Could not load pretrained weights: {e}")
    
    return model

@dataclass
class DetectionResult:
    """Structure to hold detection results"""
    is_fake: bool
    confidence: float
    fake_probability: float
    real_probability: float
    model_name: str
    processing_time: float
    metadata: Optional[Dict[str, Any]] = None

class SEResNet50Detector:
    """
    SEResNet-50 based deepfake detector
    
    This detector uses the SEResNet-50 architecture with Squeeze-and-Excitation blocks
    for enhanced feature discrimination in deepfake detection.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 model_type: str = 'pretrained',  # 'pretrained' (1000 classes) or 'complete' (2 classes)
                 device: Optional[str] = None,
                 threshold: float = 0.5,
                 input_size: int = 224,
                 debug: bool = False):
        """
        Initialize SEResNet-50 detector
        
        Args:
            model_path: Path to pretrained model. If None, downloads from torch.hub
            model_type: Type of model weights - 'pretrained' (ImageNet, 1000 classes) or 'complete' (fine-tuned, 2 classes)
            device: Device to run inference on ('cpu' or 'cuda')
            threshold: Classification threshold (default: 0.5)
            input_size: Input image size (default: 224)
            debug: If True, enable verbose logging
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        self.input_size = input_size
        self.model_name = "SEResNet-50"
        self.model_type = model_type
        self.debug = debug
        
        # Initialize model
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Define augmented transforms for ensemble
        self.augmented_transforms = [
            transforms.Compose([
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((self.input_size + 32, self.input_size + 32)),
                transforms.CenterCrop((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((self.input_size, self.input_size)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        ]
        
        if self.debug:
            logger.info(f"SEResNet-50 detector initialized on {self.device}")
    
    def _load_model(self, model_path: Optional[str] = None) -> nn.Module:
        """Load SEResNet-50 model"""
        try:
            if model_path and os.path.exists(model_path) and self.model_type == 'complete':
                # Load complete fine-tuned model for deepfake detection (2 classes)
                if self.debug:
                    logger.info(f"Loading fine-tuned model from {model_path}")
                model = se_resnet50(num_classes=2, pretrained=False)
                # Add safe globals for torchvision models
                from torchvision.models import resnet
                torch.serialization.add_safe_globals([resnet.ResNet])
                state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
                model.load_state_dict(state_dict)
            else:
                # Load pretrained ImageNet model and use as feature extractor
                if self.debug:
                    logger.info("Loading pretrained SEResNet-50 as feature extractor")
                # Load pretrained model with ImageNet weights (1000 classes)
                # Pass model_path to se_resnet50 so it loads from the specified path
                model = se_resnet50(num_classes=1000, pretrained=True, weights_path=model_path)
                
                # Get the feature dimension
                in_features = model.fc.in_features
                
                # Replace with a simpler binary classifier that preserves discriminative ability
                model.fc = nn.Sequential(
                    nn.Dropout(0.2),  # Light dropout
                    nn.Linear(in_features, 2)  # Direct binary classifier
                )
                
                # Initialize the new layers with slight bias toward real class
                for module in model.fc.modules():
                    if isinstance(module, nn.Linear):
                        # Use Xavier initialization (good for linear layers)
                        nn.init.xavier_uniform_(module.weight)
                        # Add slight bias toward real class (conservative approach)
                        if module.out_features == 2:  # Final layer
                            with torch.no_grad():
                                module.bias[0] = 0.05   # Small bias toward real
                                module.bias[1] = -0.05  # Small bias against fake
                        else:
                            nn.init.zeros_(module.bias)
                
            return model
            
        except Exception as e:
            logger.error(f"Failed to load SEResNet-50 model: {e}")
            raise
    
    def _preprocess_image(self, image: Union[Image.Image, np.ndarray, str, BinaryIO]) -> Image.Image:
        """Preprocess image for inference"""
        if isinstance(image, str):
            # File path
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # NumPy array
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = Image.fromarray(image).convert('RGB')
            else:
                image = Image.fromarray(image, mode='RGB')
        elif hasattr(image, 'read'):
            # File-like object (BinaryIO)
            from io import BytesIO
            image_data = image.read()  # type: ignore
            if hasattr(image, 'seek'):
                image.seek(0)  # type: ignore
            pil_image = Image.open(BytesIO(image_data)).convert('RGB')
            return pil_image
        elif isinstance(image, Image.Image):
            return image.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # This should never be reached, but keeping for type safety
        return image.convert('RGB') if isinstance(image, Image.Image) else Image.new('RGB', (224, 224))
    
    def detect_single(self, image: Union[Image.Image, np.ndarray, str, BinaryIO],
                     use_ensemble: bool = False) -> DetectionResult:
        """
        Detect deepfake in a single image
        
        Args:
            image: Input image (PIL Image, numpy array, file path, or file-like object)
            use_ensemble: Whether to use ensemble of augmented predictions
            
        Returns:
            DetectionResult with prediction details
        """
        start_time = time.time()
        
        try:
            # Preprocess image
            pil_image = self._preprocess_image(image)
            
            with torch.no_grad():
                if use_ensemble:
                    predictions = []
                    for transform in self.augmented_transforms:
                        tensor_img = transform(pil_image)
                        if not isinstance(tensor_img, torch.Tensor):
                            raise ValueError(f"Transform should return tensor, got {type(tensor_img)}")
                        tensor = tensor_img.unsqueeze(0).to(self.device)
                        logits = self.model(tensor)
                        probs = F.softmax(logits, dim=1)
                        predictions.append(probs.cpu().numpy())
                    
                    # Average predictions
                    avg_probs = np.mean(predictions, axis=0)[0]
                else:
                    tensor_img = self.transform(pil_image)
                    if not isinstance(tensor_img, torch.Tensor):
                        raise ValueError(f"Transform should return tensor, got {type(tensor_img)}")
                    tensor = tensor_img.unsqueeze(0).to(self.device)
                    logits = self.model(tensor)
                    avg_probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                
                # Model outputs: [fake_prob, real_prob] based on training
                # avg_probs[0] = probability of FAKE class
                # avg_probs[1] = probability of REAL class
                fake_prob = float(avg_probs[0])
                real_prob = float(avg_probs[1])
                
                # For binary classification with complementary probabilities,
                # predict based on which probability is higher
                # This is more consistent than using a fixed threshold
                is_fake = fake_prob > real_prob
                confidence = max(real_prob, fake_prob)
                
                processing_time = time.time() - start_time
                
                return DetectionResult(
                    is_fake=is_fake,
                    confidence=confidence,
                    fake_probability=fake_prob,
                    real_probability=real_prob,
                    model_name=self.model_name,
                    processing_time=processing_time,
                    metadata={
                        'prediction_method': 'argmax',  # Predict based on higher probability
                        'ensemble': use_ensemble,
                        'input_size': self.input_size
                    }
                )
                
        except Exception as e:
            logger.error(f"Error in SEResNet-50 detection: {e}")
            processing_time = time.time() - start_time
            return DetectionResult(
                is_fake=False,
                confidence=0.0,
                fake_probability=0.0,
                real_probability=0.0,
                model_name=self.model_name,
                processing_time=processing_time,
                metadata={'error': str(e)}
            )
    
    def detect_batch(self, images: List[Union[Image.Image, np.ndarray, str, BinaryIO]],
                    batch_size: int = 8, use_ensemble: bool = False) -> List[DetectionResult]:
        """
        Detect deepfakes in a batch of images
        
        Args:
            images: List of input images
            batch_size: Batch size for processing
            use_ensemble: Whether to use ensemble predictions
            
        Returns:
            List of DetectionResults
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_results = []
            
            for image in batch:
                result = self.detect_single(image, use_ensemble=use_ensemble)
                batch_results.append(result)
            
            results.extend(batch_results)
            
            if (i // batch_size + 1) % 10 == 0:
                if self.debug:
                    logger.info(f"Processed {i + len(batch)}/{len(images)} images")
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'architecture': 'SEResNet-50',
            'input_size': f"{self.input_size}x{self.input_size}",
            'num_classes': 2,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': self.device,
            'threshold': self.threshold
        }

# Factory function for easy initialization
def create_seresnet50_detector(model_path: Optional[str] = None,
                              device: Optional[str] = None,
                              threshold: float = 0.5,
                              input_size: int = 224) -> SEResNet50Detector:
    """
    Factory function to create SEResNet-50 detector
    
    Args:
        model_path: Path to pretrained model
        device: Device to run on
        threshold: Classification threshold
        input_size: Input image size
        
    Returns:
        SEResNet50Detector instance
    """
    return SEResNet50Detector(
        model_path=model_path,
        device=device,
        threshold=threshold,
        input_size=input_size
    )

if __name__ == "__main__":
    # Example usage
    detector = create_seresnet50_detector()
    print("SEResNet-50 detector created successfully!")
    print(f"Model info: {detector.get_model_info()}")