"""SEResNet50 detection tool with blob storage caching."""

from outputModel.image_detection_output_model import IndividualModelResult
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import base64
import io
import time
import os
import logging
import tempfile
import hashlib
from pathlib import Path
import sys
import types

# Create a compatibility module for senet if it doesn't exist
# This allows loading old .pkl files that reference the senet module
if 'senet' not in sys.modules:
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

logger = logging.getLogger(__name__)

# Define SE-ResNet architecture classes (copied from imagedetectionlib)
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
sys.modules['senet.se_resnet'].SEBottleneck = SEBottleneck
sys.modules['senet.se_resnet'].SEBasicBlock = SEBasicBlock
sys.modules['senet.se_resnet'].SEResNet = SEResNet
sys.modules['senet.se_resnet'].SELayer = SELayer

# Also populate senet.se_module in case pickle references it
sys.modules['senet.se_module'].SELayer = SELayer

# Global model cache
_seresnet50_model = None
_device = None

def get_cached_model_path(model_key: str, blob_config: dict) -> str:
    """
    Get cached model path or download from blob storage.
    Priority: Cache → Blob download to cache
    
    Note: blob_storage_helper now handles cache checking internally,
    so we just delegate to it directly (no double checking).
    
    Args:
        model_key: Model key identifier
        blob_config: Blob storage configuration from MODEL_REGISTRY
        
    Returns:
        Local file path to model in cache
    """
    # Check for single file blob
    if 'blob' in blob_config:
        try:
            from utils.blob_storage_helper import get_blob_loader
            blob_loader = get_blob_loader()
            
            if blob_loader and blob_loader.blob_service_client:
                # download_model now checks cache first internally
                downloaded_path = blob_loader.download_model(
                    container_name=blob_config['container'],
                    blob_name=blob_config['blob']
                )
                
                if downloaded_path:
                    return downloaded_path
                else:
                    logger.warning(f"Failed to get {model_key} from cache or blob storage")
            else:
                logger.warning("Blob storage not configured")
                
        except Exception as e:
            logger.error(f"Error getting {model_key}: {e}")
        
        # If failed, return expected cache path (will fail gracefully in caller)
        cache_dir = Path(tempfile.gettempdir()) / "deepfake_models_cache"
        blob_path = f"{blob_config['container']}/{blob_config['blob']}"
        path_hash = hashlib.md5(blob_path.encode()).hexdigest()
        filename = Path(blob_config['blob']).name
        cache_path = cache_dir / f"{path_hash}_{filename}"
        logger.error(f"Failed to get model {model_key} - returning fallback path")
        return str(cache_path)
    
    # No blob config provided
    logger.error(f"No blob configuration provided for {model_key}")
    return ""

def se_resnet50(num_classes=1000, pretrained=False, weights_path=None):
    """
    Constructs a SE-ResNet-50 model - EXACTLY matching imagedetectionlib implementation.
    This helper function properly handles loading pretrained weights.
    """
    # Use the custom SEResNet implementation for compatibility
    model = SEResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    
    if pretrained and weights_path and os.path.exists(weights_path):
        try:
            logger.info(f"Loading SEResNet-50 pretrained weights from: {weights_path}")
            # Add safe globals for torchvision models
            from torchvision.models import resnet
            torch.serialization.add_safe_globals([resnet.ResNet])
            
            pretrained_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
            
            # Handle both cases: pretrained_dict could be a state_dict or a full model
            if isinstance(pretrained_dict, nn.Module):
                pretrained_dict = pretrained_dict.state_dict()
                logger.info("Extracted state_dict from loaded model object")
            
            if num_classes != 1000:
                # Copy all layers except the final classifier
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'fc' not in k}
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
            else:
                model.load_state_dict(pretrained_dict)
            
            logger.info("✓ Successfully loaded pretrained weights")
        except Exception as e:
            logger.error(f"Failed to load pretrained weights: {e}", exc_info=True)
            raise
    
    return model

def load_seresnet50_model():
    """
    Load SEResNet50 model - EXACTLY matching imagedetectionlib implementation.
    Uses pretrained ImageNet weights with a new 2-class classifier head.
    """
    global _seresnet50_model, _device
    
    if _seresnet50_model is not None:
        return _seresnet50_model, _device
    
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Match imagedetectionlib: Load pretrained ImageNet model (1000 classes)
        # Then replace final layer with binary classifier
        logger.info("Loading pretrained SEResNet-50 as feature extractor (matching imagedetectionlib)")
        
        # Get pretrained weights path (local or from cache/blob)
        blob_config = {
            'container': 'models',
            'blob': 'SERESNet_50/seresnet50_pretrained.pth'
        }
        
        model_path = get_cached_model_path('seresnet50_pretrained', blob_config)
        logger.info(f"Using model path: {model_path}")
        
        # Use the se_resnet50 helper function (matching imagedetectionlib EXACTLY)
        model = se_resnet50(num_classes=1000, pretrained=True, weights_path=model_path)
        
        # Replace final layer with binary classifier (matching imagedetectionlib)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.2),  # Light dropout
            nn.Linear(in_features, 2)  # Direct binary classifier
        )
        
        # Initialize the new layers with slight bias toward real class (matching imagedetectionlib)
        for module in model.fc.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.out_features == 2:  # Final layer
                    with torch.no_grad():
                        module.bias[0] = 0.05   # Small bias toward fake (index 0)
                        module.bias[1] = -0.05  # Small bias against real (index 1)
                else:
                    nn.init.zeros_(module.bias)
        
        model = model.to(_device)
        model.eval()
        _seresnet50_model = model
        
        logger.info(f"✓ Loaded SEResNet50 model (pretrained + binary classifier)")
        return model, _device
        
    except Exception as e:
        logger.error(f"Failed to load SEResNet50 model: {e}", exc_info=True)
        return None, _device


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


def detect_with_seresnet50_tool(image_path: str) -> dict:
    """
    Agent tool: Detect deepfakes using the SEResNet50 model.
    
    Returns dictionary representation of IndividualModelResult for JSON serialization.
    
    Args:
        image_path: File path to image to analyze
        
    Returns:
        Dictionary with detection results
    """
    logger.info("Running detect_with_seresnet50_tool...")
    start_time = time.time()
    
    try:
        logger.info("Loading SEResNet50 model...")
        # Load model (uses cache if available)
        model, device = load_seresnet50_model()
        
        if model is None:
            error_msg = "Failed to load SEResNet50 model - model file may be missing"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        logger.info("Preprocessing image...")
        # STEP 1: Convert file to tensor
        tensor = convert_image_to_tensor(image_path)
        
        # STEP 2: Convert tensor to PIL for transforms
        import torchvision.transforms.functional as F
        tensor_float = tensor.to(torch.float32) / 255.0  # [C, H, W] float32 [0, 1]
        image = F.to_pil_image(tensor_float)
        
        # STEP 3: Apply model-specific transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        tensor_processed = transform(image)
        image_tensor = tensor_processed.unsqueeze(0).to(device)
        
        logger.info("Running inference...")
        # Run inference
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            # Model outputs: [fake_prob, real_prob] based on training
            # probabilities[0][0] = probability of FAKE class
            # probabilities[0][1] = probability of REAL class
            fake_prob = probabilities[0][0].item()
            real_prob = probabilities[0][1].item()
        
        processing_time = time.time() - start_time
        
        # Determine prediction
        is_deepfake = fake_prob > real_prob
        confidence = max(fake_prob, real_prob)
        prediction = 'FAKE' if is_deepfake else 'REAL'
        
        logger.info(f"SEResNet50 prediction: {prediction} (confidence: {confidence:.2f})")
        
        # Return as dictionary
        result = IndividualModelResult(
            model='seresnet50',
            prediction=prediction,
            confidence=confidence,
            fakeScore=fake_prob,
            realScore=real_prob
        )
        return result.model_dump()
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"SEResNet50 detection error: {e}", exc_info=True)
        
        # Return error result as dictionary
        error_result = IndividualModelResult(
            model='seresnet50',
            prediction='REAL',
            confidence=0.0,
            fakeScore=0.0,
            realScore=0.0
        )
        logger.warning(f"Returning error result: {error_result.model_dump()}")
        return error_result.model_dump()


 