"""
CREDIT: Adapted from the Nonescape repository by Aedili Labs (https://github.com/aediliclabs/nonescape)
Nonescape AI Image Detection Model

This module implements the Nonescape classifier for detecting AI-generated images.
Based on the nonescape library: https://github.com/aediliclabs/nonescape

The Nonescape models use:
- Full Model: Vision Transformer (DINOv2) + EfficientNet v2 with attention mechanism
- Mini Model: EfficientNet v2 (faster, lighter inference)

Both models are trained to detect AI-generated images with high accuracy.
"""

import torch
import torch.nn as nn
from torch import Tensor
import torchvision.models as models
import torchvision.transforms.v2 as T
from PIL import Image
from typing import Union, BinaryIO, Optional
import logging
import warnings
import time
import os
from dataclasses import dataclass

try:
    from transformers import Dinov2Model, Dinov2Config
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers library not available. Full NonescapeClassifier will not work.")

try:
    from safetensors.torch import load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    logging.warning("safetensors library not available. Cannot load .safetensors model files.")

warnings.filterwarnings("ignore", message=".*deprecated.*")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NonescapeDetectionResult:
    """Result of Nonescape AI image detection"""
    is_ai_generated: bool
    confidence: float
    authentic_prob: float
    ai_prob: float
    processing_time: float
    model_name: str
    image_path: Optional[str] = None


def preprocess_image(image: Image.Image) -> Tensor:
    """
    Preprocess image for Nonescape models.
    
    Args:
        image: PIL Image
        
    Returns:
        Preprocessed tensor ready for model input
    """
    transform = T.Compose([
        T.ToImage(),
        T.Resize(256),
        T.CenterCrop(224),
        T.JPEG(quality=100),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image)


class NonescapeClassifier(nn.Module):
    """
    ViT/EfficientNet-based fake image detector
    
    Uses a vision transformer (DINOv2) as backbone with EfficientNet v2 to compute
    attention maps for the ViT features.
    """
    
    def __init__(self, num_classes: int = 2, num_heads: int = 16, num_queries: int = 128):
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required for full NonescapeClassifier. Install with: pip install transformers")
        
        self.embedding_size = 1024
        self.num_queries = num_queries
        
        vit_backbone = Dinov2Model(Dinov2Config.from_pretrained("facebook/dinov2-large"))
        efficientnet = models.efficientnet_v2_l(weights=None, num_classes=num_queries * self.embedding_size)
        self.vit_backbone = vit_backbone
        self.query_net = efficientnet
        self.key_net = nn.Linear(self.embedding_size, self.embedding_size)
        self.value_net = nn.Linear(self.embedding_size, self.embedding_size)
        self.attention = nn.MultiheadAttention(self.embedding_size, num_heads=num_heads, batch_first=True)
        self.head = nn.Linear(self.embedding_size, num_classes)
        
        self.register_buffer("_input_mean", torch.empty((3, 1, 1)))
        self.register_buffer("_input_std", torch.empty((3, 1, 1)))
    
    @classmethod
    def from_pretrained(cls, path: str) -> 'NonescapeClassifier':
        """Load model from .safetensors file"""
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("safetensors library required. Install with: pip install safetensors")
        
        state_dict = load_file(path)
        model = cls()
        model.load_state_dict(state_dict)
        return model
    
    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        
        with torch.no_grad():
            vit_output = self.vit_backbone(x)
            vit_features = vit_output.last_hidden_state
        
        q = self.query_net.forward(x).reshape(B, self.num_queries, -1)
        k = self.key_net.forward(vit_features)
        v = self.value_net.forward(vit_features)
        
        emb, _ = self.attention.forward(q, k, v)
        emb = emb.mean(dim=1)
        logits = self.head(emb.squeeze(1))
        probs = nn.functional.softmax(logits, dim=-1)
        
        return probs


class NonescapeClassifierMini(nn.Module):
    """
    EfficientNet-based fake image detector
    
    Lightweight version using only EfficientNet v2 for faster inference.
    """
    
    def __init__(self, num_classes: int = 2, embedding_size: int = 1024, dropout: float = 0.2):
        super().__init__()
        
        self.backbone = models.efficientnet_v2_s(weights=None, num_classes=embedding_size, dropout=dropout)
        self.head = nn.Linear(embedding_size, num_classes)
    
    @classmethod
    def from_pretrained(cls, path: str) -> 'NonescapeClassifierMini':
        """Load model from .safetensors file"""
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("safetensors library required. Install with: pip install safetensors")
        
        state_dict = load_file(path)
        model = cls()
        model.load_state_dict(state_dict)
        return model
    
    def forward(self, x: Tensor) -> Tensor:
        emb = self.backbone(x)
        logits = self.head(emb)
        probs = nn.functional.softmax(logits, dim=-1)
        return probs


class NonescapeDetector:
    """
    Wrapper class for Nonescape AI image detection
    
    Supports both full and mini models:
    - Full: Higher accuracy, uses ViT + EfficientNet
    - Mini: Faster inference, uses only EfficientNet
    """
    
    def __init__(self, 
                 model_path: str,
                 use_mini: bool = True,
                 device: Optional[str] = None,
                 debug: bool = False):
        """
        Initialize Nonescape detector
        
        Args:
            model_path: Path to .safetensors model file
            use_mini: If True, use mini model (faster). If False, use full model (more accurate)
            device: Device to run inference on ('cpu', 'cuda', or None for auto-detection)
            debug: If True, enable verbose logging
        """
        self.model_path = model_path
        self.use_mini = use_mini
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.debug = debug
        
        if self.debug:
            logger.info(f"Initializing Nonescape{'Mini' if use_mini else ''} detector on device: {self.device}")
            logger.info(f"Model path: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "Download models from:\n"
                "- Full: https://nonescape.sfo2.cdn.digitaloceanspaces.com/nonescape-v0.safetensors\n"
                "- Mini: https://nonescape.sfo2.cdn.digitaloceanspaces.com/nonescape-mini-v0.safetensors"
            )
        
        self._load_model()
    
    def _load_model(self):
        """Load the Nonescape model"""
        try:
            if self.use_mini:
                self.model = NonescapeClassifierMini.from_pretrained(self.model_path)
                self.model_name = "Nonescape Mini"
            else:
                self.model = NonescapeClassifier.from_pretrained(self.model_path)
                self.model_name = "Nonescape Full"
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            if self.debug:
                logger.info(f"{self.model_name} model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Nonescape model: {e}")
            raise
    
    def preprocess_image(self, image_input: Union[str, Image.Image]) -> Image.Image:
        """Convert various image inputs to PIL Image"""
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image file not found: {image_input}")
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            image = image_input.convert('RGB')
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
        
        return image
    
    def detect(self, 
               image_input: Union[str, Image.Image, BinaryIO],
               threshold: float = 0.5) -> NonescapeDetectionResult:
        """
        Detect if image is AI-generated
        
        Args:
            image_input: Image path, PIL Image, or file-like object
            threshold: Threshold for classification (default: 0.5)
            
        Returns:
            NonescapeDetectionResult with detection results
        """
        start_time = time.time()
        
        # Preprocess image
        image = self.preprocess_image(image_input)
        image_path = image_input if isinstance(image_input, str) else None
        
        # Transform and prepare input
        tensor = preprocess_image(image).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            probs = self.model(tensor)
            authentic_prob = float(probs[0][0].cpu())
            ai_prob = float(probs[0][1].cpu())
        
        processing_time = time.time() - start_time
        
        # Classification based on threshold
        is_ai_generated = ai_prob > threshold
        confidence = max(authentic_prob, ai_prob)
        
        return NonescapeDetectionResult(
            is_ai_generated=is_ai_generated,
            confidence=confidence,
            authentic_prob=authentic_prob,
            ai_prob=ai_prob,
            processing_time=processing_time,
            model_name=self.model_name,
            image_path=image_path
        )
    
    def get_model_info(self):
        """Get information about the loaded model"""
        num_params = sum(p.numel() for p in self.model.parameters())
        model_size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024 / 1024
        
        return {
            'model_name': self.model_name,
            'model_type': 'Mini' if self.use_mini else 'Full',
            'device': str(self.device),
            'model_path': self.model_path,
            'num_parameters': num_params,
            'model_size_mb': model_size_mb
        }


if __name__ == "__main__":
    print("Nonescape AI Image Detection Module")
    print("=" * 50)
    print("\nTo use this detector, you need to download a model file:")
    print("\nMini Model (faster, recommended for most use cases):")
    print("  wget https://nonescape.sfo2.cdn.digitaloceanspaces.com/nonescape-mini-v0.safetensors")
    print("\nFull Model (higher accuracy, slower):")
    print("  wget https://nonescape.sfo2.cdn.digitaloceanspaces.com/nonescape-v0.safetensors")
    print("\nExample usage:")
    print("  detector = NonescapeDetector('nonescape-mini-v0.safetensors', use_mini=True)")
    print("  result = detector.detect('image.jpg')")
    print("  print(f'AI Generated: {result.is_ai_generated}, Confidence: {result.confidence:.2f}')")
