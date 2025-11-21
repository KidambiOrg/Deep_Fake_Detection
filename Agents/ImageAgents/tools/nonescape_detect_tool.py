"""NoEscape detection tool with blob storage caching."""

from outputModel.image_detection_output_model import IndividualModelResult
import torch
import torch.nn as nn
from safetensors.torch import load_file
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import base64
import io
import time
import os
import logging
import tempfile
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)

# Import transformers for DINOv2
try:
    from transformers import Dinov2Model, Dinov2Config
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers library not available. Full NonescapeClassifier will not work.")


class NonescapeClassifier(nn.Module):
    """
    ViT/EfficientNet-based fake image detector
    
    Uses a vision transformer (DINOv2) as backbone with EfficientNet v2 to compute
    attention maps for the ViT features.
    
    CREDIT: Adapted from the Nonescape repository by Aedili Labs (https://github.com/aediliclabs/nonescape)
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

# Global model cache
_nonescape_model = None
_device = None

def load_nonescape_model():
    """
    Load NoEscape model from local Models folder.
    """
    global _nonescape_model, _device
    
    if _nonescape_model is not None:
        return _nonescape_model, _device
    
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NonescapeClassifier()
    
    # Load from local Models folder only
    script_dir = Path(__file__).parent.parent.parent.parent
    model_path = script_dir / "Models" / "nonescape-v0.safetensors"
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"NoEscape model not found at: {model_path}\n"
            f"Please ensure the model file exists in Models/ folder.\n"
            f"Expected file: nonescape-v0.safetensors"
        )
    
    model_path = str(model_path)
    
    # Load model weights
    try:
        state_dict = load_file(model_path)
        logger.info(f"Loading NoEscape model from: {model_path}")
        logger.info(f"State dict has {len(state_dict)} keys, first 5: {list(state_dict.keys())[:5]}")
        
        model.load_state_dict(state_dict)
        logger.info("Successfully loaded NoEscape model weights")
    except Exception as e:
        logger.error(f"Failed to load model weights: {e}")
        raise
    
    model = model.to(_device)
    model.eval()
    
    _nonescape_model = model
    return model, _device


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


def detect_with_nonescape_tool(image_path: str) -> dict:
    """
    Agent tool: Detect deepfakes using the NoEscape model.
    
    Returns dictionary representation of IndividualModelResult for JSON serialization.
    
    Args:
        image_path: File path to image to analyze
        
    Returns:
        Dictionary with detection results (model, prediction, confidence, fakeScore, realScore)
    """
    logger.info("Running detect_with_nonescape_tool...")
    
    start_time = time.time()
    
    try:
        # BUG FIX: Azure Agent Framework sometimes duplicates the path on retries
        from utils.path_fix_helper import fix_duplicated_path
        image_path = fix_duplicated_path(image_path)
        logger.info("Loading NoEscape model...")
        # Load model (uses cache if available)
        model, device = load_nonescape_model()
        
        logger.info("Preprocessing image...")
        # STEP 1: Convert file to tensor
        tensor = convert_image_to_tensor(image_path)
        
        # STEP 2: Convert tensor to PIL for transforms
        import torchvision.transforms.functional as F
        tensor_float = tensor.to(torch.float32) / 255.0  # [C, H, W] float32 [0, 1]
        image = F.to_pil_image(tensor_float)
        
        # STEP 3: Apply model-specific transforms (Resize 256 -> CenterCrop 224 -> JPEG compression)
        from torchvision.transforms import v2 as T
        transform = T.Compose([
            T.ToImage(),
            T.Resize(256),
            T.CenterCrop(224),
            T.JPEG(quality=100),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        image_tensor = transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(device)
        
        logger.info("Running inference...")
        # Run inference (matching imagedetectionlib EXACTLY)
        # Index 0 = authentic (real), Index 1 = AI-generated (fake)
        with torch.no_grad():
            probs = model(image_tensor)
            real_prob = float(probs[0][0].item())  # Index 0 = authentic/real
            fake_prob = float(probs[0][1].item())  # Index 1 = AI-generated/fake
        
        processing_time = time.time() - start_time
        
        # Determine prediction based on which score is higher
        is_deepfake = fake_prob > real_prob
        confidence = max(fake_prob, real_prob)
        prediction = 'FAKE' if is_deepfake else 'REAL'
        
        # Return as dictionary (JSON serializable for Agent Framework)
        result = IndividualModelResult(
            model='nonescape-full',
            prediction=prediction,
            confidence=confidence,
            fakeScore=fake_prob,
            realScore=real_prob
        )
        return result.model_dump()
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"NoEscape detection error: {e}")
        
        # Return error result as dictionary
        error_result = IndividualModelResult(
            model='nonescape-full',
            prediction='REAL',
            confidence=0.0,
            fakeScore=0.0,
            realScore=0.0
        )
        return error_result.model_dump()


 