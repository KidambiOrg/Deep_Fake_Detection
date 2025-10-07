"""
DistilDIRE Models

This module contains the DistilDIRE model architectures for deepfake detection.
Based on: https://github.com/miraflow/DistilDIRE

DistilDIRE is a lightweight version of DIRE that reconstructs features of the
DIRE image using one-step noise from DDIM inversion.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict
from typing import Dict, Any, Optional


def convert_module_to_f16(module):
    """Convert a module to use float16 precision"""
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        module.half()
        if module.bias is not None:
            module.bias.data = module.bias.data.half()


class DistilDIRE(nn.Module):
    """
    DistilDIRE model for deepfake detection
    
    Uses 6-channel input (image + epsilon) to match the trained model checkpoint.
    """
    
    def __init__(self, device: str = 'cuda'):
        super(DistilDIRE, self).__init__()
        
        # Use ResNet50 as the backbone with 6-channel input modification
        student = models.resnet50(weights=None)
        
        # Modify the first convolutional layer to accept 6 channels (img + eps)
        original_conv1 = student.conv1
        student.conv1 = nn.Conv2d(
            6,  # 6 input channels (3 for image + 3 for epsilon)
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None
        )
        
        # Remove the last two layers (avgpool and classifier)
        self.student_backbone = nn.Sequential(OrderedDict([
            *(list(student.named_children())[:-2])
        ]))
        
        # Create a new classifier head
        self.student_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 1)
        )
        
        self.device = device
    
    def convert_to_fp16_student(self):
        """Convert the student model to float16 for efficiency"""
        self.student_backbone.apply(convert_module_to_f16)
    
    def forward(self, img: torch.Tensor, eps: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of DistilDIRE (6-channel version)
        
        Args:
            img: Input image tensor (B, 3, H, W)
            eps: Epsilon tensor from DDIM inversion (B, 3, H, W)
            
        Returns:
            Dictionary containing 'logit' and 'feature'
        """
        img = img.to(self.device)
        eps = eps.to(self.device)
        
        # Concatenate image and epsilon along channel dimension for 6-channel input
        img_tens = torch.cat([img, eps], dim=1)  # (B, 6, H, W)
        
        # Extract features using the backbone (6-channel input)
        feature = self.student_backbone(img_tens)
        
        # Get the final logit
        logit = self.student_head(feature)
        
        return {'logit': logit, 'feature': feature}


class DistilDIREOnlyEPS(DistilDIRE):
    """
    DistilDIRE variant that only uses epsilon (noise) for detection
    
    This is a faster variant that doesn't require the original image.
    """
    
    def __init__(self, device: str = 'cuda'):
        super(DistilDIREOnlyEPS, self).__init__(device)
        
        # Modify the first conv layer to accept only 3 channels (eps only)
        self.student_backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Recreate the classifier head
        self.student_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 1)
        )
    
    def forward(self, eps: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass using only epsilon
        
        Args:
            eps: Epsilon tensor from DDIM inversion (B, 3, H, W)
            
        Returns:
            Dictionary containing 'logit' and 'feature'
        """
        eps = eps.to(self.device)
        
        # Extract features using only epsilon
        feature = self.student_backbone(eps)
        
        # Get the final logit
        logit = self.student_head(feature)
        
        return {'logit': logit, 'feature': feature}


class DistilDIREOnlyIMG(DistilDIRE):
    """
    DistilDIRE variant that only uses the original image for detection
    
    This variant computes epsilon on-the-fly using a diffusion model.
    """
    
    def __init__(self, device: str = 'cuda'):
        super(DistilDIREOnlyIMG, self).__init__(device)
        
        # Keep the 6-channel input as we'll compute eps on-the-fly
        # The eps computation will be handled externally
    
    def forward(self, img: torch.Tensor, eps: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass using image (eps computed externally if not provided)
        
        Args:
            img: Input image tensor (B, 3, H, W)
            eps: Optional epsilon tensor (B, 3, H, W)
            
        Returns:
            Dictionary containing 'logit' and 'feature'
        """
        img = img.to(self.device)
        
        if eps is None:
            # If eps is not provided, create zeros (should be computed externally)
            eps = torch.zeros_like(img)
        
        eps = eps.to(self.device)
        
        # Concatenate image and noise
        img_tens = torch.cat([img, eps], dim=1)
        
        # Extract features
        feature = self.student_backbone(img_tens)
        
        # Get the final logit
        logit = self.student_head(feature)
        
        return {'logit': logit, 'feature': feature}


class DIRE(nn.Module):
    """
    Original DIRE model that works directly with DIRE images
    
    This is included for compatibility and comparison purposes.
    """
    
    def __init__(self, device: str = 'cuda'):
        super(DIRE, self).__init__()
        
        # Use ResNet50 as the backbone
        student = models.resnet50(weights=None)
        
        # Remove the last two layers
        self.student_backbone = nn.Sequential(OrderedDict([
            *(list(student.named_children())[:-2])
        ]))
        
        # Create classifier head
        self.student_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 1)
        )
        
        self.device = device
    
    def forward(self, dire: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass using DIRE image
        
        Args:
            dire: DIRE image tensor (B, 3, H, W)
            
        Returns:
            Dictionary containing 'logit' and 'feature'
        """
        dire = dire.to(self.device)
        
        # Extract features
        feature = self.student_backbone(dire)
        
        # Get the final logit
        logit = self.student_head(feature)
        
        return {'logit': logit, 'feature': feature}


def load_distil_dire_model(
    model_path: str, 
    model_type: str = 'distil_dire',
    device: str = 'cuda'
) -> DistilDIRE:
    """
    Load a pre-trained DistilDIRE model (always loads full model)
    
    Args:
        model_path: Path to the model checkpoint
        model_type: Type of model (ignored, always loads full DistilDIRE)
        device: Device to load the model on
        
    Returns:
        Loaded DistilDIRE model
    """
    # Always create the full DistilDIRE model
    model = DistilDIRE(device)
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict'] 
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present (from DataParallel)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        # Load the state dict
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"⚠️  Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"⚠️  Unexpected keys: {unexpected_keys}")
        
        model.eval()
        model.to(device)
        
        print(f"✅ Successfully loaded DistilDIRE model from {model_path}")
        return model
        
    except Exception as e:
        print(f"❌ Error loading DistilDIRE model from {model_path}: {e}")
        raise


def real_or_fake_threshold(probability: float, threshold: float = 0.5) -> str:
    """
    Convert probability to real/fake prediction
    
    Args:
        probability: Fake probability (0-1)
        threshold: Classification threshold
        
    Returns:
        "FAKE" or "REAL"
    """
    return "FAKE" if probability >= threshold else "REAL"