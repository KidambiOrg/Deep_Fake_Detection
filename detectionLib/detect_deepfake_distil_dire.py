"""
DistilDIRE Deepfake Detection

This module implements deepfake detection using DistilDIRE models.
DistilDIRE is a lightweight version of DIRE that uses DDIM inversion
to compute epsilon (noise) and combines it with the original image.

Based on: https://github.com/miraflow/DistilDIRE
Paper: DistilDIRE: A Small, Fast, Cheap and Lightweight Diffusion Synthesized Deepfake Detection
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import logging
import time
from typing import Union, BinaryIO, Optional, Dict, Any
from dataclasses import dataclass
import os
import io
from pathlib import Path

# Import local modules
from .distil_dire_models import (
    DistilDIRE, DistilDIREOnlyEPS, DistilDIREOnlyIMG, DIRE,
    load_distil_dire_model, real_or_fake_threshold
)
from .guided_diffusion_utils import (
    create_gaussian_diffusion, ModelMeanType, ModelVarType, LossType
)

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class DistilDIREDetectionResult:
    """Result of DistilDIRE detection analysis"""
    is_deepfake: bool
    confidence: float
    fake_probability: float
    processing_time: float
    image_path: Optional[str] = None
    model_name: Optional[str] = None
    model_type: Optional[str] = None
    
    def get_recommendation(self, threshold: float = 0.5) -> str:
        """Get a human-readable recommendation"""
        if self.fake_probability >= threshold:
            return f"⚠️  LIKELY DEEPFAKE (confidence: {self.confidence:.3f})"
        else:
            return f"✅ LIKELY REAL (confidence: {self.confidence:.3f})"


class DistilDIREDetector:
    """
    DistilDIRE Detector for deepfake detection
    
    This detector uses DistilDIRE models that combine original images with
    epsilon noise computed through DDIM inversion to detect deepfakes.
    """
    
    def __init__(self, 
                 model_path: str,
                 model_type: str = 'distil_dire',
                 inference_mode: str = 'full',
                 adm_model_path: Optional[str] = None,
                 device: Optional[str] = None,
                 threshold: float = 0.5):
        """
        Initialize the DistilDIRE detector
        
        Args:
            model_path: Path to the DistilDIRE model checkpoint
            model_type: Type of model ('distil_dire')
            inference_mode: Inference mode ('full', 'eps_only', 'img_only')
            adm_model_path: Path to the ADM diffusion model (required for epsilon computation)
            device: Device to run inference on ('cpu', 'cuda', or None for auto-detection)
            threshold: Threshold for deepfake classification (default: 0.5)
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        self.model_path = model_path
        self.model_type = model_type
        self.inference_mode = inference_mode
        self.model_name = f"DistilDIRE-{model_type}-{inference_mode}"
        self.threshold = threshold
        
        # Load the DistilDIRE model
        self.model = None
        self.adm_model = None
        self.diffusion = None
        
        self._load_model()
        
        # Basic sanity check
        if self.model is not None:
            logger.info(f"✅ DistilDIRE model loaded successfully: {self.model_name}")
        else:
            logger.error(f"❌ Failed to load DistilDIRE model: {self.model_name}")
        
        # Load ADM model if needed and available
        if inference_mode in ['full', 'eps_only'] and adm_model_path:
            try:
                self._load_adm_model(adm_model_path)
            except Exception as e:
                logger.warning(f"⚠️ Failed to load ADM model, continuing without epsilon computation: {e}")
                self.adm_loaded = False
    
    def _load_model(self):
        """Load the DistilDIRE model"""
        logger.info(f"🔧 Loading DistilDIRE model...")
        logger.info(f"  📁 Model path: {self.model_path}")
        logger.info(f"  🏗️ Model type: {self.model_type}")
        logger.info(f"  📱 Device: {self.device}")
        
        try:
            # Check if model file exists
            if not os.path.exists(self.model_path):
                logger.error(f"❌ Model file not found: {self.model_path}")
                self.model = None
                return
            
            file_size = os.path.getsize(self.model_path) / (1024*1024)  # MB
            logger.info(f"  📊 Model file size: {file_size:.1f} MB")
            
            self.model = load_distil_dire_model(
                self.model_path, 
                self.model_type, 
                str(self.device)
            )
            
            if self.model is not None:
                logger.info(f"✅ Model loaded successfully from: {self.model_path}")
                
                # Debug model architecture
                total_params = sum(p.numel() for p in self.model.parameters())
                logger.info(f"  📊 Total parameters: {total_params:,}")
                
                # Check first conv layer (should be 6 channels for full/img_only, 3 for eps_only)
                try:
                    if hasattr(self.model, 'student_backbone') and hasattr(self.model.student_backbone, 'conv1'):
                        conv1 = self.model.student_backbone.conv1
                        if isinstance(conv1, torch.nn.Conv2d):
                            logger.info(f"  🔧 First conv layer: {conv1.in_channels} → {conv1.out_channels}")
                            
                            expected_channels = 6 if self.inference_mode in ['full', 'img_only'] else 3
                            if conv1.in_channels != expected_channels:
                                logger.warning(f"⚠️ Channel mismatch! Expected {expected_channels}, got {conv1.in_channels}")
                        else:
                            logger.info(f"  🔧 First conv layer type: {type(conv1)}")
                except Exception as conv_e:
                    logger.info(f"  🔧 Could not inspect conv layer: {conv_e}")
                
                # Test model with dummy input
                try:
                    with torch.no_grad():
                        dummy_img = torch.randn(1, 3, 256, 256).to(self.device)
                        dummy_eps = torch.randn(1, 3, 256, 256).to(self.device)
                        
                        if self.inference_mode == 'full':
                            test_output = self.model(dummy_img, dummy_eps)
                        elif self.inference_mode == 'img_only':
                            zero_eps = torch.zeros_like(dummy_img)
                            test_output = self.model(dummy_img, zero_eps)
                        else:  # eps_only
                            zero_img = torch.zeros_like(dummy_img)
                            test_output = self.model(zero_img, dummy_eps)
                        
                        logit = test_output['logit']
                        logger.info(f"  🧪 Dummy test successful, logit shape: {logit.shape}")
                        logger.info(f"  🧪 Dummy logit value: {logit.item():.6f}")
                        
                except Exception as test_e:
                    logger.error(f"❌ Model test failed: {test_e}")
                    
            else:
                logger.error(f"❌ Model loading returned None")
                
        except Exception as e:
            logger.error(f"❌ Failed to load DistilDIRE model: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            raise
    
    def _load_adm_model(self, adm_model_path: str):
        """Load the ADM diffusion model for epsilon computation"""
        try:
            # Import necessary guided diffusion utilities
            from .guided_diffusion_utils import create_model_and_diffusion, model_and_diffusion_defaults
            
            # Create ADM model configuration matching the original DistilDIRE setup
            adm_args = {
                'attention_resolutions': '32,16,8',
                'class_cond': False,
                'diffusion_steps': 1000,
                'image_size': 256,
                'learn_sigma': True,
                'noise_schedule': 'linear',
                'num_channels': 256,
                'num_head_channels': 64,
                'num_res_blocks': 2,
                'resblock_updown': True,
                'use_fp16': False,
                'use_scale_shift_norm': True,
                'timestep_respacing': 'ddim20'
            }
            
            # Create the ADM model and diffusion
            self.adm_model, self.diffusion = create_model_and_diffusion(**adm_args)
            
            # Load the pretrained weights
            if os.path.exists(adm_model_path):
                logger.info(f"Loading ADM model from {adm_model_path}")
                state_dict = torch.load(adm_model_path, map_location="cpu")
                self.adm_model.load_state_dict(state_dict)
                self.adm_model.to(self.device)
                self.adm_model.eval()
                
                # Store the loaded flag for epsilon computation
                self.adm_loaded = True
                logger.info(f"✅ Successfully loaded ADM model for epsilon computation")
            else:
                logger.error(f"❌ ADM model file not found: {adm_model_path}")
                logger.error("Please download the ADM model using: powershell -Command \"Invoke-WebRequest -Uri 'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt' -OutFile 'Models/256x256-adm.pt'\"")
                self.adm_loaded = False
                raise FileNotFoundError(f"ADM model not found: {adm_model_path}")
            
        except Exception as e:
            logger.warning(f"⚠️  Could not load ADM model: {e}")
            logger.warning("Epsilon computation will not be available")
    
    def _is_valid_image(self, file_path_or_image: Union[str, Image.Image]) -> Optional[Image.Image]:
        """Check if the input is a valid image"""
        try:
            if isinstance(file_path_or_image, str):
                if not os.path.exists(file_path_or_image):
                    return None
                image = Image.open(file_path_or_image).convert('RGB')
            elif isinstance(file_path_or_image, Image.Image):
                image = file_path_or_image.convert('RGB')
            else:
                return None
            return image
        except Exception:
            return None
    
    def preprocess_image(self, 
                        image_input: Union[str, Image.Image, np.ndarray, BinaryIO, bytes]) -> Image.Image:
        """Convert various image inputs to PIL Image"""
        if isinstance(image_input, str):
            # File path
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            # PIL Image
            image = image_input.convert('RGB')
        elif isinstance(image_input, np.ndarray):
            # NumPy array
            if image_input.dtype != np.uint8:
                image_input = (image_input * 255).astype(np.uint8)
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                image = Image.fromarray(image_input, 'RGB')
            else:
                raise ValueError("NumPy array must be in RGB format with shape (H, W, 3)")
        elif isinstance(image_input, bytes):
            # Bytes
            image = Image.open(io.BytesIO(image_input)).convert('RGB')
        elif hasattr(image_input, 'read'):
            # File-like object
            image = Image.open(image_input).convert('RGB')
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
        
        return image
    
    def _compute_epsilon(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute epsilon using DDIM inversion with the loaded ADM model
        """
        logger.info(f"🔧 Computing epsilon - Starting epsilon computation")
        logger.info(f"  📊 Input image shape: {image.shape}")
        logger.info(f"  📊 Input image range: [{image.min().item():.3f}, {image.max().item():.3f}]")
        
        try:
            if not hasattr(self, 'adm_loaded') or not self.adm_loaded:
                logger.warning("⚠️ ADM model not loaded, returning zero epsilon")
                return torch.zeros_like(image)
                
            logger.info(f"✅ ADM model is loaded, proceeding with epsilon computation")
            
            with torch.no_grad():
                # Use t=0 (first timestep) for epsilon computation as in original DistilDIRE
                t = torch.zeros(image.shape[0], device=self.device, dtype=torch.long)
                logger.info(f"  🔧 Using timestep t=0 for {image.shape[0]} images")
                
                # Check if diffusion and adm_model are available
                if self.diffusion is None or self.adm_model is None:
                    logger.warning("⚠️ Diffusion or ADM model not available")
                    logger.info(f"  📊 Diffusion available: {self.diffusion is not None}")
                    logger.info(f"  📊 ADM model available: {self.adm_model is not None}")
                    return torch.zeros_like(image)
                
                logger.info(f"✅ Both diffusion and ADM model available")
                
                # Compute epsilon using DDIM reverse sampling
                logger.info(f"🔧 Calling DDIM reverse sampling...")
                eps = self.diffusion.ddim_reverse_sample_only_eps(
                    self.adm_model,
                    image,
                    t,
                    clip_denoised=True,
                    eta=0.0
                )
                
                logger.info(f"✅ Epsilon computed successfully!")
                logger.info(f"  📊 Epsilon shape: {eps.shape}")
                logger.info(f"  📊 Epsilon range: [{eps.min().item():.3f}, {eps.max().item():.3f}]")
                logger.info(f"  📊 Epsilon mean: {eps.mean().item():.6f}")
                logger.info(f"  📊 Epsilon std: {eps.std().item():.6f}")
                
                return eps
                
        except Exception as e:
            logger.error(f"❌ Failed to compute epsilon: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            # Return a meaningful fallback - some noise based on input
            fallback_eps = torch.randn_like(image) * 0.1
            logger.warning(f"⚠️ Using fallback epsilon with std=0.1")
            return fallback_eps
    
    def detect(self, 
               image_input: Union[str, Image.Image, np.ndarray, BinaryIO, bytes],
               threshold: Optional[float] = None) -> DistilDIREDetectionResult:
        """
        Detect deepfake in an image using DistilDIRE
        
        Args:
            image_input: Input image (file path, PIL Image, numpy array, or bytes)
            threshold: Threshold for deepfake classification (0.0-1.0). If None, uses the instance threshold.
            
        Returns:
            DistilDIREDetectionResult object with detection results
        """
        start_time = time.time()
        
        # Use instance threshold if not provided
        if threshold is None:
            threshold = self.threshold
        
        try:
            # Preprocess image
            image = self.preprocess_image(image_input)
            image_path = image_input if isinstance(image_input, str) else None
            
            # Convert to tensor and normalize to [-1, 1]
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            image_tensor = (image_tensor * 2.0) - 1.0  # Normalize to [-1, 1]
            
            # Resize to 256x256 (DistilDIRE expects this size)
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0), 
                size=(256, 256), 
                mode='bilinear', 
                align_corners=False
            )
            image_tensor = image_tensor.to(self.device)
            
            # Run inference based on inference mode
            with torch.no_grad():
                if self.model is None:
                    logger.error("Model not loaded - cannot perform inference")
                    return DistilDIREDetectionResult(
                        is_deepfake=False,  # Default to real when model not available
                        confidence=0.0,     # No confidence without model
                        fake_probability=0.5,  # Neutral probability
                        processing_time=time.time() - start_time,
                        image_path=str(image_path) if image_path else None,
                        model_name=self.model_name,
                        model_type="distil_dire"
                    )
                
                if self.inference_mode == 'full':
                    # Compute epsilon and use both image and eps
                    eps = self._compute_epsilon(image_tensor)
                    output = self.model(image_tensor, eps)
                elif self.inference_mode == 'eps_only':
                    # Only use epsilon - pass it as img parameter with zero eps
                    eps = self._compute_epsilon(image_tensor)
                    zero_img = torch.zeros_like(image_tensor)
                    output = self.model(zero_img, eps)
                elif self.inference_mode == 'img_only':
                    # Only use original image - pass zero eps
                    zero_eps = torch.zeros_like(image_tensor)
                    output = self.model(image_tensor, zero_eps)
                else:
                    raise ValueError(f"Unknown inference mode: {self.inference_mode}")
                
                # Get logits and convert to probability
                logit = output['logit']
                fake_probability = torch.sigmoid(logit).item()
                
                # EXTENSIVE DEBUG: Understanding model behavior
                logger.info(f"🔍 DistilDIRE EXTENSIVE DEBUG - Model: {self.model_name}")
                logger.info(f"  📊 Raw logit: {logit.item():.6f}")
                logger.info(f"  📊 Sigmoid output: {fake_probability:.6f}")
                logger.info(f"  📊 Inference mode: {self.inference_mode}")
                logger.info(f"  📊 Model loaded: {self.model is not None}")
                logger.info(f"  📊 ADM loaded: {hasattr(self, 'adm_model') and self.adm_model is not None}")
                logger.info(f"  📊 Diffusion loaded: {hasattr(self, 'diffusion') and self.diffusion is not None}")
                
                # Check if we're in modes that need epsilon
                if self.inference_mode in ['full', 'eps_only']:
                    has_adm = hasattr(self, 'adm_model') and self.adm_model is not None
                    logger.info(f"  🔧 Epsilon computation mode, ADM available: {has_adm}")
                
                # Determine if deepfake
                is_deepfake = fake_probability >= threshold
                confidence = abs(fake_probability - 0.5) * 2  # Scale to 0-1
            
            processing_time = time.time() - start_time
            
            return DistilDIREDetectionResult(
                is_deepfake=is_deepfake,
                confidence=confidence,
                fake_probability=fake_probability,
                processing_time=processing_time,
                image_path=image_path,
                model_name=self.model_name,
                model_type=self.inference_mode
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error in DistilDIRE detection: {e}")
            
            # Return a default result with error information
            return DistilDIREDetectionResult(
                is_deepfake=False,
                confidence=0.0,
                fake_probability=0.0,
                processing_time=processing_time,
                image_path=image_input if isinstance(image_input, str) else None,
                model_name=self.model_name,
                model_type=self.model_type
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        num_parameters = sum(p.numel() for p in self.model.parameters())
        model_size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024 / 1024
        
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'model_path': self.model_path,
            'device': str(self.device),
            'num_parameters': num_parameters,
            'model_size_mb': model_size_mb,
            'has_adm_model': self.adm_model is not None
        }