# """
# Deepfake Image Detection using multiple pre-trained models

# This module implements deepfake image detection using various pre-trained models
# from Hugging Face that can classify images as real or deepfake/AI-generated.

# Supported Models:
# 1. prithivMLMods/deepfake-detector-model-v1 (SigLIP-based)
# 2. dima806/deepfake_vs_real_image_detection (Vision Transformer)
# 3. umm-maybe/AI-image-detector (EfficientNet-based)
# 4. Organika/sdxl-detector (Stable Diffusion detector)
# 5. EfficientNet-B4/B7 (torchvision/timm)
# 6. Xception (timm)
# 7. ResNet34/50/101 (torchvision/timm)
# 8. Vision Transformer (torchvision)
# 9. DeiT (timm)
# 10. Swin Transformer (torchvision/timm)
# """

# import torch
# import torch.nn.functional as F
# from transformers import AutoImageProcessor, SiglipForImageClassification, AutoModelForImageClassification
# from torchvision import models, transforms
# from PIL import Image, ImageEnhance, ImageFilter
# import numpy as np
# import logging
# import warnings
# from typing import List, Tuple, Dict, Optional, Union, Any, BinaryIO
# import os
# import time
# import base64
# import io
# from dataclasses import dataclass
# from pathlib import Path
# import cv2
# from scipy import ndimage

# # Import Pydantic output models
# from outputModel.image_detection_output_model import ImageDetectionResult, IndividualModelResult, VotingResult

# # Try to import timm, but don't fail if not available
# try:
#     import timm
#     TIMM_AVAILABLE = True
# except ImportError:
#     TIMM_AVAILABLE = False
#     logging.warning("timm library not available. Some models will not be accessible.")

# # Try to import Nonescape detector
# try:
#     from .detect_deepfake_nonescape import NonescapeDetector
#     NONESCAPE_AVAILABLE = True
# except ImportError:
#     NONESCAPE_AVAILABLE = False
#     logging.warning("Nonescape detector not available.")



 
# # Try to import SEResNet-50 detector
# try:
#     from .detect_deepfake_seresnet50 import SEResNet50Detector
#     SERESNET50_AVAILABLE = True
# except ImportError:
#     SERESNET50_AVAILABLE = False
#     logging.warning("SEResNet-50 detector not available.")


# # Import the centralized model registry
# from .model_registry import MODEL_REGISTRY

# # Suppress warnings
# warnings.filterwarnings("ignore", message=".*torch.load.*")
# warnings.filterwarnings("ignore", message=".*deprecated.*")

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)




# def list_available_models() -> Dict[str, Dict[str, Any]]:
#     """
#     Get a list of all available deepfake detection models
    
#     Returns:
#         Dictionary of available models with their details
#     """
#     return MODEL_REGISTRY.copy()


# @dataclass
# class DeepfakeDetectionResult:
#     """Result of deepfake detection analysis"""
#     is_deepfake: bool
#     confidence: float
#     scores: Dict[str, float]
#     processing_time: float
#     image_path: Optional[str] = None
#     model_name: Optional[str] = None
    
#     def get_recommendation(self, threshold: float = 0.5) -> str:
#         """
#         Get a recommendation based on the detection results
        
#         Args:
#             threshold: The threshold used for detection
            
#         Returns:
#             A string with recommendations
#         """
#         fake_score = self.scores.get('Fake', self.scores.get('fake', 0.0))
#         real_score = self.scores.get('Real', self.scores.get('real', 0.0))
        
#         if self.is_deepfake:
#             if fake_score > 0.8:
#                 return "High confidence deepfake detection. Very likely to be fake."
#             elif fake_score > 0.6:
#                 return "Moderate confidence deepfake detection. Likely to be fake."
#             else:
#                 return "Low confidence deepfake detection. Consider manual review or using ensemble voting with multiple models."
#         else:
#             if real_score > 0.8:
#                 return "High confidence real image. Very likely to be authentic."
#             elif real_score > 0.6:
#                 return "Moderate confidence real image. Likely to be authentic."
#             elif fake_score < 0.3:
#                 return f"Low fake score ({fake_score:.2%}). Image appears real, but consider:\n" \
#                        f"  - Lowering threshold (current: {threshold})\n" \
#                        f"  - Trying a different model (use list_available_models())\n" \
#                        f"  - Using ensemble voting with multiple models"
#             else:
#                 return "Borderline case. Consider:\n" \
#                        f"  - Manual review\n" \
#                        f"  - Using ensemble voting with multiple models\n" \
#                        f"  - Adjusting threshold"


# class DeepfakeImageDetector:
#     """
#     Deepfake Image Detector supporting multiple pre-trained models
    
#     This class provides functionality to detect whether an image is real or deepfake/AI-generated
#     using various pre-trained transformer models.
#     """
    
#     def __init__(self, 
#                  model_name: str = "prithivMLMods/deepfake-detector-model-v1",
#                  device: Optional[str] = None,
#                  debug: bool = False):
#         """
#         Initialize the deepfake detector
        
#         Args:
#             model_name: Name of the pre-trained model to use. Can be:
#                 - Full Hugging Face model name (e.g., 'prithivMLMods/deepfake-detector-model-v1')
#                 - Short name from MODEL_REGISTRY (e.g., 'prithiv-v1', 'dima-vit')
#             device: Device to run inference on ('cpu', 'cuda', or None for auto-detection)
#             debug: Enable debug logging (default: False)
#         """
#         self.debug = debug
        
#         # Check if model_name is a short name in registry
#         if model_name in MODEL_REGISTRY:
#             self.model_key = model_name
#             self.model_name = MODEL_REGISTRY[model_name]['name']
#             self.model_info = MODEL_REGISTRY[model_name]
#         else:
#             self.model_key = None
#             self.model_name = model_name
#             self.model_info = None
        
#         # Set device
#         if device is None:
#             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         else:
#             self.device = torch.device(device)
        
#         if self.debug:
#             logger.info(f"Using device: {self.device}")
        
#         # Initialize components
#         self.processor = None
#         self.model = None
#         self.labels = None
        
#         # Load model
#         self._load_model()
    
#     def _load_model(self):
#         """Load the pre-trained model and processor"""
#         try:
#             # Determine model path - use local if available, otherwise HuggingFace
#             model_path = self.model_name
#             if self.model_info and self.model_info.get('local_path'):
#                 local_path = self.model_info['local_path']
#                 # Check if local path exists
#                 if os.path.exists(local_path):
#                     model_path = local_path
#                     if self.debug:
#                         logger.info(f"Using local model from: {local_path}")
#                 else:
#                     if self.debug:
#                         logger.info(f"Local path {local_path} not found, downloading from HuggingFace")
            
#             if self.debug:
#                 logger.info(f"Loading model: {model_path}")
            
#             # Load processor and model
#             self.processor = AutoImageProcessor.from_pretrained(model_path)
            
#             # Try different model architectures
#             model_loaded = False
            
#             # Try SigLIP first (for prithiv model)
#             if self.model_info and self.model_info['architecture'] == 'SigLIP':
#                 try:
#                     self.model = SiglipForImageClassification.from_pretrained(
#                         model_path,
#                         torch_dtype=torch.float32 if self.device.type == 'cpu' else torch.float16
#                     )
#                     if self.debug:
#                         logger.info("Using SiglipForImageClassification")
#                     model_loaded = True
#                 except Exception as e:
#                     logger.warning(f"Failed to load with SiglipForImageClassification: {e}")
            
#             # Fallback to AutoModel
#             if not model_loaded:
#                 try:
#                     self.model = AutoModelForImageClassification.from_pretrained(
#                         model_path,
#                         torch_dtype=torch.float32 if self.device.type == 'cpu' else torch.float16
#                     )
#                     if self.debug:
#                         logger.info("Using AutoModelForImageClassification")
#                     model_loaded = True
#                 except Exception as e:
#                     logger.error(f"Failed to load model: {e}")
#                     raise
            
#             # Move model to device
#             if self.model is not None:
#                 self.model = self.model.to(self.device)  # type: ignore
#                 self.model.eval()
                
#                 # Apply INT8 dynamic quantization for CPU inference (PyTorch models only)
#                 # Benefits: 50-75% memory reduction, 2-3x faster inference, <1% accuracy loss
#                 # Only quantize on CPU (GPU uses native FP16 which is already optimized)
#                 if self.device.type == 'cpu' and hasattr(self.model, 'parameters'):
#                     try:
#                         import torch.quantization
#                         if self.debug:
#                             logger.info("Applying INT8 dynamic quantization to model...")
                        
#                         # Quantize linear layers and attention layers (common in transformers)
#                         self.model = torch.quantization.quantize_dynamic(
#                             self.model,
#                             {torch.nn.Linear, torch.nn.MultiheadAttention},
#                             dtype=torch.qint8
#                         )
                        
#                         if self.debug:
#                             logger.info("✓ INT8 quantization applied successfully")
#                     except Exception as quant_err:
#                         # Quantization failure is non-fatal - model will still work at FP32
#                         logger.warning(f"Failed to apply quantization (continuing with FP32): {quant_err}")
                
#                 # Get label mappings
#                 self.labels = self.model.config.id2label
#             else:
#                 raise RuntimeError("Failed to load model")
            
#             if self.debug:
#                 logger.info(f"Model loaded successfully. Labels: {self.labels}")
            
#         except Exception as e:
#             logger.error(f"Error loading model: {str(e)}")
#             raise
    
#     def _detect_blur(self, image: Image.Image) -> Tuple[bool, float]:
#         """
#         Detect if image is blurred using Laplacian variance method
        
#         Args:
#             image: PIL Image
            
#         Returns:
#             Tuple of (is_blurred, blur_score)
#             Lower scores indicate more blur (threshold typically around 100)
#         """
#         # Convert PIL to numpy array
#         img_array = np.array(image)
        
#         # Convert to grayscale
#         if len(img_array.shape) == 3:
#             gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
#         else:
#             gray = img_array
        
#         # Compute Laplacian variance
#         laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
#         # Threshold for blur detection (lower = more blurred)
#         blur_threshold = 100.0
#         is_blurred = laplacian_var < blur_threshold
        
#         return is_blurred, laplacian_var
    
#     def _enhance_image(self, image: Image.Image, is_blurred: bool, blur_score: float) -> Image.Image:
#         """
#         Enhance image quality with denoising, sharpening, and contrast adjustment
        
#         Args:
#             image: PIL Image to enhance
#             is_blurred: Whether image is detected as blurred
#             blur_score: Blur detection score
            
#         Returns:
#             Enhanced PIL Image
#         """
#         enhanced = image.copy()
        
#         try:
#             # Convert to numpy for OpenCV operations
#             img_array = np.array(enhanced)
            
#             # 1. Denoise the image
#             denoised = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
#             enhanced = Image.fromarray(denoised)
            
#             # 2. If blurred, apply sharpening
#             if is_blurred or blur_score < 150:
#                 # Determine sharpening intensity based on blur level
#                 if blur_score < 50:
#                     sharpness_factor = 2.5  # Heavy blur
#                 elif blur_score < 100:
#                     sharpness_factor = 2.0  # Moderate blur
#                 else:
#                     sharpness_factor = 1.5  # Light blur
                
#                 # Apply unsharp mask for better sharpening
#                 enhanced_array = np.array(enhanced)
#                 gaussian = cv2.GaussianBlur(enhanced_array, (0, 0), 2.0)
#                 enhanced_array = cv2.addWeighted(enhanced_array, 1.5, gaussian, -0.5, 0)
#                 enhanced = Image.fromarray(np.clip(enhanced_array, 0, 255).astype(np.uint8))
                
#                 # Additional PIL-based sharpening
#                 enhancer = ImageEnhance.Sharpness(enhanced)
#                 enhanced = enhancer.enhance(sharpness_factor)
            
#             # 3. Enhance contrast
#             contrast_enhancer = ImageEnhance.Contrast(enhanced)
#             enhanced = contrast_enhancer.enhance(1.2)
            
#             # 4. Enhance color/brightness slightly
#             color_enhancer = ImageEnhance.Color(enhanced)
#             enhanced = color_enhancer.enhance(1.1)
            
#             # 5. Apply slight edge enhancement
#             enhanced = enhanced.filter(ImageFilter.EDGE_ENHANCE)
#             if (self.debug):
#                 logger.info(f"Image enhanced - Blur score: {blur_score:.2f}, Blurred: {is_blurred}")
            
#         except Exception as e:
#             logger.warning(f"Error during image enhancement: {e}. Using original image.")
#             enhanced = image
        
#         return enhanced
    
#     def preprocess_image(self, 
#                         image_input: Union[str, Image.Image, np.ndarray, BinaryIO, bytes],
#                         apply_enhancement: bool = True) -> Image.Image:
#         """
#         Preprocess image input to PIL Image format with quality enhancement
        
#         Args:
#             image_input: Image as:
#                 - File path (str)
#                 - PIL Image object
#                 - Numpy array
#                 - Base64 encoded string (detected by 'data:image' prefix)
#                 - Binary stream/file object
#                 - Raw bytes
#             apply_enhancement: Whether to apply blur detection and image enhancement
            
#         Returns:
#             PIL Image object (enhanced if apply_enhancement=True)
#         """
#         if isinstance(image_input, str):
#             # Check if it's base64 encoded image
#             if image_input.startswith('data:image'):
#                 # Extract base64 data from data URI
#                 if ',' in image_input:
#                     base64_data = image_input.split(',')[1]
#                 else:
#                     base64_data = image_input
                
#                 # Decode base64
#                 try:
#                     image_bytes = base64.b64decode(base64_data)
#                     image = Image.open(io.BytesIO(image_bytes))
#                 except Exception as e:
#                     raise ValueError(f"Invalid base64 image data: {e}")
                    
#             elif self._is_base64_string(image_input):
#                 # Raw base64 string without data URI prefix
#                 try:
#                     image_bytes = base64.b64decode(image_input)
#                     image = Image.open(io.BytesIO(image_bytes))
#                 except Exception as e:
#                     raise ValueError(f"Invalid base64 image data: {e}")
#             else:
#                 # File path
#                 if not os.path.exists(image_input):
#                     raise FileNotFoundError(f"Image file not found: {image_input}")
#                 image = Image.open(image_input)
                
#         elif isinstance(image_input, np.ndarray):
#             # Numpy array
#             image = Image.fromarray(image_input)
            
#         elif isinstance(image_input, Image.Image):
#             # Already PIL Image
#             image = image_input
            
#         elif isinstance(image_input, bytes):
#             # Raw bytes
#             try:
#                 image = Image.open(io.BytesIO(image_input))
#             except Exception as e:
#                 raise ValueError(f"Invalid image bytes: {e}")
                
#         elif hasattr(image_input, 'read'):
#             # Stream/file-like object
#             try:
#                 image = Image.open(image_input)
#             except Exception as e:
#                 raise ValueError(f"Invalid image stream: {e}")
#         else:
#             raise ValueError(f"Unsupported image input type: {type(image_input)}")
        
#         # Convert to RGB if not already
#         if image.mode != 'RGB':
#             image = image.convert('RGB')
        
#         # Apply enhancement if requested
#         if apply_enhancement:
#             is_blurred, blur_score = self._detect_blur(image)
#             image = self._enhance_image(image, is_blurred, blur_score)
        
#         return image
    
#     def _is_base64_string(self, s: str) -> bool:
#         """
#         Check if a string is a valid base64 encoded string
        
#         Args:
#             s: String to check
            
#         Returns:
#             True if valid base64, False otherwise
#         """
#         try:
#             # Check if string length is valid for base64
#             if len(s) < 4 or len(s) % 4 != 0:
#                 return False
            
#             # Try to decode and check if it results in valid image data
#             decoded = base64.b64decode(s, validate=True)
            
#             # Check if decoded data starts with common image file signatures
#             image_signatures = [
#                 b'\xff\xd8\xff',  # JPEG
#                 b'\x89PNG',       # PNG
#                 b'GIF8',          # GIF
#                 b'BM',            # BMP
#                 b'RIFF',          # WebP (RIFF container)
#             ]
            
#             return any(decoded.startswith(sig) for sig in image_signatures)
            
#         except Exception:
#             return False
    
#     def detect(self, 
#                image_input: Union[str, Image.Image, np.ndarray, BinaryIO, bytes],
#                threshold: float = 0.6) -> DeepfakeDetectionResult:
#         """
#         Detect if an image is deepfake/AI-generated
        
#         Args:
#             image_input: Image as:
#                 - File path (str)
#                 - PIL Image object
#                 - Numpy array
#                 - Base64 encoded string (with or without data URI prefix)
#                 - Binary stream/file object
#                 - Raw bytes
#             threshold: Threshold for deepfake classification (0.0 to 1.0)
            
#         Returns:
#             DeepfakeDetectionResult object with detection results
#         """
#         start_time = time.time()
        
#         try:
#             # Preprocess image
#             image = self.preprocess_image(image_input)
            
#             # Get image path if available
#             image_path = image_input if isinstance(image_input, str) else None
            
#             # Check if processor and model are loaded
#             if self.processor is None:
#                 raise RuntimeError("Processor not initialized")
#             if self.model is None:
#                 raise RuntimeError("Model not initialized")
            
#             # Process image
#             inputs = self.processor(images=image, return_tensors="pt")
#             inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
#             # Run inference
#             with torch.no_grad():
#                 outputs = self.model(**inputs)
#                 logits = outputs.logits
                
#                 # Get probabilities
#                 probabilities = F.softmax(logits, dim=-1)
                
#                 # Build scores dictionary with null check
#                 if self.labels is None:
#                     raise RuntimeError("Model labels not initialized")
                
#                 scores = {self.labels[i]: float(probabilities[0][i].cpu()) 
#                          for i in range(len(self.labels))}
                
#                 # Find the fake/deepfake score directly
#                 fake_score = 0.0
#                 fake_label = None
#                 real_score = 0.0
#                 real_label = None
#                 deepfake_indicators = ['fake', 'deepfake', 'artificial', 'generated', 'synthetic', 'ai']
#                 real_indicators = ['real', 'authentic', 'genuine', 'human', 'realism']
                
#                 # Find fake score
#                 for label, score in scores.items():
#                     label_lower = label.lower()
#                     for indicator in deepfake_indicators:
#                         if indicator in label_lower:
#                             fake_score = score
#                             fake_label = label
#                             break
#                     if fake_label:
#                         break
                
#                 # Find real score
#                 for label, score in scores.items():
#                     label_lower = label.lower()
#                     for indicator in real_indicators:
#                         if indicator in label_lower:
#                             real_score = score
#                             real_label = label
#                             break
#                     if real_label:
#                         break
                
#                 # If we found both fake and real labels, compare scores directly (ignore threshold)
#                 if fake_label and real_label:
#                     # Predict based on which score is higher
#                     is_deepfake = fake_score > real_score
#                     # Confidence is the higher of the two scores
#                     confidence = max(fake_score, real_score)
#                 elif fake_label:
#                     # Only fake label found, use threshold
#                     is_deepfake = fake_score >= threshold
#                     confidence = fake_score
#                 else:
#                     # Fallback to old logic if no fake/real labels found
#                     max_label = max(scores.keys(), key=lambda k: scores[k])
#                     confidence = scores[max_label]
#                     is_deepfake = self._is_deepfake_label(max_label, confidence, threshold)
                
#                 processing_time = time.time() - start_time
                
#                 return DeepfakeDetectionResult(
#                     is_deepfake=is_deepfake,
#                     confidence=confidence,
#                     scores=scores,
#                     processing_time=processing_time,
#                     image_path=image_path,
#                     model_name=self.model_name
#                 )
                
#         except Exception as e:
#             logger.error(f"Error during detection: {str(e)}")
#             raise
    
#     def _is_deepfake_label(self, label: str, confidence: float, threshold: float) -> bool:
#         """
#         Determine if a label indicates deepfake based on common label patterns
        
#         Args:
#             label: Predicted label
#             confidence: Confidence score
#             threshold: Classification threshold
            
#         Returns:
#             True if deepfake, False if real
#         """
#         # Common patterns for deepfake labels
#         deepfake_indicators = ['fake', 'deepfake', 'artificial', 'generated', 'synthetic', 'ai']
#         real_indicators = ['real', 'authentic', 'genuine', 'human', 'realism']
        
#         label_lower = label.lower()
        
#         # Check for deepfake indicators
#         for indicator in deepfake_indicators:
#             if indicator in label_lower:
#                 # If model says "Fake" with high confidence, it IS a deepfake
#                 return confidence >= threshold
        
#         # Check for real indicators
#         for indicator in real_indicators:
#             if indicator in label_lower:
#                 # If model says "Real" with high confidence, it is NOT a deepfake
#                 # So return True (is_deepfake) only if confidence is LOW (meaning model is uncertain)
#                 return confidence < (1.0 - threshold)
        
#         # Default: use threshold on confidence
#         # If we can't determine from label, assume higher confidence means more likely to be the predicted class
#         return confidence >= threshold
    
#     def detect_batch(self, 
#                      image_inputs: List[Any],
#                      threshold: float = 0.5) -> List[DeepfakeDetectionResult]:
#         """
#         Detect deepfakes in a batch of images
        
#         Args:
#             image_inputs: List of images supporting multiple formats:
#                 - File paths (str)
#                 - PIL Image objects
#                 - Numpy arrays
#                 - Base64 encoded strings (with or without data URI prefix)
#                 - Binary streams/file objects
#                 - Raw bytes
#             threshold: Threshold for deepfake classification
            
#         Returns:
#             List of DeepfakeDetectionResult objects
#         """
#         results = []
        
#         if self.debug:
#             logger.info(f"Processing batch of {len(image_inputs)} images")
        
#         for i, image_input in enumerate(image_inputs):
#             try:
#                 if self.debug:
#                     logger.info(f"Processing image {i+1}/{len(image_inputs)}")
#                 result = self.detect(image_input, threshold)
#                 results.append(result)
#             except Exception as e:
#                 logger.error(f"Error processing image {i+1}: {str(e)}")
#                 # Add error result
#                 results.append(DeepfakeDetectionResult(
#                     is_deepfake=False,
#                     confidence=0.0,
#                     scores={},
#                     processing_time=0.0,
#                     image_path=image_input if isinstance(image_input, str) else None
#                 ))
        
#         return results
    
#     def analyze_directory(self, 
#                          directory_path: str,
#                          extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp', '.gif','avif','webp'],
#                          threshold: float = 0.6) -> List[DeepfakeDetectionResult]:
#         """
#         Analyze all images in a directory for deepfakes
        
#         Args:
#             directory_path: Path to directory containing images
#             extensions: List of image file extensions to process
#             threshold: Threshold for deepfake classification
            
#         Returns:
#             List of DeepfakeDetectionResult objects
#         """
#         directory = Path(directory_path)
        
#         if not directory.exists():
#             raise FileNotFoundError(f"Directory not found: {directory_path}")
        
#         # Find all image files
#         image_files = []
#         for ext in extensions:
#             image_files.extend(directory.glob(f"*{ext}"))
#             # image_files.extend(directory.glob(f"*{ext.upper()}"))
        
#         if self.debug:
#             logger.info(f"Found {len(image_files)} image files in {directory_path}")
        
#         # Convert to strings
#         image_paths = [str(img) for img in image_files]
        
#         return self.detect_batch(image_paths, threshold)
    
#     def get_model_info(self) -> Dict[str, Any]:
#         """Get information about the loaded model"""
#         if self.model is None:
#             raise RuntimeError("Model not initialized")
            
#         return {
#             'model_name': self.model_name,
#             'model_key': self.model_key,
#             'device': str(self.device),
#             'labels': self.labels,
#             'num_parameters': sum(p.numel() for p in self.model.parameters()),
#             'model_size_mb': sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024 / 1024
#         }


# class EnsembleDetector:
#     """
#     Enhanced ensemble detector that uses multiple models for improved accuracy.
#     Supports HuggingFace, torchvision, and timm models.
#     """
    
#     def __init__(self, 
#                  model_keys: Optional[List[str]] = None,
#                  device: Optional[str] = None,
#                  debug: bool = False):
#         """
#         Initialize ensemble detector with multiple models
        
#         Args:
#             model_keys: List of model keys from MODEL_REGISTRY to use. 
#                        If None, uses default HuggingFace models.
#             device: Device to run inference on ('cpu', 'cuda', or None for auto-detection)
#             debug: Enable debug logging (default: False)
        
#         Example:
#             # Use HuggingFace models only
#             ensemble = EnsembleDetector(model_keys=['prithiv-v1', 'dima-vit'])
            
#             # Mix HuggingFace and torchvision models
#             ensemble = EnsembleDetector(model_keys=[
#                 'prithiv-v1', 'efficientnet_b4', 'xception', 'resnet50'
#             ])
#         """
#         self.debug = debug
        
#         if device is None:
#             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         else:
#             self.device = torch.device(device)
        
#         if model_keys is None:
#             # Use default HuggingFace models that are most reliable
#             model_keys = [ 'prithiv-v1', 'dima-vit', 'organika-sdxl']
        
#         # Validate model keys
#         invalid_keys = [key for key in model_keys if key not in MODEL_REGISTRY]
#         if invalid_keys:
#             raise ValueError(
#                 f"Invalid model keys: {invalid_keys}. "
#                 f"Available models: {list(MODEL_REGISTRY.keys())}"
#             )
        
#         self.model_keys = model_keys
#         self.detectors = {}
        
#         if self.debug:
#             logger.info(f"Initializing ensemble with models: {model_keys}")
        
#         # Load all models
#         self._load_models()
        
#         if not self.detectors:
#             raise RuntimeError("No models could be loaded for ensemble")
        
#         if self.debug:
#             logger.info(f"Ensemble initialized with {len(self.detectors)} models")
    
#     def _load_models(self):
#         """Load all specified models"""
#         for key in self.model_keys:
#             try:
#                 config = MODEL_REGISTRY[key]
                
#                 if config['source'] == 'huggingface':
#                     # Use existing HuggingFace loading logic
#                     detector = DeepfakeImageDetector(model_name=key, device=str(self.device), debug=self.debug)
#                     self.detectors[key] = detector
#                     if self.debug:
#                         logger.info(f"[OK] Loaded HuggingFace model: {key}")               
                 
                    
#                 elif config['source'] == 'torchvision':
#                     # Load torchvision model
#                     model = self._load_torchvision_model(config)
#                     self.detectors[key] = self._wrap_pytorch_model(model, key, config)
#                     if self.debug:
#                         logger.info(f"[OK] Loaded torchvision model: {key}")
                    
#                 elif config['source'] == 'timm':
#                     # Load timm model
#                     if not TIMM_AVAILABLE:
#                         logger.warning(f"⚠ Skipping {key}: timm library not installed")
#                         continue
#                     model = self._load_timm_model(config)
#                     self.detectors[key] = self._wrap_pytorch_model(model, key, config)
#                     if self.debug:
#                         logger.info(f"[OK] Loaded timm model: {key}")
                
#                 elif config['source'] == 'nonescape':
#                     # Load Nonescape model
#                     if not NONESCAPE_AVAILABLE:
#                         logger.warning(f"⚠ Skipping {key}: Nonescape detector not available")
#                         continue
                    
#                     model_path = config.get('model_path')
#                     if model_path is None:
#                         logger.warning(f"⚠ Skipping {key}: model_path not specified in config")
#                         logger.info(f"   To use {key}, download the model and set model_path:")
#                         if config['use_mini']:
#                             logger.info("   wget https://nonescape.sfo2.cdn.digitaloceanspaces.com/nonescape-mini-v0.safetensors")
#                         else:
#                             logger.info("   wget https://nonescape.sfo2.cdn.digitaloceanspaces.com/nonescape-v0.safetensors")
#                         continue
                    
#                     # Check if model file exists
#                     if not os.path.exists(model_path):
#                         logger.warning(f"⚠ Skipping {key}: Model file not found at {model_path}")
#                         logger.info(f"   To use {key}, download the model:")
#                         if config['use_mini']:
#                             logger.info("   wget https://nonescape.sfo2.cdn.digitaloceanspaces.com/nonescape-mini-v0.safetensors")
#                         else:
#                             logger.info("   wget https://nonescape.sfo2.cdn.digitaloceanspaces.com/nonescape-v0.safetensors")
#                         logger.info(f"   Then place it at: {model_path}")
#                         continue
                    
#                     # Import NonescapeDetector here to avoid unbound error
#                     from .detect_deepfake_nonescape import NonescapeDetector as NED
                    
#                     detector = NED(
#                         model_path=model_path,
#                         use_mini=config['use_mini'],
#                         device=str(self.device),
#                         debug=self.debug
#                     )
#                     self.detectors[key] = self._wrap_nonescape_detector(detector, key)
#                     if self.debug:
#                         logger.info(f"[OK] Loaded Nonescape model: {key}")
                
                
                
                
                
#                 elif config['source'] == 'seresnet50':
#                     # Load SEResNet-50 model
#                     if not SERESNET50_AVAILABLE:
#                         logger.warning(f"⚠ Skipping {key}: SEResNet-50 detector not available")
#                         continue
                    
#                     model_path = config.get('model_path')
#                     input_size = config.get('input_size', 224)
#                     threshold = config.get('recommended_threshold', 0.5)
                    
#                     # Check if model file exists (for fine-tuned models)
#                     if model_path and not os.path.exists(model_path):
#                         logger.warning(f"⚠ SEResNet-50 model not found at {model_path}, will use pretrained model from torch.hub")
#                         model_path = None
                    
#                     model_type = config.get('model_type', 'pretrained')  # Default to 'pretrained'
#                     from .detect_deepfake_seresnet50 import SEResNet50Detector as SER50
#                     detector = SER50(
#                         model_path=model_path,
#                         model_type=model_type,
#                         device=str(self.device),
#                         threshold=threshold,
#                         input_size=input_size,
#                         debug=self.debug
#                     )
#                     self.detectors[key] = self._wrap_seresnet50_detector(detector, key)
#                     if self.debug:
#                         logger.info(f"[OK] Loaded SEResNet-50 model: {key}")
                
                 
                
                 
                
                 
                    
#             except Exception as e:
#                 logger.warning(f"⚠ Failed to load model {key}: {e}")
    
     
    
#     def _load_torchvision_model(self, config: Dict) -> torch.nn.Module:
#         """Load model from torchvision"""
#         import torch.nn as nn
        
#         model = config['model_func']()
        
#         if config.get('modify_classifier', False):
#             # Modify final layer for binary classification
#             classifier_attr = config['classifier_attr']
#             in_features = config['classifier_in_features']
            
#             if classifier_attr == 'classifier':
#                 # For EfficientNet
#                 model.classifier = nn.Sequential(
#                     nn.Dropout(p=0.3, inplace=True),
#                     nn.Linear(in_features, 2)
#                 )
#             elif classifier_attr == 'fc':
#                 # For ResNet
#                 model.fc = nn.Linear(in_features, 2)
#             elif classifier_attr == 'heads':
#                 # For ViT
#                 model.heads = nn.Linear(in_features, 2)
#             elif classifier_attr == 'head':
#                 # For Swin Transformer
#                 model.head = nn.Linear(in_features, 2)
        
#         model = model.to(self.device)
#         model.eval()
#         return model
    
#     def _load_timm_model(self, config: Dict) -> torch.nn.Module:
#         """Load model from timm library"""
#         if not TIMM_AVAILABLE:
#             raise ImportError("timm library is not available")
#         import timm as timm_lib
#         model = timm_lib.create_model(
#             config['model_name'],
#             pretrained=True,
#             num_classes=2  # Binary classification
#         )
#         model = model.to(self.device)
#         model.eval()
#         return model
    
#     def _wrap_pytorch_model(self, model: torch.nn.Module, model_key: str, config: Dict):
#         """Wrap PyTorch model to match DeepfakeImageDetector interface"""
#         class PyTorchModelWrapper:
#             """Wrapper to make PyTorch models compatible with DeepfakeImageDetector interface"""
            
#             def __init__(self, model, device, input_size, model_key, model_name, debug=False):
#                 self.model = model
#                 self.device = device
#                 self.model_key = model_key
#                 self.model_name = model_name
#                 self.debug = debug
#                 self.transform = transforms.Compose([
#                     transforms.Resize((input_size, input_size)),
#                     transforms.ToTensor(),
#                     transforms.Normalize(
#                         mean=[0.485, 0.456, 0.406],
#                         std=[0.229, 0.224, 0.225]
#                     )
#                 ])
            
#             def _detect_blur(self, image: Image.Image) -> Tuple[bool, float]:
#                 """Detect if image is blurred using Laplacian variance method"""
#                 img_array = np.array(image)
#                 if len(img_array.shape) == 3:
#                     gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
#                 else:
#                     gray = img_array
#                 laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
#                 blur_threshold = 100.0
#                 is_blurred = laplacian_var < blur_threshold
#                 return is_blurred, laplacian_var
            
#             def _enhance_image(self, image: Image.Image, is_blurred: bool, blur_score: float) -> Image.Image:
#                 """Enhance image quality with denoising, sharpening, and contrast adjustment"""
#                 enhanced = image.copy()
#                 try:
#                     img_array = np.array(enhanced)
#                     denoised = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
#                     enhanced = Image.fromarray(denoised)
                    
#                     if is_blurred or blur_score < 150:
#                         if blur_score < 50:
#                             sharpness_factor = 2.5
#                         elif blur_score < 100:
#                             sharpness_factor = 2.0
#                         else:
#                             sharpness_factor = 1.5
                        
#                         enhanced_array = np.array(enhanced)
#                         gaussian = cv2.GaussianBlur(enhanced_array, (0, 0), 2.0)
#                         enhanced_array = cv2.addWeighted(enhanced_array, 1.5, gaussian, -0.5, 0)
#                         enhanced = Image.fromarray(np.clip(enhanced_array, 0, 255).astype(np.uint8))
                        
#                         enhancer = ImageEnhance.Sharpness(enhanced)
#                         enhanced = enhancer.enhance(sharpness_factor)
                    
#                     contrast_enhancer = ImageEnhance.Contrast(enhanced)
#                     enhanced = contrast_enhancer.enhance(1.2)
#                     color_enhancer = ImageEnhance.Color(enhanced)
#                     enhanced = color_enhancer.enhance(1.1)
#                     enhanced = enhanced.filter(ImageFilter.EDGE_ENHANCE)
#                     if (self.debug):
#                         logger.info(f"Image enhanced - Blur score: {blur_score:.2f}, Blurred: {is_blurred}")
#                 except Exception as e:
#                     logger.warning(f"Error during image enhancement: {e}. Using original image.")
#                     enhanced = image
#                 return enhanced
            
#             def preprocess_image(self, 
#                                image_input: Union[str, Image.Image, np.ndarray, BinaryIO, bytes],
#                                apply_enhancement: bool = True) -> Image.Image:
#                 """Preprocess image input to PIL Image format with quality enhancement"""
#                 if isinstance(image_input, str):
#                     if not os.path.exists(image_input):
#                         raise FileNotFoundError(f"Image file not found: {image_input}")
#                     image = Image.open(image_input)
#                 elif isinstance(image_input, np.ndarray):
#                     image = Image.fromarray(image_input)
#                 elif isinstance(image_input, Image.Image):
#                     image = image_input
#                 elif isinstance(image_input, bytes):
#                     image = Image.open(io.BytesIO(image_input))
#                 elif hasattr(image_input, 'read'):
#                     image = Image.open(image_input)
#                 else:
#                     raise ValueError(f"Unsupported image input type: {type(image_input)}")
                
#                 if image.mode != 'RGB':
#                     image = image.convert('RGB')
                
#                 # Apply enhancement if requested
#                 if apply_enhancement:
#                     is_blurred, blur_score = self._detect_blur(image)
#                     image = self._enhance_image(image, is_blurred, blur_score)
                
#                 return image
            
#             def detect(self, image_input: Union[str, Image.Image, np.ndarray, BinaryIO, bytes],
#                       threshold: float = 0.5) -> DeepfakeDetectionResult:
#                 """Predict if image is deepfake"""
#                 start_time = time.time()
                
#                 # Preprocess image
#                 image = self.preprocess_image(image_input)
#                 image_path = image_input if isinstance(image_input, str) else None
                
#                 # Transform and prepare input
#                 input_tensor = self.transform(image).unsqueeze(0).to(self.device)  # type: ignore
                
#                 # Run inference
#                 with torch.no_grad():
#                     outputs = self.model(input_tensor)
#                     probabilities = torch.softmax(outputs, dim=1)
#                     real_prob = float(probabilities[0][0].cpu())
#                     fake_prob = float(probabilities[0][1].cpu())
                
#                 processing_time = time.time() - start_time
                
#                 return DeepfakeDetectionResult(
#                     is_deepfake=fake_prob >= threshold,
#                     confidence=fake_prob,
#                     scores={'Real': real_prob, 'Fake': fake_prob},
#                     processing_time=processing_time,
#                     image_path=image_path,
#                     model_name=self.model_name
#                 )
        
#         return PyTorchModelWrapper(
#             model=model,
#             device=self.device,
#             input_size=config['input_size'],
#             model_key=model_key,
#             model_name=config['name'],
#             debug=self.debug
#         )
    
#     def _wrap_nonescape_detector(self, detector, model_key: str):
#         """Wrap Nonescape detector to match DeepfakeImageDetector interface"""
#         class NonescapeWrapper:
#             """Wrapper to make Nonescape detector compatible with DeepfakeImageDetector interface"""
            
#             def __init__(self, nonescape_detector, model_key):
#                 self.nonescape_detector = nonescape_detector
#                 self.model_key = model_key
#                 self.model_name = nonescape_detector.model_name
            
#             def detect(self, image_input: Union[str, Image.Image, np.ndarray, BinaryIO, bytes],
#                       threshold: float = 0.5) -> DeepfakeDetectionResult:
#                 """Detect deepfake using Nonescape model"""
#                 # Convert Nonescape result to DeepfakeDetectionResult
#                 nonescape_result = self.nonescape_detector.detect(image_input, threshold)
                
#                 return DeepfakeDetectionResult(
#                     is_deepfake=nonescape_result.is_ai_generated,
#                     confidence=nonescape_result.confidence,
#                     scores={
#                         'Real': nonescape_result.authentic_prob,
#                         'Fake': nonescape_result.ai_prob
#                     },
#                     processing_time=nonescape_result.processing_time,
#                     image_path=nonescape_result.image_path,
#                     model_name=self.model_name
#                 )
        
#         return NonescapeWrapper(detector, model_key)
    
#     def _wrap_universal_fake_v2_detector(self, detector, model_key: str):
#         """Wrap UniversalFakeDetectV2 detector to match DeepfakeImageDetector interface"""
#         class UniversalFakeV2Wrapper:
#             """Wrapper to make UniversalFakeDetectV2 detector compatible with DeepfakeImageDetector interface"""
            
#             def __init__(self, universal_detector, model_key):
#                 self.universal_detector = universal_detector
#                 self.model_key = model_key
#                 self.model_name = universal_detector.model_name
            
#             def detect(self, image_input: Union[str, Image.Image, np.ndarray, BinaryIO, bytes],
#                       threshold: float = 0.5) -> DeepfakeDetectionResult:
#                 """Detect deepfake using UniversalFakeDetectV2 model"""
#                 # Convert UniversalFakeDetectV2 result to DeepfakeDetectionResult
#                 ufv2_result = self.universal_detector.detect(image_input, threshold)
                
#                 return DeepfakeDetectionResult(
#                     is_deepfake=ufv2_result.is_fake,
#                     confidence=ufv2_result.confidence,
#                     scores={
#                         'Real': ufv2_result.real_probability,
#                         'Fake': ufv2_result.fake_probability
#                     },
#                     processing_time=ufv2_result.processing_time,
#                     image_path=ufv2_result.image_path,
#                     model_name=self.model_name
#                 )
        
#         return UniversalFakeV2Wrapper(detector, model_key)
    
#     def _wrap_distil_dire_detector(self, detector, model_key: str):
#         """Wrap DistilDIRE detector to match DeepfakeImageDetector interface"""
#         class DistilDIREWrapper:
#             """Wrapper to make DistilDIRE detector compatible with DeepfakeImageDetector interface"""
            
#             def __init__(self, distil_dire_detector, model_key):
#                 self.distil_dire_detector = distil_dire_detector
#                 self.model_key = model_key
#                 self.model_name = distil_dire_detector.model_name
            
#             def detect(self, image_input: Union[str, Image.Image, np.ndarray, BinaryIO, bytes],
#                       threshold: float = 0.5) -> DeepfakeDetectionResult:
#                 """Detect deepfake using DistilDIRE model"""
#                 # Convert DistilDIRE result to DeepfakeDetectionResult
#                 distil_dire_result = self.distil_dire_detector.detect(image_input, threshold)
                
#                 return DeepfakeDetectionResult(
#                     is_deepfake=distil_dire_result.is_deepfake,
#                     confidence=distil_dire_result.confidence,
#                     scores={
#                         'Real': 1.0 - distil_dire_result.fake_probability,
#                         'Fake': distil_dire_result.fake_probability
#                     },
#                     processing_time=distil_dire_result.processing_time,
#                     image_path=distil_dire_result.image_path,
#                     model_name=self.model_name
#                 )
        
#         return DistilDIREWrapper(detector, model_key)
    
#     def _wrap_seresnet50_detector(self, detector, model_key: str):
#         """Wrap SEResNet-50 detector to match DeepfakeImageDetector interface"""
#         class SEResNet50Wrapper:
#             """Wrapper to make SEResNet-50 detector compatible with DeepfakeImageDetector interface"""
            
#             def __init__(self, seresnet50_detector, model_key):
#                 self.seresnet50_detector = seresnet50_detector
#                 self.model_key = model_key
#                 self.model_name = seresnet50_detector.model_name
            
#             def detect(self, image_input: Union[str, Image.Image, np.ndarray, BinaryIO, bytes],
#                       threshold: float = 0.5) -> DeepfakeDetectionResult:
#                 """Detect deepfake using SEResNet-50 model"""
#                 # Convert SEResNet-50 result to DeepfakeDetectionResult
#                 seresnet50_result = self.seresnet50_detector.detect_single(image_input, use_ensemble=False)
                
#                 # Ignore threshold - predict based on which score is higher
#                 is_deepfake = seresnet50_result.fake_probability > seresnet50_result.real_probability
#                 # Confidence is the higher of the two scores
#                 confidence = max(seresnet50_result.fake_probability, seresnet50_result.real_probability)
                
#                 return DeepfakeDetectionResult(
#                     is_deepfake=is_deepfake,
#                     confidence=confidence,
#                     scores={
#                         'Real': seresnet50_result.real_probability,
#                         'Fake': seresnet50_result.fake_probability
#                     },
#                     processing_time=seresnet50_result.processing_time,
#                     image_path=image_input if isinstance(image_input, str) else None,
#                     model_name=self.model_name
#                 )
        
#         return SEResNet50Wrapper(detector, model_key)
    
     
    
#     def _wrap_efficientnet_b7_detector(self, detector, model_key: str):
#         """Wrap EfficientNet-B7 detector to match DeepfakeImageDetector interface"""
#         class EfficientNetB7Wrapper:
#             """Wrapper to make EfficientNet-B7 detector compatible with DeepfakeImageDetector interface"""
            
#             def __init__(self, efficientnet_b7_detector, model_key):
#                 self.efficientnet_b7_detector = efficientnet_b7_detector
#                 self.model_key = model_key
#                 self.model_name = efficientnet_b7_detector.model_name
            
#             def detect(self, image_input: Union[str, Image.Image, np.ndarray, BinaryIO, bytes],
#                       threshold: float = 0.5) -> DeepfakeDetectionResult:
#                 """Detect deepfake using EfficientNet-B7 model"""
#                 # Convert EfficientNet-B7 result to DeepfakeDetectionResult
#                 efficientnet_result = self.efficientnet_b7_detector.detect(image_input, threshold)
                
#                 return DeepfakeDetectionResult(
#                     is_deepfake=efficientnet_result.is_deepfake,
#                     confidence=efficientnet_result.confidence,
#                     scores=efficientnet_result.scores,
#                     processing_time=efficientnet_result.processing_time,
#                     image_path=efficientnet_result.image_path,
#                     model_name=self.model_name
#                 )
        
#         return EfficientNetB7Wrapper(detector, model_key)
    
    
    
#     def detect(self, 
#                image_input: Union[str, Image.Image, np.ndarray, BinaryIO, bytes],
#                threshold: float = 0.5,
#                voting_method: str = 'majority') -> ImageDetectionResult:
#         """
#         Detect deepfake using ensemble of models
        
#         Args:
#             image_input: Image input (same formats as DeepfakeImageDetector)
#             threshold: Threshold for each individual model
#             voting_method: Voting method to use:
#                 - 'majority': Simple majority vote
#                 - 'weighted': Weighted by fake scores (uses avg fake score vs threshold)
#                 - 'unanimous': All models must agree
#                 - 'average': Average of fake scores (RECOMMENDED for noisy images)
#                 - 'maximum': Use prediction from model with highest confidence
#                 - 'confidence_weighted': Weight by individual model confidence
            
#         Returns:
#             ImageDetectionResult with combined prediction
#         """
#         # Get predictions from all models using the user-provided threshold
#         individual_results = []
#         for key, detector in self.detectors.items():
#             try:
#                 # Use the threshold parameter passed by the user
#                 # This ensures consistent threshold across all models
#                 result = detector.detect(image_input, threshold)
#                 individual_results.append(result)
#             except Exception as e:
#                 logger.error(f"Error in model {key}: {e}")
        
#         if not individual_results:
#             raise RuntimeError("All models failed to process the image")
        
#         # Calculate voting
#         fake_votes = sum(1 for r in individual_results if r.is_deepfake)
#         real_votes = len(individual_results) - fake_votes
        
#         voting_summary = {
#             'fake': fake_votes,
#             'real': real_votes,
#             'total': len(individual_results)
#         }
        
#         # Calculate average scores
#         fake_scores = []
#         real_scores = []
#         for result in individual_results:
#             fake_score = result.scores.get('Fake', result.scores.get('fake', 0.0))
#             real_score = result.scores.get('Real', result.scores.get('real', 0.0))
#             fake_scores.append(fake_score)
#             real_scores.append(real_score)
        
#         avg_fake_score = sum(fake_scores) / len(fake_scores) if fake_scores else 0.0
#         avg_real_score = sum(real_scores) / len(real_scores) if real_scores else 0.0
        
#         # Determine final prediction based on voting method
#         if voting_method == 'majority':
#             is_deepfake = fake_votes > real_votes
            
#             # Calculate confidence based on models that agree with the final prediction
#             # This better represents the ensemble's certainty
#             if is_deepfake:
#                 # Get scores from models that predicted FAKE
#                 fake_model_scores = [fake_scores[i] for i in range(len(individual_results)) 
#                                     if individual_results[i].is_deepfake]
#                 confidence = np.mean(fake_model_scores) if fake_model_scores else avg_fake_score
#             else:
#                 # Get scores from models that predicted REAL
#                 real_model_scores = [real_scores[i] for i in range(len(individual_results)) 
#                                     if not individual_results[i].is_deepfake]
#                 confidence = np.mean(real_model_scores) if real_model_scores else avg_real_score
        
#         elif voting_method == 'weighted':
#             # Use average scores comparison - predict based on which score is higher
#             # More robust approach that considers both fake and real scores
#             is_deepfake = avg_fake_score > avg_real_score
            
#             # Calculate confidence based on the score that aligns with the final prediction
#             # Use the appropriate score type (fake or real) from models that agree
#             if is_deepfake:
#                 # Get FAKE scores from models that predicted FAKE
#                 fake_model_scores = [fake_scores[i] for i in range(len(individual_results)) 
#                                     if individual_results[i].is_deepfake]
#                 confidence = np.mean(fake_model_scores) if fake_model_scores else avg_fake_score
#             else:
#                 # Get REAL scores from models that predicted REAL
#                 real_model_scores = [real_scores[i] for i in range(len(individual_results)) 
#                                     if not individual_results[i].is_deepfake]
#                 confidence = np.mean(real_model_scores) if real_model_scores else avg_real_score
        
#         elif voting_method == 'unanimous':
#             # All models must agree
#             is_deepfake = fake_votes == len(individual_results)
            
#             # For unanimous, use average of all scores since all should agree
#             confidence = avg_fake_score if is_deepfake else avg_real_score
        
#         elif voting_method == 'average':
#             # Use average fake score (same as weighted for consistency)
#             is_deepfake = avg_fake_score >= threshold
            
#             # Calculate confidence based on models that agree with the final prediction
#             if is_deepfake:
#                 fake_model_scores = [fake_scores[i] for i in range(len(individual_results)) 
#                                     if individual_results[i].is_deepfake]
#                 confidence = np.mean(fake_model_scores) if fake_model_scores else avg_fake_score
#             else:
#                 real_model_scores = [real_scores[i] for i in range(len(individual_results)) 
#                                     if not individual_results[i].is_deepfake]
#                 confidence = np.mean(real_model_scores) if real_model_scores else avg_real_score
        
#         elif voting_method == 'maximum':
#             # Use prediction from model with highest confidence
#             max_confidence_result = max(individual_results, key=lambda r: r.confidence)
#             is_deepfake = max_confidence_result.is_deepfake
#             confidence = max_confidence_result.confidence
        
#         elif voting_method == 'confidence_weighted':
#             # Weight fake scores by each model's confidence
#             weighted_fake_sum = sum(fake_scores[i] * individual_results[i].confidence 
#                                    for i in range(len(individual_results)))
#             total_confidence = sum(r.confidence for r in individual_results)
#             weighted_avg_fake = weighted_fake_sum / total_confidence if total_confidence > 0 else 0.0
            
#             is_deepfake = weighted_avg_fake >= threshold
#             confidence = weighted_avg_fake
        
#         elif voting_method == 'adaptive_threshold':
#             # Adjust threshold based on prediction variance for better handling of uncertain cases
#             score_variance = np.var(fake_scores) if len(fake_scores) > 1 else 0.0
#             # Increase threshold when there's high variance (disagreement between models)
#             adjusted_threshold = threshold * (1 + score_variance)
#             is_deepfake = avg_fake_score >= adjusted_threshold
            
#             # Calculate confidence based on models that agree with the final prediction
#             if is_deepfake:
#                 fake_model_scores = [fake_scores[i] for i in range(len(individual_results)) 
#                                     if individual_results[i].is_deepfake]
#                 confidence = np.mean(fake_model_scores) if fake_model_scores else avg_fake_score
#             else:
#                 real_model_scores = [real_scores[i] for i in range(len(individual_results)) 
#                                     if not individual_results[i].is_deepfake]
#                 confidence = np.mean(real_model_scores) if real_model_scores else avg_real_score
        
#         elif voting_method == 'median':
#             # Use median instead of mean - more robust to outliers
#             median_fake_score = np.median(fake_scores) if fake_scores else 0.0
#             median_real_score = np.median(real_scores) if real_scores else 0.0
#             is_deepfake = median_fake_score >= threshold
#             confidence = median_fake_score if is_deepfake else median_real_score
        
#         elif voting_method == 'robust_majority':
#             # Remove extreme outliers before voting using IQR method
#             if len(fake_scores) >= 4:  # Need at least 4 models for robust statistics
#                 q1_fake, q3_fake = np.percentile(fake_scores, [25, 75])
#                 iqr_fake = q3_fake - q1_fake
#                 lower_bound = q1_fake - 1.5 * iqr_fake
#                 upper_bound = q3_fake + 1.5 * iqr_fake
                
#                 # Filter out outliers
#                 filtered_results = []
#                 for i, result in enumerate(individual_results):
#                     if lower_bound <= fake_scores[i] <= upper_bound:
#                         filtered_results.append(result)
                
#                 if filtered_results:
#                     filtered_fake_votes = sum(1 for r in filtered_results if r.is_deepfake)
#                     filtered_real_votes = len(filtered_results) - filtered_fake_votes
#                     is_deepfake = filtered_fake_votes > filtered_real_votes
                    
#                     # Calculate confidence from filtered results
#                     filtered_fake_scores = []
#                     filtered_real_scores = []
#                     for result in filtered_results:
#                         fake_score = result.scores.get('Fake', result.scores.get('fake', 0.0))
#                         real_score = result.scores.get('Real', result.scores.get('real', 0.0))
#                         filtered_fake_scores.append(fake_score)
#                         filtered_real_scores.append(real_score)
                    
#                     confidence = max(np.mean(filtered_fake_scores), np.mean(filtered_real_scores))
#                 else:
#                     # Fallback to regular majority if all are outliers
#                     is_deepfake = fake_votes > real_votes
#                     confidence = max(avg_fake_score, avg_real_score)
#             else:
#                 # Fallback to regular majority for small ensembles
#                 is_deepfake = fake_votes > real_votes
#                 confidence = max(avg_fake_score, avg_real_score)
        
#         elif voting_method == 'dynamic_consensus':
#             # Require higher agreement when models disagree strongly
#             total_models = len(individual_results)
#             disagreement_ratio = abs(fake_votes - real_votes) / total_models
            
#             # Require super-majority when there's high disagreement
#             if disagreement_ratio <= 0.2:  # Low disagreement
#                 required_majority = 0.5
#             elif disagreement_ratio <= 0.4:  # Moderate disagreement
#                 required_majority = 0.6
#             else:  # High disagreement
#                 required_majority = 0.75
            
#             fake_ratio = fake_votes / total_models
#             is_deepfake = fake_ratio >= required_majority
#             confidence = max(avg_fake_score, avg_real_score)
        
#         elif voting_method == 'reliability_weighted':
#             # Weight votes based on each model's estimated reliability
#             # These weights are based on general performance characteristics
#             model_weights = {
#                 'nonescape-full': 0.92,           # High reliability for general detection
#                 'efficientnet-b7-ns': 0.91,       # DFDC 3rd place solution - excellent for deepfakes
#                 'distil-dire-celebahq': 0.90,     # Excellent for face images  
#                 'distil-dire-eps-imagenet': 0.88, # Good general purpose
#                 'dima-vit': 0.85,                 # Vision transformer, good accuracy
#                 'universal-fake-v2': 0.82,        # Universal detector
#                 'seresnet50-pretrained': 0.80,    # Solid architecture
#                 'seresnet50': 0.80,               # Same architecture
#                 'prithiv-v1': 0.75,              # Can be overly sensitive
#                 'organika-sdxl': 0.70,            # Often too aggressive on real images
#             }
            
#             # Calculate weighted votes and scores
#             weighted_fake_votes = 0.0
#             weighted_total_votes = 0.0
#             weighted_fake_sum = 0.0
#             weighted_real_sum = 0.0
            
#             for i, (key, detector) in enumerate(self.detectors.items()):
#                 if i < len(individual_results):
#                     weight = model_weights.get(key, 0.75)  # Default weight
#                     result = individual_results[i]
                    
#                     weighted_total_votes += weight
#                     if result.is_deepfake:
#                         weighted_fake_votes += weight
                    
#                     fake_score = fake_scores[i] if i < len(fake_scores) else 0.0
#                     real_score = real_scores[i] if i < len(real_scores) else 0.0
#                     weighted_fake_sum += fake_score * weight
#                     weighted_real_sum += real_score * weight
            
#             if weighted_total_votes > 0:
#                 weighted_fake_ratio = weighted_fake_votes / weighted_total_votes
#                 weighted_avg_fake = weighted_fake_sum / weighted_total_votes
#                 weighted_avg_real = weighted_real_sum / weighted_total_votes
                
#                 is_deepfake = weighted_fake_ratio > 0.5
#                 confidence = weighted_avg_fake if is_deepfake else weighted_avg_real
#             else:
#                 # Fallback to simple majority
#                 is_deepfake = fake_votes > real_votes
#                 confidence = max(avg_fake_score, avg_real_score)
        
#         elif voting_method == 'bayesian':
#             # Use Bayesian inference to combine predictions
#             prior_fake = 0.5  # Prior probability of image being fake
            
#             # Calculate likelihood of observing these results given fake/real
#             log_likelihood_fake = 0.0
#             log_likelihood_real = 0.0
            
#             for i, result in enumerate(individual_results):
#                 # Use actual fake and real scores, not the confidence value
#                 fake_score = fake_scores[i] if i < len(fake_scores) else 0.0
#                 real_score = real_scores[i] if i < len(real_scores) else 0.0
                
#                 # Add log probabilities for Bayesian inference
#                 # Each model provides P(prediction|fake) and P(prediction|real)
#                 log_likelihood_fake += np.log(fake_score + 1e-10)
#                 log_likelihood_real += np.log(real_score + 1e-10)
            
#             # Convert back from log space for numerical stability
#             likelihood_fake = np.exp(log_likelihood_fake)
#             likelihood_real = np.exp(log_likelihood_real)
            
#             # Calculate posterior probability using Bayes' theorem
#             evidence = likelihood_fake * prior_fake + likelihood_real * (1 - prior_fake)
#             if evidence > 0:
#                 posterior_fake = (likelihood_fake * prior_fake) / evidence
#                 posterior_real = (likelihood_real * (1 - prior_fake)) / evidence
#             else:
#                 posterior_fake = prior_fake
#                 posterior_real = 1 - prior_fake
            
#             is_deepfake = posterior_fake >= threshold
#             # Confidence should be the higher posterior probability
#             confidence = max(posterior_fake, posterior_real)
        
#         elif voting_method == 'entropy_weighted':
#             # Weight based on prediction entropy (uncertainty)
#             weighted_fake_sum = 0.0
#             total_weight = 0.0
            
#             for i, result in enumerate(individual_results):
#                 fake_score = fake_scores[i] if i < len(fake_scores) else 0.0
#                 real_score = real_scores[i] if i < len(real_scores) else 0.0
                
#                 # Calculate entropy (lower entropy = more certain = higher weight)
#                 p_fake = max(fake_score, 1e-10)  # Avoid log(0)
#                 p_real = max(real_score, 1e-10)
#                 entropy = -(p_fake * np.log2(p_fake) + p_real * np.log2(p_real))
                
#                 # Convert entropy to weight (lower entropy = higher weight)
#                 weight = 1.0 / (1.0 + entropy)
                
#                 weighted_fake_sum += fake_score * weight
#                 total_weight += weight
            
#             if total_weight > 0:
#                 entropy_weighted_fake = weighted_fake_sum / total_weight
#                 is_deepfake = entropy_weighted_fake >= threshold
#                 # Confidence should represent certainty in the final prediction
#                 confidence = entropy_weighted_fake if is_deepfake else (1 - entropy_weighted_fake)
#             else:
#                 is_deepfake = avg_fake_score >= threshold
#                 confidence = avg_fake_score if is_deepfake else avg_real_score
        
#         elif voting_method == 'consensus_threshold':
#             # Adaptive threshold based on model consensus
#             # Higher consensus = lower threshold needed
#             max_votes = max(fake_votes, real_votes)
#             consensus_strength = max_votes / len(individual_results)
            
#             # Adjust threshold based on consensus (high consensus = lower threshold)
#             adjusted_threshold = threshold * (2 - consensus_strength)
            
#             is_deepfake = avg_fake_score >= adjusted_threshold
#             # Confidence should represent certainty in the final prediction
#             confidence = avg_fake_score if is_deepfake else avg_real_score
        
#         else:
#             raise ValueError(f"Unknown voting method: {voting_method}")
        
#         # Determine image filename
#         if isinstance(image_input, str):
#             image_file = os.path.basename(image_input)
#         else:
#             image_file = "uploaded_image"
        
#         # Convert individual results to IndividualModelResult Pydantic models
#         individual_model_results = []
#         for result in individual_results:
#             # Extract fake and real scores
#             fake_score = 0.0
#             real_score = 0.0
            
#             # Check for various fake label formats
#             for fake_key in ['artificial', 'Artificial', 'ARTIFICIAL',
#                             'Fake', 'fake', 'FAKE', 'deepfake', 'Deepfake', 'DEEPFAKE', 
#                             'generated', 'Generated', 'synthetic', 'Synthetic', 
#                             'AI', 'ai', 'LABEL_1', '1']:
#                 if fake_key in result.scores:
#                     fake_score = result.scores[fake_key]
#                     break
            
#             # Check for various real label formats
#             for real_key in ['human', 'Human', 'HUMAN',
#                             'Real', 'real', 'REAL', 'authentic', 'Authentic', 'AUTHENTIC',
#                             'genuine', 'Genuine', 'LABEL_0', '0']:
#                 if real_key in result.scores:
#                     real_score = result.scores[real_key]
#                     break
            
#             # If still no scores found, try to infer from available scores
#             if fake_score == 0.0 and real_score == 0.0 and result.scores:
#                 scores_list = list(result.scores.values())
#                 if len(scores_list) >= 2:
#                     real_score = scores_list[0]
#                     fake_score = scores_list[1]
            
#             individual_model_results.append(IndividualModelResult(
#                 model=result.model_name or "Unknown",
#                 prediction='FAKE' if result.is_deepfake else 'REAL',
#                 confidence=result.confidence,
#                 fakeScore=fake_score,
#                 realScore=real_score
#             ))
        
#         # Create VotingResult
#         voting_result = VotingResult(
#             fake=voting_summary.get('fake', 0),
#             real=voting_summary.get('real', 0),
#             total=voting_summary.get('total', len(individual_results))
#         )
        
#         # Create and return ImageDetectionResult
#         return ImageDetectionResult(
#             imageFile=image_file,
#             prediction='FAKE' if is_deepfake else 'REAL',
#             confidence=float(confidence),
#             averageFake=avg_fake_score,
#             averageReal=avg_real_score,
#             votingMethod=voting_method,
#             voting=voting_result,
#             individualModelResults=individual_model_results
#         )
    
#     def compare_models(self, 
#                       image_input: Union[str, Image.Image, np.ndarray, BinaryIO, bytes],
#                       threshold: float = 0.5) -> Dict[str, Any]:
#         """
#         Compare predictions from all models
        
#         Args:
#             image_input: Image input
#             threshold: Threshold for detection
            
#         Returns:
#             Dictionary with comparison results
#         """
#         results = []
#         for key, detector in self.detectors.items():
#             try:
#                 result = detector.detect(image_input, threshold)
#                 results.append({
#                     'model': result.model_name,
#                     'model_key': key,
#                     'is_deepfake': result.is_deepfake,
#                     'confidence': result.confidence,
#                     'scores': result.scores,
#                     'processing_time': result.processing_time
#                 })
#             except Exception as e:
#                 logger.error(f"Error in model {key}: {e}")
        
#         return {
#             'image': image_input if isinstance(image_input, str) else 'N/A',
#             'threshold': threshold,
#             'model_results': results,
#             'agreement': len(set(r['is_deepfake'] for r in results)) == 1
#         }
    
#     def analyze_directory(self,
#                          directory_path: str,
#                          extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp', '.gif','avif','webp'],
#                          threshold: float = 0.5,
#                          voting_method: str = 'majority') -> List[ImageDetectionResult]:
#         """
#         Analyze all images in a directory using ensemble
        
#         Args:
#             directory_path: Path to directory containing images
#             extensions: List of image file extensions to process
#             threshold: Threshold for deepfake classification
#             voting_method: Voting method to use
            
#         Returns:
#             List of ImageDetectionResult objects
#         """
#         directory = Path(directory_path)
        
#         if not directory.exists():
#             raise FileNotFoundError(f"Directory not found: {directory_path}")
        
#         # Find all image files
#         image_files = []
#         for ext in extensions:
#             image_files.extend(directory.glob(f"*{ext}"))
        
#         logger.info(f"Found {len(image_files)} image files in {directory_path}")
        
#         results = []
#         for i, image_file in enumerate(image_files):
#             try:
#                 logger.info(f"Processing {i+1}/{len(image_files)}: {image_file.name}")
#                 result = self.detect(str(image_file), threshold, voting_method)
#                 results.append(result)
#             except Exception as e:
#                 logger.error(f"Error processing {image_file.name}: {e}")
        
#         return results



