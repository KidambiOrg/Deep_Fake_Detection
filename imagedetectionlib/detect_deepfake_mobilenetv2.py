"""
MobileNetV2 Deepfake Detection Module

This module implements deepfake detection using MobileNetV2, a lightweight and efficient
architecture optimized for mobile and edge devices while maintaining high accuracy.

The model uses transfer learning from ImageNet pretrained MobileNetV2 and fine-tunes it
for binary classification (real vs fake) in deepfake detection tasks.

Key Features:
- MobileNetV2 architecture with inverted residuals and linear bottlenecks
- Efficient depthwise separable convolutions
- Pretrained on ImageNet for robust feature extraction
- Binary classification for deepfake detection
- Lightweight model suitable for deployment
- Video frame aggregation support

Reference Implementation:
Based on https://github.com/Gopika-2025/Deep_fake_detection/blob/main/scripts/video_level_aggregation.py
"""

import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import logging
import warnings
from typing import List, Tuple, Dict, Optional, Union, Any, BinaryIO
import os
import time
import tempfile
from dataclasses import dataclass
from pathlib import Path
import cv2

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class MobileNetV2Detector:
    """
    MobileNetV2 based deepfake detector
    
    This detector uses the MobileNetV2 architecture fine-tuned for deepfake detection.
    It supports both .h5 (Keras SavedModel) and .keras formats.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 threshold: float = 0.5,
                 input_size: Tuple[int, int] = (224, 224),
                 debug: bool = False):
        """
        Initialize MobileNetV2 detector
        
        Args:
            model_path: Path to pretrained model. If None, looks in default location
            threshold: Classification threshold (default: 0.5)
            input_size: Input image size (default: (224, 224))
            debug: If True, enable verbose logging
        """
        self.threshold = threshold
        self.input_size = input_size
        self.model_name = "MobileNetV2"
        self.debug = debug
        
        # Initialize model
        self.model = self._load_model(model_path)
        
        if self.debug:
            logger.info(f"MobileNetV2 detector initialized")
            logger.info(f"Input size: {self.input_size}")
    
    def _load_model(self, model_path: Optional[str] = None) -> keras.Model:
        """Load MobileNetV2 model"""
        try:
            if model_path is None:
                # Try default paths
                script_dir = Path(__file__).parent.parent
                default_paths = [
                    script_dir / "Models" / "MobileNetV2_FineTuned" / "mobilenetv2_deepfake_model.keras",
                    script_dir / "Models" / "MobileNetV2_FineTuned" / "mobilenetv2_deepfake.h5",
                    script_dir / "Models" / "MobileNetV2_FineTuned" / "mobilenetv2_finetuned.h5",
                ]
                
                for path in default_paths:
                    if path.exists():
                        model_path = str(path)
                        if self.debug:
                            logger.info(f"Found model at: {model_path}")
                        break
                
                if model_path is None:
                    raise FileNotFoundError(
                        f"MobileNetV2 model not found. Tried paths:\n" + 
                        "\n".join(f"  - {p}" for p in default_paths) +
                        "\nPlease ensure the model file exists in Models/MobileNetV2_FineTuned/"
                    )
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            if self.debug:
                logger.info(f"Loading MobileNetV2 model from: {model_path}")
            
            # Load model - supports both .h5 and .keras formats
            model = keras.models.load_model(model_path, compile=False)
            
            # Compile model with appropriate loss for binary classification
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            if self.debug:
                logger.info("✓ Model loaded successfully")
                logger.info(f"Model input shape: {model.input_shape}")
                logger.info(f"Model output shape: {model.output_shape}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load MobileNetV2 model: {e}")
            raise
    
    def _preprocess_image(self, image: Union[Image.Image, np.ndarray, str, BinaryIO]) -> np.ndarray:
        """
        Preprocess image for inference
        
        Follows the reference implementation preprocessing:
        - Resize to input_size
        - Normalize to [0, 1] range
        - Add batch dimension
        """
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
            image = pil_image
        elif isinstance(image, Image.Image):
            image = image.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Resize to model input size
        img_resized = cv2.resize(img_array, self.input_size)
        
        # Normalize to [0, 1] range (matching reference implementation)
        img_normalized = img_resized / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch
    
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
            img_batch = self._preprocess_image(image)
            
            if use_ensemble:
                # Apply multiple augmentations and average predictions
                predictions = []
                
                # Original image
                pred = self.model.predict(img_batch, verbose=0)
                predictions.append(pred)
                
                # Horizontal flip
                img_flip = np.flip(img_batch, axis=2)
                pred_flip = self.model.predict(img_flip, verbose=0)
                predictions.append(pred_flip)
                
                # Slight brightness adjustment
                img_bright = np.clip(img_batch * 1.1, 0, 1)
                pred_bright = self.model.predict(img_bright, verbose=0)
                predictions.append(pred_bright)
                
                # Average predictions
                avg_pred = np.mean(predictions, axis=0)
                prediction = avg_pred[0]
            else:
                # Single prediction
                prediction = self.model.predict(img_batch, verbose=0)[0]
            
            # Handle both binary output formats:
            # IMPORTANT: MobileNetV2 model was trained with [0=FAKE, 1=REAL] convention
            # So output closer to 1.0 means REAL, closer to 0.0 means FAKE (REVERSED from typical)
            if len(prediction.shape) == 0 or prediction.shape[0] == 1:
                # Single output: probability of REAL (reversed from typical convention)
                real_prob = float(prediction) if len(prediction.shape) == 0 else float(prediction[0])
                fake_prob = 1.0 - real_prob
            else:
                # Two outputs: [prob_real, prob_fake]
                real_prob = float(prediction[0])
                fake_prob = float(prediction[1])
            
            # Determine prediction
            is_fake = fake_prob > self.threshold
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
                    'threshold': self.threshold,
                    'ensemble': use_ensemble,
                    'input_size': self.input_size
                }
            )
            
        except Exception as e:
            logger.error(f"Error in MobileNetV2 detection: {e}")
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
    
    def detect_video_frames(self, video_path: str, frame_skip: int = 5,
                          max_frames: Optional[int] = None) -> Dict[str, Any]:
        """
        Detect deepfakes in video using frame-level aggregation
        
        Based on reference implementation's video_level_aggregation.py
        
        Args:
            video_path: Path to video file
            frame_skip: Process every Nth frame (default: 5)
            max_frames: Maximum number of frames to process
            
        Returns:
            Dictionary with video-level prediction and frame statistics
        """
        try:
            cap = cv2.VideoCapture(video_path)
            frame_idx = 0
            frames_processed = 0
            predictions = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames based on frame_skip
                if frame_idx % frame_skip == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Detect deepfake in frame
                    result = self.detect_single(frame_rgb)
                    predictions.append({
                        'frame_idx': frame_idx,
                        'fake_prob': result.fake_probability,
                        'real_prob': result.real_probability,
                        'is_fake': result.is_fake
                    })
                    
                    frames_processed += 1
                    
                    # Stop if max_frames reached
                    if max_frames and frames_processed >= max_frames:
                        break
                
                frame_idx += 1
            
            cap.release()
            
            if not predictions:
                return {
                    'error': 'No frames processed',
                    'video_path': video_path
                }
            
            # Aggregate predictions
            fake_probs = [p['fake_prob'] for p in predictions]
            real_probs = [p['real_prob'] for p in predictions]
            
            avg_fake_prob = np.mean(fake_probs)
            avg_real_prob = np.mean(real_probs)
            
            # Video-level prediction
            is_fake_video = avg_fake_prob > self.threshold
            
            return {
                'video_path': video_path,
                'total_frames': frame_idx,
                'frames_processed': frames_processed,
                'frame_skip': frame_skip,
                'prediction': 'FAKE' if is_fake_video else 'REAL',
                'confidence': max(avg_fake_prob, avg_real_prob),
                'avg_fake_probability': float(avg_fake_prob),
                'avg_real_probability': float(avg_real_prob),
                'frame_predictions': predictions
            }
            
        except Exception as e:
            logger.error(f"Error in video detection: {e}")
            return {
                'error': str(e),
                'video_path': video_path
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        total_params = self.model.count_params()
        
        return {
            'model_name': self.model_name,
            'architecture': 'MobileNetV2',
            'input_size': f"{self.input_size[0]}x{self.input_size[1]}",
            'num_classes': 2,
            'total_parameters': total_params,
            'framework': 'TensorFlow/Keras',
            'threshold': self.threshold
        }

# Factory function for easy initialization
def create_mobilenetv2_detector(model_path: Optional[str] = None,
                               threshold: float = 0.5,
                               input_size: Tuple[int, int] = (224, 224)) -> MobileNetV2Detector:
    """
    Factory function to create MobileNetV2 detector
    
    Args:
        model_path: Path to pretrained model
        threshold: Classification threshold
        input_size: Input image size
        
    Returns:
        MobileNetV2Detector instance
    """
    return MobileNetV2Detector(
        model_path=model_path,
        threshold=threshold,
        input_size=input_size
    )

if __name__ == "__main__":
    # Example usage
    detector = create_mobilenetv2_detector()
    print("MobileNetV2 detector created successfully!")
    print(f"Model info: {detector.get_model_info()}")
