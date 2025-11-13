# """
# CNN V3 FineTuned Deepfake Detection Module

# This module provides direct access to the CNN V3 FineTuned model for deepfake detection
# without using the agent framework. It's useful for batch processing and integration
# into other systems.
# """

# import os
# import sys
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from pydantic import BaseModel, Field
# import threading
# from pathlib import Path
# import logging
# import numpy as np
# from PIL import Image
# import cv2

# logger = logging.getLogger(__name__)

# # Thread-safe prediction lock
# prediction_lock = threading.Lock()

# # Global model cache
# _cnn_v3_model = None


# class PredictionResult(BaseModel):
#     """Pydantic model for prediction results"""
#     image_path: str
#     prediction: str = Field(..., pattern="^(FAKE|REAL)$")
#     confidence: float = Field(..., ge=0.0, le=1.0)
#     raw_score: float = Field(..., ge=0.0, le=1.0)


# # Constants
# IMAGE_SIZE = (224, 224)
# MODEL_PATH = r"Models\CNN_V3_FineTuned\cnn_v3_finetuned_model.keras"


# def load_cnn_v3_model(model_path: str | None = None):
#     """
#     Load CNN V3 FineTuned model (thread-safe)
    
#     Args:
#         model_path: Optional path to model file. If None, uses default path.
        
#     Returns:
#         Loaded Keras model
#     """
#     global _cnn_v3_model
    
#     with prediction_lock:
#         if _cnn_v3_model is not None:
#             return _cnn_v3_model
        
#         if model_path is None:
#             # Use default path
#             script_dir = Path(__file__).parent.parent
#             model_path = str(script_dir / MODEL_PATH)
        
#         if not os.path.exists(model_path):
#             raise FileNotFoundError(f"Model not found: {model_path}")
        
#         logger.info(f"Loading CNN V3 FineTuned model from: {model_path}")
#         model = load_model(model_path, compile=False)
        
#         # Compile model
#         model.compile(
#             optimizer='adam',
#             loss='binary_crossentropy',
#             metrics=['accuracy']
#         )
        
#         _cnn_v3_model = model
#         logger.info(f"✓ Loaded CNN V3 FineTuned model successfully")
#         logger.info(f"Model input shape: {model.input_shape}")
#         logger.info(f"Model output shape: {model.output_shape}")
        
#         return model


# def preprocess_image(image_path: str) -> tf.Tensor:
#     """
#     Load and preprocess a single image using native TensorFlow operations
    
#     Args:
#         image_path: Path to the image file
    
#     Returns:
#         Preprocessed TensorFlow tensor ready for model input
#     """
#     if not os.path.exists(image_path):
#         raise FileNotFoundError(f"Image not found: {image_path}")
    
#     # Read image file as bytes
#     img_bytes = tf.io.read_file(image_path)
    
#     # Decode image (automatically detects format: JPEG, PNG, etc.)
#     img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    
#     # Ensure image has correct shape
#     img = tf.ensure_shape(img, [None, None, 3])
    
#     # Resize to target size
#     img = tf.image.resize(img, IMAGE_SIZE)
    
#     # Normalize to [0, 1]
#     img = tf.cast(img, tf.float32) / 255.0
    
#     # Add batch dimension
#     img = tf.expand_dims(img, axis=0)
    
#     return img


# def preprocess_image_numpy(image_path: str) -> np.ndarray:
#     """
#     Load and preprocess a single image using NumPy/OpenCV
    
#     Alternative preprocessing method using NumPy for compatibility
    
#     Args:
#         image_path: Path to the image file
    
#     Returns:
#         Preprocessed numpy array ready for model input
#     """
#     from PIL import ImageFile
#     ImageFile.LOAD_TRUNCATED_IMAGES = True
    
#     # Load image from file path
#     image = Image.open(image_path)
#     image.load()
    
#     # Convert to RGB if needed
#     if image.mode != 'RGB':
#         image = image.convert('RGB')
    
#     # Convert PIL to numpy
#     img_array = np.array(image)  # [H, W, C]
    
#     # Resize to 224x224 (model input size)
#     img_resized = cv2.resize(img_array, IMAGE_SIZE)
    
#     # Normalize to [0, 1] range
#     img_normalized = img_resized / 255.0
    
#     # Add batch dimension
#     img_batch = np.expand_dims(img_normalized, axis=0)
    
#     return img_batch


# def predict_single_image(model, image_path: str, threshold: float = 0.5, use_numpy: bool = False) -> PredictionResult:
#     """
#     Make prediction on a single image (thread-safe)
    
#     Args:
#         model: Loaded Keras model
#         image_path: Path to the image file
#         threshold: Classification threshold (>threshold = REAL)
#         use_numpy: If True, use NumPy preprocessing instead of TensorFlow
    
#     Returns:
#         PredictionResult object
#     """
#     if use_numpy:
#         img = preprocess_image_numpy(image_path)
#     else:
#         img = preprocess_image(image_path)
    
#     with prediction_lock:
#         prediction = model.predict(img, verbose=0)
    
#     raw_score = float(prediction[0][0])
#     label = "REAL" if raw_score > threshold else "FAKE"
    
#     result = PredictionResult(
#         image_path=image_path,
#         prediction=label,
#         confidence=raw_score if label == "REAL" else (1.0 - raw_score),
#         raw_score=raw_score
#     )
    
#     return result


# def detect_deepfake(image_path: str, model_path: str | None = None, threshold: float = 0.5, use_numpy: bool = False) -> PredictionResult:
#     """
#     Detect deepfake in a single image
    
#     Args:
#         image_path: Path to the image file
#         model_path: Optional path to model file
#         threshold: Classification threshold (>threshold = REAL)
#         use_numpy: If True, use NumPy preprocessing instead of TensorFlow
        
#     Returns:
#         PredictionResult object
#     """
#     model = load_cnn_v3_model(model_path)
#     return predict_single_image(model, image_path, threshold, use_numpy)


# def detect_batch(image_paths: list, model_path: str | None = None, threshold: float = 0.5, use_numpy: bool = False) -> list:
#     """
#     Detect deepfakes in multiple images
    
#     Args:
#         image_paths: List of image file paths
#         model_path: Optional path to model file
#         threshold: Classification threshold (>threshold = REAL)
#         use_numpy: If True, use NumPy preprocessing instead of TensorFlow
        
#     Returns:
#         List of PredictionResult objects
#     """
#     model = load_cnn_v3_model(model_path)
#     results = []
    
#     for image_path in image_paths:
#         try:
#             result = predict_single_image(model, image_path, threshold, use_numpy)
#             results.append(result)
#         except Exception as e:
#             logger.error(f"Error processing {image_path}: {e}")
#             # Add error result
#             results.append(PredictionResult(
#                 image_path=image_path,
#                 prediction="REAL",
#                 confidence=0.0,
#                 raw_score=0.0
#             ))
    
#     return results


# def main():
#     """Main testing function"""
#     if len(sys.argv) < 2:
#         print("Usage: python detect_deepfake_cnn_v3.py <image_path>")
#         print("\nExample:")
#         print("  python detect_deepfake_cnn_v3.py path/to/image.jpg")
#         sys.exit(1)
    
#     image_path = sys.argv[1]
    
#     print("=== CNN V3 FineTuned Model Testing ===")
#     print(f"Image path: {image_path}")
#     print()
    
#     # Make prediction
#     print("Making prediction...")
#     result = detect_deepfake(image_path)
    
#     # Display results
#     print("\n=== Prediction Result ===")
#     print(f"Image: {os.path.basename(result.image_path)}")
#     print(f"Prediction: {result.prediction}")
#     print(f"Confidence: {result.confidence * 100:.2f}%")
#     print(f"Raw Score: {result.raw_score:.4f} (>0.5 = REAL, <=0.5 = FAKE)")
    
#     # Interpretation
#     print("\n=== Interpretation ===")
#     if result.prediction == "FAKE":
#         print(f"⚠️  The image appears to be FAKE with {result.confidence * 100:.2f}% confidence")
#     else:
#         print(f"✓ The image appears to be REAL with {result.confidence * 100:.2f}% confidence")
    
#     print("\n=== Testing Complete ===")


# if __name__ == "__main__":
#     main()
