"""
Test script for CNN V3 FineTuned model
This script loads the trained model and makes predictions on single images
Usage: python test_CNN_V3_FineTuned.py <image_path>
"""

import os
import sys
import tensorflow as tf
from tensorflow.keras.models import load_model
from pydantic import BaseModel, Field
import threading

# Thread-safe prediction lock
prediction_lock = threading.Lock()


class PredictionResult(BaseModel):
    """Pydantic model for prediction results"""
    image_path: str
    prediction: str = Field(..., pattern="^(fake|real)$")
    confidence: float = Field(..., ge=0.0, le=1.0)
    raw_score: float = Field(..., ge=0.0, le=1.0)


# Constants
IMAGE_SIZE = (224, 224)
MODEL_PATH = r"Models\CNN_V3_FineTuned\cnn_v3_finetuned_model.keras"


def preprocess_image(image_path: str) -> tf.Tensor:
    """
    Load and preprocess a single image using native TensorFlow operations
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Preprocessed TensorFlow tensor ready for model input
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Read image file as bytes
    img_bytes = tf.io.read_file(image_path)
    
    # Decode image (automatically detects format: JPEG, PNG, etc.)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    
    # Ensure image has correct shape
    img = tf.ensure_shape(img, [None, None, 3])
    
    # Resize to target size
    img = tf.image.resize(img, IMAGE_SIZE)
    
    # Normalize to [0, 1]
    img = tf.cast(img, tf.float32) / 255.0
    
    # Add batch dimension
    img = tf.expand_dims(img, axis=0)
    
    return img


def predict_single_image(model, image_path: str, threshold: float = 0.5) -> PredictionResult:
    """
    Make prediction on a single image (thread-safe)
    
    Args:
        model: Loaded Keras model
        image_path: Path to the image file
        threshold: Classification threshold
    
    Returns:
        PredictionResult object
    """
    img = preprocess_image(image_path)
    
    with prediction_lock:
        prediction = model.predict(img, verbose=0)
    
    raw_score = float(prediction[0][0])
    label = "real" if raw_score > threshold else "fake"
    
    result = PredictionResult(
        image_path=image_path,
        prediction=label,
        confidence=raw_score if label == "real" else (1.0 - raw_score),
        raw_score=raw_score
    )
    
    return result


def main():
    """Main testing function"""
    if len(sys.argv) < 2:
        print("Usage: python test_CNN_V3_FineTuned.py <image_path>")
        print("\nExample:")
        print("  python test_CNN_V3_FineTuned.py path/to/image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print("=== CNN V3 FineTuned Model Testing ===")
    print(f"Model path: {MODEL_PATH}")
    print(f"Image path: {image_path}")
    print()
    
    # Load model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    print("Loading model...")
    model = load_model(MODEL_PATH)
    print("Model loaded successfully")
    print()
    
    # Make prediction
    print("Making prediction...")
    result = predict_single_image(model, image_path)
    
    # Display results
    print("\n=== Prediction Result ===")
    print(f"Image: {os.path.basename(result.image_path)}")
    print(f"Prediction: {result.prediction.upper()}")
    print(f"Confidence: {result.confidence * 100:.2f}%")
    print(f"Raw Score: {result.raw_score:.4f} (>0.5 = real, <=0.5 = fake)")
    
    # Interpretation
    print("\n=== Interpretation ===")
    if result.prediction == "fake":
        print(f"⚠️  The image appears to be FAKE with {result.confidence * 100:.2f}% confidence")
    else:
        print(f"✓ The image appears to be REAL with {result.confidence * 100:.2f}% confidence")
    
    print("\n=== Testing Complete ===")


if __name__ == "__main__":
    main()
