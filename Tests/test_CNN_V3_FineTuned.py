"""
Test script for CNN V3 FineTuned model
This script loads the trained model and makes predictions on single images
Usage: python test_CNN_V3_FineTuned.py <image_path>
"""

import os
import sys
import cv2
import numpy as np
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


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load and preprocess a single image
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Preprocessed image array ready for model input
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    img = cv2.resize(img, IMAGE_SIZE)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    
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
