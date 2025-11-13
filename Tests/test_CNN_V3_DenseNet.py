"""
Test script for CNN V3 Enhanced model with analysis
This version provides more detailed diagnostics and threshold optimization
Usage: python test_CNN_V3_Enhanced.py <image_path> [--threshold 0.5] [--all-models]
"""

import os
import sys
import tensorflow as tf
from tensorflow.keras.models import load_model
from pydantic import BaseModel, Field
import threading
import json
from pathlib import Path

# Thread-safe prediction lock
prediction_lock = threading.Lock()


class PredictionResult(BaseModel):
    """Pydantic model for prediction results"""
    image_path: str
    prediction: str = Field(..., pattern="^(fake|real)$")
    confidence: float = Field(..., ge=0.0, le=1.0)
    raw_score: float = Field(..., ge=0.0, le=1.0)
    threshold_used: float = Field(..., ge=0.0, le=1.0)


# Constants
IMAGE_SIZE = (224, 224)

MODEL_PATH = r"Models\CNN_V3_FineTuned\cnn_v3_finetuned_DenseNet_model.keras"


def preprocess_image(image_path: str) -> tf.Tensor:
    """Load and preprocess a single image using native TensorFlow operations"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img_bytes = tf.io.read_file(image_path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.ensure_shape(img, [None, None, 3])
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img, axis=0)
    
    return img


def predict_with_model(model, image_path: str, threshold: float = 0.3) -> PredictionResult:
    """Make prediction using specified model"""
    img = preprocess_image(image_path)
    
    with prediction_lock:
        prediction = model.predict(img, verbose=0)
    
    raw_score = float(prediction[0][0])
    label = "real" if raw_score > threshold else "fake"
    
    result = PredictionResult(
        image_path=image_path,
        prediction=label,
        confidence=raw_score if label == "real" else (1.0 - raw_score),
        raw_score=raw_score,
        threshold_used=threshold
    )
    
    return result


def test_threshold_optimization(model, image_path: str):
    """Test different thresholds to find optimal decision boundary"""
    img = preprocess_image(image_path)
    
    with prediction_lock:
        prediction = model.predict(img, verbose=0)
    
    raw_score = float(prediction[0][0])
    
    print("\n=== Threshold Optimization Analysis ===")
    print(f"Raw model output (sigmoid): {raw_score:.4f}")
    print("\nPredictions at different thresholds:")
    print("| Threshold | Prediction | Confidence |")
    print("|-----------|------------|------------|")
    
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        label = "REAL" if raw_score > threshold else "FAKE"
        confidence = raw_score if label == "REAL" else (1.0 - raw_score)
        print(f"| {threshold:.1f}      | {label:>10} | {confidence*100:>8.2f}%   |")


def compare_models(image_path: str, threshold: float = 0.5):
    """Compare predictions from both models"""
    print("\n=== Model Comparison ===")
    
    results = {}
    
    
    
    # Test enhanced model
    if os.path.exists(MODEL_PATH):
        print(f"\nLoading enhanced model: {MODEL_PATH}")
        try:
            enhanced_model = load_model(MODEL_PATH)
            enhanced_result = predict_with_model(enhanced_model, image_path, threshold)
            results['Enhanced V3'] = enhanced_result
            
            print(f"Enhanced V3 Prediction: {enhanced_result.prediction.upper()}")
            print(f"  Raw Score: {enhanced_result.raw_score:.4f}")
            print(f"  Confidence: {enhanced_result.confidence * 100:.2f}%")
        except Exception as e:
            print(f"ERROR loading enhanced model: {e}")
    
    # Determine consensus
    if len(results) == 2:
        predictions = [r.prediction for r in results.values()]
        if predictions[0] == predictions[1]:
            consensus = predictions[0].upper()
            print(f"\n✓ CONSENSUS: {consensus}")
        else:
            print(f"\n⚠️  DISAGREEMENT between models!")
            for name, result in results.items():
                print(f"  {name}: {result.prediction.upper()}")
    
    return results


def main():
    """Main testing function"""
    if len(sys.argv) < 2:
        print("Usage: python test_CNN_V3_Enhanced.py <image_path> [--threshold 0.5] [--compare]")
        print("\nExamples:")
        print("  python test_CNN_V3_Enhanced.py path/to/image.jpg")
        print("  python test_CNN_V3_Enhanced.py path/to/image.jpg --compare")
        print("  python test_CNN_V3_Enhanced.py path/to/image.jpg --threshold 0.6")
        sys.exit(1)
    
    image_path = sys.argv[1]
    threshold = 0.5
    compare_mode = False
    
    # Parse arguments
    for arg in sys.argv[2:]:
        if arg.startswith('--threshold'):
            threshold = float(arg.split('=')[-1] if '=' in arg else sys.argv[sys.argv.index(arg) + 1])
        elif arg == '--compare':
            compare_mode = True
    
    print("=" * 80)
    print("CNN V3 Enhanced Model Testing")
    print("=" * 80)
    print(f"Image: {image_path}")
    print(f"Threshold: {threshold}")
    print()
    
    if compare_mode:
        # Compare both models
        compare_models(image_path, threshold)
    else:
        # Test enhanced model only
        if not os.path.exists(MODEL_PATH):
            print(f"ERROR: Enhanced model not found at {MODEL_PATH}")
            print("Please train the enhanced model first using Train_CNN_V3_Enhanced.py")
            sys.exit(1)
        
        print("Loading enhanced model...")
        model = load_model(MODEL_PATH)
        print("Model loaded successfully\n")
        
        # Make prediction
        print("Making prediction...")
        result = predict_with_model(model, image_path, threshold)
        
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
        
        # Show threshold analysis
        test_threshold_optimization(model, image_path)
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
