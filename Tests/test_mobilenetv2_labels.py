"""
Test MobileNetV2 label mapping to determine if labels are reversed.

This script will help us understand what the model's output represents:
- Does output closer to 1.0 mean FAKE or REAL?
- Does output closer to 0.0 mean FAKE or REAL?
"""

import sys
from pathlib import Path
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def load_model():
    """Load the MobileNetV2 model"""
    model_path = project_root / "Models" / "MobileNetV2_FineTuned" / "mobilenetv2_deepfake_model.keras"
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded successfully")
    print(f"Model output shape: {model.output_shape}")
    print(f"Last layer: {model.layers[-1].get_config()}")
    return model

def preprocess_image(image_path):
    """Preprocess image matching the detection code"""
    image = Image.open(image_path).convert('RGB')
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (224, 224))
    img_normalized = img_resized / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    return img_batch

def test_image(model, image_path, expected_label):
    """Test a single image and print results"""
    print(f"\n{'='*70}")
    print(f"Testing: {image_path}")
    print(f"Expected: {expected_label}")
    print(f"{'='*70}")
    
    img_batch = preprocess_image(image_path)
    prediction = model.predict(img_batch, verbose=0)[0]
    
    # The model outputs a single value with sigmoid activation
    output_value = float(prediction[0])
    
    print(f"\nRaw model output: {output_value:.6f}")
    print(f"\nInterpretation options:")
    print(f"  Option A (current code): output = FAKE probability")
    print(f"    - Fake probability: {output_value:.4f}")
    print(f"    - Real probability: {1.0 - output_value:.4f}")
    print(f"    - Prediction: {'FAKE' if output_value > 0.5 else 'REAL'}")
    
    print(f"\n  Option B (reversed): output = REAL probability")
    print(f"    - Fake probability: {1.0 - output_value:.4f}")
    print(f"    - Real probability: {output_value:.4f}")
    print(f"    - Prediction: {'FAKE' if output_value < 0.5 else 'REAL'}")
    
    # Determine which interpretation matches expected
    current_pred = 'FAKE' if output_value > 0.5 else 'REAL'
    reversed_pred = 'FAKE' if output_value < 0.5 else 'REAL'
    
    print(f"\n✓ Expected: {expected_label}")
    print(f"  Current code predicts: {current_pred} {'✓ CORRECT' if current_pred == expected_label else '✗ WRONG'}")
    print(f"  Reversed code predicts: {reversed_pred} {'✓ CORRECT' if reversed_pred == expected_label else '✗ WRONG'}")
    
    return {
        'image': str(image_path),
        'expected': expected_label,
        'output_value': output_value,
        'current_pred': current_pred,
        'reversed_pred': reversed_pred,
        'current_correct': current_pred == expected_label,
        'reversed_correct': reversed_pred == expected_label
    }

if __name__ == "__main__":
    print("MobileNetV2 Label Mapping Test")
    print("="*70)
    
    # Load model
    model = load_model()
    
    # Test with images (you'll need to provide paths to known real/fake images)
    results = []
    
    # TODO: Add paths to test images
    # Example format:
    # test_images = [
    #     ('path/to/known_real_image.jpg', 'REAL'),
    #     ('path/to/known_fake_image.jpg', 'FAKE'),
    # ]
    
    print("\n" + "="*70)
    print("INSTRUCTIONS FOR USER:")
    print("="*70)
    print("Please test with known real and fake images from your dataset.")
    print("Update the test_images list above with actual image paths.")
    print("="*70)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nBased on the model architecture:")
    print("- Output shape: (None, 1) with sigmoid activation")
    print("- This is a binary classifier")
    print("\nCommon conventions:")
    print("1. If trained with class labels [0=REAL, 1=FAKE]:")
    print("   → output closer to 1.0 means FAKE")
    print("   → output closer to 0.0 means REAL")
    print("   → Current code is CORRECT")
    print("\n2. If trained with class labels [0=FAKE, 1=REAL]:")
    print("   → output closer to 1.0 means REAL")
    print("   → output closer to 0.0 means FAKE")
    print("   → Current code is WRONG (labels are reversed)")
    print("\nYou need to test with known images to determine which convention was used!")
