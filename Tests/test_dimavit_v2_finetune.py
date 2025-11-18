"""
Deep Fake Detection - Vision Transformer Test Script
Based on Adieee5/deepfake-detection-ViT-CrossTraining
Using Vision Transformer (ViT) architecture

This script runs in a loop, accepting image paths for deepfake detection.
Labels: 0=Real, 1=Deepfake
Type 'exit' or press Enter with empty path to quit
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# Configuration
MODEL_PATH = r"Models\dimavit_v2_FineTuned"

def load_model():
    """Load the trained model and processor"""
    print("Loading model...")
    try:
        processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
        model = ViTForImageClassification.from_pretrained(MODEL_PATH)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        print(f"✓ Model loaded successfully on {device}")
        print(f"  Model expects {model.config.num_labels} classes")
        print(f"  Label mapping: {model.config.id2label}")
        return model, processor, device
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        sys.exit(1)

def preprocess_image(image_path, processor):
    """Preprocess image for model input"""
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"  ✓ Image loaded: {image.size}")
        
        # Use processor for preprocessing
        inputs = processor(images=image, return_tensors="pt")
        return inputs, image
    except FileNotFoundError:
        print(f"  ✗ Error: Image file not found: {image_path}")
        return None, None
    except Exception as e:
        print(f"  ✗ Error loading image: {e}")
        return None, None

def predict(image_path, model, processor, device):
    """Perform inference on a single image"""
    # Preprocess image
    inputs, original_image = preprocess_image(image_path, processor)
    
    if inputs is None:
        return None, None, None
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
    
    # Get prediction
    predicted_class_idx = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_class_idx]
    confidence = probabilities[predicted_class_idx].item()
    
    # Get all probabilities
    all_probabilities = probabilities.cpu().numpy()
    
    return predicted_label, confidence, all_probabilities

def main():
    """Main function"""
    print("=" * 80)
    print("Vision Transformer Deepfake Detection - Test Script")
    print("=" * 80)
    print(f"Model: {MODEL_PATH}")
    print("=" * 80)
    
    # Load model once
    model, processor, device = load_model()
    
    print("\nReady for inference!")
    print("Enter image path to analyze (or 'exit'/empty to quit)")
    print("-" * 80)
    
    # Main loop
    while True:
        try:
            image_path = input("\nImage path: ").strip()
            
            # Check for exit conditions
            if image_path.lower() == 'exit' or image_path == '':
                print("\nExiting...")
                break
            
            # Remove quotes if present
            image_path = image_path.strip('"').strip("'")
            
            print(f"\nAnalyzing: {image_path}")
            
            # Make prediction
            predicted_label, confidence, all_probabilities = predict(image_path, model, processor, device)
            
            if predicted_label is None:
                continue
            
            # Display results
            print("\n" + "-" * 80)
            print("RESULTS")
            print("-" * 80)
            print(f"Predicted Label: {predicted_label.upper()}")
            print(f"Confidence: {confidence * 100:.2f}%")
            
            print("\nDetailed Probabilities:")
            for idx, prob in enumerate(all_probabilities):
                label = model.config.id2label[idx]
                print(f"  {label}: {prob * 100:.2f}%")
            
            # Interpretation
            print("\nInterpretation:")
            if predicted_label == 'Deepfake':
                print("  ⚠️  This image appears to be DEEPFAKE/SYNTHETIC")
            else:
                print("  ✓  This image appears to be REAL/AUTHENTIC")
            
            if confidence < 0.6:
                print("  ⚠️  Warning: Low confidence prediction. Results may be uncertain.")
            elif confidence < 0.8:
                print("  ℹ️  Note: Moderate confidence. Consider additional verification.")
            else:
                print("  ✓  High confidence prediction.")
            
            print("-" * 80)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"\n  ✗ Unexpected error: {str(e)}")
    
    print("=" * 80)
    print("Inference session ended.")
    print("=" * 80)

if __name__ == "__main__":
    main()
