"""
Deep Fake Detection - Vision Transformer Inference Script
Based on Adieee5/deepfake-detection-ViT-CrossTraining
Using Vision Transformer (ViT) architecture: google/vit-base-patch16-224-in21k

This script runs in a loop, accepting image paths for deepfake detection.
- Labels: 0=Real, 1=Deepfake
- Type 'exit' or press Enter with empty path to quit
"""

import warnings
warnings.filterwarnings("ignore")

import os
import torch
from pathlib import Path
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch.nn.functional as F

# Configuration
MODEL_PATH = r"Models\dimavit_v2_FineTuned"
labels_list = ['Real', 'Deepfake']

print("=" * 80)
print("Vision Transformer Deepfake Detection - Inference Script")
print("=" * 80)
print(f"Loading model from: {MODEL_PATH}")

# Load model and processor
model = ViTForImageClassification.from_pretrained(MODEL_PATH)
processor = ViTImageProcessor.from_pretrained(MODEL_PATH)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print(f"Model loaded successfully on {device}")
print(f"Label mapping: Real=0, Deepfake=1")
print("=" * 80)

# Setup preprocessing transforms
image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]

_test_transforms = Compose([
    Resize((size, size)),
    ToTensor(),
    Normalize(mean=image_mean, std=image_std)
])

def predict_image(image_path):
    """Predict if an image is real or deepfake"""
    try:
        # Load and preprocess image
        img = Image.open(image_path).convert("RGB")
        pixel_values = _test_transforms(img).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(pixel_values)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            predicted_class = logits.argmax(dim=1).item()
        
        # Get results
        predicted_label = labels_list[predicted_class]
        confidence = probs[0][predicted_class].item() * 100
        
        print(f"\n  Result: {predicted_label}")
        print(f"  Confidence: {confidence:.2f}%")
        print(f"  Probabilities - Real: {probs[0][0].item()*100:.2f}%, Deepfake: {probs[0][1].item()*100:.2f}%")
        
    except FileNotFoundError:
        print(f"\n  Error: Image file not found: {image_path}")
    except Exception as e:
        print(f"\n  Error processing image: {str(e)}")

# Main loop
print("\nReady for inference!")
print("Enter image path to analyze (or 'exit'/empty to quit)")
print("-" * 80)

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
        predict_image(image_path)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        break
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")

print("=" * 80)
print("Inference session ended.")
print("=" * 80)
