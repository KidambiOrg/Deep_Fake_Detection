"""
Debug seresnet50 to see exact probability values
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

test_image = r"C:\Kidambi\WorkRelated\DeepFake\sample_images\fake_13.jpg"

print("\n" + "="*80)
print("DEBUG SERESNET50")
print("="*80)

from imagedetectionlib.detect_deepfake_seresnet50 import SEResNet50Detector
import torch

detector = SEResNet50Detector(model_path='Models/SERESNet_50/seresnet50_pretrained.pth', debug=True)

# Load and preprocess image manually
from PIL import Image
from torchvision import transforms

image = Image.open(test_image).convert('RGB')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

tensor = transform(image).unsqueeze(0).to(detector.device)

# Run model
with torch.no_grad():
    logits = detector.model(tensor)
    probs = torch.nn.functional.softmax(logits, dim=1)
    print(f"\nLogits shape: {logits.shape}")
    print(f"Logits (first 10): {logits[0][:10].cpu().numpy()}")
    print(f"\nProbs shape: {probs.shape}")
    print(f"Probs (first 10): {probs[0][:10].cpu().numpy()}")
    print(f"\nProb[0] (fake): {probs[0][0].item():.6f}")
    print(f"Prob[1] (real): {probs[0][1].item():.6f}")
    print(f"Sum of first 2: {(probs[0][0] + probs[0][1]).item():.6f}")
    
    # Now check with detect_single
    print("\n--- Using detect_single ---")
    result = detector.detect_single(test_image, use_ensemble=False)
    print(f"Fake prob: {result.fake_probability:.6f}")
    print(f"Real prob: {result.real_probability:.6f}")
    print(f"Prediction: {'FAKE' if result.is_fake else 'REAL'}")
