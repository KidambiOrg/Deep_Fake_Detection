"""
Test to see if image enhancement affects predictions.
This will test predictions WITH and WITHOUT image enhancement.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imagedetectionlib.detect_deepfake_image import DeepfakeImageDetector
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor

# Test image
test_image_path = r"C:\Kidambi\WorkRelated\DeepFake\sample_images\fake_13.jpg"

def enhance_image_like_imagedetectionlib(image_path):
    """Apply same enhancement as imagedetectionlib"""
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image)
    
    # Calculate blur score
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_blurred = laplacian_var < 100
    
    print(f"Blur score: {laplacian_var:.2f}, Is blurred: {is_blurred}")
    
    # 1. Denoise
    denoised = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
    enhanced = Image.fromarray(denoised)
    
    # 2. Sharpen if needed
    if is_blurred or laplacian_var < 150:
        if laplacian_var < 50:
            sharpness_factor = 2.5
        elif laplacian_var < 100:
            sharpness_factor = 2.0
        else:
            sharpness_factor = 1.5
        
        enhanced_array = np.array(enhanced)
        gaussian = cv2.GaussianBlur(enhanced_array, (0, 0), 2.0)
        enhanced_array = cv2.addWeighted(enhanced_array, 1.5, gaussian, -0.5, 0)
        enhanced = Image.fromarray(np.clip(enhanced_array, 0, 255).astype(np.uint8))
        
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(sharpness_factor)
    
    # 3. Enhance contrast
    contrast_enhancer = ImageEnhance.Contrast(enhanced)
    enhanced = contrast_enhancer.enhance(1.2)
    
    # 4. Enhance color
    color_enhancer = ImageEnhance.Color(enhanced)
    enhanced = color_enhancer.enhance(1.1)
    
    # 5. Edge enhancement
    enhanced = enhanced.filter(ImageFilter.EDGE_ENHANCE)
    
    return enhanced

def test_model_with_and_without_enhancement(model_name, model_path, model_labels):
    """Test a model with original and enhanced images"""
    print(f"\n{'='*80}")
    print(f"Testing: {model_name}")
    print(f"Path: {model_path}")
    print(f"{'='*80}")
    
    # Load model
    model = AutoModelForImageClassification.from_pretrained(model_path)
    processor = AutoImageProcessor.from_pretrained(model_path)
    model.eval()
    
    # Test 1: Original image (like agent tools)
    print("\n--- WITHOUT Enhancement (Agent Tool) ---")
    image = Image.open(test_image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
    
    scores_orig = {model_labels[i]: float(probs[i]) for i in range(len(model_labels))}
    print(f"Scores: {scores_orig}")
    
    # Test 2: Enhanced image (like imagedetectionlib)
    print("\n--- WITH Enhancement (imagedetectionlib) ---")
    enhanced = enhance_image_like_imagedetectionlib(test_image_path)
    
    inputs = processor(images=enhanced, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
    
    scores_enhanced = {model_labels[i]: float(probs[i]) for i in range(len(model_labels))}
    print(f"Scores: {scores_enhanced}")
    
    # Compare
    print("\n--- Comparison ---")
    for label in model_labels.values():
        diff = scores_enhanced[label] - scores_orig[label]
        print(f"{label}: {scores_orig[label]:.4f} → {scores_enhanced[label]:.4f} (Δ {diff:+.4f})")

print("\n" + "="*80)
print("IMAGE ENHANCEMENT IMPACT TEST")
print("="*80)

# Test the models that are failing
models_to_test = [
    ("dima-vit", "Models/dimavit", {0: 'Real', 1: 'Fake'}),
    ("organika-sdxl", "Models/huggingface/Organika/sdxl-detector", {0: 'artificial', 1: 'human'}),
]

for model_name, model_path, labels in models_to_test:
    test_model_with_and_without_enhancement(model_name, model_path, labels)

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
