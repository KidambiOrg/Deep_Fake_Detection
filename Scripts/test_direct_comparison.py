"""
Direct comparison: Agent tools vs imagedetectionlib for nonescape and seresnet50
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test nonescape
print("\n" + "="*80)
print("TESTING NONESCAPE")
print("="*80)

test_image = r"C:\Kidambi\WorkRelated\DeepFake\sample_images\fake_13.jpg"

print("\n--- Agent Tool ---")
from Agents.ImageAgents.tools.nonescape_detect_tool import detect_with_nonescape_tool
result = detect_with_nonescape_tool(test_image)
print(f"Prediction: {result['prediction']}")
print(f"Fake: {result['fakeScore']:.4f}, Real: {result['realScore']:.4f}")

print("\n--- imagedetectionlib ---")
from imagedetectionlib.detect_deepfake_nonescape import NonescapeDetector
detector = NonescapeDetector(model_path='Models/nonescape-v0.safetensors', use_mini=False, debug=False)
result = detector.detect(test_image)
print(f"Prediction: {'FAKE' if result.is_ai_generated else 'REAL'}")
print(f"Fake: {result.ai_prob:.4f}, Real: {result.authentic_prob:.4f}")

print("\n" + "="*80)
print("TESTING SERESNET50")
print("="*80)

print("\n--- Agent Tool ---")
from Agents.ImageAgents.tools.seresnet50_detect_tool import detect_with_seresnet50_tool
result = detect_with_seresnet50_tool(test_image)
print(f"Prediction: {result['prediction']}")
print(f"Fake: {result['fakeScore']:.4f}, Real: {result['realScore']:.4f}")

print("\n--- imagedetectionlib ---")
from imagedetectionlib.detect_deepfake_seresnet50 import SEResNet50Detector
detector = SEResNet50Detector(model_path='Models/SERESNet_50/seresnet50_pretrained.pth', debug=False)
result = detector.detect_single(test_image, use_ensemble=False)
print(f"Prediction: {'FAKE' if result.is_fake else 'REAL'}")
print(f"Fake: {result.fake_probability:.4f}, Real: {result.real_probability:.4f}")

print("\n" + "="*80)
