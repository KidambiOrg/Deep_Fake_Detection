"""
Debug why seresnet50 agent tool predicts differently
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Agents.ImageAgents.tools.seresnet50_detect_tool import detect_with_seresnet50_tool

test_image = r"C:\Kidambi\WorkRelated\DeepFake\sample_images\fake_13.jpg"

print("Testing SEResNet-50 Agent Tool")
print(f"Image: {os.path.basename(test_image)}")
print("-" * 80)

result = detect_with_seresnet50_tool(test_image)

print(f"\nResult: {result}")
print(f"\nParsed:")
print(f"  Model: {result['model']}")
print(f"  Prediction: {result['prediction']}")
print(f"  Confidence: {result['confidence']:.4f}")
print(f"  Fake: {result['fakeScore']:.4f}")
print(f"  Real: {result['realScore']:.4f}")
