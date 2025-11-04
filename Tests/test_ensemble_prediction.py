"""
Test imagedetectionlib ensemble to see what the FINAL prediction is.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imagedetectionlib.detect_deepfake_image import EnsembleDetector
import logging

logging.basicConfig(level=logging.INFO)

test_image = r"C:\Kidambi\WorkRelated\DeepFake\sample_images\fake_13.jpg"

print("\n" + "="*80)
print("IMAGEDETECTIONLIB ENSEMBLE TEST")
print("="*80)

# Test with all 5 models
model_keys = ['nonescape-full', 'seresnet50', 'prithiv-v1', 'dima-vit', 'organika-sdxl']

print(f"\nModels: {model_keys}")
print(f"Voting method: weighted (default)")
print(f"Threshold: 0.5 (default)")

ensemble = EnsembleDetector(model_keys=model_keys, debug=True)
result = ensemble.detect(test_image, threshold=0.5, voting_method='weighted')

print(f"\n{'='*80}")
print(f"FINAL ENSEMBLE RESULT")
print(f"{'='*80}")
print(f"Prediction: {result.prediction}")
print(f"Confidence: {result.confidence:.4f}")
print(f"Avg Fake: {result.averageFake:.4f}, Avg Real: {result.averageReal:.4f}")
print(f"Voting: {result.voting.fake} FAKE, {result.voting.real} REAL (total {result.voting.total})")
print(f"\nIndividual model results:")
for model_result in result.individualModelResults:
    print(f"  {model_result.model}: {model_result.prediction} (fake={model_result.fakeScore:.4f}, real={model_result.realScore:.4f})")

print(f"\n{'='*80}")
