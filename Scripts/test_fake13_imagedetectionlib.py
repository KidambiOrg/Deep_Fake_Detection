"""
Test imagedetectionlib directly with fake_13.jpg
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imagedetectionlib.detect_deepfake_image import EnsembleDetector

test_image = r"C:\Kidambi\WorkRelated\DeepFake\sample_images\fake_13.jpg"

print(f"Testing imagedetectionlib with: {os.path.basename(test_image)}")
print("="*80)

ensemble = EnsembleDetector(
    model_keys=['nonescape-full', 'seresnet50', 'prithiv-v1', 'dima-vit', 'organika-sdxl'],
    debug=False
)
result = ensemble.detect(test_image, threshold=0.5, voting_method='weighted')

print(f"\nFinal Prediction: {result.prediction}")
print(f"Confidence: {result.confidence:.4f}")
print(f"Avg Fake: {result.averageFake:.4f}, Avg Real: {result.averageReal:.4f}")
print(f"Voting: {result.voting.fake} FAKE, {result.voting.real} REAL")
print(f"\nIndividual models:")
for model in result.individualModelResults:
    print(f"  {model.model}: {model.prediction} (fake={model.fakeScore:.4f}, real={model.realScore:.4f})")
