"""
Direct comparison: What does imagedetectionlib predict vs what probabilities does it get?
This will show us the EXACT logic flow for dima-vit and organika-sdxl.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imagedetectionlib.detect_deepfake_image import DeepfakeImageDetector
import logging

# Enable debug logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test image that should be FAKE
test_image = r"C:\Kidambi\WorkRelated\DeepFake\sample_images\fake_13.jpg"

print("\n" + "="*80)
print("IMAGEDETECTIONLIB DETECTION TEST")
print("="*80)

for model_key in ['prithiv-v1', 'dima-vit', 'organika-sdxl']:
    print(f"\n{'='*80}")
    print(f"Testing: {model_key}")
    print(f"{'='*80}")
    
    try:
        detector = DeepfakeImageDetector(model_name=model_key, debug=True)
        result = detector.detect(test_image)
        
        print(f"\n✓ Detection complete")
        print(f"Prediction: {result.is_deepfake} ({'FAKE' if result.is_deepfake else 'REAL'})")
        print(f"Confidence: {result.confidence:.4f}")
        print(f"Scores: {result.scores}")
        print(f"Processing time: {result.processing_time:.3f}s")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
