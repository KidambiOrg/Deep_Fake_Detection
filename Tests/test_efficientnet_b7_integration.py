"""
Test EfficientNet-B7 Integration

This script tests the EfficientNet-B7 detector with Noisy Student weights
to ensure it's properly integrated into the ensemble.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

from detectionLib.detect_deepfake_efficientnet_b7 import EfficientNetB7Detector
from detectionLib.detect_deepfake_image import EnsembleDetector, list_available_models


def print_separator(char='=', length=80):
    """Print a separator line"""
    print(char * length)


def test_list_models():
    """Test if EfficientNet-B7 is in the available models"""
    print("\n1. CHECKING AVAILABLE MODELS")
    print_separator()
    
    models = list_available_models()
    
    if 'efficientnet-b7-ns' in models:
        print("✓ EfficientNet-B7-NS found in MODEL_REGISTRY")
        print(f"\nModel Details:")
        for key, value in models['efficientnet-b7-ns'].items():
            print(f"  {key}: {value}")
    else:
        print("✗ EfficientNet-B7-NS NOT found in MODEL_REGISTRY")
        return False
    
    return True


def test_standalone_detector():
    """Test EfficientNet-B7 detector standalone"""
    print("\n2. TESTING STANDALONE DETECTOR")
    print_separator()
    
    try:
        print("Initializing EfficientNet-B7 detector...")
        detector = EfficientNetB7Detector(
            model_path='Models/EfficientNet-B7/noisy_student_efficientnet-b7',
            debug=True
        )
        print("✓ Detector initialized successfully")
        print(f"  Model: {detector.model_name}")
        print(f"  Device: {detector.device}")
        print(f"  Input size: {detector.input_size}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to initialize detector: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ensemble_with_efficientnet():
    """Test ensemble detector with EfficientNet-B7"""
    print("\n3. TESTING ENSEMBLE WITH EFFICIENTNET-B7")
    print_separator()
    
    try:
        print("Initializing ensemble with EfficientNet-B7...")
        ensemble = EnsembleDetector(
            model_keys=['efficientnet-b7-ns'],
            debug=True
        )
        
        if 'efficientnet-b7-ns' in ensemble.detectors:
            print("✓ EfficientNet-B7-NS loaded in ensemble")
            print(f"  Total models in ensemble: {len(ensemble.detectors)}")
            return True
        else:
            print("✗ EfficientNet-B7-NS NOT loaded in ensemble")
            return False
            
    except Exception as e:
        print(f"✗ Failed to initialize ensemble: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_model_ensemble():
    """Test ensemble with multiple models including EfficientNet-B7"""
    print("\n4. TESTING MULTI-MODEL ENSEMBLE")
    print_separator()
    
    try:
        print("Initializing ensemble with multiple models...")
        model_keys = [
            'prithiv-v1',
            'efficientnet-b7-ns',
            'dima-vit'
        ]
        
        ensemble = EnsembleDetector(
            model_keys=model_keys,
            debug=True
        )
        
        print(f"\n✓ Ensemble initialized with {len(ensemble.detectors)} models")
        print("\nLoaded models:")
        for key in ensemble.detectors.keys():
            print(f"  - {key}")
        
        # Check if EfficientNet-B7 is included
        if 'efficientnet-b7-ns' in ensemble.detectors:
            print("\n✓ EfficientNet-B7-NS successfully included in ensemble")
            return True
        else:
            print("\n✗ EfficientNet-B7-NS not found in ensemble")
            return False
            
    except Exception as e:
        print(f"✗ Failed to initialize multi-model ensemble: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print_separator('=', 100)
    print("EFFICIENTNET-B7 INTEGRATION TEST")
    print_separator('=', 100)
    
    results = []
    
    # Test 1: Check if model is registered
    results.append(("Model Registration", test_list_models()))
    
    # Test 2: Standalone detector
    results.append(("Standalone Detector", test_standalone_detector()))
    
    # Test 3: Ensemble with EfficientNet-B7 only
    results.append(("Single Model Ensemble", test_ensemble_with_efficientnet()))
    
    # Test 4: Multi-model ensemble
    results.append(("Multi-Model Ensemble", test_multi_model_ensemble()))
    
    # Print summary
    print("\n" + "=" * 100)
    print("TEST SUMMARY")
    print("=" * 100)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:30s} : {status}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "=" * 100)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("EfficientNet-B7 is successfully integrated into the ensemble!")
    else:
        print("✗ SOME TESTS FAILED")
        print("Please check the errors above.")
    print("=" * 100)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
