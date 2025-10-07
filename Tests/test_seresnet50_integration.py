#!/usr/bin/env python3
"""
Test script for SEResNet-50 deepfake detection integration

This script tests the newly integrated SEResNet-50 model to ensure it works properly
with the existing deepfake detection system.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from detectionLib.detect_deepfake_seresnet50 import create_seresnet50_detector
from detectionLib.detect_deepfake_image import list_available_models, EnsembleDetector
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_seresnet50_standalone():
    """Test SEResNet-50 detector standalone"""
    print("\n" + "="*60)
    print("Testing SEResNet-50 Detector (Standalone)")
    print("="*60)
    
    try:
        # Create SEResNet-50 detector
        detector = create_seresnet50_detector()
        print(f"✓ SEResNet-50 detector created successfully")
        
        # Get model info
        model_info = detector.get_model_info()
        print(f"✓ Model info: {model_info}")
        
        print("✓ SEResNet-50 standalone test passed!")
        return True
        
    except Exception as e:
        print(f"❌ SEResNet-50 standalone test failed: {e}")
        return False

def test_seresnet50_in_registry():
    """Test if SEResNet-50 models are available in the model registry"""
    print("\n" + "="*60)
    print("Testing SEResNet-50 in Model Registry")
    print("="*60)
    
    try:
        models = list_available_models()
        
        seresnet50_models = [k for k, v in models.items() if v.get('source') == 'seresnet50']
        
        if seresnet50_models:
            print(f"✓ Found SEResNet-50 models in registry: {seresnet50_models}")
            for model in seresnet50_models:
                info = models[model]
                print(f"  - {model}: {info['description']}")
            return True
        else:
            print("❌ No SEResNet-50 models found in registry")
            return False
            
    except Exception as e:
        print(f"❌ Registry test failed: {e}")
        return False

def test_seresnet50_in_ensemble():
    """Test SEResNet-50 integration with ensemble detector"""
    print("\n" + "="*60)
    print("Testing SEResNet-50 in Ensemble Detector")
    print("="*60)
    
    try:
        # Try to create ensemble with SEResNet-50
        ensemble = EnsembleDetector(model_keys=['seresnet50-pretrained'])
        print(f"✓ Ensemble with SEResNet-50 created successfully")
        print(f"✓ Number of loaded models: {len(ensemble.detectors)}")
        
        if 'seresnet50-pretrained' in ensemble.detectors:
            print("✓ SEResNet-50 successfully loaded in ensemble")
            return True
        else:
            print("❌ SEResNet-50 not found in loaded ensemble detectors")
            return False
            
    except Exception as e:
        print(f"❌ Ensemble test failed: {e}")
        return False

def test_download_script_exists():
    """Test if the download script exists and is executable"""
    print("\n" + "="*60)
    print("Testing Download Script")
    print("="*60)
    
    script_path = project_root / "Scripts" / "download_seresnet50_models.py"
    
    if script_path.exists():
        print(f"✓ Download script found at: {script_path}")
        return True
    else:
        print(f"❌ Download script not found at: {script_path}")
        return False

def test_model_directory():
    """Test if the model directory exists"""
    print("\n" + "="*60)
    print("Testing Model Directory Structure")
    print("="*60)
    
    model_dir = project_root / "Models" / "SERESNet_50"
    
    if model_dir.exists():
        print(f"✓ SEResNet-50 model directory found at: {model_dir}")
        return True
    else:
        print(f"❌ SEResNet-50 model directory not found at: {model_dir}")
        return False

def main():
    """Run all tests"""
    print("Starting SEResNet-50 Integration Tests")
    print("="*80)
    
    tests = [
        ("Model Directory", test_model_directory),
        ("Download Script", test_download_script_exists),
        ("Model Registry", test_seresnet50_in_registry),
        ("Standalone Detector", test_seresnet50_standalone),
        ("Ensemble Integration", test_seresnet50_in_ensemble),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! SEResNet-50 integration is successful!")
        print("\nNext steps:")
        print("1. Run the download script to get the pretrained model:")
        print("   python Scripts/download_seresnet50_models.py")
        print("2. Test with actual images using the main app:")
        print("   python app_detect_image.py")
        return True
    else:
        print(f"\n❌ {total - passed} tests failed. Please check the integration.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)