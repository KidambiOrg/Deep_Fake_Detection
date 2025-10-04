"""
Test Script for Nonescape Model Integration

This script tests the Nonescape model integration in the deepfake detection ensemble.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detectionLib.detect_deepfake_image import EnsembleDetector, list_available_models


def test_nonescape_standalone():
    """Test Nonescape detector as standalone model"""
    print("\n🔍 Testing Nonescape Standalone Detector")
    print("=" * 60)
    
    try:
        from detectionLib.detect_deepfake_nonescape import NonescapeDetector
        
        # Check if model file exists
        model_path = "nonescape-mini-v0.safetensors"
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            print("\n📥 To download the model, run:")
            print("   wget https://nonescape.sfo2.cdn.digitaloceanspaces.com/nonescape-mini-v0.safetensors")
            print("\nOr for the full model:")
            print("   wget https://nonescape.sfo2.cdn.digitaloceanspaces.com/nonescape-v0.safetensors")
            return False
        
        print(f"✓ Found model file: {model_path}")
        print("Initializing Nonescape detector...")
        
        detector = NonescapeDetector(model_path=model_path, use_mini=True)
        print(f"✓ Detector initialized: {detector.model_name}")
        
        # Get model info
        info = detector.get_model_info()
        print("\nModel Information:")
        print(f"  - Model Type: {info['model_type']}")
        print(f"  - Device: {info['device']}")
        print(f"  - Parameters: {info['num_parameters']:,}")
        print(f"  - Size: {info['model_size_mb']:.2f} MB")
        
        # Test with an image if available
        test_image = "test_image.jpg"
        if os.path.exists(test_image):
            print(f"\n🖼️  Testing with image: {test_image}")
            result = detector.detect(test_image, threshold=0.5)
            
            print("\nDetection Results:")
            print(f"  - AI Generated: {result.is_ai_generated}")
            print(f"  - Confidence: {result.confidence:.4f}")
            print(f"  - Authentic Probability: {result.authentic_prob:.4f}")
            print(f"  - AI Probability: {result.ai_prob:.4f}")
            print(f"  - Processing Time: {result.processing_time:.2f}s")
        else:
            print(f"\n⚠️  No test image found at {test_image}")
            print("   Place a test image to see detection results")
        
        return True
        
    except Exception as e:
        print(f"❌ Standalone test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_nonescape_in_ensemble():
    """Test Nonescape model in ensemble"""
    print("\n\n🎯 Testing Nonescape in Ensemble")
    print("=" * 60)
    
    try:
        # Check if model file exists
        model_path = "nonescape-mini-v0.safetensors"
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            print("   Download it first to test ensemble integration")
            return False
        
        print("Configuring Nonescape model in ensemble...")
        
        # Update MODEL_REGISTRY with model path
        from detectionLib.detect_deepfake_image import MODEL_REGISTRY
        MODEL_REGISTRY['nonescape-mini']['model_path'] = model_path
        
        print("Initializing ensemble with Nonescape + other models...")
        ensemble = EnsembleDetector(
            model_keys=['prithiv-v1', 'nonescape-mini']
        )
        
        print(f"✓ Ensemble initialized with {len(ensemble.detectors)} models:")
        for key in ensemble.detectors.keys():
            print(f"  - {key}")
        
        # Test with an image if available
        test_image = "test_image.jpg"
        if os.path.exists(test_image):
            print(f"\n🖼️  Testing ensemble with image: {test_image}")
            
            result = ensemble.detect(test_image, threshold=0.5, voting_method='majority')
            
            print("\nEnsemble Results:")
            print(f"  - Final Prediction: {'DEEPFAKE' if result.is_deepfake else 'REAL'}")
            print(f"  - Confidence: {result.confidence:.4f}")
            print(f"  - Average Fake Score: {result.average_fake_score:.4f}")
            print(f"  - Average Real Score: {result.average_real_score:.4f}")
            print(f"\nVoting Summary:")
            print(f"  - Fake votes: {result.voting_summary['fake']}")
            print(f"  - Real votes: {result.voting_summary['real']}")
            print(f"  - Total models: {result.voting_summary['total']}")
            
            print("\nIndividual Model Results:")
            for i, res in enumerate(result.individual_results, 1):
                print(f"  {i}. {res.model_name}")
                print(f"     Prediction: {'DEEPFAKE' if res.is_deepfake else 'REAL'}")
                print(f"     Confidence: {res.confidence:.4f}")
                print(f"     Scores: Real={res.scores.get('Real', 0):.4f}, Fake={res.scores.get('Fake', 0):.4f}")
        else:
            print(f"\n⚠️  No test image found at {test_image}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ensemble test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_available_models():
    """Show all available models including Nonescape"""
    print("\n📋 Available Models")
    print("=" * 60)
    
    models = list_available_models()
    
    print("\nNonescape Models:")
    for key, config in models.items():
        if config['source'] == 'nonescape':
            print(f"\n  {key}:")
            print(f"    Name: {config['name']}")
            print(f"    Description: {config['description']}")
            print(f"    Architecture: {config['architecture']}")
            print(f"    Mini Model: {config['use_mini']}")
            print(f"    Threshold: {config['recommended_threshold']}")
            if config['use_mini']:
                print(f"    Download: https://nonescape.sfo2.cdn.digitaloceanspaces.com/nonescape-mini-v0.safetensors")
            else:
                print(f"    Download: https://nonescape.sfo2.cdn.digitaloceanspaces.com/nonescape-v0.safetensors")


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("NONESCAPE MODEL INTEGRATION TESTS")
    print("=" * 60)
    
    # Show available models
    show_available_models()
    
    # Test standalone
    standalone_success = test_nonescape_standalone()
    
    # Test in ensemble
    ensemble_success = test_nonescape_in_ensemble()
    
    # Summary
    print("\n\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Standalone Test: {'✓ PASSED' if standalone_success else '✗ FAILED'}")
    print(f"Ensemble Test: {'✓ PASSED' if ensemble_success else '✗ FAILED'}")
    print()
    
    if not standalone_success and not ensemble_success:
        print("💡 TIP: Download the model file first:")
        print("   wget https://nonescape.sfo2.cdn.digitaloceanspaces.com/nonescape-mini-v0.safetensors")


if __name__ == "__main__":
    main()
