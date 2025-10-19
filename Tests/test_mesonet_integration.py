"""
Test script for MesoNet deepfake detection

This script demonstrates how to use the MesoNet detector for deepfake detection.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from detectionLib.detect_deepfake_mesonet import MesoNetDetector


def test_mesonet_single_image():
    """Test MesoNet on a single image"""
    print("="*60)
    print("Testing MesoNet Detector - Single Image")
    print("="*60)
    
    # Initialize detector
    model_path = "Models/mesonet/mesonet_model.h5"
    
    if not os.path.exists(model_path):
        print(f"\n⚠ Model not found at: {model_path}")
        print("Please run: python Scripts/download_mesonet_models.py")
        print("Or download manually from:")
        print("https://github.com/muffin-123/DeepFake-Detection-mesonet_model.h5-")
        return
    
    print(f"\nInitializing MesoNet detector...")
    detector = MesoNetDetector(
        model_path=model_path,
        threshold=0.5
    )
    
    # Get model info
    info = detector.get_model_info()
    print(f"\nModel Information:")
    print(f"  Name: {info['model_name']}")
    print(f"  Device: {info['device']}")
    print(f"  Input Size: {info['input_size']}x{info['input_size']}")
    print(f"  Parameters: {info['num_parameters']:,}")
    print(f"  Model Size: {info['model_size_mb']:.2f} MB")
    print(f"  Threshold: {info['threshold']}")
    
    # Test with sample image (replace with your image path)
    test_image = "path/to/your/test_image.jpg"
    
    if not os.path.exists(test_image):
        print(f"\n⚠ Test image not found: {test_image}")
        print("Please update the test_image path in the script.")
        return
    
    print(f"\nTesting on image: {test_image}")
    result = detector.detect_single(test_image)
    
    print(f"\nDetection Results:")
    print(f"  Is Fake: {result.is_fake}")
    print(f"  Fake Probability: {result.fake_probability:.2%}")
    print(f"  Real Probability: {result.real_probability:.2%}")
    print(f"  Confidence: {result.confidence:.2%}")
    print(f"  Processing Time: {result.processing_time:.3f}s")


def test_mesonet_batch():
    """Test MesoNet on multiple images"""
    print("\n" + "="*60)
    print("Testing MesoNet Detector - Batch Processing")
    print("="*60)
    
    model_path = "Models/mesonet/mesonet_model.h5"
    
    if not os.path.exists(model_path):
        print(f"\n⚠ Model not found at: {model_path}")
        print("Please run: python Scripts/download_mesonet_models.py")
        return
    
    detector = MesoNetDetector(model_path=model_path)
    
    # Test with multiple images (replace with your image paths)
    test_images = [
        "path/to/image1.jpg",
        "path/to/image2.jpg",
        "path/to/image3.jpg"
    ]
    
    # Check if images exist
    existing_images = [img for img in test_images if os.path.exists(img)]
    
    if not existing_images:
        print("\n⚠ No test images found. Please update test_images in the script.")
        return
    
    print(f"\nProcessing {len(existing_images)} images...")
    results = detector.detect_batch(existing_images)
    
    print(f"\nBatch Results:")
    for i, result in enumerate(results):
        print(f"\nImage {i+1}: {result.image_path}")
        print(f"  Prediction: {'FAKE' if result.is_fake else 'REAL'}")
        print(f"  Confidence: {result.confidence:.2%}")
        print(f"  Fake Probability: {result.fake_probability:.2%}")
        print(f"  Processing Time: {result.processing_time:.3f}s")


def test_mesonet_with_ensemble():
    """Test MesoNet integrated with ensemble detector"""
    print("\n" + "="*60)
    print("Testing MesoNet in Ensemble Detector")
    print("="*60)
    
    try:
        from detectionLib.detect_deepfake_image import EnsembleDetector
        
        # Check if MesoNet model exists
        model_path = "Models/mesonet/mesonet_model.h5"
        if not os.path.exists(model_path):
            print(f"\n⚠ MesoNet model not found at: {model_path}")
            print("Please run: python Scripts/download_mesonet_models.py")
            return
        
        print("\nInitializing ensemble with MesoNet...")
        ensemble = EnsembleDetector(
            model_keys=['mesonet', 'prithiv-v1', 'seresnet50']
        )
        
        # Test image
        test_image = "path/to/your/test_image.jpg"
        
        if not os.path.exists(test_image):
            print(f"\n⚠ Test image not found: {test_image}")
            print("Please update the test_image path in the script.")
            return
        
        print(f"\nTesting ensemble on: {test_image}")
        result = ensemble.detect(test_image, voting_method='weighted')
        
        print(f"\nEnsemble Results:")
        print(f"  Final Prediction: {'FAKE' if result.is_deepfake else 'REAL'}")
        print(f"  Confidence: {result.confidence:.2%}")
        print(f"  Average Fake Score: {result.average_fake_score:.2%}")
        print(f"  Average Real Score: {result.average_real_score:.2%}")
        print(f"  Voting Summary: {result.voting_summary}")
        
        print(f"\nIndividual Model Results:")
        for i, ind_result in enumerate(result.individual_results):
            print(f"  {ind_result.model_name}:")
            print(f"    Prediction: {'FAKE' if ind_result.is_deepfake else 'REAL'}")
            print(f"    Confidence: {ind_result.confidence:.2%}")
            print(f"    Scores: {ind_result.scores}")
        
    except ImportError as e:
        print(f"Error importing ensemble detector: {e}")


def main():
    """Main test function"""
    print("MesoNet Deepfake Detection Test Suite")
    print("="*60 + "\n")
    
    # Run tests
    try:
        test_mesonet_single_image()
    except Exception as e:
        print(f"\nError in single image test: {e}")
    
    try:
        test_mesonet_batch()
    except Exception as e:
        print(f"\nError in batch test: {e}")
    
    try:
        test_mesonet_with_ensemble()
    except Exception as e:
        print(f"\nError in ensemble test: {e}")
    
    print("\n" + "="*60)
    print("Test Suite Complete")
    print("="*60)


if __name__ == "__main__":
    main()
