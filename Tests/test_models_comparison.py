"""
Comprehensive Model Comparison and Ensemble Testing

This script demonstrates:
1. How to list available models
2. How to test individual models
3. How to compare multiple models side-by-side
4. How to use ensemble voting for improved accuracy
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to allow importing detectionLib
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

from detectionLib.detect_deepfake_image import (
    DeepfakeImageDetector, 
    EnsembleDetector,
    list_available_models
)

def print_separator(char='=', length=80):
    print(char * length)

def test_list_models():
    """List all available models"""
    print("\n1. AVAILABLE DEEPFAKE DETECTION MODELS")
    print_separator()
    
    models = list_available_models()
    
    for key, info in models.items():
        print(f"\nModel Key: {key}")
        print(f"  Name: {info['name']}")
        print(f"  Description: {info['description']}")
        print(f"  Architecture: {info['architecture']}")
        print(f"  Recommended Threshold: {info['recommended_threshold']}")

def test_single_model(image_path, model_key='prithiv-v1', threshold=0.5):
    """Test a single model"""
    print(f"\n2. TESTING SINGLE MODEL: {model_key}")
    print_separator()
    
    try:
        detector = DeepfakeImageDetector(model_name=model_key)
        
        info = detector.get_model_info()
        print(f"\nModel Info:")
        print(f"  Name: {info['model_name']}")
        print(f"  Device: {info['device']}")
        print(f"  Labels: {info['labels']}")
        print(f"  Size: {info['model_size_mb']:.2f} MB")
        
        print(f"\nTesting image: {image_path}")
        result = detector.detect(image_path, threshold=threshold)
        
        print(f"\nResults:")
        print(f"  Is Deepfake: {result.is_deepfake}")
        print(f"  Confidence: {result.confidence:.4f} ({result.confidence*100:.2f}%)")
        print(f"  Processing Time: {result.processing_time:.3f}s")
        
        print(f"\nAll Scores:")
        for label, score in sorted(result.scores.items(), key=lambda x: x[1], reverse=True):
            print(f"  {label:20s}: {score:.6f} ({score*100:.2f}%)")
        
        print(f"\nRecommendation:")
        print(f"  {result.get_recommendation(threshold)}")
        
        return result
        
    except Exception as e:
        print(f"Error: {e}")
        return None

def test_model_comparison(image_path, model_keys=None, threshold=0.5):
    """Compare multiple models on the same image"""
    print(f"\n3. COMPARING MULTIPLE MODELS")
    print_separator()
    
    if model_keys is None:
        model_keys = ['prithiv-v1', 'dima-vit']
    
    print(f"Models to compare: {model_keys}")
    print(f"Image: {image_path}")
    print(f"Threshold: {threshold}\n")
    
    results = []
    
    for key in model_keys:
        print(f"\nTesting model: {key}...")
        try:
            detector = DeepfakeImageDetector(model_name=key)
            result = detector.detect(image_path, threshold=threshold)
            results.append({
                'model_key': key,
                'model_name': detector.model_name,
                'is_deepfake': result.is_deepfake,
                'confidence': result.confidence,
                'scores': result.scores,
                'processing_time': result.processing_time
            })
            print(f"  ✓ Complete - Deepfake: {result.is_deepfake}, Confidence: {result.confidence:.4f}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # Print comparison table
    print(f"\n{'='*100}")
    print(f"COMPARISON RESULTS")
    print(f"{'='*100}")
    print(f"{'Model':<40} {'Prediction':<15} {'Confidence':<15} {'Time (s)':<10}")
    print(f"{'-'*100}")
    
    for r in results:
        prediction = "DEEPFAKE" if r['is_deepfake'] else "REAL"
        print(f"{r['model_key']:<40} {prediction:<15} {r['confidence']:<15.4f} {r['processing_time']:<10.3f}")
    
    print(f"{'-'*100}")
    
    # Check agreement
    predictions = set(r['is_deepfake'] for r in results)
    if len(predictions) == 1:
        print(f"\n✓ All models AGREE on the prediction")
    else:
        print(f"\n✗ Models DISAGREE - Consider using ensemble voting")
        fake_count = sum(1 for r in results if r['is_deepfake'])
        real_count = len(results) - fake_count
        print(f"  Fake votes: {fake_count}, Real votes: {real_count}")
    
    return results

def test_ensemble_detection(image_path, model_keys=None, threshold=0.5):
    """Test ensemble detection with multiple voting methods"""
    print(f"\n4. ENSEMBLE DETECTION")
    print_separator()
    
    if model_keys is None:
        model_keys = ['prithiv-v1']  # Start with one model, add more if available
    
    print(f"Initializing ensemble with models: {model_keys}")
    
    try:
        ensemble = EnsembleDetector(model_keys=model_keys)
        
        print(f"\nTesting image: {image_path}")
        print(f"Threshold: {threshold}\n")
        
        # Test different voting methods
        voting_methods = ['majority', 'average', 'weighted']
        
        for method in voting_methods:
            print(f"\nVoting Method: {method.upper()}")
            print(f"{'-'*50}")
            
            try:
                result = ensemble.detect(image_path, threshold=threshold, voting_method=method)
                
                print(f"  Final Prediction: {'DEEPFAKE' if result.is_deepfake else 'REAL'}")
                print(f"  Confidence: {result.confidence:.4f} ({result.confidence*100:.2f}%)")
                print(f"  Voting Summary: {result.voting_summary}")
                print(f"  Average Fake Score: {result.average_fake_score:.4f}")
                print(f"  Average Real Score: {result.average_real_score:.4f}")
                
                print(f"\n  Individual Model Results:")
                for i, individual in enumerate(result.individual_results):
                    print(f"    Model {i+1} ({individual.model_name}):")
                    print(f"      Prediction: {'DEEPFAKE' if individual.is_deepfake else 'REAL'}")
                    print(f"      Confidence: {individual.confidence:.4f}")
                
            except Exception as e:
                print(f"  Error with {method}: {e}")
        
        return result
        
    except Exception as e:
        print(f"Error initializing ensemble: {e}")
        return None

def main():
    """Main test function"""
    print_separator('=', 100)
    print("DEEPFAKE DETECTION - MODEL COMPARISON & ENSEMBLE TESTING")
    print_separator('=', 100)
    
    # Image to test
    image_path = r'C:\Kidambi\WorkRelated\DeepFake\sample_images\fake_10.jpg'
    
    if not os.path.exists(image_path):
        print(f"\n✗ Error: Image not found: {image_path}")
        print("Please update the image_path variable in the script.")
        return
     

    try:
        ensemble = EnsembleDetector(model_keys=['dima-vit','umm-efficientnet'])
        result = ensemble.detect(image_path, threshold=0.5, voting_method='average')
        
        #         # 1. List available models
        #         test_list_models()
        
#         # 2. Test with default model
#         print("\n" + "="*100)
#         test_single_model(image_path, model_key='prithiv-v1', threshold=0.5)
        
#         # 3. Compare multiple models (if available)
#         print("\n" + "="*100)
#         # You can add more models here: ['prithiv-v1', 'dima-vit', 'umm-efficientnet']
#         test_model_comparison(image_path, model_keys=['prithiv-v1'], threshold=0.5)
        
#         # 4. Test ensemble (currently with one model, add more to see real ensemble)
#         print("\n" + "="*100)
#         test_ensemble_detection(image_path, model_keys=['prithiv-v1'], threshold=0.5)
        
#         # RECOMMENDATIONS
#         print("\n" + "="*100)
#         print("RECOMMENDATIONS FOR YOUR fake_0.jpg")
#         print("="*100)
#         print("""
# Based on the results:

# 1. CURRENT ISSUE:
#    - The default model (prithiv-v1) gives fake_0.jpg a fake score of only 20%
#    - This means the model genuinely thinks your image is real

# 2. SOLUTIONS TO TRY:

#    A. Lower the threshold:
#       detector = DeepfakeImageDetector(model_name='prithiv-v1')
#       result = detector.detect(image_path, threshold=0.15)  # More sensitive
   
#    B. Try a different model (requires downloading):
#       detector = DeepfakeImageDetector(model_name='dima-vit')
#       result = detector.detect(image_path, threshold=0.5)
   
#    C. Use ensemble voting (best accuracy):
#       ensemble = EnsembleDetector(model_keys=['prithiv-v1', 'dima-vit'])
#       result = ensemble.detect(image_path, threshold=0.5, voting_method='average')
   
#    D. Verify the ground truth:
#       - Double-check if fake_0.jpg is actually a deepfake
#       - High-quality deepfakes can fool detection models

# 3. ALTERNATIVE MODELS TO TRY:
#    - 'dima-vit': Vision Transformer based detector
#    - 'umm-efficientnet': EfficientNet-based detector  
#    - 'organika-sdxl': For Stable Diffusion generated images

# 4. HOW TO USE DIFFERENT MODELS:
#    Simply change the model_name parameter:
   
#    detector = DeepfakeImageDetector(model_name='dima-vit')
#    # or
#    detector = DeepfakeImageDetector(model_name='umm-efficientnet')
#         """)
        
        # print result for verification
        print(f"\nFinal Ensemble Result (Average Voting):")
        print(f"  Is Deepfake: {'DEEPFAKE' if result.is_deepfake else 'REAL'}")
        print(f"  Confidence: {result.confidence:.4f} ({result.confidence*100:.2f}%)")
        print(f"  Voting Summary: {result.voting_summary}")
        print(f"  Average Fake Score: {result.average_fake_score:.4f}")
        print(f"  Average Real Score: {result.average_real_score:.4f}")
        print(f"\n  Individual Model Results:")
        for i, individual in enumerate(result.individual_results):
            print(f"    Model {i+1} ({individual.model_name}):")
            print(f"      Prediction: {'DEEPFAKE' if individual.is_deepfake else 'REAL'}")
            print(f"      Confidence: {individual.confidence:.4f}")            
          

            


    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
