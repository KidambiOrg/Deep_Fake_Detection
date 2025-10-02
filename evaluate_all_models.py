"""
Comprehensive Model Evaluation Script

This script evaluates all available deepfake detection models against a test image (4.png)
and provides detailed comparison of results, performance metrics, and recommendations.
"""

import sys
import os
from pathlib import Path
import time
from typing import Dict, List, Any
import json
from datetime import datetime

# Add detectionLib to path
sys.path.insert(0, str(Path(__file__).parent))

from detectionLib.detect_deepfake_image import (
    DeepfakeImageDetector,
    EnsembleDetector,
    list_available_models,
    MODEL_REGISTRY
)


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable format"""
    if seconds < 1:
        return f"{seconds*1000:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"


def print_separator(char='=', length=80):
    """Print a separator line"""
    print(char * length)


def evaluate_single_model(model_key: str, image_path: str, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Evaluate a single model against an image
    
    Args:
        model_key: Model key from MODEL_REGISTRY
        image_path: Path to image file
        threshold: Detection threshold
        
    Returns:
        Dictionary with evaluation results
    """
    try:
        print(f"\n📊 Evaluating: {model_key}")
        print(f"   Description: {MODEL_REGISTRY[model_key]['description']}")
        
        # Initialize detector
        start_time = time.time()
        detector = DeepfakeImageDetector(model_name=model_key)
        load_time = time.time() - start_time
        
        # Run detection
        result = detector.detect(image_path, threshold=threshold)
        
        # Get model info
        model_info = detector.get_model_info()
        
        return {
            'model_key': model_key,
            'model_name': MODEL_REGISTRY[model_key]['name'],
            'architecture': MODEL_REGISTRY[model_key]['architecture'],
            'source': MODEL_REGISTRY[model_key]['source'],
            'success': True,
            'is_deepfake': result.is_deepfake,
            'confidence': result.confidence,
            'scores': result.scores,
            'processing_time': result.processing_time,
            'load_time': load_time,
            'model_size_mb': model_info['model_size_mb'],
            'num_parameters': model_info['num_parameters'],
            'recommendation': result.get_recommendation(threshold)
        }
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}")
        return {
            'model_key': model_key,
            'model_name': MODEL_REGISTRY[model_key]['name'],
            'success': False,
            'error': str(e)
        }


def print_model_result(result: Dict[str, Any]):
    """Print formatted result for a single model"""
    if not result['success']:
        print(f"   ❌ FAILED: {result['error']}")
        return
    
    # Determine emoji based on result
    emoji = "🚨" if result['is_deepfake'] else "✅"
    
    print(f"   {emoji} Result: {'DEEPFAKE' if result['is_deepfake'] else 'REAL'}")
    print(f"   📈 Confidence: {result['confidence']:.2%}")
    print(f"   📊 Scores: {', '.join([f'{k}: {v:.2%}' for k, v in result['scores'].items()])}")
    print(f"   ⏱️  Processing: {format_time(result['processing_time'])}")
    print(f"   🔄 Load Time: {format_time(result['load_time'])}")
    print(f"   💾 Model Size: {result['model_size_mb']:.2f} MB")
    print(f"   🔢 Parameters: {result['num_parameters']:,}")


def print_summary_table(results: List[Dict[str, Any]]):
    """Print a summary table of all results"""
    print_separator('=')
    print("\n📋 SUMMARY TABLE")
    print_separator('=')
    
    # Filter successful results
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    if successful:
        # Header
        print(f"\n{'Model':<25} {'Architecture':<20} {'Result':<12} {'Confidence':<12} {'Time':<10}")
        print('-' * 85)
        
        # Results
        for r in successful:
            result_str = "DEEPFAKE" if r['is_deepfake'] else "REAL"
            conf_str = f"{r['confidence']:.2%}"
            time_str = format_time(r['processing_time'])
            
            print(f"{r['model_key']:<25} {r['architecture']:<20} {result_str:<12} {conf_str:<12} {time_str:<10}")
    
    if failed:
        print(f"\n❌ Failed Models ({len(failed)}):")
        for r in failed:
            print(f"   • {r['model_key']}: {r['error']}")


def print_statistics(results: List[Dict[str, Any]]):
    """Print overall statistics"""
    successful = [r for r in results if r['success']]
    
    if not successful:
        print("\n⚠️  No successful model evaluations")
        return
    
    print_separator('=')
    print("\n📊 STATISTICS")
    print_separator('=')
    
    # Count predictions
    deepfake_count = sum(1 for r in successful if r['is_deepfake'])
    real_count = len(successful) - deepfake_count
    
    print(f"\n🔍 Total Models Evaluated: {len(successful)}")
    print(f"   • Predicted DEEPFAKE: {deepfake_count}")
    print(f"   • Predicted REAL: {real_count}")
    print(f"   • Agreement: {max(deepfake_count, real_count) / len(successful) * 100:.1f}%")
    
    # Average metrics
    avg_confidence = sum(r['confidence'] for r in successful) / len(successful)
    avg_processing_time = sum(r['processing_time'] for r in successful) / len(successful)
    avg_load_time = sum(r['load_time'] for r in successful) / len(successful)
    
    print(f"\n📈 Average Confidence: {avg_confidence:.2%}")
    print(f"⏱️  Average Processing Time: {format_time(avg_processing_time)}")
    print(f"🔄 Average Load Time: {format_time(avg_load_time)}")
    
    # Fastest and slowest
    fastest = min(successful, key=lambda x: x['processing_time'])
    slowest = max(successful, key=lambda x: x['processing_time'])
    
    print(f"\n⚡ Fastest Model: {fastest['model_key']} ({format_time(fastest['processing_time'])})")
    print(f"🐌 Slowest Model: {slowest['model_key']} ({format_time(slowest['processing_time'])})")
    
    # Most/least confident
    most_confident = max(successful, key=lambda x: x['confidence'])
    least_confident = min(successful, key=lambda x: x['confidence'])
    
    print(f"\n💪 Most Confident: {most_confident['model_key']} ({most_confident['confidence']:.2%})")
    print(f"🤔 Least Confident: {least_confident['model_key']} ({least_confident['confidence']:.2%})")


def evaluate_ensemble(image_path: str, threshold: float = 0.5):
    """Evaluate using ensemble of top models"""
    print_separator('=')
    print("\n🎯 ENSEMBLE EVALUATION")
    print_separator('=')
    
    try:
        # Use a mix of best performing models
        model_keys=[
            # HuggingFace models
            'prithiv-v1',
            'dima-vit',
            'xception71',
            'organika-sdxl',
            'vit_b_16'
            ]
        
        print(f"\nUsing ensemble with models: {', '.join(model_keys)}")
        
        ensemble = EnsembleDetector(model_keys=model_keys)
        
        # Test different voting methods
        voting_methods = ['majority', 'weighted', 'average', 'maximum']
        
        for method in voting_methods:
            print(f"\n📊 Voting Method: {method}")
            result = ensemble.detect(image_path, threshold=threshold, voting_method=method)
            
            emoji = "🚨" if result.is_deepfake else "✅"
            print(f"   {emoji} Result: {'DEEPFAKE' if result.is_deepfake else 'REAL'}")
            print(f"   📈 Confidence: {result.confidence:.2%}")
            print(f"   🗳️  Voting: {result.voting_summary['fake']} FAKE, {result.voting_summary['real']} REAL")
            print(f"   📊 Avg Fake Score: {result.average_fake_score:.2%}")
            print(f"   📊 Avg Real Score: {result.average_real_score:.2%}")
            
            # Show which model was chosen for maximum method
            if method == 'maximum':
                max_conf_model = max(result.individual_results, key=lambda r: r.confidence)
                print(f"   🏆 Selected Model: {max_conf_model.model_name} (confidence: {max_conf_model.confidence:.2%})")
            
    except Exception as e:
        print(f"   ❌ Ensemble evaluation failed: {e}")


def save_results_to_json(results: List[Dict[str, Any]], output_path: str):
    """Save results to JSON file"""
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'image': str(results[0].get('image_path', 'Unknown')) if results else 'Unknown',
        'total_models': len(results),
        'successful': len([r for r in results if r['success']]),
        'failed': len([r for r in results if not r['success']]),
        'results': results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")


def main():
    """Main evaluation function"""
    print_separator('=')
    print("🔍 DEEPFAKE DETECTION MODEL EVALUATION")
    print_separator('=')
    
    # Configuration
    image_path = "C:\\Kidambi\\WorkRelated\\DeepFake\\sample_images\\real_0.jpg"
    threshold = 0.5
    output_json = "evaluation_results.json"
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"\n❌ Error: Image file '{image_path}' not found!")
        print(f"   Please ensure the file exists in the current directory: {os.getcwd()}")
        return
    
    print(f"\n📁 Image: {image_path}")
    print(f"🎯 Threshold: {threshold}")
    print(f"📊 Total Models Available: {len(MODEL_REGISTRY)}")
    
    # Get all available models
    models = list_available_models()
    
    # Evaluate each model
    print_separator('=')
    print("\n🚀 STARTING EVALUATION")
    print_separator('=')
    
    results = []
    total_models = len(models)
    
    for i, (model_key, model_info) in enumerate(models.items(), 1):
        print(f"\n[{i}/{total_models}] {model_key}")
        result = evaluate_single_model(model_key, image_path, threshold)
        print_model_result(result)
        results.append(result)
        
        # Small delay to avoid overwhelming the system
        time.sleep(0.5)
    
    # Print summary
    print_summary_table(results)
    print_statistics(results)
    
    # Ensemble evaluation
    evaluate_ensemble(image_path, threshold)
    
    # Save results
    save_results_to_json(results, output_json)
    
    print_separator('=')
    print("\n✅ EVALUATION COMPLETE!")
    print_separator('=')


if __name__ == "__main__":
    main()
