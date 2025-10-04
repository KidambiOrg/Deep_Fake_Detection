"""
Example: Using Nonescape Model in Ensemble Detection

This example demonstrates how to integrate the Nonescape model into the
ensemble detector for improved accuracy through multiple model voting.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detectionLib.detect_deepfake_image import EnsembleDetector, MODEL_REGISTRY


def configure_nonescape_model(model_path):
    """Configure the Nonescape model path in MODEL_REGISTRY"""
    MODEL_REGISTRY['nonescape-mini']['model_path'] = model_path
    print(f"✓ Configured Nonescape Mini model: {model_path}")


def example_basic_ensemble(image_path, model_path):
    """Example 1: Basic ensemble with Nonescape + HuggingFace models"""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Ensemble (Nonescape + HuggingFace Models)")
    print("=" * 70)
    
    # Configure model
    configure_nonescape_model(model_path)
    
    # Create ensemble
    print("\nInitializing ensemble with:")
    print("  - prithiv-v1 (HuggingFace SigLIP)")
    print("  - nonescape-mini (EfficientNet v2)")
    
    ensemble = EnsembleDetector(
        model_keys=['prithiv-v1', 'nonescape-mini']
    )
    
    print(f"\n✓ Ensemble ready with {len(ensemble.detectors)} models")
    
    # Detect
    print(f"\nAnalyzing image: {image_path}")
    result = ensemble.detect(image_path, threshold=0.5, voting_method='majority')
    
    # Display results
    print("\n" + "-" * 70)
    print(f"FINAL PREDICTION: {'🤖 DEEPFAKE' if result.is_deepfake else '✅ REAL'}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"\nVoting Summary:")
    print(f"  Deepfake votes: {result.voting_summary['fake']}")
    print(f"  Real votes: {result.voting_summary['real']}")
    print(f"\nAverage Scores:")
    print(f"  Fake: {result.average_fake_score:.2%}")
    print(f"  Real: {result.average_real_score:.2%}")
    
    print("\nIndividual Model Results:")
    for i, res in enumerate(result.individual_results, 1):
        status = '🤖 DEEPFAKE' if res.is_deepfake else '✅ REAL'
        print(f"  {i}. {res.model_name}: {status} (confidence: {res.confidence:.2%})")


def example_multi_model_ensemble(image_path, model_path):
    """Example 2: Larger ensemble with multiple architectures"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Multi-Model Ensemble (Multiple Architectures)")
    print("=" * 70)
    
    # Configure model
    configure_nonescape_model(model_path)
    
    # Create ensemble with diverse models
    print("\nInitializing ensemble with diverse architectures:")
    print("  - prithiv-v1 (SigLIP)")
    print("  - dima-vit (Vision Transformer)")
    print("  - nonescape-mini (EfficientNet v2)")
    
    ensemble = EnsembleDetector(
        model_keys=['prithiv-v1', 'dima-vit', 'nonescape-mini']
    )
    
    print(f"\n✓ Ensemble ready with {len(ensemble.detectors)} models")
    
    # Test different voting methods
    voting_methods = ['majority', 'weighted', 'average', 'maximum']
    
    print(f"\nAnalyzing image: {image_path}")
    print("\nTesting different voting methods:")
    print("-" * 70)
    
    for method in voting_methods:
        result = ensemble.detect(image_path, threshold=0.5, voting_method=method)
        status = '🤖 DEEPFAKE' if result.is_deepfake else '✅ REAL'
        print(f"{method.upper():15} : {status} (confidence: {result.confidence:.2%})")


def example_compare_models(image_path, model_path):
    """Example 3: Compare individual model predictions"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Model Comparison")
    print("=" * 70)
    
    # Configure model
    configure_nonescape_model(model_path)
    
    # Create ensemble
    ensemble = EnsembleDetector(
        model_keys=['prithiv-v1', 'nonescape-mini']
    )
    
    # Compare models
    print(f"\nComparing models on: {image_path}")
    comparison = ensemble.compare_models(image_path, threshold=0.5)
    
    print("\nDetailed Comparison:")
    print("-" * 70)
    
    for i, result in enumerate(comparison['model_results'], 1):
        print(f"\n{i}. {result['model']}")
        print(f"   Prediction: {'DEEPFAKE' if result['is_deepfake'] else 'REAL'}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"   Scores: Real={result['scores'].get('Real', 0):.2%}, "
              f"Fake={result['scores'].get('Fake', 0):.2%}")
        print(f"   Processing Time: {result['processing_time']:.2f}s")
    
    agreement = "✓ AGREE" if comparison['agreement'] else "✗ DISAGREE"
    print(f"\nModel Agreement: {agreement}")


def example_batch_processing(directory_path, model_path):
    """Example 4: Batch process directory with ensemble"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Batch Processing with Ensemble")
    print("=" * 70)
    
    # Configure model
    configure_nonescape_model(model_path)
    
    # Create ensemble
    print("\nInitializing ensemble...")
    ensemble = EnsembleDetector(
        model_keys=['prithiv-v1', 'nonescape-mini']
    )
    
    # Process directory
    print(f"\nProcessing directory: {directory_path}")
    results = ensemble.analyze_directory(
        directory_path,
        threshold=0.5,
        voting_method='weighted'
    )
    
    print(f"\n✓ Processed {len(results)} images")
    
    # Summary
    deepfake_count = sum(1 for r in results if r.is_deepfake)
    real_count = len(results) - deepfake_count
    
    print("\nSummary:")
    print(f"  Deepfakes detected: {deepfake_count}")
    print(f"  Real images: {real_count}")
    
    # Top suspicious images
    if deepfake_count > 0:
        print("\nTop suspicious images:")
        sorted_results = sorted(
            [(r.individual_results[0].image_path, r.average_fake_score) 
             for r in results if r.is_deepfake],
            key=lambda x: x[1],
            reverse=True
        )
        for path, score in sorted_results[:5]:
            print(f"  - {os.path.basename(path)}: {score:.2%}")


def main():
    """Run all examples"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Examples of using Nonescape model in ensemble detection",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('-m', '--model', required=True,
                        help='Path to Nonescape model file (.safetensors)')
    parser.add_argument('-i', '--image', 
                        help='Path to test image')
    parser.add_argument('-d', '--directory',
                        help='Path to directory for batch processing')
    parser.add_argument('--example', type=int, choices=[1, 2, 3, 4],
                        help='Run specific example (1-4)')
    
    args = parser.parse_args()
    
    # Validate
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        print("\nDownload the model:")
        print("  wget https://nonescape.sfo2.cdn.digitaloceanspaces.com/nonescape-mini-v0.safetensors")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("NONESCAPE ENSEMBLE INTEGRATION EXAMPLES")
    print("=" * 70)
    
    try:
        # Run examples
        if args.example is None or args.example == 1:
            if args.image:
                example_basic_ensemble(args.image, args.model)
            else:
                print("\n⚠️  Example 1 requires --image argument")
        
        if args.example is None or args.example == 2:
            if args.image:
                example_multi_model_ensemble(args.image, args.model)
            else:
                print("\n⚠️  Example 2 requires --image argument")
        
        if args.example is None or args.example == 3:
            if args.image:
                example_compare_models(args.image, args.model)
            else:
                print("\n⚠️  Example 3 requires --image argument")
        
        if args.example is None or args.example == 4:
            if args.directory:
                example_batch_processing(args.directory, args.model)
            else:
                print("\n⚠️  Example 4 requires --directory argument")
        
        print("\n" + "=" * 70)
        print("Examples complete!")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
