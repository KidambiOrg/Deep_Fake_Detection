
from lib.detect_deepfake_image import DeepfakeImageDetector
import os 

def process_results_example():
    """Example of processing and filtering results"""
    
    print("\n\nProcessing Results Example")
    print("=" * 40)
    
    detector = DeepfakeImageDetector()
    directory_path = "C:\\Kidambi\\WorkRelated\\DeepFake\\sample_images"
    
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' not found.")
        return
    
    results = detector.analyze_directory(directory_path)
    
    # Filter high-confidence results
    high_confidence = [r for r in results if r.confidence > 0.7]
    deepfakes = [r for r in results if r.is_deepfake]
    
    # Print results in tabular format
    print(f"\nDetection Results:")
    print("=" * 120)
    print(f"{'Filename':<30} {'Status':<10} {'Confidence':<12} {'Real Score':<12} {'Fake Score':<12} {'Processing Time':<15}")
    print("-" * 120)
    
    for r in results:
        filename = os.path.basename(r.image_path) if r.image_path else "Unknown"
        status = "DEEPFAKE" if r.is_deepfake else "REAL"
        real_score = r.scores.get('Real', 0.0)
        fake_score = r.scores.get('Fake', 0.0)
        print(f"{filename:<30} {status:<10} {r.confidence:<12.3f} {real_score:<12.3f} {fake_score:<12.3f} {r.processing_time:<15.3f}s")
    
    print("-" * 120)
    
    # Summary statistics
    real_count = sum(1 for r in results if not r.is_deepfake)
    fake_count = sum(1 for r in results if r.is_deepfake)
    avg_time = sum(r.processing_time for r in results) / len(results)
    avg_confidence = sum(r.confidence for r in results) / len(results)
    
    print(f"\nSummary Statistics:")
    print(f"{'Total Images:':<20} {len(results)}")
    print(f"{'Real Images:':<20} {real_count}")
    print(f"{'Deepfake Images:':<20} {fake_count}")
    print(f"{'High Confidence (>70%):':<20} {len(high_confidence)}")
    print(f"{'Average Confidence:':<20} {avg_confidence:.3f}")
    print(f"{'Average Processing Time:':<20} {avg_time:.3f}s")
    
    if deepfakes:
        print(f"\n⚠️  Deepfake Images Detected:")
        for result in deepfakes:
            filename = os.path.basename(result.image_path) if result.image_path else "Unknown"
            print(f"  • {filename} (confidence: {result.confidence:.1%})")
    else:
        print(f"\n✅ No deepfake images detected in this directory.")


def main():
    """Run all usage examples"""
    
    print("analyze_directory Method - Usage Examples")
    print("=" * 50)
    
    try:
        # basic_usage_example()
        #advanced_usage_example() 
        process_results_example()       
       
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")


if __name__ == "__main__":
    main()