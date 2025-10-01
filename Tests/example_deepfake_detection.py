"""
Example Usage of Deepfake Image Detection

This script demonstrates how to use the DeepfakeImageDetector class
to detect deepfake images using the prithivMLMods/deepfake-detector-model-va1 model.
"""

from detectionLib.detect_deepfake_image import DeepfakeImageDetector, DeepfakeDetectionResult
from PIL import Image
import os


def example_single_image_detection():
    """Example: Detect deepfake in a single image"""
    print("=== Single Image Detection Example ===")
    
    # Initialize the detector
    detector = DeepfakeImageDetector()
    
    # Print model information
    model_info = detector.get_model_info()
    print(f"Model: {model_info['model_name']}")
    print(f"Device: {model_info['device']}")
    print(f"Labels: {model_info['labels']}")
    print(f"Parameters: {model_info['num_parameters']:,}")
    print(f"Model Size: {model_info['model_size_mb']:.1f} MB")
    print()
    
    # Example image path (replace with your image path)
    image_path = "example_image.jpg"
    
    if os.path.exists(image_path):
        try:
            # Detect deepfake
            result = detector.detect(image_path, threshold=0.5)
            
            print(f"Image: {result.image_path}")
            print(f"Is Deepfake: {result.is_deepfake}")
            print(f"Confidence: {result.confidence:.4f}")
            print(f"All Scores: {result.scores}")
            print(f"Processing Time: {result.processing_time:.4f} seconds")
            
        except Exception as e:
            print(f"Error processing image: {e}")
    else:
        print(f"Image file not found: {image_path}")
        print("Please provide a valid image path to test the detector.")


def example_batch_detection():
    """Example: Detect deepfakes in multiple images"""
    print("\n=== Batch Detection Example ===")
    
    # Initialize the detector
    detector = DeepfakeImageDetector()
    
    # Example image paths (replace with your image paths)
    image_paths = [
        "image1.jpg",
        "image2.png",
        "image3.jpeg"
    ]
    
    # Filter existing files
    existing_images = [path for path in image_paths if os.path.exists(path)]
    
    if existing_images:
        print(f"Processing {len(existing_images)} images...")
        
        try:
            # Detect deepfakes in batch
            results = detector.detect_batch(existing_images, threshold=0.5)
            
            print("\nResults:")
            print("-" * 80)
            for i, result in enumerate(results):
                print(f"Image {i+1}: {result.image_path}")
                print(f"  Is Deepfake: {result.is_deepfake}")
                print(f"  Confidence: {result.confidence:.4f}")
                print(f"  Processing Time: {result.processing_time:.4f}s")
                print()
                
        except Exception as e:
            print(f"Error processing batch: {e}")
    else:
        print("No image files found. Please add some images to test batch detection.")


def example_directory_analysis():
    """Example: Analyze all images in a directory"""
    print("\n=== Directory Analysis Example ===")
    
    # Initialize the detector
    detector = DeepfakeImageDetector()
    
    # Example directory path (replace with your directory path)
    directory_path = "test_images"
    
    if os.path.exists(directory_path):
        try:
            # Analyze directory
            results = detector.analyze_directory(
                directory_path, 
                extensions=['.jpg', '.jpeg', '.png', '.bmp'],
                threshold=0.5
            )
            
            print(f"Analyzed {len(results)} images from {directory_path}")
            print("\nSummary:")
            print("-" * 50)
            
            deepfake_count = sum(1 for r in results if r.is_deepfake)
            real_count = len(results) - deepfake_count
            
            print(f"Real Images: {real_count}")
            print(f"Deepfake Images: {deepfake_count}")
            print(f"Total Images: {len(results)}")
            
            # Show detailed results
            print("\nDetailed Results:")
            print("-" * 80)
            for result in results:
                status = "DEEPFAKE" if result.is_deepfake else "REAL"
                filename = os.path.basename(result.image_path) if result.image_path else "Unknown"
                print(f"{filename:<20} | {status:<8} | Confidence: {result.confidence:.4f}")
                
        except Exception as e:
            print(f"Error analyzing directory: {e}")
    else:
        print(f"Directory not found: {directory_path}")
        print("Please create a directory with images to test directory analysis.")


def create_test_setup():
    """Create a basic test setup with instructions"""
    print("\n=== Test Setup Instructions ===")
    print("To test the deepfake image detector, you can:")
    print()
    print("1. Single Image Test:")
    print("   - Place an image named 'example_image.jpg' in the current directory")
    print("   - Run: python example_deepfake_detection.py")
    print()
    print("2. Batch Test:")
    print("   - Place multiple images (image1.jpg, image2.png, etc.) in the current directory")
    print("   - Modify the image_paths list in example_batch_detection()")
    print()
    print("3. Directory Test:")
    print("   - Create a folder named 'test_images'")
    print("   - Add multiple images to this folder")
    print("   - Run the directory analysis example")
    print()
    print("Supported formats: .jpg, .jpeg, .png, .bmp, .gif")


if __name__ == "__main__":
    print("Deepfake Image Detection Examples")
    print("=" * 50)
    
    try:
        # Run examples
        example_single_image_detection()
        example_batch_detection()
        example_directory_analysis()
        create_test_setup()
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("\nMake sure you have installed the required dependencies:")
        print("pip install -r requirements.txt")