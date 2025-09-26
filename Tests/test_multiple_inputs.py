"""
Test the updated deepfake detector with multiple input types:
- File path
- PIL Image
- Numpy array
- Base64 encoded string
- Binary stream
- Raw bytes
"""

from lib.detect_deepfake_image import DeepfakeImageDetector
from PIL import Image
import numpy as np
import base64
import io
import os


def create_test_image():
    """Create a simple test image for testing"""
    # Create a 224x224 RGB test image
    test_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    test_image = Image.fromarray(test_array)
    return test_image, test_array


def image_to_base64(image: Image.Image, format: str = 'PNG') -> str:
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    image_bytes = buffer.getvalue()
    return base64.b64encode(image_bytes).decode('utf-8')


def image_to_data_uri(image: Image.Image, format: str = 'PNG') -> str:
    """Convert PIL Image to data URI format"""
    base64_str = image_to_base64(image, format)
    return f"data:image/{format.lower()};base64,{base64_str}"


def image_to_bytes(image: Image.Image, format: str = 'PNG') -> bytes:
    """Convert PIL Image to raw bytes"""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


def test_all_input_types():
    """Test the detector with all supported input types"""
    
    print("Testing DeepfakeImageDetector with multiple input types...")
    print("=" * 70)
    
    # Initialize detector
    print("Initializing detector...")
    detector = DeepfakeImageDetector()
    
    # Create test data
    test_image, test_array = create_test_image()
    
    # Save test image to file for file path testing
    test_file_path = "test_image_temp.png"
    test_image.save(test_file_path)
    
    # Prepare different input formats
    test_cases = [
        ("File Path", test_file_path),
        ("PIL Image", test_image),
        ("Numpy Array", test_array),
        ("Base64 String", image_to_base64(test_image)),
        ("Data URI", image_to_data_uri(test_image)),
        ("Raw Bytes", image_to_bytes(test_image)),
        ("Binary Stream", io.BytesIO(image_to_bytes(test_image)))
    ]
    
    print(f"\nTesting {len(test_cases)} different input types:")
    print("-" * 70)
    
    results = []
    
    for test_name, test_input in test_cases:
        try:
            print(f"Testing {test_name}...")
            result = detector.detect(test_input, threshold=0.5)
            
            print(f"  ✅ Success!")
            print(f"  Is Deepfake: {result.is_deepfake}")
            print(f"  Confidence: {result.confidence:.4f}")
            print(f"  Processing Time: {result.processing_time:.4f}s")
            print()
            
            results.append((test_name, True, result))
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            print()
            results.append((test_name, False, str(e)))
    
    # Clean up temporary file
    if os.path.exists(test_file_path):
        os.remove(test_file_path)
    
    # Summary
    print("Summary:")
    print("=" * 70)
    successful = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    print(f"Successful tests: {successful}/{total}")
    print()
    
    for test_name, success, result in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    return successful == total


def test_batch_processing():
    """Test batch processing with mixed input types"""
    
    print("\n" + "=" * 70)
    print("Testing Batch Processing with Mixed Input Types")
    print("=" * 70)
    
    # Initialize detector
    detector = DeepfakeImageDetector()
    
    # Create test data
    test_image1, test_array1 = create_test_image()
    test_image2, test_array2 = create_test_image()
    
    # Save test images
    test_file1 = "test_batch_1.png"
    test_file2 = "test_batch_2.png"
    test_image1.save(test_file1)
    test_image2.save(test_file2)
    
    # Create mixed batch
    mixed_batch = [
        test_file1,                           # File path
        test_image2,                          # PIL Image
        test_array1,                          # Numpy array
        image_to_base64(test_image2),         # Base64 string
        image_to_bytes(test_image1)           # Raw bytes
    ]
    
    try:
        print(f"Processing batch of {len(mixed_batch)} images with mixed types...")
        results = detector.detect_batch(mixed_batch, threshold=0.5)
        
        print("Results:")
        print("-" * 40)
        
        input_types = ["File Path", "PIL Image", "Numpy Array", "Base64", "Raw Bytes"]
        
        for i, (input_type, result) in enumerate(zip(input_types, results)):
            print(f"{i+1}. {input_type}:")
            print(f"   Is Deepfake: {result.is_deepfake}")
            print(f"   Confidence: {result.confidence:.4f}")
            print(f"   Time: {result.processing_time:.4f}s")
            print()
        
        print("✅ Batch processing successful!")
        success = True
        
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        success = False
    
    # Clean up
    for file in [test_file1, test_file2]:
        if os.path.exists(file):
            os.remove(file)
    
    return success


def demo_real_world_usage():
    """Demonstrate real-world usage examples"""
    
    print("\n" + "=" * 70)
    print("Real-World Usage Examples")
    print("=" * 70)
    
    detector = DeepfakeImageDetector()
    test_image, _ = create_test_image()
    
    print("1. Web API Usage (Base64):")
    print("-" * 30)
    
    # Simulate receiving base64 from web API
    base64_data = image_to_data_uri(test_image, 'JPEG')
    print(f"Received data URI: {base64_data[:50]}...")
    
    result = detector.detect(base64_data)
    print(f"Detection result: {'Deepfake' if result.is_deepfake else 'Real'} (Confidence: {result.confidence:.4f})")
    print()
    
    print("2. File Upload Processing:")
    print("-" * 30)
    
    # Simulate file upload as bytes
    image_bytes = image_to_bytes(test_image, 'PNG')
    print(f"Received {len(image_bytes)} bytes of image data")
    
    result = detector.detect(image_bytes)
    print(f"Detection result: {'Deepfake' if result.is_deepfake else 'Real'} (Confidence: {result.confidence:.4f})")
    print()
    
    print("3. Stream Processing:")
    print("-" * 30)
    
    # Simulate stream processing
    image_stream = io.BytesIO(image_to_bytes(test_image, 'JPEG'))
    print("Processing image from stream...")
    
    result = detector.detect(image_stream)
    print(f"Detection result: {'Deepfake' if result.is_deepfake else 'Real'} (Confidence: {result.confidence:.4f})")
    print()


if __name__ == "__main__":
    try:
        # Test all input types
        success1 = test_all_input_types()
        
        # Test batch processing
        success2 = test_batch_processing()
        
        # Demo real-world usage
        demo_real_world_usage()
        
        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        
        if success1 and success2:
            print("🎉 ALL TESTS PASSED!")
            print("The detector now supports all input types:")
            print("  ✅ File paths")
            print("  ✅ PIL Images")
            print("  ✅ Numpy arrays")
            print("  ✅ Base64 strings")
            print("  ✅ Data URIs")
            print("  ✅ Raw bytes")
            print("  ✅ Binary streams")
            print("  ✅ Batch processing with mixed types")
        else:
            print("⚠️  Some tests failed. Please check the error messages above.")
            
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()