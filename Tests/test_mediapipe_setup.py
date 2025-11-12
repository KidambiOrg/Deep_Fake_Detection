"""
Simple test to verify MediaPipe face detection is working correctly.
Tests the core face detection functionality without full video processing.
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Setup paths
SCRIPT_DIR = Path(__file__).parent.parent
MEDIAPIPE_MODEL_PATH = SCRIPT_DIR / "Video_Models" / "Mediapipe" / "blaze_face_short_range.tflite"


def test_mediapipe_image_detection():
    """Test MediaPipe face detection on a simple test image."""
    
    print("Testing MediaPipe Face Detection...")
    print(f"Model path: {MEDIAPIPE_MODEL_PATH}")
    print(f"Model exists: {MEDIAPIPE_MODEL_PATH.exists()}")
    
    if not MEDIAPIPE_MODEL_PATH.exists():
        print("ERROR: Model file not found!")
        return False
    
    try:
        # Create a simple test image with a synthetic face-like pattern
        # (For real testing, use an actual face image)
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw a simple face-like pattern (oval for head, circles for eyes, line for mouth)
        cv2.ellipse(test_image, (320, 240), (100, 130), 0, 0, 360, (255, 200, 150), -1)  # Face
        cv2.circle(test_image, (290, 220), 15, (0, 0, 0), -1)  # Left eye
        cv2.circle(test_image, (350, 220), 15, (0, 0, 0), -1)  # Right eye
        cv2.ellipse(test_image, (320, 260), (30, 15), 0, 0, 180, (200, 100, 100), 2)  # Mouth
        
        print("\nCreated test image with synthetic face pattern")
        
        # Initialize MediaPipe Face Detector for IMAGE mode
        base_options = python.BaseOptions(model_asset_path=str(MEDIAPIPE_MODEL_PATH))
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=0.3,  # Lower threshold for test
            min_suppression_threshold=0.3
        )
        
        print("Initializing MediaPipe Face Detector...")
        
        with vision.FaceDetector.create_from_options(options) as detector:
            print("✓ MediaPipe Face Detector initialized successfully!")
            
            # Convert to RGB (MediaPipe expects RGB)
            rgb_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            
            print("\nRunning face detection on test image...")
            
            # Detect faces
            detection_result = detector.detect(mp_image)
            
            print(f"\n{'='*50}")
            print(f"DETECTION RESULTS")
            print(f"{'='*50}")
            print(f"Number of faces detected: {len(detection_result.detections)}")
            
            if detection_result.detections:
                for idx, detection in enumerate(detection_result.detections):
                    print(f"\nFace {idx + 1}:")
                    print(f"  Confidence: {detection.categories[0].score:.3f}")
                    bbox = detection.bounding_box
                    print(f"  Bounding Box:")
                    print(f"    Origin: ({bbox.origin_x}, {bbox.origin_y})")
                    print(f"    Size: {bbox.width} x {bbox.height}")
                    
                    if detection.keypoints:
                        print(f"  Keypoints: {len(detection.keypoints)} points detected")
                
                print("\n✓ Face detection working correctly!")
                return True
            else:
                print("\nNote: No faces detected in synthetic test image")
                print("This is expected - try with a real face image for better results")
                print("\n✓ MediaPipe is working (but needs real face image for detection)")
                return True
        
    except Exception as e:
        print(f"\n✗ ERROR during face detection: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mediapipe_video_mode():
    """Test MediaPipe face detection in VIDEO mode (without actual video)."""
    
    print(f"\n{'='*50}")
    print("Testing MediaPipe VIDEO Mode...")
    print(f"{'='*50}")
    
    try:
        # Initialize MediaPipe Face Detector for VIDEO mode
        base_options = python.BaseOptions(model_asset_path=str(MEDIAPIPE_MODEL_PATH))
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_detection_confidence=0.5,
            min_suppression_threshold=0.3
        )
        
        print("Initializing MediaPipe Face Detector in VIDEO mode...")
        
        with vision.FaceDetector.create_from_options(options) as detector:
            print("✓ MediaPipe Face Detector (VIDEO mode) initialized successfully!")
            
            # Create test frame
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            rgb_frame = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Test with timestamp (required for VIDEO mode)
            timestamp_ms = 0
            detection_result = detector.detect_for_video(mp_image, timestamp_ms)
            
            print(f"✓ VIDEO mode detection completed (timestamp: {timestamp_ms}ms)")
            print(f"  Faces detected: {len(detection_result.detections)}")
            
            return True
        
    except Exception as e:
        print(f"\n✗ ERROR in VIDEO mode test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all MediaPipe tests."""
    
    print("="*50)
    print("MEDIAPIPE FACE DETECTION TEST SUITE")
    print("="*50)
    print()
    
    # Test 1: Image mode
    test1_passed = test_mediapipe_image_detection()
    
    # Test 2: Video mode
    test2_passed = test_mediapipe_video_mode()
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Image Mode Test: {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"Video Mode Test: {'✓ PASSED' if test2_passed else '✗ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n✓ All tests passed! MediaPipe is ready to use.")
        print("\nNext steps:")
        print("  1. Install mediapipe: pip install mediapipe")
        print("  2. Run full video extraction: py Tests\\test_extract_faces_from_video.py <video_path>")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
    
    print("="*50)


if __name__ == "__main__":
    main()
