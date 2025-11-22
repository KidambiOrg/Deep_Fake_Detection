"""
Test script to extract faces from video using OpenCV DNN face detector.
Logic:
1. Extract frames from video file
2. Send frames to detect faces in parallel
3. If faces found, save with cropping, else ignore
4. Cleanup - Remove duplicate faces (same person) using DeepFace VGG-Face
"""

import cv2
import os
from pathlib import Path
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from dotenv import load_dotenv
import urllib.request
import time

from deduplicate_faces import deduplicate_faces

# Load environment variables from .env file
SCRIPT_DIR = Path(__file__).parent.parent
ENV_FILE = SCRIPT_DIR / ".env"
load_dotenv(ENV_FILE)

# Configure OpenCV DNN face detector model paths
MODELS_DIR = SCRIPT_DIR / "Video_Models" / "OpenCV_DNN"
PROTOTXT_PATH = MODELS_DIR / "deploy.prototxt"
CAFFEMODEL_PATH = MODELS_DIR / "res10_300x300_ssd_iter_140000.caffemodel"

# Model URLs for automatic download
PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
CAFFEMODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"


def download_models():
    """Download OpenCV DNN face detector models if not present."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    if not PROTOTXT_PATH.exists():
        print(f"Downloading deploy.prototxt...")
        urllib.request.urlretrieve(PROTOTXT_URL, PROTOTXT_PATH)
        print(f"✓ Downloaded to {PROTOTXT_PATH}")
    
    if not CAFFEMODEL_PATH.exists():
        print(f"Downloading res10_300x300_ssd_iter_140000.caffemodel...")
        urllib.request.urlretrieve(CAFFEMODEL_URL, CAFFEMODEL_PATH)
        print(f"✓ Downloaded to {CAFFEMODEL_PATH}")


def load_face_detector():
    """Load OpenCV DNN face detector model."""
    download_models()
    
    net = cv2.dnn.readNetFromCaffe(str(PROTOTXT_PATH), str(CAFFEMODEL_PATH))
    return net


def process_single_face(face_data: tuple, faces_folder: str, min_confidence: float, 
                       min_face_size: int, face_expansion_ratio: float, min_blur_threshold: float, 
                       stats_lock: Lock, stats: dict) -> None:
    """
    Process a single detected face from OpenCV DNN detection.
    
    Args:
        face_data: Tuple of (frame_number, face_idx, confidence, x, y, w, h, frame_img)
        faces_folder: Output folder for face images
        min_confidence: Minimum confidence threshold
        min_face_size: Minimum face size
        min_blur_threshold: Minimum blur threshold (Laplacian variance)
        stats_lock: Thread lock for updating statistics
        stats: Shared statistics dictionary
    """
    frame_number, face_idx, confidence, x, y, w, h, frame_img = face_data
    
    # Skip low confidence detections (already filtered, but double-check)
    if confidence < min_confidence:
        with stats_lock:
            stats['faces_rejected_low_confidence'] += 1
        print(f"  [Frame {frame_number}] Skipped face {face_idx + 1}: Low confidence ({confidence:.2f} < {min_confidence})")
        return
    
    # Validate face dimensions (must be positive)
    if w <= 0 or h <= 0:
        with stats_lock:
            stats['faces_rejected_low_confidence'] += 1
        print(f"  [Frame {frame_number}] Skipped face {face_idx + 1}: Invalid dimensions ({w}x{h})")
        return
    
    aspect_ratio = w / h
    if aspect_ratio < 0.4 or aspect_ratio > 2.0:
        with stats_lock:
            stats['faces_rejected_low_confidence'] += 1
        print(f"  [Frame {frame_number}] Skipped face {face_idx + 1}: Invalid aspect ratio ({aspect_ratio:.2f})")
        return
    
    # Expand the bounding box to include full head, hair, and ears
    # The detected face box typically captures eyes/nose/mouth but not full head
    
    # Calculate face center
    face_center_x = x + w // 2
    face_center_y = y + h // 2
    
    # Expand dimensions to capture full head:
    # - Width: 60% wider (30% on each side for ears)
    # - Height above: 100% of face height (to capture all hair/forehead)
    # - Height below: 40% of face height (to capture chin/neck)
    width_expansion = 0.6  # 60% wider
    height_top_expansion = 1.0  # 100% above for full hair
    height_bottom_expansion = 0.4  # 40% below for chin/neck
    
    # Calculate expanded width
    expanded_width = int(w * (1 + width_expansion))
    
    # Calculate total expanded height
    expanded_height = int(h * (1 + height_top_expansion + height_bottom_expansion))
    
    # Make it a square using the larger dimension
    crop_size = max(expanded_width, expanded_height)
    
    # Add 5% padding all around for gap
    padding_factor = 0.05
    padding = int(crop_size * padding_factor)
    final_size = crop_size + (padding * 2)
    
    # Calculate crop center point
    # Shift center upward to account for hair being above detected face
    # The detected face is usually in the lower 2/3 of the full head
    vertical_shift = int(h * 0.3)  # Shift up by 30% of face height
    crop_center_x = face_center_x
    crop_center_y = face_center_y - vertical_shift
    
    # Calculate crop coordinates
    x_start = crop_center_x - final_size // 2
    y_start = crop_center_y - final_size // 2
    
    # Ensure we don't go out of bounds
    frame_h, frame_w = frame_img.shape[:2]
    
    # Clamp to frame boundaries
    x_start = max(0, x_start)
    y_start = max(0, y_start)
    x_end = min(frame_w, x_start + final_size)
    y_end = min(frame_h, y_start + final_size)
    
    # Adjust if we're at the edge to maintain square aspect ratio
    actual_width = x_end - x_start
    actual_height = y_end - y_start
    
    if actual_width < final_size:
        # Hit right edge, shift left
        x_start = max(0, x_end - final_size)
    if actual_height < final_size:
        # Hit bottom edge, shift up
        y_start = max(0, y_end - final_size)
    
    # Recalculate end coordinates after adjustment
    x_end = min(frame_w, x_start + final_size)
    y_end = min(frame_h, y_start + final_size)
    
    # Crop the square face region
    face_img = frame_img[y_start:y_end, x_start:x_end]
    
    # Ensure it's uint8
    if face_img.dtype == np.float32 or face_img.dtype == np.float64:
        face_img = (face_img * 255).astype(np.uint8)
    
    # Check face size
    h, w = face_img.shape[:2]
    if h < min_face_size or w < min_face_size:
        with stats_lock:
            stats['faces_rejected_small'] += 1
        print(f"  [Frame {frame_number}] Skipped face {face_idx + 1}: Too small ({w}x{h} < {min_face_size})")
        return
    
    # Check image sharpness using Laplacian variance
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(face_img, cv2.CV_64F).var()
    
    # if blur_score < min_blur_threshold:
    #     with stats_lock:
    #         stats['faces_rejected_blurry'] += 1
    #     print(f"  [Frame {frame_number}] Skipped face {face_idx + 1}: Too blurry (score: {blur_score:.2f} < {min_blur_threshold})")
    #     return
    
    # Generate filename
    filename = f"face_frame{frame_number:06d}_face{face_idx:02d}_conf{confidence:.2f}.jpg"
    filepath = os.path.join(faces_folder, filename)
    
    # Save face image
    cv2.imwrite(filepath, face_img)
    
    with stats_lock:
        stats['faces_extracted'] += 1
    
    print(f"  ✓ [Frame {frame_number}] Extracted face {face_idx + 1} (confidence: {confidence:.2f}, size: {w}x{h}, blur: {blur_score:.2f})")


def process_frame(frame_data: tuple, faces_folder: str, detector, 
                  min_confidence: float, min_face_size: int,
                  face_expansion_ratio: float, min_blur_threshold: float,
                  stats_lock: Lock, stats: dict, max_workers: int, detector_lock: Lock) -> None:
    """
    Process a single frame to extract faces using OpenCV DNN. If multiple faces detected, process them in parallel.
    
    Args:
        frame_data: Tuple of (frame_number, frame_image)
        faces_folder: Output folder for face images
        detector: OpenCV DNN face detector network
        min_confidence: Minimum confidence threshold
        min_face_size: Minimum face size
        face_expansion_ratio: Ratio to expand face bounding box (e.g., 0.2 = 20% expansion)
        min_blur_threshold: Minimum Laplacian variance for sharpness (lower = more blurry)
        stats_lock: Thread lock for updating statistics
        stats: Shared statistics dictionary
        max_workers: Number of parallel workers for face processing
        detector_lock: Thread lock for detector access (OpenCV DNN is not thread-safe)
    """
    frame_number, frame = frame_data
    
    try:
        h, w = frame.shape[:2]
        
        # Prepare frame for OpenCV DNN face detector
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        
        # Detect faces using OpenCV DNN (thread-safe access with lock)
        with detector_lock:
            detector.setInput(blob)
            detections = detector.forward()
        
        # Collect valid faces (confidence > threshold)
        valid_faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > min_confidence:
                # Get bounding box coordinates (normalized 0-1, format: [x1, y1, x2, y2])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure box is within frame boundaries
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                
                # Calculate width and height
                face_w = endX - startX
                face_h = endY - startY
                
                # Skip invalid boxes (must have positive dimensions)
                if face_w <= 0 or face_h <= 0:
                    continue
                
                # Store: (x, y, width, height, confidence)
                valid_faces.append((startX, startY, face_w, face_h, confidence))
        
        # Check if any faces were detected
        if not valid_faces:
            with stats_lock:
                stats['frames_without_faces'] += 1
            if frame_number % 20 == 0:  # Only print occasional messages to avoid spam
                print(f"  [Frame {frame_number}] No faces detected")
            return
        
        # Mark frame as having faces
        with stats_lock:
            stats['frames_with_faces'] += 1
        
        # If multiple faces detected, process them in parallel
        if len(valid_faces) > 1:
            print(f"  [Frame {frame_number}] Detected {len(valid_faces)} faces, processing in parallel...")
            
            # Prepare face data for parallel processing
            # valid_faces format: (x, y, width, height, confidence)
            face_data_list = [
                (frame_number, face_idx, confidence, x, y, face_w, face_h, frame) 
                for face_idx, (x, y, face_w, face_h, confidence) in enumerate(valid_faces)
            ]
            
            # Process faces in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as face_executor:
                face_futures = [
                    face_executor.submit(
                        process_single_face,
                        face_data,
                        faces_folder,
                        min_confidence,
                        min_face_size,
                        face_expansion_ratio,
                        min_blur_threshold,
                        stats_lock,
                        stats
                    )
                    for face_data in face_data_list
                ]
                
                # Wait for all face processing to complete
                for future in as_completed(face_futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"  [Frame {frame_number}] Error processing face: {e}")
        else:
            # Single face, process directly
            # valid_faces format: (x, y, width, height, confidence)
            x, y, face_w, face_h, confidence = valid_faces[0]
            process_single_face(
                (frame_number, 0, confidence, x, y, face_w, face_h, frame),
                faces_folder,
                min_confidence,
                min_face_size,
                face_expansion_ratio,
                min_blur_threshold,
                stats_lock,
                stats
            )
    
    except Exception as e:
        # Error during face detection
        with stats_lock:
            stats['frames_without_faces'] += 1
        if frame_number % 20 == 0:  # Only print occasional errors to avoid spam
            print(f"  [Frame {frame_number}] Error detecting faces: {e}")


def extract_faces_from_video(video_path: str, output_folder: str | None = None, 
                             min_confidence: float = 0.50, min_face_size: int = 50,
                             face_expansion_ratio: float = 0.2, min_blur_threshold: float = 100.0,
                             max_frames: int | None = None, max_workers: int = 4):
    """
    Extract all faces from all frames in a video file using OpenCV DNN with parallel processing.
    
    Logic:
    1. Extract frames from video file
    2. Send frames to detect faces in parallel
    3. If faces found, save with cropping, else ignore
    
    Args:
        video_path: Path to the video file
        output_folder: Folder to save extracted face images (defaults to same folder as script)
        min_confidence: Minimum confidence threshold for face detection (default: 0.50)
        min_face_size: Minimum face size in pixels (default: 50)
        face_expansion_ratio: Ratio to expand face bounding box to include hair/ears/chin (default: 0.2 = 20%)
        min_blur_threshold: Minimum Laplacian variance for sharpness filtering (default: 100.0, lower = more blurry)
        max_frames: Maximum number of frames to process (default: None = all frames)
        max_workers: Number of parallel workers (default: 4)
    
    Returns:
        Dictionary with extraction statistics
    """
    # Validate video file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Setup output folder
    if output_folder is None:
        output_folder = r"C:\Kidambi\WorkRelated\DeepFake\sample_images\extracted_faces"
    
    # Use output folder directly (don't create subfolder)
    faces_folder = output_folder
    os.makedirs(faces_folder, exist_ok=True)
    
    # Load OpenCV DNN face detector
    detector = load_face_detector()
    
    # Open video file
    video_capture = cv2.VideoCapture(video_path)
    
    if not video_capture.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")
    
    # Get video properties
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    
    # Limit frames to process
    frames_to_process = total_frames if max_frames is None else min(total_frames, max_frames)
    
    print(f"Processing video: {video_path}")
    print(f"Total frames in video: {total_frames}")
    print(f"Frames to process: {frames_to_process}")
    print(f"FPS: {fps}")
    print(f"Output folder: {faces_folder}")
    print(f"Detector: OpenCV DNN (res10_300x300_ssd)")
    print(f"Min confidence: {min_confidence}")
    print(f"Parallel workers: {max_workers}")
    print("-" * 50)
    
    # Step 1: Extract frames from video file
    print(f"\nStep 1: Extracting {frames_to_process} frames from video...")
    step1_start = time.time()
    frames = []
    frame_count = 0
    
    try:
        while frame_count < frames_to_process:
            ret, frame = video_capture.read()
            
            if not ret:
                break
            
            frame_count += 1
            frames.append((frame_count, frame.copy()))
            
            if frame_count % 20 == 0:
                print(f"  Extracted {frame_count}/{frames_to_process} frames...")
        
        step1_elapsed = time.time() - step1_start
        print(f"✓ Extracted {len(frames)} frames (Time: {step1_elapsed:.2f}s)")
    
    finally:
        video_capture.release()
    
    # Step 2: Send frames to detect faces in parallel
    print(f"\nStep 2: Detecting faces in {len(frames)} frames (parallel processing)...")
    step2_start = time.time()
    
    # Shared statistics and detector access with thread-safe locks
    stats_lock = Lock()
    detector_lock = Lock()  # Lock for thread-safe detector access
    stats = {
        'faces_extracted': 0,
        'frames_with_faces': 0,
        'frames_without_faces': 0,
        'faces_rejected_low_confidence': 0,
        'faces_rejected_small': 0,
        'faces_rejected_blurry': 0
    }
    
    # Process frames in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for frame_data in frames:
            futures.append(
                executor.submit(
                    process_frame,
                    frame_data,
                    faces_folder,
                    detector,
                    min_confidence,
                    min_face_size,
                    face_expansion_ratio,
                    min_blur_threshold,
                    stats_lock,
                    stats,
                    max_workers,
                    detector_lock  # Pass detector lock for thread-safe access
                )
            )
        
        # Wait for all tasks to complete
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 10 == 0:
                print(f"  Progress: {completed}/{len(frames)} frames processed...")
            try:
                future.result()
            except Exception as e:
                print(f"  Error processing frame: {e}")
    
    step2_elapsed = time.time() - step2_start
    print(f"✓ Face detection completed (Time: {step2_elapsed:.2f}s)")
    
    # Calculate total time
    total_time = step1_elapsed + step2_elapsed
    
    # Print summary
    print("\n" + "=" * 50)
    print("EXTRACTION SUMMARY")
    print("=" * 50)
    print(f"Total frames processed: {frame_count}")
    print(f"Frames with valid faces: {stats['frames_with_faces']}")
    print(f"Frames without faces: {stats['frames_without_faces']}")
    print(f"Total faces extracted: {stats['faces_extracted']}")
    print(f"\nQuality Filters:")
    print(f"  Rejected (low confidence): {stats['faces_rejected_low_confidence']}")
    print(f"  Rejected (too small): {stats['faces_rejected_small']}")
    print(f"  Rejected (too blurry): {stats['faces_rejected_blurry']}")
    print(f"\nPerformance:")
    print(f"  Step 1 (Frame Extraction): {step1_elapsed:.2f}s")
    print(f"  Step 2 (Face Detection): {step2_elapsed:.2f}s")
    print(f"  Total Time: {total_time:.2f}s")
    print(f"  Avg Time per Frame: {total_time/frame_count:.3f}s")
    print(f"\nOutput folder: {faces_folder}")
    print("=" * 50)
    
    # Step 3: Cleanup - Remove duplicate faces (same person)
    step3_elapsed = 0
    dedup_results = None
    total_time_with_cleanup = total_time
    
    if stats['faces_extracted'] > 0:
        print(f"\nStep 3: Cleanup - Removing duplicate faces (same person)...")
        step3_start = time.time()
        
        try:
            dedup_results = deduplicate_faces(
                faces_folder=faces_folder,
                model_name="Facenet",
                distance_metric="cosine",
                detector_backend="yolov11n",
                fallback_detector="retinaface"
            )
            
            step3_elapsed = time.time() - step3_start
            
            # Update total time
            total_time_with_cleanup = total_time + step3_elapsed
            
            print(f"\n✓ Cleanup completed (Time: {step3_elapsed:.2f}s)")
            print(f"  Total Time (with cleanup): {total_time_with_cleanup:.2f}s")
            
        except Exception as e:
            step3_elapsed = time.time() - step3_start
            dedup_results = None
            print(f"\n⚠ Cleanup failed: {e}")
            total_time_with_cleanup = total_time
    else:
        print(f"\nStep 3: Cleanup - Skipped (no faces extracted)")
    
    return {
        'total_frames': frame_count,
        'frames_with_faces': stats['frames_with_faces'],
        'frames_without_faces': stats['frames_without_faces'],
        'faces_extracted': stats['faces_extracted'],
        'faces_rejected_low_confidence': stats['faces_rejected_low_confidence'],
        'faces_rejected_small': stats['faces_rejected_small'],
        'faces_rejected_blurry': stats['faces_rejected_blurry'],
        'output_folder': faces_folder,
        'time_frame_extraction': step1_elapsed,
        'time_face_detection': step2_elapsed,
        'time_cleanup': step3_elapsed if 'step3_elapsed' in locals() else 0,
        'time_total': total_time_with_cleanup,
        'time_per_frame': total_time/frame_count,
        'deduplication': dedup_results
    }


def main():
    """
    Main function to run the face extraction from video using OpenCV DNN.
    
    Logic:
    1. Extract frames from video file
    2. Send frames to detect faces in parallel
    3. If faces found, save with cropping, else ignore
    4. Cleanup - Remove duplicate faces (same person) using VGG-Face
    
    Usage:
        py test_extract_faces_from_video.py <video_path> [min_confidence] [face_expansion_ratio] [min_blur_threshold] [max_frames] [max_workers]
    
    Examples:
        py test_extract_faces_from_video.py video.mp4
        py test_extract_faces_from_video.py video.mp4 0.5
        py test_extract_faces_from_video.py video.mp4 0.5 0.2
        py test_extract_faces_from_video.py video.mp4 0.5 0.2 100.0
        py test_extract_faces_from_video.py video.mp4 0.5 0.2 100.0 200
        py test_extract_faces_from_video.py video.mp4 0.5 0.2 100.0 200 8
    """
    # Example usage - update with your video path
    video_path = r"path\to\your\video.mp4"  # Update this path
    
    # Check if video path was provided as command line argument
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    
    # Quality thresholds
    min_confidence = 0.5  # OpenCV DNN default confidence threshold
    face_expansion_ratio = 0.2  # Expand face box by 20% to include hair/ears/chin
    min_blur_threshold = 60.0  # Laplacian variance threshold for sharpness (lower = more blurry)
    max_frames = None  # Process limited frames (set to None for all frames)
    max_workers = 16 # Number of parallel workers
    
    # Check if thresholds were provided as command line arguments
    if len(sys.argv) > 2:
        min_confidence = float(sys.argv[2])
    
    if len(sys.argv) > 3:
        face_expansion_ratio = float(sys.argv[3])
    
    if len(sys.argv) > 4:
        min_blur_threshold = float(sys.argv[4])
    
    if len(sys.argv) > 5:
        max_frames = int(sys.argv[5]) if sys.argv[5].lower() != 'none' else None
    
    if len(sys.argv) > 6:
        max_workers = int(sys.argv[6])
    
    print(f"Quality Settings:")
    print(f"  Min Confidence: {min_confidence}")
    print(f"  Face Expansion Ratio: {face_expansion_ratio}")
    print(f"  Min Blur Threshold: {min_blur_threshold}")
    print(f"  Max Frames: {max_frames if max_frames else 'All'}")
    print(f"  Parallel Workers: {max_workers}")
    print(f"  Detector: OpenCV DNN (res10_300x300_ssd)")
    print()
    
    try:
        results = extract_faces_from_video(
            video_path=video_path,
            min_confidence=min_confidence,
            face_expansion_ratio=face_expansion_ratio,
            min_blur_threshold=min_blur_threshold,
            max_frames=max_frames,
            max_workers=max_workers
        )
        print("\nFace extraction completed successfully!")
        
    except Exception as e:
        print(f"\nError during face extraction: {str(e)}")
        raise


if __name__ == "__main__":
    main()
