"""
Test script to extract faces from video using DeepFace library.
"""

import cv2
import os
from deepface import DeepFace
from pathlib import Path
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from dotenv import load_dotenv

# Load environment variables from .env file
SCRIPT_DIR = Path(__file__).parent.parent
ENV_FILE = SCRIPT_DIR / ".env"
load_dotenv(ENV_FILE)

# Configure DeepFace to use models from Video_Models folder
# Get paths from environment variables or use defaults
DEEPFACE_HOME = os.getenv('DEEPFACE_HOME', 'Video_Models/RetinaFace')
VGG_MODELS_DIR_PATH = os.getenv('VGG_MODELS_DIR', 'Video_Models/VGG')

# Convert to absolute paths
RETINAFACE_MODELS_DIR = SCRIPT_DIR / DEEPFACE_HOME
VGG_MODELS_DIR = SCRIPT_DIR / VGG_MODELS_DIR_PATH

# Set environment variable to tell DeepFace where to find models
# Will switch between RetinaFace and VGG directories as needed
os.environ['DEEPFACE_HOME'] = str(RETINAFACE_MODELS_DIR)


def process_single_face(face_data: tuple, faces_folder: str, min_confidence: float, 
                       min_face_size: int, stats_lock: Lock, stats: dict) -> None:
    """
    Process a single detected face.
    
    Args:
        face_data: Tuple of (frame_number, face_idx, face_obj, frame_img)
        faces_folder: Output folder for face images
        min_confidence: Minimum confidence threshold
        min_face_size: Minimum face size
        stats_lock: Thread lock for updating statistics
        stats: Shared statistics dictionary
    """
    frame_number, face_idx, face_obj, frame_img = face_data
    
    confidence = face_obj.get('confidence', 0)
    
    # Skip low confidence detections
    if confidence < min_confidence:
        with stats_lock:
            stats['faces_rejected_low_confidence'] += 1
        print(f"  [Frame {frame_number}] Skipped face {face_idx + 1}: Low confidence ({confidence:.2f} < {min_confidence})")
        return
    
    # Get the face region (bounding box) from DeepFace
    facial_area = face_obj.get('facial_area')
    
    if facial_area is None:
        return
    
    # Extract bounding box coordinates
    x = facial_area.get('x', 0)
    y = facial_area.get('y', 0)
    w = facial_area.get('w', 0)
    h = facial_area.get('h', 0)
    
    # Validate face proportions (faces should have reasonable aspect ratio)
    # Typical face aspect ratio is between 0.5 and 1.5 (width/height)
    if w == 0 or h == 0:
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
    
    # frame_img is now passed as part of face_data
    
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
    
    # Generate filename
    filename = f"face_frame{frame_number:06d}_face{face_idx:02d}_conf{confidence:.2f}.jpg"
    filepath = os.path.join(faces_folder, filename)
    
    # Save face image
    cv2.imwrite(filepath, face_img)
    
    with stats_lock:
        stats['faces_extracted'] += 1
    
    print(f"  ✓ [Frame {frame_number}] Extracted face {face_idx + 1} (confidence: {confidence:.2f}, size: {w}x{h})")


def process_frame(frame_data: tuple, faces_folder: str, detector_backend: str, 
                  min_confidence: float, min_face_size: int,
                  stats_lock: Lock, stats: dict, max_workers: int) -> None:
    """
    Process a single frame to extract faces. If multiple faces detected, process them in parallel.
    
    Args:
        frame_data: Tuple of (frame_number, frame_image)
        faces_folder: Output folder for face images
        detector_backend: Face detection backend
        min_confidence: Minimum confidence threshold
        min_face_size: Minimum face size
        stats_lock: Thread lock for updating statistics
        stats: Shared statistics dictionary
        max_workers: Number of parallel workers for face processing
    """
    frame_number, frame = frame_data
    
    try:
        # Extract faces from frame using DeepFace with enforce_detection=True
        face_objs = DeepFace.extract_faces(
            img_path=frame,
            detector_backend=detector_backend,
            enforce_detection=True,  # Will raise exception if no face detected
            align=True
        )
        
        # Mark frame as having faces
        with stats_lock:
            stats['frames_with_faces'] += 1
        
        # If multiple faces detected, process them in parallel
        if len(face_objs) > 1:
            print(f"  [Frame {frame_number}] Detected {len(face_objs)} faces, processing in parallel...")
            
            # Prepare face data for parallel processing
            face_data_list = [(frame_number, face_idx, face_obj, frame) 
                             for face_idx, face_obj in enumerate(face_objs)]
            
            # Process faces in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as face_executor:
                face_futures = [
                    face_executor.submit(
                        process_single_face,
                        face_data,
                        faces_folder,
                        min_confidence,
                        min_face_size,
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
            # Single face, process directly (no need for parallel)
            process_single_face(
                (frame_number, 0, face_objs[0], frame),
                faces_folder,
                min_confidence,
                min_face_size,
                stats_lock,
                stats
            )
    
    except Exception as e:
        # No face detected or other error - skip this frame
        with stats_lock:
            stats['frames_without_faces'] += 1
        if frame_number % 20 == 0:  # Only print occasional errors to avoid spam
            print(f"  [Frame {frame_number}] No faces detected")


def extract_faces_from_video(video_path: str, output_folder: str | None = None, detector_backend: str = 'opencv', 
                             min_confidence: float = 0.90, min_face_size: int = 50,
                             max_frames: int | None = None, max_workers: int = 4):
    """
    Extract all faces from all frames in a video file using DeepFace with parallel processing.
    
    Args:
        video_path: Path to the video file
        output_folder: Folder to save extracted face images (defaults to same folder as script)
        detector_backend: Face detection backend to use (opencv, ssd, dlib, mtcnn, retinaface, mediapipe, yolov8, yunet, centerface)
        min_confidence: Minimum confidence threshold for face detection (default: 0.90)
        min_face_size: Minimum face size in pixels (default: 50)
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
    print(f"Detector backend: {detector_backend}")
    print(f"Parallel workers: {max_workers}")
    print("-" * 50)
    
    # Step 1: Collect all frames
    print(f"\nStep 1: Collecting {frames_to_process} frames from video...")
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
                print(f"  Collected {frame_count}/{frames_to_process} frames...")
        
        print(f"✓ Collected {len(frames)} frames")
    
    finally:
        video_capture.release()
    
    # Step 2: Process frames in parallel
    print(f"\nStep 2: Processing {len(frames)} frames in parallel...")
    
    # Shared statistics with thread-safe access
    stats_lock = Lock()
    stats = {
        'faces_extracted': 0,
        'frames_with_faces': 0,
        'frames_without_faces': 0,
        'faces_rejected_low_confidence': 0,
        'faces_rejected_small': 0
    }
    
    # Process frames in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_frame,
                frame_data,
                faces_folder,
                detector_backend,
                min_confidence,
                min_face_size,
                stats_lock,
                stats,
                max_workers
            )
            for frame_data in frames
        ]
        
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
    print(f"\nOutput folder: {faces_folder}")
    print("=" * 50)
    
    return {
        'total_frames': frame_count,
        'frames_with_faces': stats['frames_with_faces'],
        'frames_without_faces': stats['frames_without_faces'],
        'faces_extracted': stats['faces_extracted'],
        'faces_rejected_low_confidence': stats['faces_rejected_low_confidence'],
        'faces_rejected_small': stats['faces_rejected_small'],
        'output_folder': faces_folder
    }


def main():
    """
    Main function to run the face extraction from video.
    
    Usage:
        py test_extract_faces_from_video.py <video_path> [detector] [min_confidence] [max_frames] [max_workers]
    
    Examples:
        py test_extract_faces_from_video.py video.mp4
        py test_extract_faces_from_video.py video.mov retinaface
        py test_extract_faces_from_video.py video.mp4 mtcnn 0.95
        py test_extract_faces_from_video.py video.mp4 mtcnn 0.90 200 8
    """
    # Example usage - update with your video path
    video_path = r"path\to\your\video.mp4"  # Update this path
    
    # Check if video path was provided as command line argument
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    
    # Detector backends available:
    # 'opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe', 
    # 'yolov8n', 'yolov8m', 'yolov8l', 'yunet', 'centerface'
    detector = 'retinaface'  # Default, more accurate face detection
    
    # Check if detector backend was provided as command line argument
    if len(sys.argv) > 2:
        detector = sys.argv[2]
    
    # Quality thresholds
    min_confidence = 0.95  # Higher = more strict (reject more false positives)
    max_frames = 10 #None  # Process all frames (set to int to limit)
    max_workers = 4  # Number of parallel workers
    
    # Check if thresholds were provided as command line arguments
    if len(sys.argv) > 3:
        min_confidence = float(sys.argv[3])
    
    if len(sys.argv) > 4:
        max_frames = int(sys.argv[4])
    
    if len(sys.argv) > 5:
        max_workers = int(sys.argv[5])
    
    print(f"Quality Settings:")
    print(f"  Min Confidence: {min_confidence}")
    print(f"  Max Frames: {max_frames}")
    print(f"  Parallel Workers: {max_workers}")
    print(f"  Detector: {detector}")
    print()
    
    try:
        results = extract_faces_from_video(
            video_path=video_path,
            detector_backend=detector,
            min_confidence=min_confidence,
            max_frames=max_frames,
            max_workers=max_workers
        )
        print("\nFace extraction completed successfully!")
        
    except Exception as e:
        print(f"\nError during face extraction: {str(e)}")
        raise


if __name__ == "__main__":
    main()
