"""Face detection tool for extracting faces from frames."""

import cv2
import os
import numpy as np
import logging
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import urllib.request

logger = logging.getLogger(__name__)

# OpenCV DNN face detector paths
SCRIPT_DIR = Path(__file__).parent.parent.parent.parent
MODELS_DIR = SCRIPT_DIR / "Video_Models" / "OpenCV_DNN"
PROTOTXT_PATH = MODELS_DIR / "deploy.prototxt"
CAFFEMODEL_PATH = MODELS_DIR / "res10_300x300_ssd_iter_140000.caffemodel"

PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
CAFFEMODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"


def download_models():
    """Download OpenCV DNN face detector models if not present."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    if not PROTOTXT_PATH.exists():
        logger.info("Downloading deploy.prototxt...")
        urllib.request.urlretrieve(PROTOTXT_URL, PROTOTXT_PATH)
    
    if not CAFFEMODEL_PATH.exists():
        logger.info("Downloading res10_300x300_ssd_iter_140000.caffemodel...")
        urllib.request.urlretrieve(CAFFEMODEL_URL, CAFFEMODEL_PATH)


def load_face_detector():
    """Load OpenCV DNN face detector model."""
    download_models()
    return cv2.dnn.readNetFromCaffe(str(PROTOTXT_PATH), str(CAFFEMODEL_PATH))


def process_single_face(
    face_data: tuple, 
    faces_folder: str, 
    min_confidence: float,
    min_face_size: int,
    stats_lock: Lock,
    stats: dict
) -> None:
    """Process and save a single detected face."""
    frame_number, face_idx, confidence, x, y, w, h, frame_img = face_data
    
    if confidence < min_confidence:
        with stats_lock:
            stats['faces_rejected_low_confidence'] += 1
        return
    
    # Expand face crop to include hair, ears, chin
    face_center_x = x + w // 2
    face_center_y = y + h // 2
    
    width_expansion = 0.6
    height_top_expansion = 1.0
    height_bottom_expansion = 0.4
    
    expanded_width = int(w * (1 + width_expansion))
    expanded_height = int(h * (1 + height_top_expansion + height_bottom_expansion))
    crop_size = max(expanded_width, expanded_height)
    
    padding_factor = 0.05
    padding = int(crop_size * padding_factor)
    final_size = crop_size + (padding * 2)
    
    vertical_shift = int(h * 0.3)
    crop_center_x = face_center_x
    crop_center_y = face_center_y - vertical_shift
    
    x_start = crop_center_x - final_size // 2
    y_start = crop_center_y - final_size // 2
    
    frame_h, frame_w = frame_img.shape[:2]
    x_start = max(0, x_start)
    y_start = max(0, y_start)
    x_end = min(frame_w, x_start + final_size)
    y_end = min(frame_h, y_start + final_size)
    
    actual_width = x_end - x_start
    actual_height = y_end - y_start
    
    if actual_width < final_size:
        x_start = max(0, x_end - final_size)
    if actual_height < final_size:
        y_start = max(0, y_end - final_size)
    
    x_end = min(frame_w, x_start + final_size)
    y_end = min(frame_h, y_start + final_size)
    
    face_img = frame_img[y_start:y_end, x_start:x_end]
    
    if face_img.dtype == np.float32 or face_img.dtype == np.float64:
        face_img = (face_img * 255).astype(np.uint8)
    
    h_crop, w_crop = face_img.shape[:2]
    if h_crop < min_face_size or w_crop < min_face_size:
        with stats_lock:
            stats['faces_rejected_small'] += 1
        return
    
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(face_img, cv2.CV_64F).var()
    
    filename = f"face_frame{frame_number:06d}_face{face_idx:02d}_conf{confidence:.2f}.jpg"
    filepath = os.path.join(faces_folder, filename)
    cv2.imwrite(filepath, face_img)
    
    with stats_lock:
        stats['faces_extracted'] += 1


def process_frame(
    frame_data: tuple,
    faces_folder: str,
    detector,
    min_confidence: float,
    min_face_size: int,
    stats_lock: Lock,
    stats: dict,
    max_workers: int,
    detector_lock: Lock
) -> None:
    """Process a single frame to extract faces."""
    frame_number, frame = frame_data
    
    try:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        
        with detector_lock:
            detector.setInput(blob)
            detections = detector.forward()
        
        valid_faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence < min_confidence:
                continue
            
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)
            
            face_w = endX - startX
            face_h = endY - startY
            
            if face_w <= 0 or face_h <= 0:
                continue
            
            valid_faces.append((startX, startY, face_w, face_h, confidence))
        
        if not valid_faces:
            with stats_lock:
                stats['frames_without_faces'] += 1
            return
        
        with stats_lock:
            stats['frames_with_faces'] += 1
        
        if len(valid_faces) > 1:
            face_data_list = [
                (frame_number, face_idx, confidence, x, y, face_w, face_h, frame) 
                for face_idx, (x, y, face_w, face_h, confidence) in enumerate(valid_faces)
            ]
            
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
                
                for future in as_completed(face_futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Error processing face: {e}")
        else:
            x, y, face_w, face_h, confidence = valid_faces[0]
            process_single_face(
                (frame_number, 0, confidence, x, y, face_w, face_h, frame),
                faces_folder,
                min_confidence,
                min_face_size,
                stats_lock,
                stats
            )
    
    except Exception as e:
        with stats_lock:
            stats['frames_without_faces'] += 1


def detect_faces_in_frames_tool(
    keyframes_folder: str,
    output_folder: str,
    min_confidence: float = 0.50,
    min_face_size: int = 50,
    max_workers: int = 4
) -> dict:
    """
    Agent tool: Detect and extract faces from saved key frame images.
    
    Args:
        keyframes_folder: Folder containing saved key frame images
        output_folder: Folder to save extracted face images
        min_confidence: Minimum confidence threshold
        min_face_size: Minimum face size in pixels
        max_workers: Number of parallel workers
    
    Returns:
        Dictionary with face detection statistics
    """
    logger.info(f"Detecting faces in key frames from: {keyframes_folder}")
    
    os.makedirs(output_folder, exist_ok=True)
    detector = load_face_detector()
    
    # Load all key frame files from folder
    keyframe_files = sorted([
        f for f in os.listdir(keyframes_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    
    if not keyframe_files:
        logger.warning(f"No key frame images found in: {keyframes_folder}")
        return {
            'faces_extracted': 0,
            'frames_with_faces': 0,
            'frames_without_faces': 0,
            'faces_rejected_low_confidence': 0,
            'faces_rejected_small': 0,
            'output_folder': output_folder
        }
    
    logger.info(f"Found {len(keyframe_files)} key frame images to process")
    
    stats_lock = Lock()
    detector_lock = Lock()
    stats = {
        'faces_extracted': 0,
        'frames_with_faces': 0,
        'frames_without_faces': 0,
        'faces_rejected_low_confidence': 0,
        'faces_rejected_small': 0,
    }
    
    # Load frames from disk and prepare data list
    frame_data_list = []
    for idx, filename in enumerate(keyframe_files):
        file_path = os.path.join(keyframes_folder, filename)
        frame = cv2.imread(file_path)
        if frame is not None:
            frame_data_list.append((idx, frame))
        else:
            logger.warning(f"Failed to load frame: {file_path}")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for frame_data in frame_data_list:
            futures.append(
                executor.submit(
                    process_frame,
                    frame_data,
                    output_folder,
                    detector,
                    min_confidence,
                    min_face_size,
                    stats_lock,
                    stats,
                    max_workers,
                    detector_lock
                )
            )
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
    
    logger.info(f"Face detection complete. Extracted {stats['faces_extracted']} faces")
    
    # Add output_folder to stats for next agent
    stats['output_folder'] = output_folder

    return stats
