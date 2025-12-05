"""Key frame extraction tool for video processing."""

import cv2
import numpy as np
import logging
import os
import json
from typing import List, Tuple

logger = logging.getLogger(__name__)


def compute_hsv_hist(frame: np.ndarray, bins: tuple[int, int, int] = (32, 32, 32)) -> np.ndarray:
    """Compute a normalized HSV histogram for the frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, None, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
    return hist.flatten().astype(np.float32)


def compare_histograms(hist_a: np.ndarray, hist_b: np.ndarray, method: str = "chisq") -> float:
    """Return distance between histograms using chi-square or correlation distance."""
    method_lower = method.lower()
    if method_lower == "chisq":
        return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CHISQR))
    if method_lower == "corr":
        correlation = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
        return float(1.0 - correlation)
    raise ValueError("Unknown histogram comparison method. Use 'chisq' or 'corr'.")


def extract_key_frames_tool(
    video_path: str,
    output_folder: str,
    max_frames: int | None = None,
    method: str = "chisq",
    threshold: float = 0.6,
    min_gap_sec: float = 1.0,
    resize_width: int | None = 640,
    resize_height: int | None = None,
) -> dict:
    """
    Agent tool: Extract key frames from video and save them to disk.
    
    Args:
        video_path: Path to video file
        output_folder: Folder to save extracted key frames
        max_frames: Maximum frames to evaluate (None = all)
        method: Histogram comparison method ('chisq' or 'corr')
        threshold: Histogram distance threshold for key frame selection
        min_gap_sec: Minimum time gap between key frames in seconds
        resize_width: Optional resize width for histogram computation
        resize_height: Optional resize height
    
    Returns:
        Dictionary with key frame file paths and metadata
    """
    logger.info(f"Extracting key frames from: {video_path}")
    
    # Create output folder for key frames
    keyframes_folder = os.path.join(output_folder, "keyframes")
    os.makedirs(keyframes_folder, exist_ok=True)
    
    # Fixed histogram bins (32, 32, 32) - cannot be passed as parameter due to Azure OpenAI schema limitations
    histogram_bins = (32, 32, 32)
    
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    evaluated = 0
    key_frames: List[dict] = []
    last_hist: np.ndarray | None = None
    last_timestamp = -1e9

    while True:
        if max_frames is not None and evaluated >= max_frames:
            break

        ret, frame = capture.read()
        if not ret:
            break

        evaluated += 1
        timestamp_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
        frame_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        timestamp = timestamp_ms / 1000.0 if timestamp_ms else (frame_index / fps if fps else 0.0)

        if resize_width is not None:
            aspect_ratio = frame.shape[0] / frame.shape[1]
            target_height = resize_height if resize_height is not None else max(1, int(resize_width * aspect_ratio))
            scaled_frame = cv2.resize(frame, (resize_width, target_height), interpolation=cv2.INTER_AREA)
        else:
            scaled_frame = frame

        hist = compute_hsv_hist(scaled_frame, histogram_bins)
        should_select = last_hist is None
        if not should_select:
            distance = compare_histograms(hist, last_hist, method)
            if (timestamp - last_timestamp) >= min_gap_sec and distance > threshold:
                should_select = True

        if should_select:
            # Save frame to disk instead of keeping in memory
            frame_filename = f"keyframe_{frame_index:06d}_{timestamp:.3f}s.jpg"
            frame_path = os.path.join(keyframes_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
            
            # Store metadata instead of arrays
            key_frames.append({
                "frame_index": frame_index,
                "timestamp": timestamp,
                "file_path": frame_path
            })
            
            last_hist = hist
            last_timestamp = timestamp

    capture.release()
    
    logger.info(f"Extracted {len(key_frames)} key frames from {evaluated} evaluated frames")
    logger.info(f"Key frames saved to: {keyframes_folder}")
    
    metadata = {
        "key_frames": key_frames,
        "keyframes_folder": keyframes_folder,
        "total_frames": total_frames,
        "frames_evaluated": evaluated,
        "fps": fps,
        "output_folder": output_folder
    }

    try:
        metadata_path = os.path.join(keyframes_folder, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2)
        logger.debug(f"Key frame metadata written to {metadata_path}")
    except Exception as metadata_err:
        logger.warning(f"Failed to persist key frame metadata: {metadata_err}")
    
    return metadata
