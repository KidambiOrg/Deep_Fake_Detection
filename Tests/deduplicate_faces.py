"""
Face deduplication script using DeepFace embeddings.
Compares face images in a folder and deletes duplicates (same person).
"""

import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # Torch is optional; fall back to CPU if missing
    torch = None

# Set custom DeepFace home directory BEFORE importing DeepFace
SCRIPT_DIR = Path(__file__).parent.parent
CUSTOM_DEEPFACE_HOME = SCRIPT_DIR / "Video_Models" / "RetinaFace"
os.environ['DEEPFACE_HOME'] = str(CUSTOM_DEEPFACE_HOME)

from deepface import DeepFace
from deepface.modules.verification import find_threshold as deepface_find_threshold


def _configure_cuda_device() -> tuple[str, str]:
    """Ensure DeepFace uses GPU when available, otherwise fall back to CPU."""
    if torch is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return "cpu", "PyTorch not installed"

    try:
        if torch.cuda.is_available():
            device_index = torch.cuda.current_device()
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_index)
            device_name = torch.cuda.get_device_name(device_index)
            return f"cuda:{device_index}", device_name
    except Exception as exc:
        # Any CUDA probing error should not abort the flow; log and fall back.
        fallback_reason = f"CUDA check failed: {exc}"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return "cpu", fallback_reason

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    return "cpu", "CUDA not available"


def _extract_embedding(
    image_path: str,
    model_name: str,
    detector_backend: str,
    fallback_detector: str | None,
) -> tuple[np.ndarray | None, bool]:
    """Return embedding using preferred detector; fall back if necessary."""
    fallback_used = False

    try:
        representations: Any = DeepFace.represent(
            img_path=image_path,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=False,
        )
    except Exception as primary_error:
        if fallback_detector and fallback_detector != detector_backend:
            print(
                f"  ⚠ Detector '{detector_backend}' failed for {os.path.basename(image_path)}: {primary_error}. "
                f"Trying '{fallback_detector}'..."
            )
            try:
                representations = DeepFace.represent(
                    img_path=image_path,
                    model_name=model_name,
                    detector_backend=fallback_detector,
                    enforce_detection=False,
                )
                fallback_used = True
            except Exception as secondary_error:
                print(
                    f"  ⚠ Fallback detector '{fallback_detector}' also failed for {os.path.basename(image_path)}: {secondary_error}"
                )
                return None, False
        else:
            print(
                f"  ⚠ Detector '{detector_backend}' failed for {os.path.basename(image_path)}: {primary_error}"
            )
            return None, False

    if isinstance(representations, list) and representations:
        return np.array(representations[0]["embedding"], dtype=np.float32), fallback_used

    if isinstance(representations, dict) and "embedding" in representations:
        return np.array(representations["embedding"], dtype=np.float32), fallback_used

    return None, fallback_used


def _compute_distance(vec1: np.ndarray, vec2: np.ndarray, metric: str) -> float:
    """Compute distance between embeddings using standard metrics."""
    if metric == "cosine":
        vec1_norm = np.linalg.norm(vec1)
        vec2_norm = np.linalg.norm(vec2)
        if vec1_norm == 0 or vec2_norm == 0:
            return 1.0
        cosine_similarity = np.dot(vec1, vec2) / (vec1_norm * vec2_norm)
        return float(1.0 - cosine_similarity)

    if metric == "euclidean":
        return float(np.linalg.norm(vec1 - vec2))

    if metric == "euclidean_l2":
        vec1_norm = vec1 / np.linalg.norm(vec1) if np.linalg.norm(vec1) else vec1
        vec2_norm = vec2 / np.linalg.norm(vec2) if np.linalg.norm(vec2) else vec2
        return float(np.linalg.norm(vec1_norm - vec2_norm))

    raise ValueError(f"Unsupported distance metric: {metric}")


def deduplicate_faces(
    faces_folder: str,
    model_name: str = "Facenet",
    distance_metric: str = "cosine",
    threshold: float | None = None,
    detector_backend: str = "yolov11n",
    fallback_detector: str | None = "retinaface",
):
    """
    Remove duplicate faces from a folder using DeepFace verification.
    
    Logic:
    - Compare each face with all previously kept faces
    - If face matches any kept face (same person), delete it
    - If face is unique, keep it for future comparisons
    
    Args:
        faces_folder: Folder containing extracted face images
        model_name: DeepFace model to use (default: Facenet - fast & accurate)
        distance_metric: Distance metric (cosine, euclidean, euclidean_l2)
        threshold: Custom threshold for verification (None = use model default)
        detector_backend: Detector to align faces before embedding (default: yolov11n)
        fallback_detector: Detector to use if preferred backend fails (default: retinaface)
    
    Returns:
        Dictionary with deduplication statistics
    """
    print(f"\n{'='*50}")
    print("FACE DEDUPLICATION")
    print(f"{'='*50}")
    print(f"Folder: {faces_folder}")
    device_label, device_description = _configure_cuda_device()

    active_detector = detector_backend
    embedding_fallback_backend = (
        fallback_detector if fallback_detector and fallback_detector != active_detector else None
    )

    print(f"Model: {model_name}")
    print(f"Distance Metric: {distance_metric}")
    print(f"Detector Backend: {active_detector}")
    if embedding_fallback_backend:
        print(f"Fallback Detector: {embedding_fallback_backend}")
    print(f"Threshold: {threshold if threshold else 'Auto (model default)'}")
    print(f"Compute Device: {device_label} ({device_description})")
    print("-" * 50)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    all_faces = [
        os.path.join(faces_folder, f) 
        for f in os.listdir(faces_folder) 
        if os.path.splitext(f.lower())[1] in image_extensions
    ]
    
    threshold_value = (
        threshold if threshold is not None else deepface_find_threshold(model_name, distance_metric)
    )

    if not all_faces:
        print("No face images found in folder")
        print(f"Resolved Threshold: {threshold_value:.4f}")
        return {
            'total_faces': 0,
            'unique_faces': 0,
            'duplicates_removed': 0,
            'detector_fallbacks': 0,
            'embedding_failures': 0,
            'time_elapsed': 0.0,
            'threshold_used': threshold_value
        }
    
    all_faces.sort()  # Sort for consistent processing order
    total_faces = len(all_faces)
    print(f"\nTotal faces to process: {total_faces}")
    print(f"Resolved Threshold: {threshold_value:.4f}")
    
    start_time = time.time()
    embeddings: list[tuple[str, np.ndarray | None, bool]] = [
        (face_path, None, False) for face_path in all_faces
    ]
    embedding_failures = 0
    detector_fallbacks = 0

    print(f"\nStep 1: Building embeddings with {active_detector} alignment...")

    # Process sequentially to avoid GPU model duplication in threads
    for idx, face_path in enumerate(all_faces, 1):
        try:
            embedding, used_fallback = _extract_embedding(
                face_path,
                model_name,
                active_detector,
                embedding_fallback_backend,
            )
        except Exception as exc:
            embedding = None
            used_fallback = False
            print(
                f"  ⚠ Embedding task failed for {os.path.basename(face_path)}: {exc}"
            )

        if used_fallback:
            detector_fallbacks += 1

        if embedding is None:
            embedding_failures += 1
            print(f"  ⚠ Unable to compute embedding for {os.path.basename(face_path)}")

        embeddings[idx - 1] = (face_path, embedding, used_fallback)

        if idx % 10 == 0 or idx == 1 or idx == total_faces:
            print(f"  Embedding progress: {idx}/{total_faces}")

    print(f"\nStep 2: Comparing embeddings for duplicates...")
    unique_faces: list[str] = []
    unique_embeddings: list[np.ndarray | None] = []
    duplicates_removed = 0

    for idx, (face_path, embedding, _used_fallback) in enumerate(embeddings, 1):
        if idx % 10 == 0 or idx == 1:
            print(f"  Dedup progress: {idx}/{total_faces}")

        if embedding is None:
            unique_faces.append(face_path)
            unique_embeddings.append(None)
            continue

        is_duplicate = False
        for unique_idx, stored_embedding in enumerate(unique_embeddings):
            if stored_embedding is None:
                continue

            distance_value = _compute_distance(embedding, stored_embedding, distance_metric)
            if distance_value <= threshold_value:
                is_duplicate = True
                duplicates_removed += 1
                try:
                    os.remove(face_path)
                except OSError:
                    pass
                print(
                    f"  ✗ Removed duplicate: {os.path.basename(face_path)} "
                    f"(matches {os.path.basename(unique_faces[unique_idx])}, distance: {distance_value:.4f})"
                )
                break

        if not is_duplicate:
            unique_faces.append(face_path)
            unique_embeddings.append(embedding)
            if idx % 10 == 0 or idx == 1:
                print(f"  ✓ Kept unique face: {os.path.basename(face_path)}")

    elapsed_time = time.time() - start_time
    
    # Print summary
    print(f"\n{'='*50}")
    print("DEDUPLICATION SUMMARY")
    print(f"{'='*50}")
    print(f"Total faces processed: {total_faces}")
    print(f"Unique faces kept: {len(unique_faces)}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Detector fallbacks used: {detector_fallbacks}")
    print(f"Embedding failures: {embedding_failures}")
    print(f"Threshold used: {threshold_value:.4f}")
    print(f"\nTime elapsed: {elapsed_time:.2f}s")
    print(f"Avg time per face: {elapsed_time/total_faces:.3f}s")
    print(f"{'='*50}")
    
    return {
        'total_faces': total_faces,
        'unique_faces': len(unique_faces),
        'duplicates_removed': duplicates_removed,
        'detector_fallbacks': detector_fallbacks,
        'embedding_failures': embedding_failures,
        'time_elapsed': elapsed_time,
        'threshold_used': threshold_value
    }


def main():
    """
    Main function to run face deduplication.
    
    Usage:
        py deduplicate_faces.py <faces_folder> [model_name] [distance_metric] [threshold]
    
    Examples:
        py deduplicate_faces.py C:\\path\\to\\faces
        py deduplicate_faces.py C:\\path\\to\\faces VGG-Face
        py deduplicate_faces.py C:\\path\\to\\faces VGG-Face cosine
        py deduplicate_faces.py C:\\path\\to\\faces VGG-Face cosine 0.4
        py deduplicate_faces.py C:\\path\\to\\faces Facenet cosine 0.4 yolov11n retinaface
    """
    if len(sys.argv) < 2:
        print("Error: Please provide faces folder path")
        print("Usage: py deduplicate_faces.py <faces_folder> [model_name] [distance_metric] [threshold]")
        sys.exit(1)
    
    faces_folder = sys.argv[1]
    
    if not os.path.exists(faces_folder):
        print(f"Error: Folder not found: {faces_folder}")
        sys.exit(1)
    
    # Optional parameters
    model_name = sys.argv[2] if len(sys.argv) > 2 else "Facenet"
    distance_metric = sys.argv[3] if len(sys.argv) > 3 else "cosine"
    threshold = float(sys.argv[4]) if len(sys.argv) > 4 else None
    detector_backend = sys.argv[5] if len(sys.argv) > 5 else "yolov11n"
    fallback_detector = sys.argv[6] if len(sys.argv) > 6 else "retinaface"
    
    try:
        results = deduplicate_faces(
            faces_folder=faces_folder,
            model_name=model_name,
            distance_metric=distance_metric,
            threshold=threshold,
            detector_backend=detector_backend,
            fallback_detector=fallback_detector,
        )
        print("\nFace deduplication completed successfully!")
        
    except Exception as e:
        print(f"\nError during face deduplication: {str(e)}")
        raise


if __name__ == "__main__":
    main()
