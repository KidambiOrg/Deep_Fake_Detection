"""Face deduplication tool using DeepFace."""

import logging
import json
import os
from utils.deduplicate_faces import deduplicate_faces

logger = logging.getLogger(__name__)


def deduplicate_faces_tool(
    faces_folder: str,
    model_name: str = "Facenet",
    distance_metric: str = "cosine",
    threshold: float | None = None,
    detector_backend: str = "yolov11n",
    fallback_detector: str = "retinaface"
) -> dict:
    """
    Agent tool: Remove duplicate faces using DeepFace embeddings.
    
    Args:
        faces_folder: Folder containing extracted face images
        model_name: DeepFace model (default: Facenet)
        distance_metric: Distance metric (cosine, euclidean, euclidean_l2)
        threshold: Custom threshold (None = use model default)
        detector_backend: Primary detector (default: yolov11n)
        fallback_detector: Fallback detector (default: retinaface)
    
    Returns:
        Dictionary with deduplication statistics
    """
    logger.info(f"Deduplicating faces in: {faces_folder}")
    
    results = deduplicate_faces(
        faces_folder=faces_folder,
        model_name=model_name,
        distance_metric=distance_metric,
        threshold=threshold,
        detector_backend=detector_backend,
        fallback_detector=fallback_detector
    )
    
    logger.info(f"Deduplication complete. {results['duplicates_removed']} duplicates removed")
    
    # Add output_folder for final result
    results['output_folder'] = faces_folder

    return results
