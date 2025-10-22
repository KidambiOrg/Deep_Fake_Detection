"""
Services module for Deep Fake Detection
Contains service modules for image and text detection
"""

from .app_detect_image import process_image, create_ensemble_detector, process_image_data
from .app_detect_text import TextDetectionApp

__all__ = [
    'process_image',
    'create_ensemble_detector', 
    'process_image_data',
    'TextDetectionApp'
]
