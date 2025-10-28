"""
Services module for Deep Fake Detection
Contains service modules for image and text detection
"""

from .app_detect_image import process_image, get_ensemble_detector, process_image_data
from .app_detect_text import get_text_ensemble, detect_text, detect_text_json

__all__ = [
    'process_image',
    'get_ensemble_detector', 
    'process_image_data',
    'get_text_ensemble',
    'detect_text',
    'detect_text_json'
]
