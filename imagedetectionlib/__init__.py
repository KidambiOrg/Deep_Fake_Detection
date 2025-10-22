"""
Image Detection Library for deepfake and AI-generated image detection
"""
from .detect_deepfake_image import (
    DeepfakeImageDetector,
    EnsembleDetector,
    DeepfakeDetectionResult,
    EnsembleResult,
    list_available_models
)

__all__ = [
    'DeepfakeImageDetector',
    'EnsembleDetector',
    'DeepfakeDetectionResult',
    'EnsembleResult',
    'list_available_models'
]
