"""
Output models for deepfake detection results
"""
from .image_detection_output_model import (
    DetectionOutput,
    ImageDetectionResult,
    VotingResult,
    IndividualModelResult,
    ExifData
)

__all__ = [
    'DetectionOutput',
    'ImageDetectionResult',
    'VotingResult',
    'IndividualModelResult',
    'ExifData'
]
