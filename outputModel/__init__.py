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
from .text_detection_output_model import (
    TextDetectionResult,
    IndividualModelScore
)

__all__ = [
    'DetectionOutput',
    'ImageDetectionResult',
    'VotingResult',
    'IndividualModelResult',
    'ExifData',
    'TextDetectionResult',
    'IndividualModelScore'
]
