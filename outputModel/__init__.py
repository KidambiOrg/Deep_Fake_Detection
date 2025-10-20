"""
Output models for deepfake detection results
"""
from .detection_output_model import (
    DetectionOutput,
    ImageDetectionResult,
    VotingResult,
    IndividualModelResult
)

__all__ = [
    'DetectionOutput',
    'ImageDetectionResult',
    'VotingResult',
    'IndividualModelResult'
]
