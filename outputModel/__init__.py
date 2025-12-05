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
from .video_detection_output_model import (
    KeyFrameMetadata,
    KeyFrameExtractionOutput,
    FaceDetectionStats,
    FaceDeduplicationStats,
    VideoProcessingSummary,
)

__all__ = [
    'DetectionOutput',
    'ImageDetectionResult',
    'VotingResult',
    'IndividualModelResult',
    'ExifData',
    'TextDetectionResult',
    'IndividualModelScore',
    'KeyFrameMetadata',
    'KeyFrameExtractionOutput',
    'FaceDetectionStats',
    'FaceDeduplicationStats',
    'VideoProcessingSummary',
]
