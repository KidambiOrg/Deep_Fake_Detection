"""Pydantic models for video deepfake detection workflow outputs."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class KeyFrameMetadata(BaseModel):
    """Metadata about a selected key frame."""

    frame_index: int = Field(description="Index of the frame in the original video")
    timestamp: float = Field(description="Timestamp (seconds) at which the key frame occurs")
    file_path: str = Field(description="Absolute path to the saved key frame image")


class KeyFrameExtractionOutput(BaseModel):
    """Output payload produced by the key-frame extraction agent."""

    key_frames: List[KeyFrameMetadata] = Field(
        default_factory=list,
        description="List of extracted key frame metadata entries",
    )
    keyframes_folder: str = Field(description="Directory containing saved key frame images")
    total_frames: Optional[int] = Field(
        default=None,
        description="Total number of frames in the source video if available",
    )
    frames_evaluated: Optional[int] = Field(
        default=None,
        description="How many frames were evaluated while selecting key frames",
    )
    fps: Optional[float] = Field(
        default=None,
        description="Frames-per-second used for timestamp calculations",
    )
    output_folder: Optional[str] = Field(
        default=None,
        description="Base workflow output folder propagated to downstream agents",
    )


class FaceDetectionStats(BaseModel):
    """Output payload produced by the face detection agent."""

    faces_extracted: int = Field(default=0, description="Number of cropped faces saved to disk")
    frames_with_faces: int = Field(default=0, description="Frames that contained at least one face")
    frames_without_faces: int = Field(default=0, description="Frames that contained no detectable faces")
    faces_rejected_low_confidence: int = Field(
        default=0, description="Faces discarded because confidence fell below threshold"
    )
    faces_rejected_small: int = Field(
        default=0, description="Faces discarded because bounding box size was too small"
    )
    output_folder: str = Field(description="Directory where extracted faces were saved")


class FaceDeduplicationStats(BaseModel):
    """Output payload produced by the face deduplication agent."""

    total_faces: int = Field(default=0, description="Number of face images analysed")
    unique_faces: int = Field(default=0, description="Unique faces retained after deduplication")
    duplicates_removed: int = Field(default=0, description="Duplicate faces removed")
    detector_fallbacks: int = Field(default=0, description="Count of fallback detector invocations")
    embedding_failures: int = Field(default=0, description="Images for which embeddings could not be computed")
    time_elapsed: float = Field(default=0.0, description="Time taken for deduplication in seconds")
    threshold_used: Optional[float] = Field(default=None, description="Distance threshold applied during deduplication")
    output_folder: Optional[str] = Field(default=None, description="Directory containing the final deduplicated face set")


class VideoProcessingSummary(BaseModel):
    """Aggregated output returned by the video workflow."""

    keyframes: Optional[KeyFrameExtractionOutput] = Field(
        default=None,
        description="Key frame extraction stage output (if available)",
    )
    detection: Optional[FaceDetectionStats] = Field(
        default=None,
        description="Face detection stage output (if available)",
    )
    deduplication: FaceDeduplicationStats = Field(
        description="Deduplication statistics from the final workflow stage"
    )
    output_folder: str = Field(description="Root workflow output directory for artefacts")
    time_total: float = Field(description="Total wall-clock time spent in the workflow")

    def to_dict(self) -> dict:
        """Convert the video processing summary to a native dictionary."""
        return self.model_dump()

    def to_json(self, indent: int = 2) -> str:
        """Convert the video processing summary to a JSON string."""
        return self.model_dump_json(indent=indent)
