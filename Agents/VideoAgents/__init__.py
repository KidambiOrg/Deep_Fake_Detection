"""Video Processing Agents for face extraction from videos.

This package provides sequential workflow for video face extraction:
1. KeyFrameExtractorAgent - Extracts key frames from video
2. FaceDetectorAgent - Detects faces in frames
3. FaceDeduplicatorAgent - Removes duplicate faces

Usage:
    from Agents.VideoAgents import run_video_processing_workflow
    
    result = await run_video_processing_workflow(
        video_path="video.mp4",
        output_folder="faces"
    )
"""

from Agents.VideoAgents.video_workflow import run_video_processing_workflow

__all__ = ['run_video_processing_workflow']
