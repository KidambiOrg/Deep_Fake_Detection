"""
Test script to extract faces from video using Agent Framework workflow.
"""

import sys
import os
from pathlib import Path
import asyncio

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Agents.VideoAgents import run_video_processing_workflow


async def main():
    """Main function to test video processing workflow."""
    
    # Example usage
    video_path = r"path\to\your\video.mp4"
    
    # Check if video path was provided as command line argument
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    # Quality thresholds
    min_confidence = 0.5
    min_face_size = 50
    max_frames = None  # Process all frames
    max_workers = 16
    keyframe_threshold = 0.6
    keyframe_method = "chisq"
    keyframe_min_gap = 1.0
    
    # Get parameters from command line if provided
    if len(sys.argv) > 2:
        min_confidence = float(sys.argv[2])
    if len(sys.argv) > 3:
        max_frames = int(sys.argv[3]) if sys.argv[3].lower() != 'none' else None
    if len(sys.argv) > 4:
        max_workers = int(sys.argv[4])
    if len(sys.argv) > 5:
        keyframe_threshold = float(sys.argv[5])
    if len(sys.argv) > 6:
        keyframe_method = sys.argv[6]
    if len(sys.argv) > 7:
        keyframe_min_gap = float(sys.argv[7])
    
    print(f"Quality Settings:")
    print(f"  Min Confidence: {min_confidence}")
    print(f"  Min Face Size: {min_face_size}")
    print(f"  Max Frames: {max_frames if max_frames else 'All'}")
    print(f"  Parallel Workers: {max_workers}")
    print(f"Key Frame Settings:")
    print(f"  Threshold: {keyframe_threshold}")
    print(f"  Method: {keyframe_method}")
    print(f"  Min Gap (s): {keyframe_min_gap}")
    print()
    
    try:
        # Run workflow
        summary = await run_video_processing_workflow(
            video_path=video_path,
            min_confidence=min_confidence,
            min_face_size=min_face_size,
            max_frames=max_frames,
            max_workers=max_workers,
            keyframe_threshold=keyframe_threshold,
            keyframe_method=keyframe_method,
            keyframe_min_gap=keyframe_min_gap
        )
        
        results = summary.model_dump()
        detection_stats = results.get('detection') or {}
        dedup_stats = results.get('deduplication') or {}

        print("\nFace extraction completed successfully!")
        print("\nResults:")
        print(f"  Key Frames: {len((results.get('keyframes') or {}).get('key_frames', []))}")
        print(f"  Faces Extracted: {detection_stats.get('faces_extracted', 0)}")
        print(f"  Total Faces Analysed: {dedup_stats.get('total_faces', 0)}")
        print(f"  Unique Faces Kept: {dedup_stats.get('unique_faces', 0)}")
        print(f"  Duplicates Removed: {dedup_stats.get('duplicates_removed', 0)}")
        print(f"  Output Folder: {results.get('output_folder', 'N/A')}")
        print(f"  Total Time: {results.get('time_total', 0.0):.2f}s")
        
    except Exception as e:
        print(f"\nError during face extraction: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
