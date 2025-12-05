"""Video Processing Workflow using Sequential Builder.

Sequential workflow for face extraction from videos:
1. KeyFrameExtractorAgent - Extracts key frames
2. FaceDetectorAgent - Detects faces in frames
3. FaceDeduplicatorAgent - Removes duplicate faces
"""

from agent_framework import SequentialBuilder
from Agents.VideoAgents.keyframe_extractor_agent import create_keyframe_extractor_agent
from Agents.VideoAgents.face_detector_agent import create_face_detector_agent
from Agents.VideoAgents.face_deduplicator_agent import create_face_deduplicator_agent
from outputModel.video_detection_output_model import (
    KeyFrameExtractionOutput,
    FaceDetectionStats,
    FaceDeduplicationStats,
    VideoProcessingSummary,
)
from pydantic import ValidationError
import json
import logging
import os
from pathlib import Path
import time
import threading

logger = logging.getLogger(__name__)


def _extract_tool_output(message_obj):
    """Extract JSON payload from agent chat message structures."""

    if message_obj is None:
        return None

    # Convert dataclass/pydantic objects to dictionaries when possible
    candidate = message_obj
    if hasattr(candidate, "model_dump"):
        try:
            candidate = candidate.model_dump()
        except Exception as dump_err:  # pragma: no cover - defensive logging
            logger.warning(f"Failed model_dump on message: {dump_err}")
            candidate = message_obj
    elif hasattr(candidate, "to_dict"):
        try:
            candidate = candidate.to_dict()
        except Exception as to_dict_err:  # pragma: no cover
            logger.warning(f"Failed to_dict on message: {to_dict_err}")
            candidate = message_obj
    elif hasattr(candidate, "__dict__") and not isinstance(candidate, dict):
        candidate = candidate.__dict__

    if not isinstance(candidate, dict):
        return None

    # Content may be nested; look for output_text entries containing JSON
    content = candidate.get("content")
    if isinstance(content, list):
        for idx, content_item in enumerate(content):
            if hasattr(content_item, "model_dump"):
                try:
                    content_item = content_item.model_dump()
                except Exception:
                    content_item = getattr(content_item, "__dict__", content_item)
            elif hasattr(content_item, "to_dict"):
                try:
                    content_item = content_item.to_dict()
                except Exception:
                    content_item = getattr(content_item, "__dict__", content_item)
            elif hasattr(content_item, "__dict__") and not isinstance(content_item, dict):
                content_item = content_item.__dict__

            if not isinstance(content_item, dict):
                continue

            item_type = content_item.get("type")
            if item_type == "tool_result":
                inner = content_item.get("content")
                if isinstance(inner, list):
                    for inner_item in inner:
                        if hasattr(inner_item, "model_dump"):
                            try:
                                inner_item = inner_item.model_dump()
                            except Exception:
                                inner_item = getattr(inner_item, "__dict__", inner_item)
                        elif hasattr(inner_item, "to_dict"):
                            try:
                                inner_item = inner_item.to_dict()
                            except Exception:
                                inner_item = getattr(inner_item, "__dict__", inner_item)
                        elif hasattr(inner_item, "__dict__") and not isinstance(inner_item, dict):
                            inner_item = inner_item.__dict__

                        if not isinstance(inner_item, dict):
                            continue

                        json_value = inner_item.get("json") or inner_item.get("output_json")
                        if isinstance(json_value, (dict, list)):
                            return json_value
                        if isinstance(json_value, str):
                            parsed_json = _try_parse_json(json_value)
                            if parsed_json is not None:
                                return parsed_json

                        text_value = inner_item.get("text") or inner_item.get("output_text")
                        if isinstance(text_value, str) and text_value.strip():
                            try:
                                return json.loads(text_value)
                            except json.JSONDecodeError as decode_err:
                                logger.warning(
                                    f"Failed to decode tool output JSON (content index {idx}): {decode_err}"
                                )
                                continue

            # Some messages may store text at the same level
            text_value = content_item.get("text") or content_item.get("output_text")
            if isinstance(text_value, str) and text_value.strip():
                parsed_text = _try_parse_json(text_value)
                if parsed_text is not None:
                    return parsed_text
                else:
                    logger.warning(
                        "Failed to decode JSON text in content index %s", idx
                    )

            arguments_value = content_item.get("arguments")
            if isinstance(arguments_value, str) and arguments_value.strip():
                parsed_arguments = _try_parse_json(arguments_value)
                if parsed_arguments is not None:
                    return parsed_arguments

    # Fallback: check for top-level text field
    top_text = candidate.get("text")
    if isinstance(top_text, str) and top_text.strip():
        parsed_text = _try_parse_json(top_text)
        if parsed_text is not None:
            return parsed_text
        logger.warning("Failed to decode top-level message text")

    return None

def _try_parse_json(text: str):
    """Attempt strict then relaxed JSON parsing for text payloads."""

    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = text[first : last + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    return None


def _safe_snapshot(obj):
    """Return a lightweight representation of workflow objects for logging."""

    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, (list, tuple)):
        return [_safe_snapshot(item) for item in obj]

    if isinstance(obj, dict):
        return {
            key: _safe_snapshot(value)
            for key, value in obj.items()
        }

    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass

    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass

    if hasattr(obj, "__dict__") and not isinstance(obj, dict):
        try:
            return {
                key: value
                for key, value in vars(obj).items()
                if not key.startswith("_")
            }
        except Exception:
            pass

    return str(obj)


# Global agent cache - initialized once and reused
_cached_video_agents = None
_video_agents_lock = threading.Lock()


def get_or_create_video_agents():
    """
    Get or create video processing agents with singleton pattern.
    Agents are loaded only once and cached globally.
    Thread-safe initialization prevents duplicate loading.
    
    Returns:
        Tuple of (keyframe_agent, face_detector_agent, face_deduplicator_agent)
    """
    global _cached_video_agents
    
    # Fast path: if agents already exist, return immediately
    if _cached_video_agents is not None:
        logger.info("Reusing cached video agents")
        return _cached_video_agents
    
    # Slow path: need to create agents (with lock to prevent concurrent initialization)
    with _video_agents_lock:
        # Double-check after acquiring lock
        if _cached_video_agents is not None:
            logger.info("Reusing cached video agents (after lock)")
            return _cached_video_agents
        
        logger.info("Initializing video processing agents for the first time...")
        
        # Create agents once
        keyframe_agent = create_keyframe_extractor_agent()
        face_detector_agent = create_face_detector_agent()
        face_deduplicator_agent = create_face_deduplicator_agent()
        
        # Cache the agents globally
        _cached_video_agents = (
            keyframe_agent,
            face_detector_agent,
            face_deduplicator_agent
        )
        


        def _safe_snapshot(obj):
            """Return a lightweight representation of workflow objects for logging."""

            if obj is None:
                return None

            if isinstance(obj, (dict, list, str, int, float, bool)):
                return obj

            if hasattr(obj, "model_dump"):
                try:
                    return obj.model_dump()
                except Exception:
                    pass

            if hasattr(obj, "to_dict"):
                try:
                    return obj.to_dict()
                except Exception:
                    pass

            if hasattr(obj, "__dict__") and not isinstance(obj, dict):
                try:
                    return {
                        key: value
                        for key, value in vars(obj).items()
                        if not key.startswith("_")
                    }
                except Exception:
                    pass

            return str(obj)
        logger.info("All video processing agents created and cached successfully")
        return _cached_video_agents


async def run_video_processing_workflow(
    video_path: str,
    output_folder: str | None = None,
    min_confidence: float = 0.50,
    min_face_size: int = 50,
    max_frames: int | None = None,
    max_workers: int = 4,
    keyframe_threshold: float = 0.6,
    keyframe_method: str = "chisq",
    keyframe_min_gap: float = 1.0
) -> VideoProcessingSummary:
    """
    Run sequential video processing workflow.
    
    Args:
        video_path: Path to video file
        output_folder: Output folder for extracted faces
        min_confidence: Minimum face detection confidence
        min_face_size: Minimum face size in pixels
        max_frames: Maximum frames to process (None = all)
        max_workers: Number of parallel workers
        keyframe_threshold: Histogram threshold for key frame selection
        keyframe_method: Histogram comparison method
        keyframe_min_gap: Minimum gap between key frames (seconds)
    
    Returns:
        Dictionary with processing results
    """
    logger.info(f"Starting video processing workflow for: {video_path}")
    start_time = time.time()
    
    temp_video_cleanup = None
    
    try:
        # Validate video path
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Track if video is in temp folder for cleanup
        if "temp" in video_path.lower() or "tmp" in video_path.lower():
            temp_video_cleanup = video_path
        
        # Setup output folder
        if output_folder is None:
            output_folder = str(Path(video_path).parent / "extracted_faces")
        
        os.makedirs(output_folder, exist_ok=True)
        
        # Get or create cached agents
        logger.info("Getting video processing agents...")
        keyframe_agent, face_detector_agent, face_deduplicator_agent = get_or_create_video_agents()
        
        # Build sequential workflow
        logger.info("Building sequential workflow...")
        workflow = SequentialBuilder().participants([
            keyframe_agent,
            face_detector_agent,
            face_deduplicator_agent
        ]).build()
        
        # Prepare workflow input as JSON string (Agent Framework requirement)
        import json
        workflow_input_dict = {
            "video_path": video_path,
            "output_folder": output_folder,
            "max_frames": max_frames,
            "method": keyframe_method,
            "threshold": keyframe_threshold,
            "min_gap_sec": keyframe_min_gap,
            "min_confidence": min_confidence,
            "min_face_size": min_face_size,
            "max_workers": max_workers
        }
        workflow_input = json.dumps(workflow_input_dict)
        
        logger.info("Running workflow...")
        workflow_result = await workflow.run(workflow_input)

        try:
            events = workflow_result.get_events()
            logger.debug(
                "Workflow events type: %s, count: %s",
                type(events),
                len(events) if events is not None else 0,
            )
        except AttributeError:
            events = None
            logger.debug("Workflow result has no get_events() method")
        
        # Extract results from workflow events
        outputs = workflow_result.get_outputs()
        logger.info(
            "Workflow outputs type: %s, length: %s",
            type(outputs),
            len(outputs) if outputs else 0,
        )

        if not outputs:
            raise RuntimeError("Workflow did not produce any output")

        def _coerce_payload(output_obj):
            """Convert workflow output object into a JSON-compatible dictionary."""
            # First, attempt to pull structured payloads from agent chat messages
            extracted = _extract_tool_output(output_obj)
            if extracted is not None:
                return extracted

            if isinstance(output_obj, list):
                logger.debug("Output is list with %s elements; scanning for payload", len(output_obj))
                for nested in reversed(output_obj):
                    nested_payload = _coerce_payload(nested)
                    if isinstance(nested_payload, dict):
                        return nested_payload
                return None

            if isinstance(output_obj, (KeyFrameExtractionOutput, FaceDetectionStats, FaceDeduplicationStats)):
                return output_obj.model_dump()

            if isinstance(output_obj, dict):
                return output_obj

            if isinstance(output_obj, str):
                text_value = output_obj.strip()
                if not text_value:
                    return None
                parsed = _try_parse_json(text_value)
                if parsed is None:
                    logger.debug("Failed to parse string output as JSON after relaxed parsing")
                return parsed

            if hasattr(output_obj, "text"):
                text_value = getattr(output_obj, "text")
                if isinstance(text_value, str) and text_value.strip():
                    parsed = _try_parse_json(text_value)
                    if parsed is not None:
                        return parsed

            if hasattr(output_obj, "model_dump"):
                try:
                    return output_obj.model_dump()
                except Exception:  # pragma: no cover - defensive
                    pass

            if hasattr(output_obj, "to_dict"):
                try:
                    return output_obj.to_dict()
                except Exception:  # pragma: no cover - defensive
                    pass

            if hasattr(output_obj, "__dict__") and not isinstance(output_obj, dict):
                candidate = {
                    key: value
                    for key, value in vars(output_obj).items()
                    if not key.startswith("_")
                }
                if candidate:
                    return candidate

            return None

        keyframe_output: KeyFrameExtractionOutput | None = None
        detection_output: FaceDetectionStats | None = None
        dedup_output: FaceDeduplicationStats | None = None
        last_dict_payload: dict | None = None

        for idx, raw_entry in enumerate(outputs):
            snapshot = _safe_snapshot(raw_entry)
            if isinstance(snapshot, dict):
                snapshot_keys = list(snapshot.keys())
            else:
                snapshot_keys = type(snapshot)
            logger.debug("Workflow raw output[%s] snapshot keys/type: %s", idx, snapshot_keys)

            payload = _coerce_payload(raw_entry)
            logger.debug("Processed workflow output %s -> payload keys: %s", idx, list(payload.keys()) if isinstance(payload, dict) else type(payload))

            if not isinstance(payload, dict):
                continue

            try:
                nested_payload = payload.get("result") if isinstance(payload, dict) else None
                if isinstance(nested_payload, dict):
                    logger.debug("Found nested 'result' payload; merging for detection")
                    payload_candidate = nested_payload
                else:
                    payload_candidate = payload

                last_dict_payload = payload_candidate

                if (
                    keyframe_output is None
                    and "key_frames" in payload_candidate
                    and "keyframes_folder" in payload_candidate
                ):
                    keyframe_output = KeyFrameExtractionOutput.model_validate(payload_candidate)
                    logger.info("Captured key frame extraction output from workflow")
                    continue

                if (
                    detection_output is None
                    and {"faces_extracted", "frames_with_faces", "frames_without_faces"}.issubset(payload_candidate.keys())
                ):
                    detection_output = FaceDetectionStats.model_validate(payload_candidate)
                    logger.info("Captured face detection output from workflow")
                    continue

                if (
                    {"duplicates_removed", "unique_faces", "total_faces"}.issubset(payload_candidate.keys())
                    and "output_folder" in payload_candidate
                ):
                    dedup_output = FaceDeduplicationStats.model_validate(payload_candidate)
                    logger.info("Captured deduplication output from workflow")
                    continue
            except ValidationError as validation_err:
                logger.warning(
                    "Failed to validate workflow payload for stage detection (index %s): %s",
                    idx,
                    validation_err,
                )
                continue

        if keyframe_output is None:
            keyframe_metadata_paths = [
                Path(output_folder) / "keyframes" / "metadata.json",
            ]
            for metadata_path in keyframe_metadata_paths:
                if metadata_path.exists():
                    try:
                        with open(metadata_path, "r", encoding="utf-8") as metadata_file:
                            metadata_data = json.load(metadata_file)
                        keyframe_output = KeyFrameExtractionOutput.model_validate(metadata_data)
                        logger.info("Loaded key frame metadata from %s", metadata_path)
                        break
                    except Exception as metadata_err:
                        logger.warning(
                            "Failed to load key frame metadata from %s: %s",
                            metadata_path,
                            metadata_err,
                        )

        if detection_output is None:
            detection_stats_path = Path(output_folder) / "face_detection_stats.json"
            if detection_stats_path.exists():
                try:
                    with open(detection_stats_path, "r", encoding="utf-8") as stats_file:
                        detection_data = json.load(stats_file)
                    detection_output = FaceDetectionStats.model_validate(detection_data)
                    logger.info("Loaded face detection stats from %s", detection_stats_path)
                except Exception as detection_err:
                    logger.warning(
                        "Failed to load face detection stats from %s: %s",
                        detection_stats_path,
                        detection_err,
                    )

        if dedup_output is None:
            if last_dict_payload is not None:
                try:
                    dedup_output = FaceDeduplicationStats.model_validate(last_dict_payload)
                    logger.warning(
                        "Deduplication output not explicitly detected; using fallback payload keys: %s",
                        list(last_dict_payload.keys()),
                    )
                except ValidationError:
                    pass

        if dedup_output is None:
            dedup_stats_path = Path(output_folder) / "deduplication_stats.json"
            if dedup_stats_path.exists():
                try:
                    with open(dedup_stats_path, "r", encoding="utf-8") as stats_file:
                        dedup_data = json.load(stats_file)
                    dedup_output = FaceDeduplicationStats.model_validate(dedup_data)
                    logger.info("Loaded deduplication stats from %s", dedup_stats_path)
                except Exception as dedup_err:
                    logger.warning(
                        "Failed to load deduplication stats from %s: %s",
                        dedup_stats_path,
                        dedup_err,
                    )

        if dedup_output is None:
            logger.error(
                "Could not locate deduplication output. Output snapshots: %s",
                [
                    _safe_snapshot(entry)
                    for entry in outputs
                ],
            )
            raise RuntimeError("Could not locate deduplication output in workflow results")

        elapsed_time = time.time() - start_time
        resolved_output_folder = dedup_output.output_folder or output_folder

        summary = VideoProcessingSummary(
            keyframes=keyframe_output,
            detection=detection_output,
            deduplication=dedup_output,
            output_folder=resolved_output_folder,
            time_total=elapsed_time,
        )

        logger.info("Workflow completed in %.2fs", elapsed_time)
        return summary
    
    except Exception as e:
        logger.error(f"Error in video processing workflow: {e}")
        raise
    
    finally:
        # Cleanup: Delete video file if it was in temp folder
        if temp_video_cleanup and os.path.exists(temp_video_cleanup):
            try:
                os.unlink(temp_video_cleanup)
                logger.info(f"Cleaned up temporary video file: {temp_video_cleanup}")
            except Exception as cleanup_err:
                logger.warning(f"Failed to clean up video file: {cleanup_err}")
