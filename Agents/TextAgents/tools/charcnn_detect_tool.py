"""CharCNN detection tool for text deepfake detection."""

from agent_framework import ai_function
from outputModel.text_detection_output_model import IndividualModelScore
import logging

logger = logging.getLogger(__name__)

_charcnn_detector = None

def load_charcnn_detector():
    """
    Load CharCNN detector with caching.
    Downloads model from blob storage on first use and caches it.
    """
    global _charcnn_detector
    
    if _charcnn_detector is not None:
        logger.info("Using cached CharCNN detector")
        return _charcnn_detector
    
    logger.info("Loading CharCNN detector from blob storage...")
    from textdetectionlib.detector_charcnn import DetectorCharCNN
    
    # Initialize with blob storage loading enabled
    # This will download from blob storage if models aren't cached locally
    _charcnn_detector = DetectorCharCNN(load_from_blob=True)
    
    logger.info("CharCNN detector loaded and cached successfully")
    return _charcnn_detector

@ai_function
def detect_with_charcnn_tool(text: str) -> dict:
    """Agent tool: Detect AI-generated text using CharCNN model.
    Returns dictionary representation for JSON serialization."""
    try:
        detector = load_charcnn_detector()
        result = detector.detect(text)
        
        score = IndividualModelScore(
            model_name='CharCNN',
            prediction=result.prediction,
            confidence=result.confidence,
            bot_score=result.bot_score,
            human_score=result.human_score
        )
        return score.model_dump()
    except Exception as e:
        logger.error(f"CharCNN detection error: {e}")
        error_score = IndividualModelScore(
            model_name='CharCNN',
            prediction='human',
            confidence=0.0,
            bot_score=0.0,
            human_score=0.0
        )
        return error_score.model_dump()

def detect_with_charcnn(text: str) -> IndividualModelScore:
    """Wrapper function for CharCNN detection."""
    return detect_with_charcnn_tool(text)
