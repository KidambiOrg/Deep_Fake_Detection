"""BoW-ML detection tool for text deepfake detection."""

from agent_framework import ai_function
from outputModel.text_detection_output_model import IndividualModelScore
import logging

logger = logging.getLogger(__name__)

_bow_ml_detector = None

def load_bow_ml_detector():
    """
    Load BoW-ML detector with caching.
    Downloads model from blob storage on first use and caches it.
    """
    global _bow_ml_detector
    
    if _bow_ml_detector is not None:
        logger.info("Using cached BoW-ML detector")
        return _bow_ml_detector
    
    logger.info("Loading BoW-ML detector from blob storage...")
    from textdetectionlib.detector_bow_ml import DetectorBowML
    
    # Initialize with blob storage loading enabled
    # This will download from blob storage if models aren't cached locally
    _bow_ml_detector = DetectorBowML(load_from_blob=True)
    
    logger.info("BoW-ML detector loaded and cached successfully")
    return _bow_ml_detector

@ai_function
def detect_with_bow_ml_tool(text: str) -> dict:
    """Agent tool: Detect AI-generated text using BoW-ML model.
    Returns dictionary representation for JSON serialization."""
    try:
        detector = load_bow_ml_detector()
        result = detector.detect(text)
        
        score = IndividualModelScore(
            model_name='BoW-ML',
            prediction=result.prediction,
            confidence=result.confidence,
            bot_score=result.bot_score,
            human_score=result.human_score
        )
        return score.model_dump()
    except Exception as e:
        logger.error(f"BoW-ML detection error: {e}")
        error_score = IndividualModelScore(
            model_name='BoW-ML',
            prediction='human',
            confidence=0.0,
            bot_score=0.0,
            human_score=0.0
        )
        return error_score.model_dump()

def detect_with_bow_ml(text: str) -> IndividualModelScore:
    """Wrapper function for BoW-ML detection."""
    return detect_with_bow_ml_tool(text)
