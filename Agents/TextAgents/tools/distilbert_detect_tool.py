"""DistilBERT detection tool for text deepfake detection."""

from agent_framework import ai_function
from outputModel.text_detection_output_model import IndividualModelScore
import logging

logger = logging.getLogger(__name__)

_distilbert_detector = None

def load_distilbert_detector():
    """
    Load DistilBERT detector with caching.
    Downloads model from blob storage on first use and caches it.
    """
    global _distilbert_detector
    
    if _distilbert_detector is not None:
        logger.info("Using cached DistilBERT detector")
        return _distilbert_detector
    
    logger.info("Loading DistilBERT detector from blob storage...")
    from textdetectionlib.detector_distilbert import DetectorDistilBertML
    
    # Initialize with blob storage loading enabled
    # This will download from blob storage if models aren't cached locally
    _distilbert_detector = DetectorDistilBertML(load_from_blob=True)
    
    logger.info("DistilBERT detector loaded and cached successfully")
    return _distilbert_detector

@ai_function
def detect_with_distilbert_tool(text: str) -> dict:
    """Agent tool: Detect AI-generated text using DistilBERT model.
    Returns dictionary representation for JSON serialization."""
    try:
        detector = load_distilbert_detector()
        result = detector.detect(text)
        
        score = IndividualModelScore(
            model_name='DistilBERT',
            prediction=result.prediction,
            confidence=result.confidence,
            bot_score=result.bot_score,
            human_score=result.human_score
        )
        return score.model_dump()
    except Exception as e:
        logger.error(f"DistilBERT detection error: {e}")
        error_score = IndividualModelScore(
            model_name='DistilBERT',
            prediction='human',
            confidence=0.0,
            bot_score=0.0,
            human_score=0.0
        )
        return error_score.model_dump()

def detect_with_distilbert(text: str) -> IndividualModelScore:
    """Wrapper function for DistilBERT detection."""
    return detect_with_distilbert_tool(text)
