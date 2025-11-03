"""BERT-ML detection tool for text deepfake detection."""

from agent_framework import ai_function
from outputModel.text_detection_output_model import IndividualModelScore
import logging

logger = logging.getLogger(__name__)

_bert_ml_detector = None

def load_bert_ml_detector():
    """
    Load BERT-ML detector with caching.
    Downloads model from blob storage on first use and caches it.
    """
    global _bert_ml_detector
    
    if _bert_ml_detector is not None:
        logger.info("Using cached BERT-ML detector")
        return _bert_ml_detector
    
    logger.info("Loading BERT-ML detector from blob storage...")
    from textdetectionlib.detector_bert_ml import DetectorBertML
    
    # Initialize with blob storage loading enabled
    # This will download from blob storage if models aren't cached locally
    _bert_ml_detector = DetectorBertML(load_from_blob=True)
    
    logger.info("BERT-ML detector loaded and cached successfully")
    return _bert_ml_detector

@ai_function
def detect_with_bert_ml_tool(text: str) -> dict:
    """Agent tool: Detect AI-generated text using BERT-ML model.
    Returns dictionary representation for JSON serialization."""
    try:
        detector = load_bert_ml_detector()
        result = detector.detect(text)
        
        score = IndividualModelScore(
            model_name='BERT-ML',
            prediction=result.prediction,
            confidence=result.confidence,
            bot_score=result.bot_score,
            human_score=result.human_score
        )
        return score.model_dump()
    except Exception as e:
        logger.error(f"BERT-ML detection error: {e}")
        error_score = IndividualModelScore(
            model_name='BERT-ML',
            prediction='human',
            confidence=0.0,
            bot_score=0.0,
            human_score=0.0
        )
        return error_score.model_dump()

def detect_with_bert_ml(text: str) -> IndividualModelScore:
    """Wrapper function for BERT-ML detection."""
    return detect_with_bert_ml_tool(text)
