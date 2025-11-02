"""BoW-ML detection tool for text deepfake detection."""

from agent_framework import ai_function
from outputModel.text_detection_output_model import IndividualModelScore
import logging

logger = logging.getLogger(__name__)

_bow_ml_detector = None

def load_bow_ml_detector():
    """Load BoW-ML detector with caching."""
    global _bow_ml_detector
    
    if _bow_ml_detector is not None:
        return _bow_ml_detector
    
    from textdetectionlib.detector_bow_ml import DetectorBowML
    _bow_ml_detector = DetectorBowML()
    
    logger.info("Loaded BoW-ML detector")
    return _bow_ml_detector

@ai_function
def detect_with_bow_ml_tool(text: str) -> IndividualModelScore:
    """Agent tool: Detect AI-generated text using BoW-ML model."""
    try:
        detector = load_bow_ml_detector()
        result = detector.detect(text)
        
        return IndividualModelScore(
            model='bow-ml',
            fakeScore=result.fake_score,
            realScore=result.real_score
        )
    except Exception as e:
        logger.error(f"BoW-ML detection error: {e}")
        return IndividualModelScore(model='bow-ml', fakeScore=0.0, realScore=0.0)

def detect_with_bow_ml(text: str) -> IndividualModelScore:
    """Wrapper function for BoW-ML detection."""
    return detect_with_bow_ml_tool(text)
