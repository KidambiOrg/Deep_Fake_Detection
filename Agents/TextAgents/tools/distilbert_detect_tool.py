"""DistilBERT detection tool for text deepfake detection."""

from agent_framework import ai_function
from outputModel.text_detection_output_model import IndividualModelScore
import logging

logger = logging.getLogger(__name__)

_distilbert_detector = None

def load_distilbert_detector():
    """Load DistilBERT detector with caching."""
    global _distilbert_detector
    
    if _distilbert_detector is not None:
        return _distilbert_detector
    
    from textdetectionlib.detector_distilbert import DetectorDistilBertML
    _distilbert_detector = DetectorDistilBertML()
    
    logger.info("Loaded DistilBERT detector")
    return _distilbert_detector

@ai_function
def detect_with_distilbert_tool(text: str) -> IndividualModelScore:
    """Agent tool: Detect AI-generated text using DistilBERT model."""
    try:
        detector = load_distilbert_detector()
        result = detector.detect(text)
        
        return IndividualModelScore(
            model='distilbert',
            fakeScore=result.fake_score,
            realScore=result.real_score
        )
    except Exception as e:
        logger.error(f"DistilBERT detection error: {e}")
        return IndividualModelScore(model='distilbert', fakeScore=0.0, realScore=0.0)

def detect_with_distilbert(text: str) -> IndividualModelScore:
    """Wrapper function for DistilBERT detection."""
    return detect_with_distilbert_tool(text)
