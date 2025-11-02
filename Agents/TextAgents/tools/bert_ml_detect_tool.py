"""BERT-ML detection tool for text deepfake detection."""

from agent_framework import ai_function
from outputModel.text_detection_output_model import IndividualModelScore
import logging

logger = logging.getLogger(__name__)

_bert_ml_detector = None

def load_bert_ml_detector():
    """Load BERT-ML detector with caching."""
    global _bert_ml_detector
    
    if _bert_ml_detector is not None:
        return _bert_ml_detector
    
    from textdetectionlib.detector_bert_ml import DetectorBertML
    _bert_ml_detector = DetectorBertML()
    
    logger.info("Loaded BERT-ML detector")
    return _bert_ml_detector

@ai_function
def detect_with_bert_ml_tool(text: str) -> IndividualModelScore:
    """Agent tool: Detect AI-generated text using BERT-ML model."""
    try:
        detector = load_bert_ml_detector()
        result = detector.detect(text)
        
        return IndividualModelScore(
            model='bert-ml',
            fakeScore=result.fake_score,
            realScore=result.real_score
        )
    except Exception as e:
        logger.error(f"BERT-ML detection error: {e}")
        return IndividualModelScore(model='bert-ml', fakeScore=0.0, realScore=0.0)

def detect_with_bert_ml(text: str) -> IndividualModelScore:
    """Wrapper function for BERT-ML detection."""
    return detect_with_bert_ml_tool(text)
