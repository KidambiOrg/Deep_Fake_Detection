"""CharCNN detection tool for text deepfake detection."""

from agent_framework import ai_function
from outputModel.text_detection_output_model import IndividualModelScore
import logging

logger = logging.getLogger(__name__)

_charcnn_detector = None

def load_charcnn_detector():
    """Load CharCNN detector with caching."""
    global _charcnn_detector
    
    if _charcnn_detector is not None:
        return _charcnn_detector
    
    from textdetectionlib.detector_charcnn import DetectorCharCNN
    _charcnn_detector = DetectorCharCNN()
    
    logger.info("Loaded CharCNN detector")
    return _charcnn_detector

@ai_function
def detect_with_charcnn_tool(text: str) -> IndividualModelScore:
    """Agent tool: Detect AI-generated text using CharCNN model."""
    try:
        detector = load_charcnn_detector()
        result = detector.detect(text)
        
        return IndividualModelScore(
            model='charcnn',
            fakeScore=result.fake_score,
            realScore=result.real_score
        )
    except Exception as e:
        logger.error(f"CharCNN detection error: {e}")
        return IndividualModelScore(model='charcnn', fakeScore=0.0, realScore=0.0)

def detect_with_charcnn(text: str) -> IndividualModelScore:
    """Wrapper function for CharCNN detection."""
    return detect_with_charcnn_tool(text)
