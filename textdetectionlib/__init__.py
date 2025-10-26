"""
Text Detection Library for AI-generated text detection
"""
from .detector_bert_ml import DetectorBertML, BertMLResult
from .detector_bow_ml import DetectorBowML, BowMLResult
from .detector_charcnn import DetectorCharCNN, CharCNNResult
from .detector_distilbert import DetectorDistilBertML, DistilBertMLResult
from .text_detection_ensemble import TextDetectionEnsemble
from .model_registry import (
    TEXT_MODEL_REGISTRY,
    list_available_text_models,
    get_text_model_info,
    get_text_models_by_architecture
)

# Import output models for convenience
from outputModel.text_detection_output_model import TextDetectionResult, IndividualModelScore

__all__ = [
    'DetectorBertML',
    'BertMLResult',
    'DetectorBowML',
    'BowMLResult',
    'DetectorCharCNN',
    'CharCNNResult',
    'DetectorDistilBertML',
    'DistilBertMLResult',
    'TextDetectionEnsemble',
    'TextDetectionResult',
    'IndividualModelScore',
    'TEXT_MODEL_REGISTRY',
    'list_available_text_models',
    'get_text_model_info',
    'get_text_models_by_architecture'
]
