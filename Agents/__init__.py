"""Agent Framework Workflows for Deepfake Detection.

This package provides concurrent workflow implementations for deepfake detection
using Microsoft Agent Framework SDK.

Image Detection:
    - NoEscape, SEResNet50, Prithiv V1, DimaVit, Organika SDXL, MobileNetV2, MobileNetV2 CNN, Prithiv Open Deepfake models
    - Ensemble voting with 14+ methods (majority, weighted, bayesian, etc.)
    - Returns ImageDetectionResult with markdown report

Text Detection:
    - BERT-ML, BoW-ML, CharCNN, DistilBERT models
    - Ensemble averaging
    - Returns TextDetectionResult with markdown report

Usage:
    from Agents import run_image_detection_workflow, run_text_detection_workflow
    
    # Image detection
    result = await run_image_detection_workflow(
        image_path="image.jpg",
        threshold=0.5,
        voting_method='majority'
    )
    
    # Text detection
    result = await run_text_detection_workflow(
        text="Sample text...",
        threshold=0.5
    )
"""

from Agents.ImageAgents.image_workflow import run_image_detection_workflow
from Agents.TextAgents.text_workflow import run_text_detection_workflow

__all__ = [
    'run_image_detection_workflow',
    'run_text_detection_workflow'
]
