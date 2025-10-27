"""
Main application for text deepfake detection
Takes string input and outputs JSON with classification results
"""
import json
import argparse
import sys
import os
import logging
from typing import Optional, Literal
from pathlib import Path
import shutil

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from textdetectionlib.text_detection_ensemble import TextDetectionEnsemble
from outputModel.text_detection_output_model import TextDetectionResult, TextMetadata
from utils.blob_storage_helper import get_blob_loader
from utils.text_analysis_helper import add_metadata_and_markdown
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TextDetectionApp:
    """
    Application for detecting AI-generated text using an ensemble of models
    """
    
    def __init__(
        self,
        models_dir: Optional[str] = None,
        voting_method: Literal["majority", "weighted", "unanimous"] = "weighted",
        use_blob_storage: bool = True
    ):
        """
        Initialize the text detection application
        
        Args:
            models_dir: Directory containing trained models
            voting_method: Voting strategy for ensemble ("majority", "weighted", "unanimous")
            use_blob_storage: Whether to download models from Azure Blob Storage
        """
        self.models_dir = models_dir or "textmodels"
        self.voting_method = voting_method
        self.use_blob_storage = use_blob_storage
        
        # Configure detector paths with load_from_blob option
        bert_config = {
            "model_path": f"{self.models_dir}/bert_ml_classifier.pkl" if not use_blob_storage else None,
            "pretrained_weights": "bert-base-cased",
            "load_from_blob": use_blob_storage
        }
        
        bow_config = {
            "model_path": f"{self.models_dir}/bow_ml_classifier.pkl" if not use_blob_storage else None,
            "vectorizer_path": f"{self.models_dir}/bow_ml_vectorizer.pkl" if not use_blob_storage else None,
            "load_from_blob": use_blob_storage
        }
        
        charcnn_config = {
            "model_path": f"{self.models_dir}/charcnn_model.h5" if not use_blob_storage else None,
            "tokenizer_path": f"{self.models_dir}/charcnn_tokenizer.pkl" if not use_blob_storage else None,
            "load_from_blob": use_blob_storage
        }
        
        distilbert_config = {
            "model_path": f"{self.models_dir}/distilbert_ml_classifier.pkl" if not use_blob_storage else None,
            "pretrained_weights": "distilbert-base-cased",
            "load_from_blob": use_blob_storage
        }
        
        # Model weights for weighted voting (based on expected performance)
        weights = {
            "BERT_ML": 0.30,
            "BoW_ML": 0.20,
            "CharCNN": 0.20,
            "DistilBERT": 0.30
        }
        
        # Initialize ensemble
        logger.info("Initializing text detection ensemble...")
        self.ensemble = TextDetectionEnsemble(
            bert_config=bert_config,
            bow_config=bow_config,
            charcnn_config=charcnn_config,
            distilbert_config=distilbert_config,
            voting_method=voting_method,
            weights=weights
        )
        
        logger.info("Text detection app initialized successfully")
    
    def detect_text(self, text: str) -> TextDetectionResult:
        """
        Detect if text is AI-generated or human-written
        
        Args:
            text: Input text to analyze
            
        Returns:
            TextDetectionResult with detection results, metadata, and markdown
        """
        if not text or not text.strip():
            raise ValueError("Empty text provided")
        
        # Run detection
        logger.info(f"Analyzing text (length: {len(text)} characters)...")
        result: TextDetectionResult = self.ensemble.detect(text)
        
        # Add metadata and generate markdown
        result = add_metadata_and_markdown(result)
        
        logger.info(f"Detection complete: {result.prediction} (confidence: {result.confidence:.4f})")
        return result
    
    def detect_text_json(self, text: str, indent: int = 2) -> str:
        """
        Detect text and return result as JSON string
        
        Args:
            text: Input text to analyze
            indent: JSON indentation level
            
        Returns:
            JSON string with detection results
        """
        result = self.detect_text(text)
        return result.to_json(indent=indent)


def main():
    """
    Main entry point for command-line interface
    """
    parser = argparse.ArgumentParser(
        description="Detect AI-generated text using ensemble of ML models"
    )
    parser.add_argument(
        "text",
        type=str,
        nargs="?",
        help="Text to analyze (if not provided, reads from stdin)"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Directory containing trained models"
    )
    parser.add_argument(
        "--voting-method",
        type=str,
        default="weighted",
        choices=["majority", "weighted", "unanimous"],
        help="Voting method for ensemble (default: weighted)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (if not specified, prints to stdout)"
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level (default: 2)"
    )
    
    args = parser.parse_args()
    
    # Get input text
    if args.text:
        input_text = args.text
    else:
        # Read from stdin
        logger.info("Reading text from stdin...")
        input_text = sys.stdin.read()
    
    if not input_text or not input_text.strip():
        logger.error("No text provided")
        sys.exit(1)
    
    # Initialize app
    try:
        app = TextDetectionApp(
            models_dir=args.models_dir,
            voting_method=args.voting_method
        )
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        sys.exit(1)
    
    # Run detection
    try:
        result_json = app.detect_text_json(input_text, indent=args.indent)
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during detection: {e}")
        
        # Provide helpful instructions for common errors
        if "All detectors failed" in str(e) or "No trained model" in str(e):
            logger.info(
                "Models need to be trained before use. "
                "Train models using: python train_text_detectors.py --dataset <path_to_training_data.csv> "
                "Your dataset should have 'text' and 'label' columns (0=human, 1=bot/AI-generated)."
            )
        sys.exit(1)
    
    # Output results
    if args.output:
        logger.info(f"Writing results to {args.output}")
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(result_json)
    else:
        print(result_json)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
