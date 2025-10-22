"""
Main application for text deepfake detection
Takes string input and outputs JSON with classification results
"""
import json
import argparse
import sys
import logging
from typing import Optional, Dict, Any
from dataclasses import asdict

from textdetectionlib.text_detection_ensemble import TextDetectionEnsemble, EnsembleResult

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TextDetectionApp:
    """
    Application for detecting AI-generated text using an ensemble of models
    """
    
    def __init__(
        self,
        models_dir: Optional[str] = None,
        voting_method: str = "weighted"
    ):
        """
        Initialize the text detection application
        
        Args:
            models_dir: Directory containing trained models
            voting_method: Voting strategy for ensemble ("majority", "weighted", "unanimous")
        """
        self.models_dir = models_dir or "Models/text_detection"
        self.voting_method = voting_method
        
        # Configure detector paths
        bert_config = {
            "model_path": f"{self.models_dir}/bert_ml_classifier.pkl",
            "pretrained_weights": "bert-base-cased"
        }
        
        bow_config = {
            "model_path": f"{self.models_dir}/bow_ml_classifier.pkl",
            "vectorizer_path": f"{self.models_dir}/bow_ml_vectorizer.pkl"
        }
        
        charcnn_config = {
            "model_path": f"{self.models_dir}/charcnn_model.h5",
            "tokenizer_path": f"{self.models_dir}/charcnn_tokenizer.pkl"
        }
        
        distilbert_config = {
            "model_path": f"{self.models_dir}/distilbert_ml_classifier.pkl",
            "pretrained_weights": "distilbert-base-cased"
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
    
    def detect_text(self, text: str) -> Dict[str, Any]:
        """
        Detect if text is AI-generated or human-written
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with detection results
        """
        if not text or not text.strip():
            return {
                "error": "Empty text provided",
                "text": text,
                "prediction": None,
                "metrics": None
            }
        
        try:
            # Run detection
            logger.info(f"Analyzing text (length: {len(text)} characters)...")
            result: EnsembleResult = self.ensemble.detect(text)
            
            # Format output
            output = {
                "text": result.text,
                "prediction": result.prediction,
                "confidence": round(result.confidence, 4),
                "metrics": {
                    "average_bot_score": round(result.average_bot_score, 4),
                    "average_human_score": round(result.average_human_score, 4),
                    "voting_method": result.voting_method,
                    "voting_breakdown": result.voting_breakdown
                },
                "individual_model_results": [
                    {
                        "model": score.model_name,
                        "prediction": score.prediction,
                        "confidence": round(score.confidence, 4),
                        "bot_score": round(score.bot_score, 4),
                        "human_score": round(score.human_score, 4)
                    }
                    for score in result.individual_scores
                ]
            }
            
            logger.info(f"Detection complete: {result.prediction} (confidence: {result.confidence:.4f})")
            return output
        
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error during detection: {error_msg}")
            
            # Provide helpful instructions for common errors
            help_msg = None
            if "All detectors failed" in error_msg or "No trained model" in error_msg:
                help_msg = (
                    "Models need to be trained before use. "
                    "Train models using: python train_text_detectors.py --dataset <path_to_training_data.csv> "
                    "Your dataset should have 'text' and 'label' columns (0=human, 1=bot/AI-generated)."
                )
            
            return {
                "error": error_msg,
                "help": help_msg,
                "text": text,
                "prediction": None,
                "metrics": None
            }
    
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
        return json.dumps(result, indent=indent)


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
    result_json = app.detect_text_json(input_text, indent=args.indent)
    
    # Output results
    if args.output:
        logger.info(f"Writing results to {args.output}")
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(result_json)
    else:
        print(result_json)
    
    # Parse result to determine exit code
    result = json.loads(result_json)
    if "error" in result and result["error"]:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
