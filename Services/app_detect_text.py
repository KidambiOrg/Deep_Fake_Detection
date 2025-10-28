"""
Main application for text deepfake detection
Takes string input and outputs JSON with classification results
"""
import json
import argparse
import sys
import os
import logging
import threading
import time
from typing import Optional, Literal
from pathlib import Path

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from textdetectionlib.text_detection_ensemble import TextDetectionEnsemble
from outputModel.text_detection_output_model import TextDetectionResult, TextMetadata
from utils.blob_storage_helper import get_blob_loader
from utils.text_analysis_helper import add_metadata_and_markdown

logger = logging.getLogger(__name__)

# Global ensemble detector - initialized lazily to avoid cold start timeout
_text_ensemble = None
_text_ensemble_loading = False
_text_ensemble_lock = threading.Lock()


def download_text_models_from_blob_storage():
    """Download text models from Azure Blob Storage if configured.
    Returns dictionary of downloaded model paths.
    """
    blob_loader = get_blob_loader()
    if not blob_loader or not blob_loader.blob_service_client:
        logger.info("Blob storage not configured, using local text models")
        return {}
    
    logger.info("Checking for text models in Azure Blob Storage...")
    
    downloaded_paths = {}
    
    # Download models from text model registry
    from textdetectionlib.model_registry import TEXT_MODEL_REGISTRY
    
    for model_key, model_info in TEXT_MODEL_REGISTRY.items():
        if 'model_files' not in model_info:
            continue
        
        model_files = model_info['model_files']
        downloaded_paths[model_key] = {}
        
        # Download each model file
        for file_key, file_config in model_files.items():
            if 'blob_storage' not in file_config:
                continue
            
            blob_config = file_config['blob_storage']
            container = blob_config.get('container')
            blob_name = blob_config.get('blob')
            
            if not container or not blob_name:
                continue
            
            logger.info(f"Downloading {model_key}/{file_key} from blob storage...")
            local_path = blob_loader.download_model(
                container_name=container,
                blob_name=blob_name
            )
            
            if local_path:
                downloaded_paths[model_key][file_key] = local_path
                logger.info(f"Downloaded {model_key}/{file_key} to: {local_path}")
            else:
                logger.warning(f"Failed to download {model_key}/{file_key}")
    
    return downloaded_paths


def get_text_ensemble(
    models_dir: str = "TextModels",
    voting_method: Literal["majority", "weighted", "unanimous"] = "weighted",
    debug: bool = False,
    timeout: int = 300
):
    """
    Get or initialize the global text ensemble detector.
    Uses lazy loading to avoid cold start timeouts.
    Downloads models from Azure Blob Storage if configured.
    
    Args:
        models_dir: Directory containing local models (fallback)
        voting_method: Voting strategy for ensemble
        debug: Enable debug logging
        timeout: Maximum time to wait for loading (seconds)
    
    Returns:
        TextDetectionEnsemble instance or None if still loading
    """
    global _text_ensemble, _text_ensemble_loading, _text_ensemble_lock
    
    # If already loaded, return immediately
    if _text_ensemble is not None:
        return _text_ensemble
    
    # Prevent multiple simultaneous initialization attempts
    with _text_ensemble_lock:
        # Double-check after acquiring lock
        if _text_ensemble is not None:
            return _text_ensemble
        
        # Check if already loading
        if _text_ensemble_loading:
            logger.info("Text models are already being loaded by another request")
            return None
        
        # Mark as loading
        _text_ensemble_loading = True
    
    try:
        logger.info("Starting text ensemble detector initialization...")
        
        # Download models from blob storage if configured
        downloaded_paths = download_text_models_from_blob_storage()
        
        # Configure detector paths - use downloaded paths if available, otherwise fall back to local paths
        bert_config = {
            "model_path": downloaded_paths.get("bert_ml", {}).get("classifier") or f"{models_dir}/bert_ml_classifier.pkl",
            "pretrained_weights": "bert-base-cased",
            "load_from_blob": False  # Already downloaded above
        }
        
        bow_config = {
            "model_path": downloaded_paths.get("bow_ml", {}).get("classifier") or f"{models_dir}/bow_ml_classifier.pkl",
            "vectorizer_path": downloaded_paths.get("bow_ml", {}).get("vectorizer") or f"{models_dir}/bow_ml_vectorizer.pkl",
            "load_from_blob": False  # Already downloaded above
        }
        
        charcnn_config = {
            "model_path": downloaded_paths.get("charcnn", {}).get("model") or f"{models_dir}/charcnn_model.h5",
            "tokenizer_path": downloaded_paths.get("charcnn", {}).get("tokenizer") or f"{models_dir}/charcnn_tokenizer.pkl",
            "load_from_blob": False  # Already downloaded above
        }
        
        distilbert_config = {
            "model_path": downloaded_paths.get("distilbert_ml", {}).get("classifier") or f"{models_dir}/distilbert_ml_classifier.pkl",
            "pretrained_weights": "distilbert-base-cased",
            "load_from_blob": False  # Already downloaded above
        }
        
        # Model weights for weighted voting (based on expected performance)
        weights = {
            "BERT_ML": 0.30,
            "BoW_ML": 0.20,
            "CharCNN": 0.20,
            "DistilBERT": 0.30
        }
        
        # Initialize ensemble
        start_time = time.time()
        _text_ensemble = TextDetectionEnsemble(
            bert_config=bert_config,
            bow_config=bow_config,
            charcnn_config=charcnn_config,
            distilbert_config=distilbert_config,
            voting_method=voting_method,
            weights=weights
        )
        elapsed = time.time() - start_time
        
        logger.info(f"Text ensemble detector initialized successfully in {elapsed:.2f} seconds")
        return _text_ensemble
        
    except Exception as e:
        logger.error(f"Failed to initialize text ensemble detector: {e}")
        logger.exception(e)
        return None
    finally:
        # Mark loading as complete
        with _text_ensemble_lock:
            _text_ensemble_loading = False


def detect_text(
    text: str,
    voting_method: Literal["majority", "weighted", "unanimous"] = "weighted"
) -> TextDetectionResult:
    """
    Detect if text is AI-generated or human-written
    
    Args:
        text: Input text to analyze
        voting_method: Voting strategy for ensemble
        
    Returns:
        TextDetectionResult with detection results, metadata, and markdown
    """
    if not text or not text.strip():
        raise ValueError("Empty text provided")
    
    # Get ensemble detector
    ensemble = get_text_ensemble(voting_method=voting_method)
    
    if ensemble is None:
        raise RuntimeError("Text detection models not loaded. Call warmup endpoint first or try again.")
    
    # Run detection
    logger.info(f"Analyzing text (length: {len(text)} characters)...")
    result: TextDetectionResult = ensemble.detect(text)
    
    # Add metadata and generate markdown
    result = add_metadata_and_markdown(result)
    
    logger.info(f"Detection complete: {result.prediction} (confidence: {result.confidence:.4f})")
    return result


def detect_text_json(
    text: str,
    voting_method: Literal["majority", "weighted", "unanimous"] = "weighted",
    indent: int = 2
) -> str:
    """
    Detect text and return result as JSON string
    
    Args:
        text: Input text to analyze
        voting_method: Voting strategy for ensemble
        indent: JSON indentation level
        
    Returns:
        JSON string with detection results
    """
    result = detect_text(text, voting_method=voting_method)
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
        default="TextModels",
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
    
    # Pre-load models
    try:
        logger.info("Loading text detection models...")
        ensemble = get_text_ensemble(
            models_dir=args.models_dir,
            voting_method=args.voting_method
        )
        if ensemble is None:
            logger.error("Failed to load text detection models")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to initialize text ensemble: {e}")
        sys.exit(1)
    
    # Run detection
    try:
        result_json = detect_text_json(
            input_text,
            voting_method=args.voting_method,
            indent=args.indent
        )
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
