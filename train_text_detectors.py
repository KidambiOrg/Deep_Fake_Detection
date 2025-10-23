"""
Training script for text deepfake detectors
Trains all four models (BERT_ML, BoW_ML, CharCNN, DistilBERT) on provided dataset
Supports parallel training for faster execution
"""
import argparse
import logging
import os
import sys
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from textdetectionlib.detector_bert_ml import DetectorBertML
from textdetectionlib.detector_bow_ml import DetectorBowML
from textdetectionlib.detector_charcnn import DetectorCharCNN
from textdetectionlib.detector_distilbert import DetectorDistilBertML

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_dataset(
    dataset_path: str,
    text_column: str = "text",
    label_column: str = "label"
) -> Tuple[List[str], List[str]]:
    """
    Load dataset from CSV file
    
    Args:
        dataset_path: Path to CSV file
        text_column: Name of column containing text
        label_column: Name of column containing labels (human, ai)
        
    Returns:
        Tuple of (texts, labels)
    """
    logger.info(f"Loading dataset from {dataset_path}")
    
    # Check file extension
    file_ext = Path(dataset_path).suffix.lower()
    
    if file_ext == '.csv':
        df = pd.read_csv(dataset_path)
    elif file_ext in ['.json', '.jsonl']:
        df = pd.read_json(dataset_path, lines=(file_ext == '.jsonl'))
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Use .csv, .json, or .jsonl")
    
    # Validate columns
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in dataset. Available columns: {df.columns.tolist()}")
    
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataset. Available columns: {df.columns.tolist()}")
    
    # Extract data and handle missing values
    texts = df[text_column].astype(str).tolist()
    labels_raw = df[label_column].astype(str).str.lower().str.strip().tolist()
    
    # Filter out rows with missing labels and log them
    valid_indices = []
    invalid_rows = []
    
    for i, label in enumerate(labels_raw):
        if label in ['nan', 'none', '', 'null']:
            invalid_rows.append((i + 2, texts[i], label))  # +2 because of header and 0-indexing
        else:
            valid_indices.append(i)
    
    if invalid_rows:
        logger.warning(f"Found {len(invalid_rows)} rows with missing or invalid labels:")
        for row_num, text, label in invalid_rows:
            # Truncate long text for readability
            text_preview = text[:100] + "..." if len(text) > 100 else text
            logger.warning(f"  Row {row_num}: label='{label}', text='{text_preview}'")
    
    texts = [texts[i] for i in valid_indices]
    labels = [labels_raw[i] for i in valid_indices]
    
    # Validate labels
    unique_labels = set(labels)
    valid_labels = {'human', 'ai'}
    if not unique_labels.issubset(valid_labels):
        raise ValueError(f"Labels must be 'human' or 'ai'. Found: {unique_labels}")
    
    logger.info(f"Loaded {len(texts)} samples")
    logger.info(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")
    
    return texts, labels


def train_bert_ml_wrapper(args_tuple):
    """Wrapper function for multiprocessing"""
    train_texts, train_labels, output_dir, pretrained_weights = args_tuple
    return train_bert_ml(train_texts, train_labels, output_dir, pretrained_weights)


def train_bert_ml(
    train_texts: List[str],
    train_labels: List[str],
    output_dir: str,
    pretrained_weights: str = "bert-base-cased"
) -> Dict[str, Any]:
    """Train BERT_ML detector"""
    # Setup logging for this process
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("=" * 80)
        logger.info("Training BERT_ML detector")
        logger.info("=" * 80)
        
        # Convert string labels to integers (human=0, ai/bot=1)
        label_map = {'human': 0, 'ai': 1, 'bot': 1}
        numeric_labels = [label_map[label] for label in train_labels]
        
        detector = DetectorBertML(pretrained_weights=pretrained_weights)
        detector.train(train_texts, numeric_labels)
        
        # Save model
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "bert_ml_classifier.pkl")
        detector.save_model(model_path)
        
        logger.info(f"BERT_ML model saved to {model_path}")
        return {"success": True, "model": "bert", "path": model_path}
    except Exception as e:
        logger.error(f"Failed to train BERT_ML: {e}")
        return {"success": False, "model": "bert", "error": str(e)}


def train_bow_ml_wrapper(args_tuple):
    """Wrapper function for multiprocessing"""
    train_texts, train_labels, output_dir = args_tuple
    return train_bow_ml(train_texts, train_labels, output_dir)


def train_bow_ml(
    train_texts: List[str],
    train_labels: List[str],
    output_dir: str
) -> Dict[str, Any]:
    """Train BoW_ML detector"""
    # Setup logging for this process
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("=" * 80)
        logger.info("Training BoW_ML detector")
        logger.info("=" * 80)
        
        # Convert string labels to integers (human=0, ai/bot=1)
        label_map = {'human': 0, 'ai': 1, 'bot': 1}
        numeric_labels = [label_map[label] for label in train_labels]
        
        detector = DetectorBowML()
        detector.train(train_texts, numeric_labels)
        
        # Save model and vectorizer separately
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "bow_ml_classifier.pkl")
        vectorizer_path = os.path.join(output_dir, "bow_ml_vectorizer.pkl")
        detector.save_model(model_path)
        detector.save_vectorizer(vectorizer_path)
        
        logger.info(f"BoW_ML model saved to {model_path}")
        logger.info(f"BoW_ML vectorizer saved to {vectorizer_path}")
        return {"success": True, "model": "bow", "path": model_path}
    except Exception as e:
        logger.error(f"Failed to train BoW_ML: {e}")
        return {"success": False, "model": "bow", "error": str(e)}


def train_charcnn_wrapper(args_tuple):
    """Wrapper function for multiprocessing"""
    train_texts, train_labels, output_dir, epochs, batch_size = args_tuple
    return train_charcnn(train_texts, train_labels, output_dir, epochs, batch_size)


def train_charcnn(
    train_texts: List[str],
    train_labels: List[str],
    output_dir: str,
    epochs: int = 10,
    batch_size: int = 32
) -> Dict[str, Any]:
    """Train CharCNN detector"""
    # Setup logging for this process
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("=" * 80)
        logger.info("Training CharCNN detector")
        logger.info("=" * 80)
        
        # Convert string labels to integers (human=0, ai/bot=1)
        label_map = {'human': 0, 'ai': 1, 'bot': 1}
        numeric_labels = [label_map[label] for label in train_labels]
        
        detector = DetectorCharCNN()
        detector.train(train_texts, numeric_labels, epochs=epochs, batch_size=batch_size)
        
        # Save model and tokenizer separately
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "charcnn_model.h5")
        tokenizer_path = os.path.join(output_dir, "charcnn_tokenizer.pkl")
        detector.save_model(model_path)
        detector.save_tokenizer(tokenizer_path)
        
        logger.info(f"CharCNN model saved to {model_path}")
        logger.info(f"CharCNN tokenizer saved to {tokenizer_path}")
        return {"success": True, "model": "charcnn", "path": model_path}
    except Exception as e:
        logger.error(f"Failed to train CharCNN: {e}")
        return {"success": False, "model": "charcnn", "error": str(e)}


def train_distilbert_ml_wrapper(args_tuple):
    """Wrapper function for multiprocessing"""
    train_texts, train_labels, output_dir, pretrained_weights = args_tuple
    return train_distilbert_ml(train_texts, train_labels, output_dir, pretrained_weights)


def train_distilbert_ml(
    train_texts: List[str],
    train_labels: List[str],
    output_dir: str,
    pretrained_weights: str = "distilbert-base-cased"
) -> Dict[str, Any]:
    """Train DistilBERT_ML detector"""
    # Setup logging for this process
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("=" * 80)
        logger.info("Training DistilBERT_ML detector")
        logger.info("=" * 80)
        
        # Convert string labels to integers (human=0, ai/bot=1)
        label_map = {'human': 0, 'ai': 1, 'bot': 1}
        numeric_labels = [label_map[label] for label in train_labels]
        
        detector = DetectorDistilBertML(pretrained_weights=pretrained_weights)
        detector.train(train_texts, numeric_labels)
        
        # Save model
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "distilbert_ml_classifier.pkl")
        detector.save_model(model_path)
        
        logger.info(f"DistilBERT_ML model saved to {model_path}")
        return {"success": True, "model": "distilbert", "path": model_path}
    except Exception as e:
        logger.error(f"Failed to train DistilBERT_ML: {e}")
        return {"success": False, "model": "distilbert", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Train text deepfake detection models"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to training dataset (CSV, JSON, or JSONL format)"
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Name of column containing text (default: text)"
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="label",
        help="Name of column containing labels: 'human' or 'ai' (default: label)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Models/text_detection",
        help="Directory to save trained models (default: Models/text_detection)"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=["bert", "bow", "charcnn", "distilbert", "all"],
        default=["all"],
        help="Which models to train (default: all)"
    )
    parser.add_argument(
        "--charcnn-epochs",
        type=int,
        default=10,
        help="Number of epochs for CharCNN training (default: 10)"
    )
    parser.add_argument(
        "--charcnn-batch-size",
        type=int,
        default=32,
        help="Batch size for CharCNN training (default: 32)"
    )
    
    args = parser.parse_args()
    
    # Load dataset
    try:
        train_texts, train_labels = load_dataset(
            args.dataset,
            args.text_column,
            args.label_column
        )
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)
    
    # Determine which models to train
    models_to_train = args.models
    if "all" in models_to_train:
        models_to_train = ["bert", "bow", "charcnn", "distilbert"]
    
    # Prepare training tasks with wrapper functions for multiprocessing
    training_tasks = []
    for model_name in models_to_train:
        if model_name == "bert":
            training_tasks.append(("bert", train_bert_ml_wrapper, (train_texts, train_labels, args.output_dir, "bert-base-cased")))
        elif model_name == "bow":
            training_tasks.append(("bow", train_bow_ml_wrapper, (train_texts, train_labels, args.output_dir)))
        elif model_name == "charcnn":
            training_tasks.append(("charcnn", train_charcnn_wrapper, (
                train_texts,
                train_labels,
                args.output_dir,
                args.charcnn_epochs,
                args.charcnn_batch_size
            )))
        elif model_name == "distilbert":
            training_tasks.append(("distilbert", train_distilbert_ml_wrapper, (train_texts, train_labels, args.output_dir, "distilbert-base-cased")))
    
    # Train models in parallel
    total_count = len(training_tasks)
    results = []
    
    logger.info("=" * 80)
    logger.info(f"Starting parallel training for {total_count} models...")
    logger.info("=" * 80)
    
    # Use ProcessPoolExecutor for true parallelism with deep learning models
    with ProcessPoolExecutor(max_workers=min(4, total_count)) as executor:
        # Submit all training tasks
        future_to_model = {}
        for model_name, train_func, args_tuple in training_tasks:
            future = executor.submit(train_func, args_tuple)
            future_to_model[future] = model_name
        
        # Collect results as they complete
        for future in as_completed(future_to_model):
            model_name = future_to_model[future]
            try:
                result = future.result()
                results.append(result)
                if result["success"]:
                    logger.info(f"✓ {model_name.upper()} training completed successfully")
                else:
                    logger.error(f"✗ {model_name.upper()} training failed: {result.get('error', 'Unknown error')}")
            except Exception as e:
                logger.error(f"✗ {model_name.upper()} training failed with exception: {e}")
                logger.exception(e)
                results.append({"success": False, "model": model_name, "error": str(e)})
    
    # Summary
    success_count = sum(1 for r in results if r["success"])
    logger.info("=" * 80)
    logger.info(f"Training complete: {success_count}/{total_count} models trained successfully")
    
    # List successful models
    if success_count > 0:
        logger.info("\nSuccessfully trained models:")
        for result in results:
            if result["success"]:
                logger.info(f"  ✓ {result['model'].upper()}: {result.get('path', 'N/A')}")
    
    # List failed models
    failed_count = total_count - success_count
    if failed_count > 0:
        logger.info(f"\nFailed models ({failed_count}):")
        for result in results:
            if not result["success"]:
                logger.info(f"  ✗ {result['model'].upper()}: {result.get('error', 'Unknown error')}")
    
    logger.info(f"\nModels directory: {args.output_dir}")
    logger.info("=" * 80)
    
    if success_count == 0:
        sys.exit(1)


if __name__ == "__main__":
    # Required for multiprocessing on Windows
    multiprocessing.freeze_support()
    main()
