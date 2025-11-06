"""
Train Logistic Regression Meta-Classifier using real image folder - OPTIMIZED VERSION.

This script:
1. Loads images from sample_images folder
2. Runs detection to get base model predictions
3. Trains LR meta-classifier on the results
4. Saves the trained model

Optimized to handle timeouts and process in batches.

File naming convention:
- fake_*.* = FAKE images
- real_*.* = REAL images
"""

import sys
import os
import asyncio
from pathlib import Path
from typing import List, Tuple, Dict
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.logistic_regression_voter import LogisticRegressionVoter
from Agents import run_image_detection_workflow


async def collect_predictions(image_path: str, timeout: int = 60):
    """
    Run detection on an image and collect model predictions.
    
    Args:
        image_path: Path to the image file
        timeout: Timeout in seconds
        
    Returns:
        List of model prediction dictionaries or None on error
    """
    try:
        print(f"  {Path(image_path).name:<20} ", end='', flush=True)
        start_time = time.time()
        
        # Run detection with timeout
        result = await asyncio.wait_for(
            run_image_detection_workflow(
                image_path=image_path,
                threshold=0.20,
                voting_method='majority'
            ),
            timeout=timeout
        )
        
        # Extract individual model results
        individual_results = result.individualModelResults
        
        # Convert to format needed for LR training
        predictions = [
            {
                'model': r.model,
                'fakeScore': r.fakeScore,
                'realScore': r.realScore,
                'confidence': r.confidence
            }
            for r in individual_results
        ]
        
        elapsed = time.time() - start_time
        print(f"✓ ({elapsed:.1f}s)")
        return predictions
        
    except asyncio.TimeoutError:
        print(f"✗ Timeout ({timeout}s)")
        return None
    except Exception as e:
        print(f"✗ Error: {str(e)[:50]}")
        return None


async def collect_training_data(image_folder: str, max_images: int = None) -> List[Tuple[List[Dict], bool]]:
    """
    Collect training data from image folder.
    
    Args:
        image_folder: Path to folder containing images
        max_images: Maximum number of images to process (None = all)
        
    Returns:
        List of (predictions, is_fake) tuples
    """
    training_data = []
    folder_path = Path(image_folder)
    
    if not folder_path.exists():
        raise ValueError(f"Image folder not found: {folder_path}")
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    all_files = [f for f in folder_path.iterdir() 
                 if f.is_file() and f.suffix.lower() in image_extensions]
    
    # Separate fake and real images
    fake_files = sorted([f for f in all_files if f.stem.lower().startswith('fake_')])
    real_files = sorted([f for f in all_files if f.stem.lower().startswith('real_')])
    
    # Limit if requested
    if max_images:
        fake_files = fake_files[:max_images//2]
        real_files = real_files[:max_images//2]
    
    print(f"\nProcessing {len(fake_files)} fake and {len(real_files)} real images")
    print(f"(Limited to first {max_images} total images)" if max_images else "(Processing all images)")
    
    # Process fake images
    print("\n" + "="*80)
    print("Processing FAKE images:")
    print("="*80)
    
    for i, image_file in enumerate(fake_files, 1):
        print(f"  [{i}/{len(fake_files)}] ", end='')
        predictions = await collect_predictions(str(image_file), timeout=120)
        if predictions:
            training_data.append((predictions, True))  # True = FAKE
    
    # Process real images
    print("\n" + "="*80)
    print("Processing REAL images:")
    print("="*80)
    
    for i, image_file in enumerate(real_files, 1):
        print(f"  [{i}/{len(real_files)}] ", end='')
        predictions = await collect_predictions(str(image_file), timeout=120)
        if predictions:
            training_data.append((predictions, False))  # False = REAL
    
    return training_data


async def main():
    """Train the Logistic Regression meta-classifier on image folder."""
    print("="*80)
    print("Training Logistic Regression Meta-Classifier from Image Folder")
    print("="*80)
    
    image_folder = r"C:\Kidambi\WorkRelated\DeepFake\sample_images"
    
    # Process only first 20 images to avoid timeouts (10 fake + 10 real)
    max_images = 20
    
    # Step 1: Collect training data
    print("\n1. Collecting training data from images...")
    print(f"   Folder: {image_folder}")
    
    training_data = await collect_training_data(image_folder, max_images=max_images)
    
    if not training_data:
        print("\n✗ No training data collected!")
        return
    
    n_fake = sum(1 for _, is_fake in training_data if is_fake)
    n_real = len(training_data) - n_fake
    
    print(f"\n✓ Successfully collected {len(training_data)} samples ({n_fake} FAKE, {n_real} REAL)")
    
    if len(training_data) < 4:
        print("\n⚠ Warning: Very few training samples. Results may not be reliable.")
    
    # Step 2: Train the model
    print("\n2. Training Logistic Regression meta-classifier...")
    voter = LogisticRegressionVoter()
    
    save_path = Path(__file__).parent / 'Models' / 'lr_meta_classifier.pkl'
    metrics = voter.train(training_data, save_path=str(save_path))
    
    # Step 3: Show results
    print("\n" + "="*80)
    print("Training Results:")
    print("="*80)
    print(f"  Accuracy:  {metrics['accuracy']:.2%}")
    print(f"  Precision: {metrics['precision']:.2%}")
    print(f"  Recall:    {metrics['recall']:.2%}")
    print(f"  Samples:   {metrics['n_samples']} ({metrics['n_fake']} fake, {metrics['n_real']} real)")
    
    print("\n" + "="*80)
    print("Model Coefficients (Learned Weights):")
    print("="*80)
    
    # Display learned weights
    for i, model_name in enumerate(voter.model_order):
        fake_coef = voter.model.coef_[0][i * 2]
        real_coef = voter.model.coef_[0][i * 2 + 1]
        print(f"  {model_name:<20} Fake: {fake_coef:>7.3f}  |  Real: {real_coef:>7.3f}")
    
    print(f"\n  Intercept: {voter.model.intercept_[0]:.3f}")
    
    print("\n" + "="*80)
    print(f"Model saved to: {save_path}")
    print("="*80)
    
    # Step 4: Test on a few samples
    print("\n" + "="*80)
    print("Testing on Training Data (showing first 5 samples):")
    print("="*80)
    
    for i, (results, true_label) in enumerate(training_data[:min(5, len(training_data))]):
        is_fake, confidence, details = voter.predict(results)
        prediction = 'FAKE' if is_fake else 'REAL'
        actual = 'FAKE' if true_label else 'REAL'
        correct = '✓' if is_fake == true_label else '✗'
        
        print(f"\n  Sample {i+1}: {actual}")
        print(f"    Predicted: {prediction} ({confidence:.1%} confidence) {correct}")
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print("\nThe model has been trained on real detection results from your image folder.")
    print("It will now be used automatically when voting_method='logistic_regression'")
    
    print("\n" + "="*80)
    print("Summary:")
    print("="*80)
    print(f"✓ Processed {n_fake + n_real} images total")
    print(f"✓ Training accuracy: {metrics['accuracy']:.1%}")
    print(f"✓ Model saved and ready to use")
    print("\nThe meta-classifier has learned which base models are most reliable")
    print("based on patterns from your actual image dataset.")


if __name__ == '__main__':
    asyncio.run(main())
