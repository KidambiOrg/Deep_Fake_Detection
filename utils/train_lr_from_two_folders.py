"""
Train Logistic Regression Meta-Classifier using separate fake/real folders.

This script:
1. Loads images from separate fake and real folders
2. Runs detection to get base model predictions from all 7 models:
   - NoEscape (nonescape-full)
   - SEResNet50 (seresnet50)
   - Prithiv V1 (prithiv-v1)
   - DimaVit (dima-vit)
   - Organika SDXL (organika-sdxl)
   - MobileNetV2 (mobilenetv2)
   - MobileNetV2 CNN (mobilenetv2_cnn)
3. Trains LR meta-classifier on the results (14 features: 7 models × 2 scores each)
4. Saves the trained model

Folder structure:
- fake_folder: All images are FAKE
- real_folder: All images are REAL
"""

import sys
import os
import asyncio
from pathlib import Path
from typing import List, Tuple, Dict
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logistic_regression_voter import LogisticRegressionVoter
from Agents import run_image_detection_workflow


async def collect_predictions(image_path: str, timeout: int = 120):
    """
    Run detection on an image and collect model predictions.
    
    Args:
        image_path: Path to the image file
        timeout: Timeout in seconds
        
    Returns:
        List of model prediction dictionaries or None on error
    """
    import shutil
    import tempfile
    
    temp_file = None
    try:
        print(f"  {Path(image_path).name:<30} ", end='', flush=True)
        start_time = time.time()
        
        # Copy to temp file to preserve original
        temp_dir = tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, f"train_{Path(image_path).name}")
        shutil.copy2(image_path, temp_file)
        
        # Run detection with timeout using temp file
        result = await asyncio.wait_for(
            run_image_detection_workflow(
                image_path=temp_file,
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
    finally:
        # Always clean up temp file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception:
                pass  # Ignore cleanup errors


async def collect_training_data(
    fake_folder: str, 
    real_folder: str, 
    max_images_per_class: int = 100
) -> List[Tuple[List[Dict], bool]]:
    """
    Collect training data from separate fake and real folders.
    
    Args:
        fake_folder: Path to folder containing FAKE images
        real_folder: Path to folder containing REAL images
        max_images_per_class: Maximum images per class (fake/real)
        
    Returns:
        List of (predictions, is_fake) tuples
    """
    training_data = []
    
    fake_path = Path(fake_folder)
    real_path = Path(real_folder)
    
    if not fake_path.exists():
        raise ValueError(f"Fake folder not found: {fake_path}")
    if not real_path.exists():
        raise ValueError(f"Real folder not found: {real_path}")
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    
    fake_files = sorted([f for f in fake_path.iterdir() 
                         if f.is_file() and f.suffix.lower() in image_extensions])
    real_files = sorted([f for f in real_path.iterdir() 
                         if f.is_file() and f.suffix.lower() in image_extensions])
    
    print(f"\nFound {len(fake_files)} fake images and {len(real_files)} real images")
    
    # Limit to max_images_per_class
    fake_files = fake_files[:max_images_per_class]
    real_files = real_files[:max_images_per_class]
    
    print(f"Processing {len(fake_files)} fake and {len(real_files)} real images (max={max_images_per_class} per class)")
    
    # Process fake images
    print("\n" + "="*80)
    print("Processing FAKE images:")
    print("="*80)
    
    successful_fake = 0
    for i, image_file in enumerate(fake_files, 1):
        print(f"  [{i}/{len(fake_files)}] ", end='')
        predictions = await collect_predictions(str(image_file), timeout=120)
        if predictions:
            training_data.append((predictions, True))  # True = FAKE
            successful_fake += 1
        
        # Progress update every 10 images
        if i % 10 == 0:
            print(f"\n  Progress: {successful_fake}/{i} successful\n")
    
    # Process real images
    print("\n" + "="*80)
    print("Processing REAL images:")
    print("="*80)
    
    successful_real = 0
    for i, image_file in enumerate(real_files, 1):
        print(f"  [{i}/{len(real_files)}] ", end='')
        predictions = await collect_predictions(str(image_file), timeout=120)
        if predictions:
            training_data.append((predictions, False))  # False = REAL
            successful_real += 1
        
        # Progress update every 10 images
        if i % 10 == 0:
            print(f"\n  Progress: {successful_real}/{i} successful\n")
    
    print(f"\n✓ Successfully collected {len(training_data)} samples ({successful_fake} FAKE, {successful_real} REAL)")
    
    return training_data


async def main():
    """Train the Logistic Regression meta-classifier on separate folders."""
    print("="*80)
    print("Training Logistic Regression Meta-Classifier from Fake/Real Folders")
    print("="*80)
    
    fake_folder = r"C:\Kidambi\WorkRelated\DeepFake\sample_images\testthresholdvoting\fake"
    real_folder = r"C:\Kidambi\WorkRelated\DeepFake\sample_images\testthresholdvoting\real"

    # Process images per class for good training data
    # With 7 models running concurrently per image:
    # - 100 images per class (~2-3 hours total)
    # - 500 images per class (~8-12 hours total) - recommended for production
    # This balances training time with having enough samples for reliable LR weights
    max_images_per_class = 500
    
    # Step 1: Collect training data
    print("\n1. Collecting training data from folders...")
    print(f"   Fake folder: {fake_folder}")
    print(f"   Real folder: {real_folder}")
    print(f"   Max images per class: {max_images_per_class}")
    
    training_data = await collect_training_data(
        fake_folder=fake_folder,
        real_folder=real_folder,
        max_images_per_class=max_images_per_class
    )
    
    if not training_data:
        print("\n✗ No training data collected!")
        return
    
    n_fake = sum(1 for _, is_fake in training_data if is_fake)
    n_real = len(training_data) - n_fake
    
    print(f"\n✓ Successfully collected {len(training_data)} samples ({n_fake} FAKE, {n_real} REAL)")
    
    if len(training_data) < 20:
        print("\n⚠ Warning: Very few training samples. Results may not be reliable.")
        print("   Recommend at least 50-100 samples per class for good performance.")
    
    # Step 2: Train the model
    print("\n2. Training Logistic Regression meta-classifier...")
    voter = LogisticRegressionVoter()
    
    save_path = Path(__file__).parent.parent / 'Models' / 'lr_meta_classifier.pkl'
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
    print("\nThe model has been trained on real detection results from your image folders.")
    print("It will now be used automatically when voting_method='logistic_regression'")
    
    print("\n" + "="*80)
    print("Summary:")
    print("="*80)
    print(f"✓ Processed {n_fake + n_real} images total")
    print(f"✓ Training accuracy: {metrics['accuracy']:.1%}")
    print(f"✓ Model saved and ready to use")
    print("\nThe meta-classifier has learned which base models are most reliable")
    print("based on patterns from your actual image dataset.")
    
    # Recommendation
    if metrics['accuracy'] >= 0.90:
        print("\n✓ EXCELLENT: Accuracy >= 90%, model is production-ready!")
    elif metrics['accuracy'] >= 0.80:
        print("\n✓ GOOD: Accuracy >= 80%, model should work well in production")
    elif metrics['accuracy'] >= 0.70:
        print("\n⚠ FAIR: Accuracy >= 70%, consider collecting more training data")
    else:
        print("\n⚠ LOW: Accuracy < 70%, recommend collecting more diverse training data")


if __name__ == '__main__':
    asyncio.run(main())
