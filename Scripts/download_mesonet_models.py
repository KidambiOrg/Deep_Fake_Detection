#!/usr/bin/env python3
"""
Download MesoNet models for deepfake detection

This script downloads the pretrained MesoNet model from GitHub
and saves it to the Models/mesonet directory.

Model source: https://github.com/muffin-123/DeepFake-Detection-mesonet_model.h5-
"""

import os
import sys
import urllib.request
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_file(url: str, destination: Path, description: str = "file"):
    """
    Download a file with progress indication
    
    Args:
        url: URL to download from
        destination: Path to save the file
        description: Description for logging
    """
    try:
        logger.info(f"Downloading {description} from:")
        logger.info(f"  {url}")
        logger.info(f"Saving to: {destination}")
        
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(downloaded * 100.0 / total_size, 100.0)
                downloaded_mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                sys.stdout.write(f"\r  Progress: {percent:.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, destination, reporthook=report_progress)
        print()  # New line after progress
        logger.info(f"✓ Downloaded {description} successfully")
        
    except Exception as e:
        logger.error(f"✗ Failed to download {description}: {e}")
        raise


def download_mesonet_model(model_dir: str = None):
    """
    Download MesoNet model from GitHub
    
    Args:
        model_dir (str): Directory to save the model. If None, uses Models/mesonet
    """
    if model_dir is None:
        # Get the script directory and go up to the project root
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        model_dir_path = project_root / "Models" / "mesonet"
    else:
        model_dir_path = Path(model_dir)
    
    # Create directory if it doesn't exist
    model_dir_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("MesoNet Model Download")
    logger.info("="*60)
    logger.info(f"Target directory: {model_dir_path}")
    logger.info("")
    
    # Model information
    model_info = {
        'name': 'mesonet_model.h5',
        'url': 'https://github.com/muffin-123/DeepFake-Detection-mesonet_model.h5-/raw/refs/heads/main/mesonet_model.h5',
        'description': 'MesoNet pretrained model (Keras H5 format)'
    }
    
    model_file = model_dir_path / model_info['name']
    
    # Check if model already exists
    if model_file.exists():
        logger.info(f"Model already exists: {model_file}")
        user_input = input("Do you want to re-download? (y/n): ").strip().lower()
        if user_input != 'y':
            logger.info("Using existing model file.")
            return model_file
        else:
            logger.info("Re-downloading model...")
    
    try:
        # Download the model
        download_file(
            url=model_info['url'],
            destination=model_file,
            description=model_info['description']
        )
        
        # Create model info file
        info_file = model_dir_path / "model_info.txt"
        with open(info_file, 'w') as f:
            f.write("MesoNet Model Information\n")
            f.write("="*50 + "\n\n")
            f.write("Model: MesoNet\n")
            f.write("Architecture: Compact CNN for deepfake detection\n")
            f.write("Input size: 256x256x3 (facial frames)\n")
            f.write("Output: Binary classification (Real/Fake)\n")
            f.write("Training dataset: UADFV_dataset\n\n")
            f.write("Paper: MesoNet: a Compact Facial Video Forgery Detection Network\n")
            f.write("URL: https://arxiv.org/abs/1809.00888\n\n")
            f.write("Model source: https://github.com/muffin-123/DeepFake-Detection-mesonet_model.h5-\n")
            f.write("Dataset source: https://www.kaggle.com/datasets/adityakeshri9234/uadfv-dataset\n\n")
            f.write("Files:\n")
            f.write(f"- {model_info['name']}: Pretrained MesoNet model (Keras H5 format)\n")
            f.write("- model_info.txt: This information file\n\n")
            f.write("Usage:\n")
            f.write("  from detectionLib.detect_deepfake_mesonet import MesoNetDetector\n")
            f.write(f"  detector = MesoNetDetector(model_path='{model_file}')\n")
            f.write("  result = detector.detect_single('image.jpg')\n")
        
        logger.info(f"✓ Model info saved to: {info_file}")
        
        # Display file information
        file_size_mb = model_file.stat().st_size / (1024 * 1024)
        logger.info("")
        logger.info("="*60)
        logger.info("Download Complete!")
        logger.info("="*60)
        logger.info(f"Model file: {model_file}")
        logger.info(f"File size: {file_size_mb:.2f} MB")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. The model is in Keras H5 format")
        logger.info("2. To use with PyTorch, you need TensorFlow installed:")
        logger.info("   pip install tensorflow")
        logger.info("3. Or convert the model to PyTorch format (.pth)")
        logger.info("")
        logger.info("Usage example:")
        logger.info("  from detectionLib.detect_deepfake_mesonet import MesoNetDetector")
        logger.info(f"  detector = MesoNetDetector(model_path=r'{model_file}')")
        logger.info("  result = detector.detect_single('your_image.jpg')")
        
        return model_file
        
    except Exception as e:
        logger.error(f"Failed to download MesoNet model: {e}")
        sys.exit(1)


def create_readme(model_dir_path: Path):
    """Create a README file with usage instructions"""
    readme_path = model_dir_path / "README.md"
    
    readme_content = """# MesoNet Model

## About MesoNet

MesoNet is a lightweight CNN architecture specifically designed for deepfake detection. 
It analyzes images at the mesoscopic level to detect subtle artifacts introduced by 
deepfake generation algorithms.

### Key Features
- **Compact architecture**: Lightweight model suitable for real-time detection
- **Mesoscopic analysis**: Detects subtle inconsistencies in facial features
- **Input size**: 256×256 RGB facial frames
- **Binary classification**: Real vs Fake

### Paper Reference
**MesoNet: a Compact Facial Video Forgery Detection Network**  
Authors: Darius Afchar, Vincent Nozick, Junichi Yamagishi, Isao Echizen  
arXiv: https://arxiv.org/abs/1809.00888

## Model Files

- `mesonet_model.h5`: Pretrained MesoNet model in Keras H5 format
- `model_info.txt`: Detailed model information
- `README.md`: This file

## Training Dataset

The model was trained on the **UADFV_dataset**, which contains:
- Real videos from authentic sources
- Deepfake videos generated using various methods
- Extracted frames resized to 256×256

Dataset source: [UADFV_dataset on Kaggle](https://www.kaggle.com/datasets/adityakeshri9234/uadfv-dataset)

## Usage

### Basic Usage

```python
from detectionLib.detect_deepfake_mesonet import MesoNetDetector

# Initialize detector with pretrained model
detector = MesoNetDetector(
    model_path='Models/mesonet/mesonet_model.h5',
    threshold=0.5
)

# Detect deepfake in single image
result = detector.detect_single('path/to/image.jpg')

print(f"Is Fake: {result.is_fake}")
print(f"Fake Probability: {result.fake_probability:.2%}")
print(f"Real Probability: {result.real_probability:.2%}")
print(f"Confidence: {result.confidence:.2%}")
```

### Batch Processing

```python
# Process multiple images
images = ['image1.jpg', 'image2.jpg', 'image3.jpg']
results = detector.detect_batch(images)

for i, result in enumerate(results):
    print(f"Image {i+1}: {'FAKE' if result.is_fake else 'REAL'} "
          f"(confidence: {result.confidence:.2%})")
```

### Directory Analysis

```python
# Analyze all images in a directory
results = detector.analyze_directory('path/to/images/')

# Get summary statistics
total = len(results)
fakes = sum(1 for r in results if r.is_fake)
print(f"Total images: {total}")
print(f"Detected fakes: {fakes} ({fakes/total*100:.1f}%)")
```

## Requirements

To use the Keras H5 model format, you need TensorFlow:

```bash
pip install tensorflow
```

Or convert the model to PyTorch format (.pth) for PyTorch-only environments.

## Performance Characteristics

- **Speed**: Fast inference (~10-50ms per image on GPU)
- **Memory**: Low memory footprint (~8-16 MB)
- **Accuracy**: Good performance on face-focused deepfakes
- **Best for**: Facial deepfakes, real-time detection scenarios

## Integration with Ensemble

MesoNet can be integrated into the ensemble detector:

```python
from detectionLib.detect_deepfake_image import EnsembleDetector

# MesoNet will be added to the ensemble automatically if configured
ensemble = EnsembleDetector(
    model_keys=['mesonet', 'prithiv-v1', 'xception', 'seresnet50']
)

result = ensemble.detect('image.jpg', voting_method='weighted')
```

## Limitations

- **Optimized for faces**: Best performance on facial images
- **Input size**: Requires 256×256 resolution
- **Training data**: Performance depends on similarity to UADFV dataset
- **Newer methods**: May need retraining for latest deepfake techniques

## Citation

If you use this model, please cite:

```bibtex
@article{afchar2018mesonet,
  title={MesoNet: a Compact Facial Video Forgery Detection Network},
  author={Afchar, Darius and Nozick, Vincent and Yamagishi, Junichi and Echizen, Isao},
  journal={arXiv preprint arXiv:1809.00888},
  year={2018}
}
```

## Model Source

Original model: https://github.com/muffin-123/DeepFake-Detection-mesonet_model.h5-

## License

Please refer to the original repository for license information.
"""
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    logger.info(f"✓ README created: {readme_path}")


def main():
    """Main function to download MesoNet models"""
    logger.info("Starting MesoNet model download...")
    logger.info("")
    
    try:
        # Download the model
        model_file = download_mesonet_model()
        
        # Create README
        create_readme(model_file.parent)
        
        logger.info("")
        logger.info("="*60)
        logger.info("✓ MesoNet Setup Complete!")
        logger.info("="*60)
        logger.info("")
        logger.info("You can now use MesoNet for deepfake detection!")
        logger.info("")
        logger.info("Quick start:")
        logger.info("  python")
        logger.info("  >>> from detectionLib.detect_deepfake_mesonet import MesoNetDetector")
        logger.info(f"  >>> detector = MesoNetDetector(model_path=r'{model_file}')")
        logger.info("  >>> result = detector.detect_single('your_image.jpg')")
        logger.info("")
        
    except KeyboardInterrupt:
        logger.info("\n✗ Download cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
