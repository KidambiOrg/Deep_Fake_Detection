# Deep Fake Detection
A library to detect deep fake images and AI-generated text. 

## Supported Models

### Image Detection
- **EfficientNet-B7 (Noisy Student)** - DFDC 3rd place solution - high accuracy deepfake detection
- **MesoNet** - Lightweight CNN for mesoscopic deepfake detection
- **SEResNet-50** - Squeeze-and-Excitation ResNet for enhanced features
- **Xception** - Depthwise separable convolutions
- **DistilDIRE** - Diffusion-based reconstruction analysis
- **Nonescape** - ViT + EfficientNet ensemble
- **UniversalFakeDetectV2** - CLIP-based universal detector
- **HuggingFace Models** - Various transformer-based detectors

## Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt

# Optional: For MesoNet Keras model support
pip install tensorflow
```

### Download Models

Download the MesoNet pretrained model:
```bash
python Scripts/download_mesonet_models.py
```

### Basic Usage

```python
from detectionLib.detect_deepfake_mesonet import MesoNetDetector

# Initialize MesoNet detector
detector = MesoNetDetector(
    model_path='Models/mesonet/mesonet_model.h5',
    threshold=0.5
)

# Detect deepfake in an image
result = detector.detect_single('path/to/image.jpg')
print(f"Is Fake: {result.is_fake} (confidence: {result.confidence:.2%})")
```

### Ensemble Detection

For improved accuracy, use multiple models:

```python
from detectionLib.detect_deepfake_image import EnsembleDetector

ensemble = EnsembleDetector(
    model_keys=['efficientnet-b7-ns', 'mesonet', 'seresnet50', 'xception']
)

result = ensemble.detect('image.jpg', voting_method='weighted')
print(f"Ensemble Prediction: {'FAKE' if result.is_deepfake else 'REAL'}")
```

### Using EfficientNet-B7 (Noisy Student)

The EfficientNet-B7 model with Noisy Student weights is based on the 3rd place solution from the Deepfake Detection Challenge:

```python
from detectionLib.detect_deepfake_efficientnet_b7 import EfficientNetB7Detector

# Initialize detector with pretrained Noisy Student weights
detector = EfficientNetB7Detector(
    input_size=224,  # Optimized for face crops
    debug=True
)

# Detect deepfake
result = detector.detect('path/to/image.jpg', threshold=0.5)
print(f"Is Deepfake: {result.is_deepfake} (confidence: {result.confidence:.2%})")
```

## Model Documentation

- **MesoNet Integration**: See [MESONET_INTEGRATION.md](MESONET_INTEGRATION.md) for detailed documentation

## References
- https://ar5iv.labs.arxiv.org/html/2006.07397
- https://semafor.dsri.org/analytic-catalog/
- https://github.com/aediliclabs/nonescape/tree/master
- MesoNet Paper: https://arxiv.org/abs/1809.00888
- MesoNet Model: https://github.com/muffin-123/DeepFake-Detection-mesonet_model.h5-
- EfficientNet-B7 DFDC Solution: https://github.com/NTech-Lab/deepfake-detection-challenge
- Noisy Student Paper: https://arxiv.org/abs/1911.04252