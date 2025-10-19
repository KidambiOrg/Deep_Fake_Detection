# Deep Fake Detection
A library to detect deep fake images and AI-generated text. 

## Supported Models

### Image Detection
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
    model_keys=['mesonet', 'seresnet50', 'xception']
)

result = ensemble.detect('image.jpg', voting_method='weighted')
print(f"Ensemble Prediction: {'FAKE' if result.is_deepfake else 'REAL'}")
```

## Model Documentation

- **MesoNet Integration**: See [MESONET_INTEGRATION.md](MESONET_INTEGRATION.md) for detailed documentation

## References
- https://ar5iv.labs.arxiv.org/html/2006.07397
- https://semafor.dsri.org/analytic-catalog/
- https://github.com/aediliclabs/nonescape/tree/master
- MesoNet Paper: https://arxiv.org/abs/1809.00888
- MesoNet Model: https://github.com/muffin-123/DeepFake-Detection-mesonet_model.h5-