# MobileNetV2 CNN - Quick Reference Guide

## Usage Examples

### 1. Direct Detection Library Usage

```python
from imagedetectionlib.detect_deepfake_mobilenetv2_cnn import create_mobilenetv2_cnn_detector

# Create detector
detector = create_mobilenetv2_cnn_detector()

# Detect from image file
result = detector.detect_single('path/to/image.jpg')

print(f"Prediction: {'FAKE' if result.is_fake else 'REAL'}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Fake Probability: {result.fake_probability:.4f}")
print(f"Real Probability: {result.real_probability:.4f}")
```

### 2. Batch Processing

```python
from imagedetectionlib.detect_deepfake_mobilenetv2_cnn import create_mobilenetv2_cnn_detector

detector = create_mobilenetv2_cnn_detector()

# Process multiple images
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = detector.detect_batch(image_paths, batch_size=8)

for path, result in zip(image_paths, results):
    print(f"{path}: {'FAKE' if result.is_fake else 'REAL'} ({result.confidence:.2%})")
```

### 3. Video Frame Analysis

```python
from imagedetectionlib.detect_deepfake_mobilenetv2_cnn import create_mobilenetv2_cnn_detector

detector = create_mobilenetv2_cnn_detector()

# Analyze video frames
video_result = detector.detect_video_frames(
    'path/to/video.mp4',
    frame_skip=5,      # Process every 5th frame
    max_frames=100     # Process up to 100 frames
)

print(f"Video Prediction: {video_result['prediction']}")
print(f"Confidence: {video_result['confidence']:.2%}")
print(f"Frames Processed: {video_result['frames_processed']}")
```

### 4. Full Workflow with All Models (Recommended)

```python
import asyncio
from Agents import run_image_detection_workflow

async def analyze_image():
    result = await run_image_detection_workflow(
        image_path='path/to/image.jpg',
        threshold=0.20,
        voting_method='optimized_weighted'  # Uses all 7 models including CNN
    )
    
    print(f"Ensemble Prediction: {result.ensemblePrediction}")
    print(f"Confidence: {result.ensembleConfidence:.2%}")
    print(f"Voting Method: {result.votingMethod}")
    
    # Individual model results (includes mobilenetv2_cnn)
    for model_result in result.individualModelResults:
        print(f"{model_result.model}: {model_result.prediction} ({model_result.confidence:.2%})")

# Run the workflow
asyncio.run(analyze_image())
```

### 5. Agent Tool Usage (Azure AI Foundry)

```python
from Agents.ImageAgents.tools.mobilenetv2_cnn_detect_tool import detect_with_mobilenetv2_cnn_tool

# Use in agent context
result_dict = detect_with_mobilenetv2_cnn_tool('path/to/image.jpg')

# Returns dictionary format:
# {
#     'model': 'mobilenetv2_cnn',
#     'prediction': 'FAKE' or 'REAL',
#     'confidence': 0.85,
#     'fakeScore': 0.85,
#     'realScore': 0.15
# }
```

## Model Information

### Get Model Details

```python
from imagedetectionlib.model_registry import get_model_info

# Get CNN model info from registry
info = get_model_info('mobilenetv2_cnn')

print(f"Name: {info['display_name']}")
print(f"Architecture: {info['architecture']}")
print(f"Model Path: {info['model_path']}")
print(f"Output Format: {info['output_format']}")
print(f"Label Convention: {info['label_convention']}")
```

### List All Available Models

```python
from imagedetectionlib.model_registry import list_available_models

models = list_available_models()

for key, model_info in models.items():
    print(f"{key}: {model_info['display_name']}")
    
# Output includes:
# mobilenetv2_cnn: MobileNetV2 CNN Detector
# mobilenetv2: MobileNetV2 Detector
# nonescape-full: NoEscape Detector
# seresnet50: SEResNet50 Detector
# ... etc
```

## Configuration Options

### Custom Threshold

```python
detector = create_mobilenetv2_cnn_detector(threshold=0.3)  # Lower threshold
```

### Ensemble Predictions

```python
# Use ensemble of augmented predictions for higher accuracy
result = detector.detect_single('image.jpg', use_ensemble=True)
```

### Custom Model Path

```python
detector = create_mobilenetv2_cnn_detector(
    model_path='custom/path/to/mobilenetv2_cnn.h5'
)
```

### Debug Mode

```python
detector = create_mobilenetv2_cnn_detector(debug=True)  # Verbose logging
```

## Understanding Output

### Label Convention

The MobileNetV2 CNN model uses **standard binary classification**:
- **Sigmoid output close to 1.0** = FAKE (positive class)
- **Sigmoid output close to 0.0** = REAL (negative class)

### Output Format

```python
DetectionResult(
    is_fake=True,              # Boolean prediction
    confidence=0.89,           # Max of (fake_prob, real_prob)
    fake_probability=0.89,     # Probability of FAKE
    real_probability=0.11,     # Probability of REAL (1 - fake_prob)
    model_name='MobileNetV2_CNN',
    processing_time=1.05,      # Seconds
    metadata={
        'threshold': 0.5,
        'ensemble': False,
        'input_size': (224, 224),
        'output_format': 'single_sigmoid'
    }
)
```

## Performance Tips

1. **Use Batch Processing** for multiple images:
   ```python
   results = detector.detect_batch(image_paths, batch_size=16)
   ```

2. **Enable Model Caching** (automatic in agent tools):
   - Models are cached globally after first load
   - Subsequent calls are much faster

3. **Adjust Frame Skip** for videos:
   ```python
   # Process every 10th frame for faster processing
   result = detector.detect_video_frames(video_path, frame_skip=10)
   ```

4. **Use Ensemble Voting** for production:
   ```python
   # Full workflow with all 7 models
   result = await run_image_detection_workflow(
       image_path='image.jpg',
       voting_method='optimized_weighted'
   )
   ```

## Troubleshooting

### Model Not Found Error

```python
# Ensure model file exists at:
# Models/MobileNETV2_CNN_FineTuned/mobilenetv2_deepfake.h5

import os
model_path = 'Models/MobileNETV2_CNN_FineTuned/mobilenetv2_deepfake.h5'
print(f"Model exists: {os.path.exists(model_path)}")
```

### Blob Storage Issues

If blob storage download fails:
1. Check Azure credentials are configured
2. Verify blob storage container exists
3. Upload model using: `python Scripts/upload_models_to_blob.py`

### Memory Issues

For large batch processing:
```python
# Reduce batch size
results = detector.detect_batch(images, batch_size=4)
```

## Integration with Voting Methods

The CNN model is automatically included in all ensemble voting methods:

```python
# These voting methods now use 7 models (including CNN)
await run_image_detection_workflow(image_path, voting_method='majority')
await run_image_detection_workflow(image_path, voting_method='weighted')
await run_image_detection_workflow(image_path, voting_method='bayesian')
await run_image_detection_workflow(image_path, voting_method='optimized_weighted')
# ... and 10+ more voting methods
```

## Next Steps

1. **Upload to Blob Storage**: `python Scripts/upload_models_to_blob.py`
2. **Retrain LR Voter**: `python utils/train_lr_from_two_folders.py`
3. **Test Full Workflow**: `python Services/app_detect_image_agent.py test_image.jpg`

## Additional Resources

- **Integration Summary**: See `MOBILENETV2_CNN_INTEGRATION.md`
- **Test Suite**: Run `python Tests/test_mobilenetv2_cnn_integration.py`
- **Reference**: https://github.com/Gopika-2025/Deep_fake_detection
