"""
Lightweight Model Registry Module

This module contains only the model registry metadata without any heavy ML library imports.
Use this for fast listing of available models without loading any actual model weights.
"""

# Model Registry - Available deepfake detection models
MODEL_REGISTRY = {
    # HuggingFace Models
    'prithiv-v1': {
        'name': 'prithivMLMods/deepfake-detector-model-v1',
        'display_name': 'Prithiv V1 Detector',
        'description': 'SigLIP-based deepfake detector (default)',
        'architecture': 'SigLIP',
        'source': 'huggingface',
        'labels': {0: 'Fake', 1: 'Real'},
        'recommended_threshold': 0.5,
        'local_path': 'Models/huggingface/prithivMLMods/deepfake-detector-model-v1',
        'blob_storage': {
            'container': 'models',
            'path': 'huggingface/prithivMLMods/deepfake-detector-model-v1'
        }
    },
    'dima-vit': {
        'name': 'dima806/deepfake_vs_real_image_detection',
        'display_name': 'DimaVit Detector',
        'description': 'Vision Transformer based detector',
        'architecture': 'ViT',
        'source': 'huggingface',
        'labels': None,
        'recommended_threshold': 0.5,
        'local_path': 'Models/dimavit',  # Local pre-downloaded weights
        'blob_storage': {
            'container': 'models',
            'path': 'huggingface/dima806/deepfake_vs_real_image_detection'
        }
    },
    'organika-sdxl': {
        'name': 'Organika/sdxl-detector',
        'display_name': 'Organika SDXL Detector',
        'description': 'Specialized for Stable Diffusion generated images',
        'architecture': 'Custom',
        'source': 'huggingface',
        'labels': None,
        'recommended_threshold': 0.5,
        'local_path': 'Models/huggingface/Organika/sdxl-detector',
        'blob_storage': {
            'container': 'models',
            'path': 'huggingface/Organika/sdxl-detector'
        }
    },
    'prithiv-open-deep-fake': {
        'name': 'prithivMLmods/open-deepfake-detection',
        'display_name': 'Prithiv Open Deepfake Detector',
        'description': 'Open source SigLIP-based deepfake detector',
        'architecture': 'SigLIP',
        'source': 'huggingface',
        'labels': {0: 'Fake', 1: 'Real'},
        'recommended_threshold': 0.5,
        'local_path': 'Models/huggingface/prithiv_open_deep_fake',
        'blob_storage': {
            'container': 'models',
            'path': 'huggingface/prithiv_open_deep_fake'
        }
    },
    
    # Nonescape Models   
    'nonescape-full': {
        'name': 'nonescape-v0',
        'display_name': 'NoEscape Detector',
        'description': 'Full Nonescape model for high accuracy detection',
        'architecture': 'Nonescape',
        'source': 'nonescape',
        'model_path': 'Models/nonescape-v0.safetensors',
        'use_mini': False,
        'recommended_threshold': 0.5,
        'blob_storage': {
            'container': 'models',
            'blob': 'nonescape/nonescape-v0.safetensors'
        }
    },
    
    
    # SEResNet-50 Models    
    'seresnet50': {
        'name': 'SEResNet-50 Fine-tuned',
        'display_name': 'SEResNet50 Detector',
        'description': 'SEResNet-50 fine-tuned for deepfake detection',
        'architecture': 'SEResNet-50',
        'source': 'seresnet50',
        'model_type': 'pretrained',  # Changed from 'complete' to 'pretrained'
        'model_path': 'Models/SERESNet_50/seresnet50_pretrained.pth',  # Use pretrained weights
        'recommended_threshold': 0.5,
        'local_path': 'Models/SERESNet_50',  # Directory containing all model files
        'blob_storage': {
            'container': 'models',
            'path': 'SERESNet_50'  # Upload entire directory
        }
    },
    
    # MobileNetV2 Models
    'mobilenetv2': {
        'name': 'MobileNetV2 Fine-tuned',
        'display_name': 'MobileNetV2 Detector',
        'description': 'Lightweight MobileNetV2 fine-tuned for deepfake detection',
        'architecture': 'MobileNetV2',
        'source': 'mobilenetv2',
        'model_type': 'finetuned',
        'model_path': 'Models/MobileNetV2_FineTuned/mobilenetv2_deepfake_model.keras',
        'recommended_threshold': 0.5,
        'local_path': 'Models/MobileNetV2_FineTuned',
        'blob_storage': {
            'container': 'models',
            'path': 'MobileNetV2_FineTuned'
        }
    },
    
    'mobilenetv2_cnn': {
        'name': 'MobileNetV2 CNN Fine-tuned',
        'display_name': 'MobileNetV2 CNN Detector',
        'description': 'Lightweight MobileNetV2 with custom CNN layers fine-tuned for deepfake detection',
        'architecture': 'MobileNetV2_CNN',
        'source': 'mobilenetv2_cnn',
        'model_type': 'cnn_finetuned',
        'model_path': 'Models/MobileNETV2_CNN_FineTuned/mobilenetv2_deepfake.h5',
        'recommended_threshold': 0.5,
        'output_format': 'single_sigmoid',
        'label_convention': '1.0=FAKE, 0.0=REAL',
        'local_path': 'Models/MobileNETV2_CNN_FineTuned',
        'blob_storage': {
            'container': 'models',
            'path': 'MobileNETV2_CNN_FineTuned'
        }
    },
    
    
  
}


def list_available_models():
    """
    Get a list of all available deepfake detection models
    
    Returns:
        Dictionary of available models with their details
    """
    return MODEL_REGISTRY.copy()


def get_model_info(model_key):
    """
    Get information about a specific model
    
    Args:
        model_key: The key of the model to get info for
        
    Returns:
        Dictionary with model information or None if not found
    """
    return MODEL_REGISTRY.get(model_key)


def get_models_by_source(source):
    """
    Get all models from a specific source
    
    Args:
        source: The source to filter by (e.g., 'huggingface', 'timm', 'torchvision')
        
    Returns:
        Dictionary of models from the specified source
    """
    return {
        key: info for key, info in MODEL_REGISTRY.items()
        if info.get('source') == source
    }


def get_models_by_architecture(architecture):
    """
    Get all models with a specific architecture
    
    Args:
        architecture: The architecture to filter by (e.g., 'EfficientNet', 'ViT', 'Xception')
        
    Returns:
        Dictionary of models with the specified architecture
    """
    return {
        key: info for key, info in MODEL_REGISTRY.items()
        if info.get('architecture') == architecture
    }
