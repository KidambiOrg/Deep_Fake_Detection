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
        'description': 'SigLIP-based deepfake detector (default)',
        'architecture': 'SigLIP',
        'source': 'huggingface',
        'labels': {0: 'Fake', 1: 'Real'},
        'recommended_threshold': 0.5,
    },
    'dima-vit': {
        'name': 'dima806/deepfake_vs_real_image_detection',
        'description': 'Vision Transformer based detector',
        'architecture': 'ViT',
        'source': 'huggingface',
        'labels': None,
        'recommended_threshold': 0.5,
    },   
    'organika-sdxl': {
        'name': 'Organika/sdxl-detector',
        'description': 'Specialized for Stable Diffusion generated images',
        'architecture': 'Custom',
        'source': 'huggingface',
        'labels': None,
        'recommended_threshold': 0.5,
    },
    
    # Nonescape Models   
    'nonescape-full': {
        'name': 'nonescape-v0',
        'description': 'Full Nonescape model for high accuracy detection',
        'architecture': 'Nonescape',
        'source': 'nonescape',
        'model_path': 'Models/nonescape-v0.safetensors',
        'use_mini': False,
        'recommended_threshold': 0.5,
    },
    
    # Universal Fake Detect V2
    'universal-fake-v2': {
        'name': 'UniversalFakeDetect V2',
        'description': 'Universal detector for various types of AI-generated images',
        'architecture': 'ResNet50',
        'source': 'universal_fake_v2',
        'recommended_threshold': 0.5,
    },
    
    
    
    # SEResNet-50 Models    
    'seresnet50': {
        'name': 'SEResNet-50 Fine-tuned',
        'description': 'SEResNet-50 fine-tuned for deepfake detection',
        'architecture': 'SEResNet-50',
        'source': 'seresnet50',
        'model_type': 'pretrained',  # Changed from 'complete' to 'pretrained'
        'model_path': 'Models/SERESNet_50/seresnet50_pretrained.pth',  # Use pretrained weights
        'recommended_threshold': 0.5,
    },
    
    
    # FTCN Model
    'ftcn': {
        'name': 'FTCN',
        'description': 'Fully Temporal Convolutional Network for deepfake detection',
        'architecture': 'FTCN',
        'source': 'ftcn',
        'model_path': 'Models/FTCN/ftcn_tt.pth',
        'recommended_threshold': 0.5,
    }
    
    
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
