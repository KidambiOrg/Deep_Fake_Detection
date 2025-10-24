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
    'umm-efficientnet': {
        'name': 'umm-maybe/AI-image-detector',
        'description': 'EfficientNet-based AI image detector',
        'architecture': 'EfficientNet',
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
    'nonescape-mini': {
        'name': 'nonescape-mini-v0',
        'description': 'Lightweight Nonescape model for fast detection',
        'architecture': 'Nonescape',
        'source': 'nonescape',
        'model_path': 'Models/nonescape-mini-v0.safetensors',
        'use_mini': True,
        'recommended_threshold': 0.5,
    },
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
    
    # DistilDIRE Models
    'distil-dire-celebahq': {
        'name': 'DistilDIRE CelebA-HQ',
        'description': 'DistilDIRE trained on CelebA-HQ (good for faces/portraits)',
        'architecture': 'DistilDIRE',
        'source': 'distil_dire',
        'model_type': 'celebahq',
        'recommended_threshold': 0.5,
    },
    'distil-dire-celebahq-full': {
        'name': 'DistilDIRE CelebA-HQ Full',
        'description': 'Full DistilDIRE model trained on CelebA-HQ',
        'architecture': 'DistilDIRE',
        'source': 'distil_dire',
        'model_type': 'celebahq-full',
        'recommended_threshold': 0.5,
    },
    'distil-dire-eps-imagenet': {
        'name': 'DistilDIRE ImageNet (EPS)',
        'description': 'Fast general-purpose DistilDIRE detection',
        'architecture': 'DistilDIRE',
        'source': 'distil_dire',
        'model_type': 'eps-imagenet',
        'recommended_threshold': 0.5,
    },
    
    # SEResNet-50 Models
    'seresnet50-pretrained': {
        'name': 'SEResNet-50 Pretrained',
        'description': 'SEResNet-50 pretrained on ImageNet',
        'architecture': 'SEResNet-50',
        'source': 'seresnet50',
        'model_type': 'pretrained',
        'recommended_threshold': 0.5,
    },
    'seresnet50': {
        'name': 'SEResNet-50 Fine-tuned',
        'description': 'SEResNet-50 fine-tuned for deepfake detection',
        'architecture': 'SEResNet-50',
        'source': 'seresnet50',
        'model_type': 'pretrained',  # Changed from 'complete' to 'pretrained'
        'model_path': 'Models/SERESNet_50/seresnet50_pretrained.pth',  # Use pretrained weights
        'recommended_threshold': 0.5,
    },
    
    # EfficientNet-B7 Model (Noisy Student)
    'efficientnet-b7-ns': {
        'name': 'EfficientNet-B7-NoisyStudent',
        'description': 'EfficientNet-B7 with Noisy Student weights (DFDC 3rd place solution - high accuracy)',
        'architecture': 'EfficientNet-B7',
        'source': 'efficientnet_b7',
        'model_path': 'Models/EfficientNet-B7/efficientnet_b7_ns_pytorch.pth',
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
    },
    
    # TorchVision/Timm Models
    'efficientnet_b4': {
        'name': 'efficientnet_b4',
        'description': 'EfficientNet-B4 from torchvision (pretrained on ImageNet)',
        'architecture': 'EfficientNet',
        'source': 'torchvision',
        'input_size': 380,
        'modify_classifier': True,
        'classifier_attr': 'classifier',
        'classifier_in_features': 1792,
        'recommended_threshold': 0.5,
    },
    'efficientnet_b7': {
        'name': 'efficientnet_b7',
        'description': 'EfficientNet-B7 from torchvision (high accuracy, larger model)',
        'architecture': 'EfficientNet',
        'source': 'torchvision',
        'input_size': 600,
        'modify_classifier': True,
        'classifier_attr': 'classifier',
        'classifier_in_features': 2560,
        'recommended_threshold': 0.5,
    },
    'efficientnet_b4_timm': {
        'name': 'efficientnet_b4',
        'description': 'EfficientNet-B4 from timm (pretrained on ImageNet)',
        'architecture': 'EfficientNet',
        'source': 'timm',
        'model_name': 'efficientnet_b4',
        'input_size': 380,
        'modify_classifier': True,
        'recommended_threshold': 0.5,
    },
    'efficientnet_b7_timm': {
        'name': 'efficientnet_b7',
        'description': 'EfficientNet-B7 from timm (high accuracy, larger model)',
        'architecture': 'EfficientNet',
        'source': 'timm',
        'model_name': 'efficientnet_b7',
        'input_size': 600,
        'modify_classifier': True,
        'recommended_threshold': 0.5,
    },
    'xception': {
        'name': 'xception',
        'description': 'Xception architecture (excellent for deepfake detection)',
        'architecture': 'Xception',
        'source': 'timm',
        'model_name': 'xception',
        'input_size': 299,
        'modify_classifier': True,
        'recommended_threshold': 0.5,
    },
    'xception41': {
        'name': 'xception41',
        'description': 'Xception-41 variant (balanced performance)',
        'architecture': 'Xception',
        'source': 'timm',
        'model_name': 'xception41',
        'input_size': 299,
        'modify_classifier': True,
        'recommended_threshold': 0.5,
    },
    'xception65': {
        'name': 'xception65',
        'description': 'Xception-65 variant (higher capacity)',
        'architecture': 'Xception',
        'source': 'timm',
        'model_name': 'xception65',
        'input_size': 299,
        'modify_classifier': True,
        'recommended_threshold': 0.5,
    },
    'xception71': {
        'name': 'xception71',
        'description': 'Xception-71 variant (highest capacity)',
        'architecture': 'Xception',
        'source': 'timm',
        'model_name': 'xception71',
        'input_size': 299,
        'modify_classifier': True,
        'recommended_threshold': 0.5,
    },
    'resnet34': {
        'name': 'resnet34',
        'description': 'ResNet-34 from torchvision (balanced speed and accuracy)',
        'architecture': 'ResNet',
        'source': 'torchvision',
        'input_size': 224,
        'modify_classifier': True,
        'classifier_attr': 'fc',
        'classifier_in_features': 512,
        'recommended_threshold': 0.5,
    },
    'resnet50': {
        'name': 'resnet50',
        'description': 'ResNet-50 from torchvision (pretrained on ImageNet)',
        'architecture': 'ResNet',
        'source': 'torchvision',
        'input_size': 224,
        'modify_classifier': True,
        'classifier_attr': 'fc',
        'classifier_in_features': 2048,
        'recommended_threshold': 0.5,
    },
    'resnet101': {
        'name': 'resnet101',
        'description': 'ResNet-101 from torchvision (pretrained on ImageNet)',
        'architecture': 'ResNet',
        'source': 'torchvision',
        'input_size': 224,
        'modify_classifier': True,
        'classifier_attr': 'fc',
        'classifier_in_features': 2048,
        'recommended_threshold': 0.5,
    },
    'vit_b_16': {
        'name': 'vit_b_16',
        'description': 'Vision Transformer B/16 from torchvision',
        'architecture': 'ViT',
        'source': 'torchvision',
        'input_size': 224,
        'modify_classifier': True,
        'classifier_attr': 'heads',
        'classifier_in_features': 768,
        'recommended_threshold': 0.5,
    },
    'deit_base': {
        'name': 'deit_base_patch16_224',
        'description': 'DeiT Base (Data-efficient Image Transformer) - efficient training',
        'architecture': 'DeiT',
        'source': 'timm',
        'model_name': 'deit_base_patch16_224',
        'input_size': 224,
        'modify_classifier': True,
        'recommended_threshold': 0.5,
    },
    'deit_small': {
        'name': 'deit_small_patch16_224',
        'description': 'DeiT Small (Data-efficient Image Transformer) - faster variant',
        'architecture': 'DeiT',
        'source': 'timm',
        'model_name': 'deit_small_patch16_224',
        'input_size': 224,
        'modify_classifier': True,
        'recommended_threshold': 0.5,
    },
    'deit_tiny': {
        'name': 'deit_tiny_patch16_224',
        'description': 'DeiT Tiny (Data-efficient Image Transformer) - lightweight',
        'architecture': 'DeiT',
        'source': 'timm',
        'model_name': 'deit_tiny_patch16_224',
        'input_size': 224,
        'modify_classifier': True,
        'recommended_threshold': 0.5,
    },
    'swin_b': {
        'name': 'swin_b',
        'description': 'Swin Transformer Base from torchvision (hierarchical vision transformer)',
        'architecture': 'Swin Transformer',
        'source': 'torchvision',
        'input_size': 224,
        'modify_classifier': True,
        'classifier_attr': 'head',
        'classifier_in_features': 1024,
        'recommended_threshold': 0.5,
    },
    'swin_t': {
        'name': 'swin_t',
        'description': 'Swin Transformer Tiny from torchvision (faster, lighter version)',
        'architecture': 'Swin Transformer',
        'source': 'torchvision',
        'input_size': 224,
        'modify_classifier': True,
        'classifier_attr': 'head',
        'classifier_in_features': 768,
        'recommended_threshold': 0.5,
    },
    'swin_v2_b': {
        'name': 'swin_v2_b',
        'description': 'Swin Transformer V2 Base from torchvision (improved version)',
        'architecture': 'Swin Transformer',
        'source': 'torchvision',
        'input_size': 256,
        'modify_classifier': True,
        'classifier_attr': 'head',
        'classifier_in_features': 1024,
        'recommended_threshold': 0.5,
    },
    'swin_base_timm': {
        'name': 'swin_base_patch4_window7_224',
        'description': 'Swin Transformer Base from timm (hierarchical vision transformer)',
        'architecture': 'Swin Transformer',
        'source': 'timm',
        'model_name': 'swin_base_patch4_window7_224',
        'input_size': 224,
        'modify_classifier': True,
        'recommended_threshold': 0.5,
    },
    'swin_large_timm': {
        'name': 'swin_large_patch4_window7_224',
        'description': 'Swin Transformer Large from timm (high capacity model)',
        'architecture': 'Swin Transformer',
        'source': 'timm',
        'model_name': 'swin_large_patch4_window7_224',
        'input_size': 224,
        'modify_classifier': True,
        'recommended_threshold': 0.5,
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
