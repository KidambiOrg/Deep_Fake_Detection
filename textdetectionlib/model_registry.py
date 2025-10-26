"""
Text Detection Model Registry Module

This module contains the registry metadata for text detection models
without any heavy ML library imports.
"""

# Text Detection Model Registry
TEXT_MODEL_REGISTRY = {
    # Character-level CNN model
    'charcnn': {
        'name': 'CharCNN',
        'description': 'Character-level CNN for deepfake text detection',
        'architecture': 'CNN',
        'source': 'trained',
        'model_files': {
            'model': {
                'filename': 'charcnn_model.h5',
                'blob_storage': {
                    'container': 'textmodels',
                    'blob': 'charcnn/charcnn_model.h5'
                }
            },
            'tokenizer': {
                'filename': 'charcnn_tokenizer.pkl',
                'blob_storage': {
                    'container': 'textmodels',
                    'blob': 'charcnn/charcnn_tokenizer.pkl'
                }
            }
        },
        'local_path': 'textmodels',
        'labels': {'human': 0, 'bot': 1}
    },
    
    # BERT-based ML classifier
    'bert_ml': {
        'name': 'BERT ML',
        'description': 'BERT-based ML classifier for deepfake text detection',
        'architecture': 'BERT + LogisticRegression',
        'source': 'trained',
        'model_files': {
            'classifier': {
                'filename': 'bert_ml_classifier.pkl',
                'blob_storage': {
                    'container': 'textmodels',
                    'blob': 'bert_ml/bert_ml_classifier.pkl'
                }
            }
        },
        'local_path': 'textmodels',
        'labels': {'human': 0, 'bot': 1}
    },
    
    # DistilBERT-based ML classifier
    'distilbert_ml': {
        'name': 'DistilBERT ML',
        'description': 'DistilBERT-based ML classifier for deepfake text detection',
        'architecture': 'DistilBERT + LogisticRegression',
        'source': 'trained',
        'model_files': {
            'classifier': {
                'filename': 'distilbert_ml_classifier.pkl',
                'blob_storage': {
                    'container': 'textmodels',
                    'blob': 'distilbert_ml/distilbert_ml_classifier.pkl'
                }
            }
        },
        'local_path': 'textmodels',
        'labels': {'human': 0, 'bot': 1}
    },
    
    # Bag of Words ML classifier
    'bow_ml': {
        'name': 'BoW ML',
        'description': 'Bag of Words ML classifier for deepfake text detection',
        'architecture': 'TF-IDF + LogisticRegression',
        'source': 'trained',
        'model_files': {
            'classifier': {
                'filename': 'bow_ml_classifier.pkl',
                'blob_storage': {
                    'container': 'textmodels',
                    'blob': 'bow_ml/bow_ml_classifier.pkl'
                }
            },
            'vectorizer': {
                'filename': 'bow_ml_vectorizer.pkl',
                'blob_storage': {
                    'container': 'textmodels',
                    'blob': 'bow_ml/bow_ml_vectorizer.pkl'
                }
            }
        },
        'local_path': 'textmodels',
        'labels': {'human': 0, 'bot': 1}
    }
}


def list_available_text_models():
    """
    Get a list of all available text detection models
    
    Returns:
        Dictionary of available text models with their details
    """
    return TEXT_MODEL_REGISTRY.copy()


def get_text_model_info(model_key):
    """
    Get information about a specific text model
    
    Args:
        model_key: The key of the model to get info for
        
    Returns:
        Dictionary with model information or None if not found
    """
    return TEXT_MODEL_REGISTRY.get(model_key)


def get_text_models_by_architecture(architecture):
    """
    Get all text models with a specific architecture
    
    Args:
        architecture: The architecture to filter by
        
    Returns:
        Dictionary of models with the specified architecture
    """
    return {
        key: info for key, info in TEXT_MODEL_REGISTRY.items()
        if info.get('architecture') == architecture
    }
