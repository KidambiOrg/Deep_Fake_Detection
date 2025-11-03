"""
Script to check actual label mappings in HuggingFace models
This will help us understand why the agent predictions don't match imagedetectionlib
"""

import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_model_labels(model_name, model_path):
    """Check what labels a model actually has"""
    print(f"\n{'='*80}")
    print(f"Checking: {model_name}")
    print(f"Path: {model_path}")
    print(f"{'='*80}")
    
    try:
        if not os.path.exists(model_path):
            print(f"❌ Model path does not exist: {model_path}")
            return
        
        print("Loading model...")
        model = AutoModelForImageClassification.from_pretrained(model_path)
        processor = AutoImageProcessor.from_pretrained(model_path)
        
        print(f"\n✓ Model loaded successfully")
        print(f"\nModel config id2label:")
        print(f"  {model.config.id2label}")
        
        print(f"\nLabel analysis:")
        for idx, label in model.config.id2label.items():
            print(f"  Index {idx}: '{label}'")
            
            # Check what category this label falls into
            label_lower = label.lower()
            fake_indicators = ['fake', 'deepfake', 'artificial', 'generated', 'synthetic', 'ai']
            real_indicators = ['real', 'authentic', 'genuine', 'human', 'realism']
            
            is_fake = any(indicator in label_lower for indicator in fake_indicators)
            is_real = any(indicator in label_lower for indicator in real_indicators)
            
            if is_fake:
                print(f"    → Detected as FAKE class")
            elif is_real:
                print(f"    → Detected as REAL class")
            else:
                print(f"    → ⚠️ NOT DETECTED (no matching keywords!)")
        
        print(f"\n{'='*80}\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"{'='*80}\n")

# Check all HuggingFace models
print("\n" + "="*80)
print("CHECKING MODEL LABEL CONFIGURATIONS")
print("="*80)

models_to_check = [
    ("prithiv-v1", "Models/huggingface/prithivMLMods/deepfake-detector-model-v1"),
    ("dima-vit", "Models/dimavit"),
    ("organika-sdxl", "Models/huggingface/Organika/sdxl-detector"),
]

for model_name, model_path in models_to_check:
    check_model_labels(model_name, model_path)

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
