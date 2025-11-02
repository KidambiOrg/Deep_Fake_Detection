"""
Quick test to verify SEResNet50 model loading
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_seresnet50_loading():
    """Test if SEResNet50 model loads correctly"""
    print("\n=== Testing SEResNet50 Model Loading ===\n")
    
    try:
        from Agents.ImageAgents.tools.seresnet50_detect_tool import load_seresnet50_model
        
        print("Attempting to load SEResNet50 model...")
        model, device = load_seresnet50_model()
        
        if model is not None:
            print(f"\n✅ Model loaded successfully!")
            print(f"   Device: {device}")
            print(f"   Model type: {type(model)}")
        else:
            print(f"\n❌ Model loading failed - returned None")
            print(f"   Check logs above for details")
            
    except Exception as e:
        print(f"\n❌ Exception during model loading:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_seresnet50_loading()
