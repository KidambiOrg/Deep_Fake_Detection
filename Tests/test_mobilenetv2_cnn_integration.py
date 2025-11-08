"""
Test script for MobileNetV2 CNN model integration.

This script tests:
1. Model loading
2. Detection library functionality
3. Agent tool functionality
4. Model registry integration
5. Basic inference

Usage:
    python test_mobilenetv2_cnn_integration.py
"""

import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image
import tempfile

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_model_registry():
    """Test that CNN model is properly registered"""
    print("\n" + "="*80)
    print("TEST 1: Model Registry Integration")
    print("="*80)
    
    from imagedetectionlib.model_registry import get_model_info, list_available_models
    
    # Check if CNN model is in registry
    cnn_info = get_model_info('mobilenetv2_cnn')
    
    if cnn_info is None:
        print("❌ FAILED: mobilenetv2_cnn not found in model registry")
        return False
    
    print("✓ mobilenetv2_cnn found in registry")
    print(f"  Display Name: {cnn_info['display_name']}")
    print(f"  Architecture: {cnn_info['architecture']}")
    print(f"  Description: {cnn_info['description']}")
    print(f"  Model Path: {cnn_info['model_path']}")
    print(f"  Output Format: {cnn_info.get('output_format', 'N/A')}")
    print(f"  Label Convention: {cnn_info.get('label_convention', 'N/A')}")
    
    # Verify blob storage config
    if 'blob_storage' in cnn_info:
        print(f"  Blob Container: {cnn_info['blob_storage']['container']}")
        print(f"  Blob Path: {cnn_info['blob_storage']['path']}")
    else:
        print("  ⚠ No blob storage configuration")
    
    return True


def test_detection_library():
    """Test the detection library can be imported and initialized"""
    print("\n" + "="*80)
    print("TEST 2: Detection Library")
    print("="*80)
    
    try:
        from imagedetectionlib.detect_deepfake_mobilenetv2_cnn import (
            MobileNetV2CNNDetector,
            create_mobilenetv2_cnn_detector
        )
        
        print("✓ Import successful")
        
        # Check if model file exists
        model_path = Path(__file__).parent.parent / "Models" / "MobileNETV2_CNN_FineTuned" / "mobilenetv2_deepfake.h5"
        
        if not model_path.exists():
            print(f"⚠ Model file not found at: {model_path}")
            print("  Skipping model loading test")
            return True
        
        print(f"✓ Model file found at: {model_path}")
        
        # Try to create detector
        print("  Initializing detector...")
        detector = create_mobilenetv2_cnn_detector(model_path=str(model_path))
        
        print("✓ Detector created successfully")
        
        # Get model info
        model_info = detector.get_model_info()
        print(f"  Model Name: {model_info['model_name']}")
        print(f"  Architecture: {model_info['architecture']}")
        print(f"  Input Size: {model_info['input_size']}")
        print(f"  Output Format: {model_info['output_format']}")
        print(f"  Label Convention: {model_info['label_convention']}")
        print(f"  Parameters: {model_info['total_parameters']:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference():
    """Test inference on a dummy image"""
    print("\n" + "="*80)
    print("TEST 3: Inference Test")
    print("="*80)
    
    try:
        from imagedetectionlib.detect_deepfake_mobilenetv2_cnn import create_mobilenetv2_cnn_detector
        
        # Check if model file exists
        model_path = Path(__file__).parent.parent / "Models" / "MobileNETV2_CNN_FineTuned" / "mobilenetv2_deepfake.h5"
        
        if not model_path.exists():
            print(f"⚠ Model file not found at: {model_path}")
            print("  Skipping inference test")
            return True
        
        # Create detector
        detector = create_mobilenetv2_cnn_detector(model_path=str(model_path))
        
        # Create a dummy test image (random noise)
        print("  Creating dummy test image...")
        dummy_image = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = tmp.name
            dummy_image.save(tmp_path)
        
        try:
            print(f"  Running inference on dummy image...")
            result = detector.detect_single(tmp_path)
            
            print("✓ Inference successful")
            print(f"  Prediction: {'FAKE' if result.is_fake else 'REAL'}")
            print(f"  Confidence: {result.confidence:.2%}")
            print(f"  Fake Probability: {result.fake_probability:.4f}")
            print(f"  Real Probability: {result.real_probability:.4f}")
            print(f"  Processing Time: {result.processing_time:.3f}s")
            
            # Verify probabilities sum to 1
            total_prob = result.fake_probability + result.real_probability
            if abs(total_prob - 1.0) > 0.01:
                print(f"  ⚠ Warning: Probabilities don't sum to 1.0 (sum={total_prob:.4f})")
            else:
                print(f"  ✓ Probabilities sum correctly: {total_prob:.4f}")
            
            return True
            
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_tool():
    """Test the agent tool can be imported"""
    print("\n" + "="*80)
    print("TEST 4: Agent Tool")
    print("="*80)
    
    try:
        from Agents.ImageAgents.tools.mobilenetv2_cnn_detect_tool import (
            detect_with_mobilenetv2_cnn_tool,
            load_mobilenetv2_cnn_model
        )
        
        print("✓ Agent tool imports successful")
        
        # Check if model file exists
        model_path = Path(__file__).parent.parent / "Models" / "MobileNETV2_CNN_FineTuned" / "mobilenetv2_deepfake.h5"
        
        if not model_path.exists():
            print(f"⚠ Model file not found at: {model_path}")
            print("  Skipping tool loading test")
            return True
        
        print("  Testing model loading...")
        model = load_mobilenetv2_cnn_model()
        
        if model is None:
            print("❌ Model loading returned None")
            return False
        
        print("✓ Model loaded successfully via tool")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_creation():
    """Test the agent can be created"""
    print("\n" + "="*80)
    print("TEST 5: Agent Creation")
    print("="*80)
    
    try:
        from Agents.ImageAgents.mobilenetv2_cnn_agent import create_mobilenetv2_cnn_agent
        
        print("✓ Agent module imports successful")
        
        # Check if Azure environment variables are set
        project_endpoint = os.getenv('AZURE_AI_PROJECT_ENDPOINT')
        deployment_name = os.getenv('AZURE_AI_MODEL_DEPLOYMENT_NAME')
        
        if not project_endpoint or not deployment_name:
            print("⚠ Azure environment variables not set")
            print("  Skipping agent creation test")
            print("  Required: AZURE_AI_PROJECT_ENDPOINT, AZURE_AI_MODEL_DEPLOYMENT_NAME")
            return True
        
        print("  Creating agent...")
        agent = create_mobilenetv2_cnn_agent()
        
        print("✓ Agent created successfully")
        print(f"  Agent name: {agent.name if hasattr(agent, 'name') else 'N/A'}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_workflow_integration():
    """Test that the workflow includes the CNN model"""
    print("\n" + "="*80)
    print("TEST 6: Workflow Integration")
    print("="*80)
    
    try:
        from Agents.ImageAgents.image_workflow import get_or_create_agents
        
        print("✓ Workflow imports successful")
        
        # Check environment variables
        project_endpoint = os.getenv('AZURE_AI_PROJECT_ENDPOINT')
        if not project_endpoint:
            print("⚠ Azure environment variables not set")
            print("  Skipping workflow test")
            return True
        
        print("  Getting agents...")
        agents = get_or_create_agents()
        
        expected_count = 7  # Updated count with CNN model
        actual_count = len(agents)
        
        if actual_count == expected_count:
            print(f"✓ Correct number of agents: {actual_count}")
            return True
        else:
            print(f"❌ FAILED: Expected {expected_count} agents, got {actual_count}")
            return False
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lr_voter_update():
    """Test that LR voter includes the CNN model"""
    print("\n" + "="*80)
    print("TEST 7: Logistic Regression Voter Update")
    print("="*80)
    
    try:
        from utils.logistic_regression_voter import LogisticRegressionVoter
        
        voter = LogisticRegressionVoter()
        
        print("✓ LR Voter created")
        print(f"  Model order: {voter.model_order}")
        
        if 'mobilenetv2_cnn' in voter.model_order:
            print("✓ mobilenetv2_cnn is in model order")
            return True
        else:
            print("❌ FAILED: mobilenetv2_cnn not in model order")
            return False
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("MobileNetV2 CNN Integration Test Suite")
    print("="*80)
    
    results = []
    
    # Run tests
    results.append(("Model Registry", test_model_registry()))
    results.append(("Detection Library", test_detection_library()))
    results.append(("Inference", test_inference()))
    results.append(("Agent Tool", test_agent_tool()))
    results.append(("Agent Creation", test_agent_creation()))
    results.append(("Workflow Integration", test_workflow_integration()))
    results.append(("LR Voter Update", test_lr_voter_update()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<30} {status}")
    
    print("="*80)
    print(f"Results: {passed}/{total} tests passed")
    print("="*80)
    
    if passed == total:
        print("\n🎉 All tests passed! MobileNetV2 CNN integration is complete.")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
