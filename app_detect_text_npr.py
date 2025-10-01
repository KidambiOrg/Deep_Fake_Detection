"""
Test Script for DetectLLM-NPR Implementation

This script tests the detect_ai_text_npr.py implementation with sample texts
to verify that the NPR (Normalized Perturbation Rank) algorithm works correctly.
"""

import sys
import os
import time

# Add the current directory to the path so we can import our module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from detectionLib.detect_deepfake_text_npr import DetectLLMNPR, NPRDetectionResult
    print("✅ Successfully imported DetectLLM-NPR module")
except ImportError as e:
    print(f"❌ Failed to import DetectLLM-NPR module: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)


def test_model_recommendations():
    """Test the model recommendation system."""
    print("\n🎯 Testing Model Recommendations")
    print("-" * 40)
    
    try:
        recommendations = DetectLLMNPR.get_model_recommendations()
        
        for use_case, config in recommendations.items():
            print(f"{use_case.upper()}:")
            print(f"  Scoring Model: {config['scoring_model']}")
            print(f"  Perturbation Model: {config['perturbation_model']}")
            print(f"  Description: {config['description']}")
            print(f"  Accuracy: {config['accuracy']}")
            print(f"  Speed: {config['speed']}")
            print()
        
        print("✅ Model recommendations test completed")
        return True
        
    except Exception as e:
        print(f"❌ Model recommendations test failed: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality with small models."""
    print("\n🔧 Testing Basic Functionality")
    print("-" * 40)
    
    # Use smaller models for faster testing
    print("Initializing DetectLLM-NPR with small models for testing...")
    
    try:
        detector = DetectLLMNPR(
            scoring_model="gpt2",  # Smaller model for faster testing
            perturbation_model="t5-small",  # Smaller model for faster testing
            random_seed=42  # For reproducible results
        )
        print("✅ Detector initialized successfully")
        
        # Test with a simple text
        test_text = "The quick brown fox jumps over the lazy dog. This is a simple test sentence."
        
        print(f"Testing with text: '{test_text[:50]}...'")
        
        start_time = time.time()
        result = detector.detect(
            test_text,
            threshold=0.1,
            num_perturbations=5,  # Very few for quick testing
            perturbation_strategy="random"
        )
        end_time = time.time()
        
        print("✅ Detection completed successfully")
        print(f"   Result: {'AI-generated' if result.is_ai_generated else 'Human-written'}")
        print(f"   NPR Score: {result.npr_score:.4f}")
        print(f"   Confidence: {result.confidence:.4f}")
        print(f"   Processing Time: {end_time - start_time:.2f}s")
        print(f"   Perturbations Used: {result.num_perturbations_used}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def test_perturbation_generation():
    """Test perturbation generation without full detection."""
    print("\n🔄 Testing Perturbation Generation")
    print("-" * 40)
    
    try:
        detector = DetectLLMNPR(
            scoring_model="gpt2",
            perturbation_model="t5-small",
            random_seed=42
        )
        
        test_text = "Machine learning algorithms can process large datasets efficiently."
        print(f"Original text: {test_text}")
        
        # Test different perturbation strategies
        strategies = ["random", "mixed"]
        
        for strategy in strategies:
            print(f"\nTesting {strategy} perturbation strategy:")
            
            perturbations = detector.generate_perturbations(
                test_text,
                num_perturbations=3,
                perturbation_strategy=strategy
            )
            
            for i, perturbed in enumerate(perturbations[:3], 1):
                print(f"  {i}. {perturbed}")
        
        print("✅ Perturbation generation test completed")
        return True
        
    except Exception as e:
        print(f"❌ Perturbation generation test failed: {e}")
        return False


def test_log_probability_computation():
    """Test log probability computation."""
    print("\n📊 Testing Log Probability Computation")
    print("-" * 40)
    
    try:
        detector = DetectLLMNPR(
            scoring_model="gpt2",
            perturbation_model="t5-small"
        )
        
        test_texts = [
            "This is a simple sentence.",
            "Artificial intelligence is transforming society.",
            "The cat sat on the mat."
        ]
        
        print("Computing log probabilities:")
        for text in test_texts:
            log_prob = detector.compute_log_probability(text)
            print(f"  '{text}' -> {log_prob:.4f}")
        
        print("✅ Log probability computation test completed")
        return True
        
    except Exception as e:
        print(f"❌ Log probability computation test failed: {e}")
        return False


def interactive_test():
    """Interactive testing mode."""
    print("\n🎮 Interactive Testing Mode")
    print("-" * 40)
    print("Enter your own text to test the NPR detector!")
    print("(Type 'quit' to exit)")
    
    try:
        detector = DetectLLMNPR(
            scoring_model="gpt2-xl",
            perturbation_model="t5-3b",
            random_seed=42
        )
        
        while True:
            user_text = input("\n📝 Enter text to analyze: ").strip()
            
            if user_text.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_text:
                print("Please enter some text to analyze.")
                continue
            
            print("Analyzing...")
            result = detector.detect(
                user_text,
                threshold=0.1,
                num_perturbations=20,
                perturbation_strategy="mixed"
            )
            
            prediction = "🤖 AI-generated" if result.is_ai_generated else "👤 Human-written"
            print(f"\n📊 Analysis Result:")
            print(f"   Prediction: {prediction}")
            print(f"   NPR Score: {result.npr_score:+.4f}")
            print(f"   Confidence: {result.confidence:.4f}")
            print(f"   Processing Time: {result.processing_time:.2f}s")
            
            # Interpretation
            if result.npr_score > 0.2:
                interpretation = "Strong AI indication"
            elif result.npr_score > 0.1:
                interpretation = "Moderate AI indication"
            elif result.npr_score > 0.0:
                interpretation = "Weak AI indication"
            else:
                interpretation = "Human-like characteristics"
            
            print(f"   Interpretation: {interpretation}")
    
    except Exception as e:
        print(f"❌ Interactive test failed: {e}")


def main():
    """Run all tests."""
    print("🧪 DetectLLM-NPR Test Suite")
    print("=" * 50)
    
    # Run tests
    tests = [
        ("Model Recommendations", test_model_recommendations),
        ("Log Probability Computation", test_log_probability_computation),
        ("Perturbation Generation", test_perturbation_generation),
        ("Basic Functionality", test_basic_functionality),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔬 Running {test_name} Test...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} test PASSED")
            else:
                print(f"❌ {test_name} test FAILED")
        except Exception as e:
            print(f"❌ {test_name} test ERROR: {e}")
    
    print(f"\n📋 Test Summary")
    print("=" * 30)
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The DetectLLM-NPR implementation is working correctly.")
        
        # Offer interactive testing
        try:
            response = input("\nWould you like to try interactive testing? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                interactive_test()
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        print("\nPossible issues:")
        print("1. Missing dependencies - run: pip install -r requirements.txt")
        print("2. Insufficient GPU memory - the models will fallback to CPU")
        print("3. Network issues downloading models")


if __name__ == "__main__":
    try:
        # main()
        interactive_test()
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user. Goodbye!")
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()