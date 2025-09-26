"""
Enhanced DetectGPT with High-Accuracy Model Configurations

This script demonstrates different model combinations for better accuracy.
Models are ranked from fastest to most accurate.
"""

from lib.detect_gpt import DetectGPT, DetectionResult
import time

def test_model_configurations():
    """Test different model configurations for accuracy comparison."""
    
    # Sample text for testing
    test_text = """
    Artificial intelligence has emerged as a transformative technology that is reshaping 
    industries and society. Machine learning algorithms can now process vast amounts of 
    data to identify patterns and make predictions with remarkable accuracy. Furthermore, 
    natural language processing capabilities enable computers to understand and generate 
    human-like text, facilitating more intuitive human-computer interactions.
    """
    
    # Model configurations from fastest to most accurate
    configurations = {
        "Fast (Current)": {
            "model_name": "gpt2-medium",
            "mask_model_name": "t5-small",
            "description": "Good balance of speed and accuracy"
        },
        
        "Better Accuracy": {
            "model_name": "gpt2-large", 
            "mask_model_name": "t5-base",
            "description": "Improved accuracy with larger models"
        },
        
        "High Accuracy": {
            "model_name": "gpt2-xl",
            "mask_model_name": "t5-large", 
            "description": "Best accuracy but slower (requires GPU recommended)"
        },
        
        "Alternative - Distil": {
            "model_name": "distilgpt2",
            "mask_model_name": "t5-small",
            "description": "Faster inference, good for quick testing"
        },
        
        "Alternative - Modern": {
            "model_name": "microsoft/DialoGPT-medium",
            "mask_model_name": "t5-base",
            "description": "More conversational, good for chat-like text"
        }
    }
    
    print("🔬 Testing Different Model Configurations")
    print("=" * 60)
    
    results = {}
    
    for config_name, config in configurations.items():
        print(f"\n📊 Testing: {config_name}")
        print(f"Description: {config['description']}")
        print(f"Models: {config['model_name']} + {config['mask_model_name']}")
        
        try:
            start_time = time.time()
            
            # Initialize detector with specific configuration
            detector = DetectGPT(
                model_name=config['model_name'],
                mask_model_name=config['mask_model_name']
            )
            
            # Run detection with fewer perturbations for speed
            result = detector.detect(test_text.strip(), num_perturbations=5)
            
            load_time = time.time() - start_time
            
            # Store results
            results[config_name] = {
                'result': result,
                'load_time': load_time,
                'config': config
            }
            
            # Display results
            prediction = "🤖 AI-generated" if result.is_ai_generated else "👤 Human-written"
            print(f"   Result: {prediction}")
            print(f"   Confidence: {result.confidence:.3f}")
            print(f"   Score: {result.criterion_score:+.4f}")
            print(f"   Load Time: {load_time:.1f}s")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            print(f"   (Model may not be available or requires more memory)")
    
    return results

def recommend_best_models():
    """Provide recommendations for different use cases."""
    
    recommendations = {
        "🚀 For Speed (Development/Testing)": {
            "scoring_model": "distilgpt2",
            "mask_model": "t5-small",
            "pros": ["Fastest loading", "Low memory usage", "Good for prototyping"],
            "cons": ["Lower accuracy", "Less robust"],
            "use_case": "Quick testing, development, resource-constrained environments"
        },
        
        "⚖️ For Balanced Performance (Recommended)": {
            "scoring_model": "gpt2-large", 
            "mask_model": "t5-base",
            "pros": ["Good accuracy", "Reasonable speed", "Better perturbations"],
            "cons": ["Higher memory usage", "Slower than small models"],
            "use_case": "Production use, good balance of accuracy and speed"
        },
        
        "🎯 For Maximum Accuracy (Research)": {
            "scoring_model": "gpt2-xl",
            "mask_model": "t5-large",
            "pros": ["Highest accuracy", "Best perturbation quality", "Most robust"],
            "cons": ["Very slow", "High memory usage", "May need GPU"],
            "use_case": "Research, final validation, when accuracy is critical"
        },
        
        "💬 For Conversational Text": {
            "scoring_model": "microsoft/DialoGPT-medium",
            "mask_model": "t5-base", 
            "pros": ["Better for chat/dialogue", "Good conversational understanding"],
            "cons": ["Specialized for conversations", "May not work well for formal text"],
            "use_case": "Detecting AI in chat messages, social media, forums"
        },
        
        "📚 For Academic/Formal Text": {
            "scoring_model": "gpt2-large",
            "mask_model": "google/flan-t5-base",
            "pros": ["Better instruction following", "Good for formal text", "Improved reasoning"],
            "cons": ["Slightly different API", "May need code adjustments"],
            "use_case": "Academic papers, formal documents, structured text"
        }
    }
    
    print("\n🎯 Model Recommendations by Use Case")
    print("=" * 60)
    
    for category, info in recommendations.items():
        print(f"\n{category}")
        print(f"Models: {info['scoring_model']} + {info['mask_model']}")
        print(f"Use Case: {info['use_case']}")
        print("Pros:", " | ".join(info['pros']))
        print("Cons:", " | ".join(info['cons']))

def create_enhanced_detector_example():
    """Example of creating an enhanced detector with better models."""
    
    print("\n🔧 Creating Enhanced DetectGPT Configuration")
    print("=" * 50)
    
    # Configuration for better accuracy
    enhanced_config = {
        "model_name": "gpt2-large",           # Larger scoring model
        "mask_model_name": "t5-base",         # Better perturbation model
        "device": "cpu"                       # Use "cuda" if you have GPU
    }
    
    print("Enhanced Configuration:")
    for key, value in enhanced_config.items():
        print(f"  {key}: {value}")
    
    print("\nTo use this configuration:")
    print("""
# Create enhanced detector
detector = DetectGPT(
    model_name="gpt2-large",
    mask_model_name="t5-base"
)

# For even better accuracy, use more perturbations
result = detector.detect(
    text, 
    num_perturbations=50,  # More perturbations = better accuracy
    threshold=0.1          # Fine-tune threshold if needed
)
""")

def advanced_tips():
    """Advanced tips for improving DetectGPT accuracy."""
    
    print("\n💡 Advanced Tips for Better Accuracy")
    print("=" * 45)
    
    tips = [
        {
            "tip": "Use More Perturbations",
            "details": "Increase num_perturbations from 10 to 50-100 for better estimates",
            "code": "result = detector.detect(text, num_perturbations=100)"
        },
        {
            "tip": "Adjust Mask Ratio", 
            "details": "Try different mask_ratio values (0.1-0.3) for different text types",
            "code": "perturbations = detector.generate_perturbations(text, mask_ratio=0.2)"
        },
        {
            "tip": "Fine-tune Threshold",
            "details": "Adjust classification threshold based on your specific use case",
            "code": "# Lower threshold = more sensitive to AI text\nresult.is_ai_generated = result.criterion_score > 0.05"
        },
        {
            "tip": "Ensemble Methods",
            "details": "Use multiple model combinations and average their scores",
            "code": "# Run with different models and average the criterion scores"
        },
        {
            "tip": "Text Preprocessing",
            "details": "Clean and normalize text before detection for consistency",
            "code": "text = text.strip().replace('\\n', ' ')  # Remove extra whitespace"
        }
    ]
    
    for i, tip in enumerate(tips, 1):
        print(f"\n{i}. {tip['tip']}")
        print(f"   {tip['details']}")
        print(f"   Example: {tip['code']}")

if __name__ == "__main__":
    print("🎯 Enhanced DetectGPT: Better Models for Higher Accuracy")
    print("=" * 65)
    
    # Show recommendations first
    recommend_best_models()
    
    # Show configuration example
    create_enhanced_detector_example()
    
    # Advanced tips
    advanced_tips()
    
    print("\n" + "=" * 65)
    print("🚀 Ready to test different configurations?")
    
    try:
        response = input("Test model configurations? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            test_model_configurations()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")