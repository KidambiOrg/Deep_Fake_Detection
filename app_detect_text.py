"""
Simple example demonstrating how to use the DetectGPT implementation
for detecting AI-generated text.
"""

from time import time
from lib.detect_gpt import DetectGPT, DetectionResult


def interactive_mode():
    """Interactive mode for testing custom text."""
    print("\n🎯 Interactive Mode")
    print("Enter your own text to analyze (or 'quit' to exit):")
    print("💡 Note: Using conservative threshold (0.8) to avoid misclassifying formal human text")
    
    detector = None
    
    while True:
        try:
            user_text = input("\n📝 Enter text: ").strip()
            
            if user_text.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if not user_text:
                print("Please enter some text to analyze.")
                continue
            
            # Initialize detector if not already done
            if detector is None:
                print("Loading models...")
                detector = DetectGPT(
                    model_name="gpt2-xl",
                    mask_model_name="t5-large",
                    random_seed=42  # Fixed seed for consistent results
                )
                print("✅ Ready! (Using improved threshold and fixed seed for better results)\n")
            
            
            # start timer
            start_time = time()
            # Analyze the text with conservative threshold
            result = detector.detect(user_text, 
                                     threshold=0.8,  # More conservative threshold to reduce false positives
                                     num_perturbations=50,  # More perturbations for stability
                                     early_stopping=True)  # Fast analysis
            
            # End timer
            end_time = time()
            elapsed_time = end_time - start_time
            
            prediction = "🤖 AI-generated" if result.is_ai_generated else "👤 Human-written"
            
            # Interpret the score with conservative thresholds
            if result.criterion_score > 1.5:
                confidence_desc = "Very high confidence - AI"
            elif result.criterion_score > 1.2:
                confidence_desc = "High confidence - AI"
            elif result.criterion_score > 0.8:
                confidence_desc = "Moderate confidence - AI"
            elif result.criterion_score > 0.3:
                confidence_desc = "Possible AI (but could be formal human text)"
            elif result.criterion_score > -0.3:
                confidence_desc = "Uncertain (mixed characteristics)"
            elif result.criterion_score > -0.8:
                confidence_desc = "Moderate confidence - Human"
            elif result.criterion_score > -1.5:
                confidence_desc = "High confidence - Human"
            else:
                confidence_desc = "Very high confidence - Human"
            
            print(f"\n📊 Analysis Result:")
            print(f"   Prediction: {prediction}")
            print(f"   Interpretation: {confidence_desc}")
            print(f"   DetectGPT Score: {result.criterion_score:+.4f} (>0.8=AI, <0.8=Human)")
            print(f"   Confidence: {result.confidence:.3f}")
            print(f"   Threshold Used: 0.8 (conservative - reduces false positives)")
            print(f"   Analysis Time: {elapsed_time:.2f} seconds")
            
            # Additional notes
            if 0.3 < result.criterion_score <= 0.8:
                print(f"   💭 Note: Score between 0.3-0.8 - could be formal human text or AI")
            elif abs(result.criterion_score - 0.8) < 0.2:
                print(f"   💭 Note: Score is near threshold - characteristics are mixed")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🔬 DetectGPT: AI-Generated Text Detection")
    print("=" * 45)
    print("🎯 Now using conservative threshold (0.8) for better accuracy!")
    print("   Issue identified: Formal/polite human text was being misclassified as AI")
    print("   Solution: More conservative threshold to reduce false positives")
    print("   Note: Formal business language can score higher due to structure")
    
    # Optionally run interactive mode
    try:
        interactive_mode()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")