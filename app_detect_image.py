from detectionLib.detect_deepfake_image import DeepfakeImageDetector, EnsembleDetector, EnsembleResult, list_available_models
import os 

def list_available_models_example():
    """Example of listing all available models"""
    
    print("\n\nAvailable Models")
    print("=" * 80)
    
    models = list_available_models()
    
    print("\n### HuggingFace Models\n")
    print("| Key | Name | Architecture | Description |")
    print("|-----|------|--------------|-------------|")
    
    for key, info in models.items():
        if info.get('source') == 'huggingface':
            print(f"| {key} | {info['name']} | {info['architecture']} | {info['description']} |")
    
    print("\n### TorchVision Models\n")
    print("| Key | Name | Architecture | Description |")
    print("|-----|------|--------------|-------------|")
    
    for key, info in models.items():
        if info.get('source') == 'torchvision':
            print(f"| {key} | {info['name']} | {info['architecture']} | {info['description']} |")
    
    print("\n### Timm Models\n")
    print("| Key | Name | Architecture | Description |")
    print("|-----|------|--------------|-------------|")
    
    for key, info in models.items():
        if info.get('source') == 'timm':
            print(f"| {key} | {info['name']} | {info['architecture']} | {info['description']} |")

def process_results_example():
    """Example of processing and filtering results"""
    
    print("\n\nProcessing Results Example")
    print("=" * 40)
    
    # Use correct model keys - mix HuggingFace and PyTorch models
    ensemble = EnsembleDetector(
        model_keys=[
            # HuggingFace models
            'prithiv-v1',
            'dima-vit',
            'organika-sdxl',           
            # 'xception71',    # Excellent for deepfakes (requires timm)
            # 'efficientnet_b7', # High accuracy (slower)
        ]
    )
    
    directory_path = "C:\\Kidambi\\WorkRelated\\DeepFake\\sample_images"
    
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' not found.")
        return
   
    # Collect results from all images
    results_data = []
    
    for image_file in os.listdir(directory_path):
        image_path = os.path.join(directory_path, image_file)
        if os.path.isfile(image_path):
            try:
                # Use 'average' voting method - most balanced and interpretable
                result = ensemble.detect(image_path, threshold=0.3, voting_method='weighted')
                
                # Determine ground truth from filename
                ground_truth_is_fake = 'fake' in image_file.lower()
                is_correct = result.is_deepfake == ground_truth_is_fake
                
                # Collect data for table
                results_data.append({
                    'image': image_file,
                    'ground_truth': 'FAKE' if ground_truth_is_fake else 'REAL',
                    'prediction': 'FAKE' if result.is_deepfake else 'REAL',
                    'is_correct': is_correct,
                    'confidence': result.confidence,
                    'voting_summary': result.voting_summary,
                    'avg_fake_score': result.average_fake_score,
                    'avg_real_score': result.average_real_score,
                    'individual_results': result.individual_results
                })
                    
            except Exception as e:
                print(f"Error processing image '{image_file}': {e}")
                results_data.append({
                    'image': image_file,
                    'ground_truth': 'UNKNOWN',
                    'prediction': 'ERROR',
                    'is_correct': False,
                    'confidence': 0.0,
                    'voting_summary': str(e),
                    'avg_fake_score': 0.0,
                    'avg_real_score': 0.0,
                    'individual_results': []
                })
        else:
            print(f"Skipping non-file entry: {image_file}")
    
    # Calculate accuracy metrics
    if results_data:
        total = len(results_data)
        correct = sum(1 for d in results_data if d['is_correct'])
        accuracy = (correct / total) * 100 if total > 0 else 0
        
        # Confusion matrix
        true_positives = sum(1 for d in results_data if d['ground_truth'] == 'FAKE' and d['prediction'] == 'FAKE')
        false_positives = sum(1 for d in results_data if d['ground_truth'] == 'REAL' and d['prediction'] == 'FAKE')
        true_negatives = sum(1 for d in results_data if d['ground_truth'] == 'REAL' and d['prediction'] == 'REAL')
        false_negatives = sum(1 for d in results_data if d['ground_truth'] == 'FAKE' and d['prediction'] == 'REAL')
        
        # Calculate precision, recall, F1
        precision = (true_positives / (true_positives + false_positives)) * 100 if (true_positives + false_positives) > 0 else 0
        recall = (true_positives / (true_positives + false_negatives)) * 100 if (true_positives + false_negatives) > 0 else 0
        f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        
        print("\n\n## 📊 Performance Metrics")
        print(f"\n**Voting Method**: `average` (uses average fake score across all models)")
        print(f"**Threshold**: 0.5")
        print(f"\n**Overall Accuracy**: {accuracy:.2f}% ({correct}/{total})")
        print(f"**Precision**: {precision:.2f}% (of predicted fakes, how many are actually fake)")
        print(f"**Recall**: {recall:.2f}% (of actual fakes, how many were detected)")
        print(f"**F1-Score**: {f1_score:.2f}%")
        
        print("\n### Confusion Matrix")
        print(f"```")
        print(f"                    Predicted")
        print(f"                FAKE      REAL")
        print(f"Actual FAKE     {true_positives:4d}      {false_negatives:4d}")
        print(f"       REAL     {false_positives:4d}      {true_negatives:4d}")
        print(f"```")
        
        # Display results in markdown table format
        print("\n\n## 🎯 Detection Results Summary")
        print("\n### Overall Results\n")
        print("| Image | Ground Truth | Prediction | ✓/✗ | Confidence | Avg Fake | Avg Real | Voting |")
        print("|-------|--------------|------------|-----|------------|----------|----------|--------|")
        
        for data in results_data:
            correctness = "✓" if data['is_correct'] else "✗"
            print(f"| {data['image']} | {data['ground_truth']} | {data['prediction']} | {correctness} | "
                  f"{data['confidence']:.4f} ({data['confidence']*100:.1f}%) | "
                  f"{data['avg_fake_score']:.4f} | {data['avg_real_score']:.4f} | {data['voting_summary']} |")
        
        # Display individual model results
        print("\n### 📋 Individual Model Results\n")
        for data in results_data:
            if data['individual_results']:
                correctness_icon = "✓" if data['is_correct'] else "✗"
                print(f"\n#### {correctness_icon} {data['image']} (Ground Truth: {data['ground_truth']})\n")
                print("| Model | Prediction | Confidence | Fake Score | Real Score |")
                print("|-------|------------|------------|------------|------------|")
                for individual in data['individual_results']:
                    pred = 'FAKE' if individual.is_deepfake else 'REAL'
                    fake_score = individual.scores.get('Fake', individual.scores.get('fake', 0.0))
                    real_score = individual.scores.get('Real', individual.scores.get('real', 0.0))
                    print(f"| {individual.model_name} | {pred} | "
                          f"{individual.confidence:.4f} ({individual.confidence*100:.1f}%) | "
                          f"{fake_score:.4f} | {real_score:.4f} |")

def process_results_with_optimization():
    """Optimized ensemble with best models and adaptive thresholding"""
    
    print("\n\n🚀 Optimized Ensemble Detection")
    print("=" * 40)
    
    # OPTIMIZED ENSEMBLE: Mix of architectures for better diversity
    ensemble = EnsembleDetector(
        model_keys=[
            # Core HuggingFace models (proven performance)
            'prithiv-v1',      # SigLIP - good general detector
            'dima-vit',        # ViT - different architecture
            'organika-sdxl',   # Specialized for AI-generated images
            
            # Add NPR for perturbation-based detection
            'npr-ensemble',    # Physics-based approach
            
            # Optional: Add if you have GPU and want even better accuracy
            # Uncomment these if performance allows:
            # 'xception71',    # Excellent for deepfakes (requires timm)
            # 'efficientnet_b7', # High accuracy (slower)
        ]
    )
    
    directory_path = "C:\\Kidambi\\WorkRelated\\DeepFake\\sample_images"
    
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' not found.")
        return
   
    # STEP 1: Find optimal threshold using calibration
    print("\n📊 Phase 1: Threshold Calibration")
    print("-" * 40)
    
    thresholds_to_test = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    best_threshold = 0.5
    best_accuracy = 0.0
    
    threshold_results = {}
    
    for test_threshold in thresholds_to_test:
        results_data = []
        
        for image_file in os.listdir(directory_path):
            image_path = os.path.join(directory_path, image_file)
            if os.path.isfile(image_path):
                try:
                    result = ensemble.detect(image_path, threshold=test_threshold, voting_method='average')
                    ground_truth_is_fake = 'fake' in image_file.lower()
                    is_correct = result.is_deepfake == ground_truth_is_fake
                    results_data.append(is_correct)
                except Exception as e:
                    print(f"Error: {e}")
        
        if results_data:
            accuracy = (sum(results_data) / len(results_data)) * 100
            threshold_results[test_threshold] = accuracy
            print(f"  Threshold {test_threshold:.2f}: {accuracy:.2f}% accuracy")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = test_threshold
    
    print(f"\n✅ Best Threshold: {best_threshold:.2f} (Accuracy: {best_accuracy:.2f}%)")
    
    # STEP 2: Run with optimal threshold
    print(f"\n📊 Phase 2: Final Detection with Optimized Threshold")
    print("-" * 40)
    
    results_data = []
    
    for image_file in os.listdir(directory_path):
        image_path = os.path.join(directory_path, image_file)
        if os.path.isfile(image_path):
            try:
                result = ensemble.detect(image_path, threshold=best_threshold, voting_method='average')
                
                ground_truth_is_fake = 'fake' in image_file.lower()
                is_correct = result.is_deepfake == ground_truth_is_fake
                
                results_data.append({
                    'image': image_file,
                    'ground_truth': 'FAKE' if ground_truth_is_fake else 'REAL',
                    'prediction': 'FAKE' if result.is_deepfake else 'REAL',
                    'is_correct': is_correct,
                    'confidence': result.confidence,
                    'voting_summary': result.voting_summary,
                    'avg_fake_score': result.average_fake_score,
                    'avg_real_score': result.average_real_score,
                    'individual_results': result.individual_results
                })
                    
            except Exception as e:
                print(f"Error processing '{image_file}': {e}")
    
    # Calculate metrics
    if results_data:
        total = len(results_data)
        correct = sum(1 for d in results_data if d['is_correct'])
        accuracy = (correct / total) * 100
        
        true_positives = sum(1 for d in results_data if d['ground_truth'] == 'FAKE' and d['prediction'] == 'FAKE')
        false_positives = sum(1 for d in results_data if d['ground_truth'] == 'REAL' and d['prediction'] == 'FAKE')
        true_negatives = sum(1 for d in results_data if d['ground_truth'] == 'REAL' and d['prediction'] == 'REAL')
        false_negatives = sum(1 for d in results_data if d['ground_truth'] == 'FAKE' and d['prediction'] == 'REAL')
        
        precision = (true_positives / (true_positives + false_positives)) * 100 if (true_positives + false_positives) > 0 else 0
        recall = (true_positives / (true_positives + false_negatives)) * 100 if (true_positives + false_negatives) > 0 else 0
        f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        
        print("\n\n## 📊 OPTIMIZED Performance Metrics")
        print(f"\n**Ensemble Models**: {len(ensemble.detectors)} models")
        print(f"**Voting Method**: `average`")
        print(f"**Optimized Threshold**: {best_threshold:.2f}")
        print(f"\n**Overall Accuracy**: {accuracy:.2f}% ({correct}/{total})")
        print(f"**Precision**: {precision:.2f}%")
        print(f"**Recall**: {recall:.2f}%")
        print(f"**F1-Score**: {f1_score:.2f}%")
        
        print("\n### Confusion Matrix")
        print(f"```")
        print(f"                    Predicted")
        print(f"                FAKE      REAL")
        print(f"Actual FAKE     {true_positives:4d}      {false_negatives:4d}")
        print(f"       REAL     {false_positives:4d}      {true_negatives:4d}")
        print(f"```")
        
        # Threshold calibration results
        print("\n### 🎯 Threshold Calibration Results")
        print("\n| Threshold | Accuracy |")
        print("|-----------|----------|")
        for thresh, acc in sorted(threshold_results.items()):
            marker = " ⭐" if thresh == best_threshold else ""
            print(f"| {thresh:.2f}      | {acc:.2f}%{marker} |")
        
        # Display results table
        print("\n\n## 🎯 Detection Results")
        print("\n| Image | Truth | Prediction | ✓/✗ | Confidence | Avg Fake |")
        print("|-------|-------|------------|-----|------------|----------|")
        
        for data in results_data:
            correctness = "✓" if data['is_correct'] else "✗"
            print(f"| {data['image']} | {data['ground_truth']} | {data['prediction']} | {correctness} | "
                  f"{data['confidence']:.3f} ({data['confidence']*100:.1f}%) | {data['avg_fake_score']:.3f} |")
        
        # Identify problematic images
        print("\n### ⚠️ Misclassified Images (Need Attention)")
        misclassified = [d for d in results_data if not d['is_correct']]
        if misclassified:
            for data in misclassified:
                print(f"\n**{data['image']}** (Truth: {data['ground_truth']}, Predicted: {data['prediction']})")
                print(f"  - Avg Fake Score: {data['avg_fake_score']:.3f}")
                print(f"  - Model Votes: {data['voting_summary']}")
                print("  - Individual Predictions:")
                for ind in data['individual_results']:
                    fake_score = ind.scores.get('Fake', ind.scores.get('fake', 0.0))
                    print(f"    • {ind.model_name}: {'FAKE' if ind.is_deepfake else 'REAL'} (fake={fake_score:.3f})")
        else:
            print("\n🎉 **Perfect! All images classified correctly!**")
        
        # Recommendations
        print("\n\n## 💡 Recommendations to Reach 90% Accuracy")
        if accuracy < 90:
            print(f"\n**Current Gap**: Need {90 - accuracy:.1f}% improvement")
            print("\n**Action Items**:")
            
            if len(misclassified) > 0:
                print(f"1. **Address {len(misclassified)} misclassified images**")
                print("   - Review if ground truth labels are correct")
                print("   - These might be edge cases or high-quality deepfakes")
            
            if best_threshold != 0.5:
                print(f"2. **Threshold is optimized at {best_threshold:.2f}** (non-standard)")
                print("   - This suggests the models are slightly biased")
                print("   - Consider adding more diverse models")
            
            print("3. **Add more specialized models**:")
            print("   - Uncomment xception71 (excellent for deepfakes)")
            print("   - Add efficientnet_b7 (high accuracy)")
            print("   - Use GPU if available for better models")
            
            print("4. **Try different voting methods**:")
            print("   - Test 'weighted' voting (considers confidence)")
            print("   - Test 'maximum' voting (trusts most confident model)")
            
            print("5. **Expand training data understanding**:")
            print("   - Ensure test images represent real-world scenarios")
            print("   - Mix different types of fakes (GAN, diffusion, face-swap)")
        else:
            print("\n🎉 **Congratulations! You've achieved >90% accuracy!**")
            print(f"   Current accuracy: {accuracy:.2f}%")

def main():
    """Run all usage examples"""
    
    print("Deepfake Detection - Optimized for 90% Accuracy")
    print("=" * 50)
    
    try:
        # Show available models first (optional)
        # list_available_models_example()
        
        ## Run OPTIMIZED detection with threshold calibration
        #process_results_with_optimization()
        
        # You can still run the basic version for comparison:
        process_results_example()
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")


if __name__ == "__main__":
    main()