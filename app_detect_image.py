from detectionLib.detect_deepfake_image import DeepfakeImageDetector, EnsembleDetector, EnsembleResult, list_available_models
import os 
import sys


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


def create_ensemble_detector():
    """Create and return an ensemble detector with configured models"""
    return EnsembleDetector(
        model_keys=[
            # HuggingFace models
            'prithiv-v1',
            'dima-vit',
            'organika-sdxl',           
            'nonescape-full',            
            'universal-fake-v2',
            #'xception71',    # Excellent for deepfakes (requires timm)
            # 'efficientnet_b7', # High accuracy (slower)
        ]
    )


def process_image_data(image_path, ensemble, threshold=0.26, voting_method='weighted'):
    """Process a single image and return result data
    
    Args:
        image_path: Path to the image file
        ensemble: EnsembleDetector instance
        threshold: Detection threshold
        voting_method: Voting method for ensemble
        
    Returns:
        Dictionary containing detection results or None if error
    """
    try:
        result = ensemble.detect(image_path, threshold=threshold, voting_method=voting_method)
        
        # Determine ground truth from filename
        image_file = os.path.basename(image_path)
        ground_truth_is_fake = 'fake' in image_file.lower()
        is_correct = result.is_deepfake == ground_truth_is_fake
        
        return {
            'image': image_file,
            'ground_truth': 'FAKE' if ground_truth_is_fake else 'REAL',
            'prediction': 'FAKE' if result.is_deepfake else 'REAL',
            'is_correct': is_correct,
            'confidence': result.confidence,
            'voting_summary': result.voting_summary,
            'avg_fake_score': result.average_fake_score,
            'avg_real_score': result.average_real_score,
            'individual_results': result.individual_results
        }
    except Exception as e:
        image_file = os.path.basename(image_path)
        print(f"Error processing image '{image_file}': {e}")
        return {
            'image': image_file,
            'ground_truth': 'UNKNOWN',
            'prediction': 'ERROR',
            'is_correct': False,
            'confidence': 0.0,
            'voting_summary': str(e),
            'avg_fake_score': 0.0,
            'avg_real_score': 0.0,
            'individual_results': []
        }


def display_overall_results_table(results_data):
    """Display overall results in a markdown table format
    
    Args:
        results_data: List of result dictionaries
    """
    print("| Image | Ground Truth | Prediction | ✓/✗ | Confidence | Avg Fake | Avg Real | Voting |")
    print("|-------|--------------|------------|-----|------------|----------|----------|--------|")
    
    for data in results_data:
        correctness = "✓" if data['is_correct'] else "✗"
        print(f"| {data['image']} | {data['ground_truth']} | {data['prediction']} | {correctness} | "
              f"{data['confidence']:.4f} ({data['confidence']*100:.1f}%) | "
              f"{data['avg_fake_score']:.4f} | {data['avg_real_score']:.4f} | {data['voting_summary']} |")


def display_individual_model_results(results_data):
    """Display individual model results for each image
    
    Args:
        results_data: List of result dictionaries
    """
    print("\n### 📋 Individual Model Results\n")
    for data in results_data:
        if data['individual_results']:
            correctness_icon = "✓" if data['is_correct'] else "✗"
            print(f"\n#### {correctness_icon} {data['image']} (Ground Truth: {data['ground_truth']})\n")
            print("| Model | Prediction | Confidence | Fake Score | Real Score |")
            print("|-------|------------|------------|------------|------------|")
            for individual in data['individual_results']:
                pred = 'FAKE' if individual.is_deepfake else 'REAL'
                
                # Extract fake and real scores from the model's scores dictionary
                fake_score = 0.0
                real_score = 0.0
                
                # Check for various fake label formats (artificial, fake, etc.)
                for fake_key in ['artificial', 'Artificial', 'ARTIFICIAL',
                                'Fake', 'fake', 'FAKE', 'deepfake', 'Deepfake', 'DEEPFAKE', 
                                'generated', 'Generated', 'synthetic', 'Synthetic', 
                                'AI', 'ai', 'LABEL_1', '1']:
                    if fake_key in individual.scores:
                        fake_score = individual.scores[fake_key]
                        break
                
                # Check for various real label formats (human, real, authentic, etc.)
                for real_key in ['human', 'Human', 'HUMAN',
                                'Real', 'real', 'REAL', 'authentic', 'Authentic', 'AUTHENTIC',
                                'genuine', 'Genuine', 'LABEL_0', '0']:
                    if real_key in individual.scores:
                        real_score = individual.scores[real_key]
                        break
                
                # If still no scores found, try to infer from available scores
                if fake_score == 0.0 and real_score == 0.0 and individual.scores:
                    # Print debug info to help identify the issue
                    print(f"| {individual.model_name} | DEBUG: Available labels: {list(individual.scores.keys())} |")
                    # Try to use first two scores if available
                    scores_list = list(individual.scores.values())
                    if len(scores_list) >= 2:
                        # Assume first is real, second is fake (common pattern)
                        real_score = scores_list[0]
                        fake_score = scores_list[1]
                
                print(f"| {individual.model_name} | {pred} | "
                      f"{individual.confidence:.4f} ({individual.confidence*100:.1f}%) | "
                      f"{fake_score:.4f} | {real_score:.4f} |")


def calculate_and_display_metrics(results_data, voting_method='weighted', threshold=0.26):
    """Calculate and display performance metrics
    
    Args:
        results_data: List of result dictionaries
        voting_method: Voting method used
        threshold: Threshold value used
    """
    if not results_data:
        return
    
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
    print(f"\n**Voting Method**: `{voting_method}`")
    
    # Explain voting method
    if voting_method == 'weighted' or voting_method == 'average':
        print(f"- Uses **average fake score** across all models (avg fake score >= {threshold} → FAKE)")
        print(f"- More robust for noisy images than simple majority voting")
    elif voting_method == 'majority':
        print(f"- Uses **simple majority** vote (most models agree)")
    elif voting_method == 'confidence_weighted':
        print(f"- Weights fake scores by each model's confidence")
    elif voting_method == 'maximum':
        print(f"- Uses prediction from model with highest confidence")
    elif voting_method == 'unanimous':
        print(f"- All models must agree")
    
    print(f"\n**Threshold**: {threshold}")
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


def process_single_image(image_path):
    """Process a single image and display results"""
    
    print("\n\nProcessing Single Image")
    print("=" * 40)
    
    if not os.path.exists(image_path):
        print(f"Image '{image_path}' not found.")
        return
    
    if not os.path.isfile(image_path):
        print(f"Path '{image_path}' is not a file.")
        return
    
    threshold=0.26
    voting_method='weighted'  # Changed from 'confidence_weighted' to 'weighted'

    # Create ensemble detector
    ensemble = create_ensemble_detector()
    
    # Process the image
    data = process_image_data(image_path, ensemble, threshold=threshold, voting_method=voting_method)
    
    if data:
        # Display results
        print("\n\n## 🎯 Detection Results")
        print("\n### Overall Result\n")
        display_overall_results_table([data])
        display_individual_model_results([data])

def process_images_in_a_directory(directory_path=None):
    """Example of processing and filtering results"""
    
    print("\n\nProcessing Results Example")
    print("=" * 40)
    
    if directory_path is None:
        directory_path = "C:\\Kidambi\\WorkRelated\\DeepFake\\sample_images"
    
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' not found.")
        return
    
    if not os.path.isdir(directory_path):
        print(f"Path '{directory_path}' is not a directory.")
        return
    
    # Create ensemble detector
    ensemble = create_ensemble_detector()
    
    # Collect results from all images
    results_data = []

    threshold=0.26
    voting_method='weighted'  # Changed from 'confidence_weighted' to 'weighted'

    
    for image_file in os.listdir(directory_path):
        image_path = os.path.join(directory_path, image_file)
        if os.path.isfile(image_path):
            # Process each image
            data = process_image_data(image_path, ensemble, threshold=threshold, voting_method=voting_method)
            if data:
                results_data.append(data)
        else:
            print(f"Skipping non-file entry: {image_file}")
    
    # Calculate and display metrics
    if results_data:
        calculate_and_display_metrics(results_data, voting_method=voting_method, threshold=threshold)
        
        # Display results in markdown table format
        print("\n\n## 🎯 Detection Results Summary")
        print("\n### Overall Results\n")
        display_overall_results_table(results_data)
        display_individual_model_results(results_data)

 

def main(path=None):
    """Run deepfake detection on an image or directory
    
    Args:
        path: Path to an image file or directory. If None, uses default directory.
    """
    
    print("Deepfake Detection - Optimized for 90% Accuracy")
    print("=" * 50)
    
    try:
        # Show available models first (optional)
        # list_available_models_example()
        
        if path is None:
            # Use default directory
            process_images_in_a_directory()
        elif os.path.isfile(path):
            # Single image
            print(f"\n📷 Processing single image: {path}")
            process_single_image(path)
        elif os.path.isdir(path):
            print(f"\n📁 Processing directory: {path}")
            process_images_in_a_directory(path)
        else:
            print(f"❌ Error: Path '{path}' does not exist or is not accessible.")
        
    except Exception as e:
        print(f"❌ Error running detection: {e}")


if __name__ == "__main__":
    # Check if argument is provided via command line
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()