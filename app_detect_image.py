from detectionLib.detect_deepfake_image import DeepfakeImageDetector, EnsembleDetector, EnsembleResult
import os 

def process_results_example():
    """Example of processing and filtering results"""
    
    print("\n\nProcessing Results Example")
    print("=" * 40)
    
    ensemble = EnsembleDetector(model_keys=['prithiv-v1', 'dima-vit','organika-sdxl','efficientnet_b4','xception','efficientnet_b4','imagenet','resnet50','resnet101'])
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
                result = ensemble.detect(image_path, threshold=0.5, voting_method='weighted')
                
                # Collect data for table
                results_data.append({
                    'image': image_file,
                    'prediction': 'DEEPFAKE' if result.is_deepfake else 'REAL',
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
                    'prediction': 'ERROR',
                    'confidence': 0.0,
                    'voting_summary': str(e),
                    'avg_fake_score': 0.0,
                    'avg_real_score': 0.0,
                    'individual_results': []
                })
        else:
            print(f"Skipping non-file entry: {image_file}")
    
    # Display results in markdown table format
    if results_data:
        print("\n\n## Detection Results Summary")
        print("\n### Overall Results\n")
        print("| Image | Prediction | Confidence | Voting Summary | Avg Fake Score | Avg Real Score |")
        print("|-------|------------|------------|----------------|----------------|----------------|")
        
        for data in results_data:
            print(f"| {data['image']} | {data['prediction']} | {data['confidence']:.4f} ({data['confidence']*100:.2f}%) | {data['voting_summary']} | {data['avg_fake_score']:.4f} | {data['avg_real_score']:.4f} |")
        
        # Display individual model results
        print("\n### Individual Model Results\n")
        for data in results_data:
            if data['individual_results']:
                print(f"\n#### {data['image']}\n")
                print("| Model | Prediction | Confidence |")
                print("|-------|------------|------------|")
                for individual in data['individual_results']:
                    pred = 'DEEPFAKE' if individual.is_deepfake else 'REAL'
                    print(f"| {individual.model_name} | {pred} | {individual.confidence:.4f} ({individual.confidence*100:.2f}%) |")
    
def main():
    """Run all usage examples"""
    
    print("analyze_directory Method - Usage Examples")
    print("=" * 50)
    
    try:
        # basic_usage_example()
        #advanced_usage_example() 
        process_results_example()       
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")


if __name__ == "__main__":
    main()