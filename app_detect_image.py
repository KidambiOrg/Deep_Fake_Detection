from detectionLib.detect_deepfake_image import DeepfakeImageDetector, EnsembleDetector, EnsembleResult, list_available_models
from outputModel.detection_output_model import (
    DetectionOutput,
    ImageDetectionResult,
    VotingResult,
    IndividualModelResult,
    ExifData,
    C2paData,
    C2paManifestSummary
)
from utils.exif_extractor import ExifExtractor
from utils.c2pa_extractor import C2paExtractor
import os 
import sys
import json
import threading
import time
from io import BytesIO
from PIL import Image

# Global ensemble detector - initialized lazily to avoid cold start timeout
_ensemble_detector = None
_ensemble_loading = False
_ensemble_lock = threading.Lock()

def get_ensemble_detector(debug=False, timeout=300):
    """Get or create the global ensemble detector instance with lazy loading
    
    This ensures the ensemble is created only once, with proper handling
    for concurrent requests during initialization.
    
    Args:
        debug: Enable debug logging (default: False)
        timeout: Maximum wait time in seconds for ensemble to load (default: 300)
        
    Returns:
        EnsembleDetector instance or None if still loading
    """
    global _ensemble_detector, _ensemble_loading
    
    # If already loaded, return immediately
    if _ensemble_detector is not None:
        return _ensemble_detector
    
    with _ensemble_lock:
        # Double-check after acquiring lock
        if _ensemble_detector is not None:
            return _ensemble_detector
        
        # If another thread is loading, wait for it
        if _ensemble_loading:
            if debug:
                print("Ensemble is being loaded by another thread, waiting...", file=sys.stderr)
            
            # Wait for the loading to complete
            start_time = time.time()
            while _ensemble_loading and (time.time() - start_time) < timeout:
                time.sleep(1)
            
            # Check if loading completed
            if _ensemble_detector is not None:
                return _ensemble_detector
            else:
                raise TimeoutError("Ensemble loading timed out")
        
        # This thread will do the loading
        _ensemble_loading = True
        if debug:
            print("Starting ensemble detector initialization...", file=sys.stderr)
    
    try:
        # Load the ensemble (outside the lock to allow other threads to wait)
        detector = create_ensemble_detector(debug=debug)
        
        with _ensemble_lock:
            _ensemble_detector = detector
            _ensemble_loading = False
            if debug:
                print("Ensemble detector loaded successfully!", file=sys.stderr)
        
        return _ensemble_detector
        
    except Exception as e:
        with _ensemble_lock:
            _ensemble_loading = False
        if debug:
            print(f"Error loading ensemble detector: {e}", file=sys.stderr)
        raise


def process_image(image_bytes):
    """Process an image from bytes and return detection results
    
    Args:
        image_bytes: Image data as bytes
        
    Returns:
        DetectionOutput Pydantic model with detection results as JSON string
    """
    try:
        # Get the global ensemble detector (lazy loaded on first call)
        ensemble = get_ensemble_detector(debug=False)
        
        if ensemble is None:
            return json.dumps({
                "error": "Models are still loading. Please try again in a moment.",
                "status": "loading"
            })

        # Convert bytes to PIL Image
        image = Image.open(BytesIO(image_bytes))
        
        # Save temporarily to process (some models require file path)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = temp_file.name
            image.save(temp_path)
        
        try:
        
            # Process the image
            threshold = 0.26
            voting_method = 'weighted'
        
            
            result = ensemble.detect(temp_path, threshold=threshold, voting_method=voting_method)
            
            # Create result data
            data = {
                'image': 'uploaded_image.png',
                'image_path': temp_path,
                'ground_truth': 'UNKNOWN',
                'prediction': 'FAKE' if result.is_deepfake else 'REAL',
                'is_correct': False,
                'confidence': result.confidence,
                'voting_summary': result.voting_summary,
                'avg_fake_score': result.average_fake_score,
                'avg_real_score': result.average_real_score,
                'individual_results': result.individual_results,
                'has_c2pa': False,
                'c2pa_info': {}
            }
            
            # Convert to Pydantic model
            output_model = convert_to_pydantic_model([data], voting_method=voting_method)
            return output_model.to_json()
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        return json.dumps({"error": f"Error processing image: {str(e)}"})


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
    
    print("\n### SEResNet-50 Models\n")
    print("| Key | Name | Architecture | Description |")
    print("|-----|------|--------------|-------------|")
    
    for key, info in models.items():
        if info.get('source') == 'seresnet50':
            print(f"| {key} | {info['name']} | {info['architecture']} | {info['description']} |")
    
    print("\n### MesoNet Models\n")
    print("| Key | Name | Architecture | Description |")
    print("|-----|------|--------------|-------------|")
    
    for key, info in models.items():
        if info.get('source') == 'mesonet':
            print(f"| {key} | {info['name']} | {info['architecture']} | {info['description']} |")


def list_voting_methods():
    """Display all available voting methods with descriptions"""
    print("\n\n## 🗳️ Available Voting Methods")
    print("=" * 80)
    
    methods = {
        "Basic Methods": {
            "majority": "Simple majority vote - most models agree wins",
            "weighted": "Compares average fake vs real scores - robust and balanced",
            "average": "Same as weighted method",
            "unanimous": "All models must agree - very conservative",
            "maximum": "Use prediction from most confident model",
            "confidence_weighted": "Weight scores by model confidence"
        },
        "Advanced Statistical Methods": {
            "adaptive_threshold": "Adjusts threshold based on prediction variance",
            "median": "Uses median instead of mean - robust to outliers",
            "robust_majority": "Excludes extreme outliers before voting (IQR method)",
            "dynamic_consensus": "Requires higher agreement when models disagree",
            "entropy_weighted": "Weights by prediction certainty (low entropy = high weight)"
        },
        "Machine Learning Methods": {
            "reliability_weighted": "Weights based on historical model accuracy",
            "bayesian": "Bayesian inference combining predictions probabilistically",
            "consensus_threshold": "Adapts threshold based on consensus strength"
        }
    }
    
    for category, category_methods in methods.items():
        print(f"\n### {category}")
        print("| Method | Description |")
        print("|--------|-------------|")
        for method, description in category_methods.items():
            print(f"| `{method}` | {description} |")
    
    print(f"\n### [TIP] Recommendations by Use Case")
    print("| Use Case | Recommended Method | Why |")
    print("|----------|-------------------|-----|")
    print("| General purpose | `reliability_weighted` | Accounts for model strengths |")
    print("| High accuracy needed | `bayesian` | Most sophisticated combination |") 
    print("| Outlier models present | `robust_majority` | Excludes extreme predictions |")
    print("| Conservative detection | `dynamic_consensus` | Requires strong agreement |")
    print("| Real-time/speed critical | `majority` | Fastest computation |")
    print("| Mixed image types | `adaptive_threshold` | Adapts to uncertainty |")


def create_ensemble_detector(debug=False):
    """Create and return an ensemble detector with configured models
    
    Args:
        debug: Enable debug logging (default: False)
    """
    return EnsembleDetector(
        model_keys=[
            # HuggingFace models
             'prithiv-v1',
             'dima-vit',
             'organika-sdxl',           
            'nonescape-full',            
            #'universal-fake-v2',
            ## DistilDIRE models (lightweight and efficient)
            #'distil-dire-celebahq',     # Good for face/portrait images
            #'distil-dire-celebahq-full',
            # #'distil-dire-eps-imagenet', # Fast general-purpose detection
            # SEResNet-50 models (excellent architecture for deepfake detection)
            # # 'seresnet50-pretrained',    # Uses pretrained model from torch.hub
            'seresnet50',              # Fine-tuned model (if available)
            # MesoNet model (lightweight CNN for mesoscopic detection)
            # 'mesonet',                 # Compact and fast, good for faces
            # 'xception71',    # Excellent for deepfakes (requires timm)
            # 'efficientnet_b7', # High accuracy (slower)
            # EfficientNet-B7 with Noisy Student (DFDC 3rd place solution)
            #'efficientnet-b7-ns',      # High accuracy, larger model - excellent for deepfakes
        ],
        debug=debug
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
        
        # Extract C2PA data for markdown report
        has_c2pa = False
        c2pa_info = {}
        try:
            c2pa_extractor = C2paExtractor()
            if c2pa_extractor.available:
                c2pa_raw = c2pa_extractor.extract_c2pa_data(image_path)
                has_c2pa = c2pa_raw.get('hasC2pa', False)
                c2pa_info = c2pa_raw
        except Exception:
            pass  # C2PA extraction is optional, don't fail the whole process
        
        return {
            'image': image_file,
            'image_path': image_path,  # Store full path for EXIF and C2PA extraction
            'ground_truth': 'FAKE' if ground_truth_is_fake else 'REAL',
            'prediction': 'FAKE' if result.is_deepfake else 'REAL',
            'is_correct': is_correct,
            'confidence': result.confidence,
            'voting_summary': result.voting_summary,
            'avg_fake_score': result.average_fake_score,
            'avg_real_score': result.average_real_score,
            'individual_results': result.individual_results,
            'has_c2pa': has_c2pa,
            'c2pa_info': c2pa_info
        }
    except Exception as e:
        image_file = os.path.basename(image_path)
        print(f"Error processing image '{image_file}': {e}")
        return {
            'image': image_file,
            'image_path': image_path,  # Store full path for EXIF and C2PA extraction
            'ground_truth': 'UNKNOWN',
            'prediction': 'ERROR',
            'is_correct': False,
            'confidence': 0.0,
            'voting_summary': str(e),
            'avg_fake_score': 0.0,
            'avg_real_score': 0.0,
            'individual_results': [],
            'has_c2pa': False,
            'c2pa_info': {}
        }


def convert_to_pydantic_model(results_data, voting_method='weighted'):
    """Convert results data to Pydantic model
    
    Args:
        results_data: List of result dictionaries
        voting_method: Voting method used
        
    Returns:
        DetectionOutput Pydantic model
    """
    image_results = []
    
    for data in results_data:
        # Process individual model results
        individual_models = []
        for individual in data.get('individual_results', []):
            # Extract fake and real scores from the model's scores dictionary
            fake_score = 0.0
            real_score = 0.0
            
            # Check for various fake label formats
            for fake_key in ['artificial', 'Artificial', 'ARTIFICIAL',
                            'Fake', 'fake', 'FAKE', 'deepfake', 'Deepfake', 'DEEPFAKE', 
                            'generated', 'Generated', 'synthetic', 'Synthetic', 
                            'AI', 'ai', 'LABEL_1', '1']:
                if fake_key in individual.scores:
                    fake_score = individual.scores[fake_key]
                    break
            
            # Check for various real label formats
            for real_key in ['human', 'Human', 'HUMAN',
                            'Real', 'real', 'REAL', 'authentic', 'Authentic', 'AUTHENTIC',
                            'genuine', 'Genuine', 'LABEL_0', '0']:
                if real_key in individual.scores:
                    real_score = individual.scores[real_key]
                    break
            
            # If still no scores found, try to infer from available scores
            if fake_score == 0.0 and real_score == 0.0 and individual.scores:
                scores_list = list(individual.scores.values())
                if len(scores_list) >= 2:
                    real_score = scores_list[0]
                    fake_score = scores_list[1]
            
            individual_models.append(IndividualModelResult(
                model=individual.model_name,
                prediction='FAKE' if individual.is_deepfake else 'REAL',
                confidence=individual.confidence,
                fakeScore=fake_score,
                realScore=real_score
            ))
        
        # Extract voting information
        voting_summary = data.get('voting_summary', {})
        if isinstance(voting_summary, dict):
            voting = VotingResult(
                fake=voting_summary.get('fake', 0),
                real=voting_summary.get('real', 0),
                total=voting_summary.get('total', len(individual_models))
            )
        else:
            # If voting_summary is not a dict, create default voting
            voting = VotingResult(
                fake=0,
                real=0,
                total=len(individual_models)
            )
        
        # Extract EXIF data
        exif_data = None
        image_path = data.get('image_path')
        if image_path and os.path.exists(image_path):
            try:
                exif_dict = ExifExtractor.extract_exif(image_path)
                exif_data = ExifData(**exif_dict)
            except Exception as e:
                # If EXIF extraction fails, use default
                exif_data = ExifData(hasExif=False, additionalData={'error': str(e)})
        
        # Extract C2PA data
        c2pa_data = None
        if image_path and os.path.exists(image_path):
            try:
                c2pa_extractor = C2paExtractor()
                c2pa_raw = c2pa_extractor.extract_c2pa_data(image_path)
                
                # Create C2paManifestSummary if active manifest exists
                manifest_summary = None
                if c2pa_raw.get('activeManifest'):
                    manifest_summary = C2paManifestSummary(**c2pa_raw['activeManifest'])
                
                # Create C2paData model
                c2pa_data = C2paData(
                    hasC2pa=c2pa_raw.get('hasC2pa', False),
                    validationState=c2pa_raw.get('validationState'),
                    activeManifest=manifest_summary,
                    manifests=c2pa_raw.get('manifests'),
                    validationResults=c2pa_raw.get('validationResults'),
                    isEmbedded=c2pa_raw.get('isEmbedded'),
                    remoteUrl=c2pa_raw.get('remoteUrl'),
                    error=c2pa_raw.get('error')
                )
            except Exception as e:
                # If C2PA extraction fails, create default with error
                c2pa_data = C2paData(hasC2pa=False, error=f"C2PA extraction failed: {str(e)}")
        
        # Create image result
        image_result = ImageDetectionResult(
            imageFile=data.get('image', ''),
            prediction=data.get('prediction', 'UNKNOWN'),
            confidence=data.get('confidence', 0.0),
            averageFake=data.get('avg_fake_score', 0.0),
            averageReal=data.get('avg_real_score', 0.0),
            votingMethod=voting_method,
            voting=voting,
            exifData=exif_data,
            c2paData=c2pa_data,
            individualModelResults=individual_models
        )
        image_results.append(image_result)
    
    # Generate markdown content
    markdown_content = generate_markdown_report(results_data, voting_method)
    
    # Create final output
    output = DetectionOutput(
        markdown=markdown_content,
        totalImages=len(results_data),
        images=image_results
    )
    
    return output


def generate_markdown_report(results_data, voting_method='weighted', threshold=0.26):
    """Generate markdown formatted report from results data
    
    Args:
        results_data: List of result dictionaries
        voting_method: Voting method used
        threshold: Threshold value used
        
    Returns:
        Markdown formatted string
    """
    if not results_data:
        return ""
    
    lines = []
    
    # Header
    lines.append("# Deepfake Detection Results")
    lines.append("")
    
    # Performance Metrics
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
    
    lines.append("## [STATS] Performance Metrics")
    lines.append("")
    lines.append(f"**Voting Method**: `{voting_method}`")
    lines.append("")
    
    # Explain voting method
    if voting_method == 'weighted' or voting_method == 'average':
        lines.append(f"- Compares **average fake score vs average real score** (avg fake > avg real → FAKE)")
        lines.append(f"- More robust for noisy images than simple majority voting")
    elif voting_method == 'majority':
        lines.append(f"- Uses **simple majority** vote (most models agree)")
    elif voting_method == 'confidence_weighted':
        lines.append(f"- Weights fake scores by each model's confidence")
    elif voting_method == 'maximum':
        lines.append(f"- Uses prediction from model with highest confidence")
    elif voting_method == 'unanimous':
        lines.append(f"- All models must agree")
    elif voting_method == 'adaptive_threshold':
        lines.append(f"- Adjusts threshold based on prediction variance (disagreement between models)")
        lines.append(f"- Higher variance → higher threshold needed")
    elif voting_method == 'median':
        lines.append(f"- Uses **median fake score** instead of average (more robust to outliers)")
    elif voting_method == 'robust_majority':
        lines.append(f"- Excludes extreme outliers before majority voting using IQR method")
        lines.append(f"- Best for handling models with very different sensitivities")
    elif voting_method == 'dynamic_consensus':
        lines.append(f"- Requires higher agreement when models disagree strongly")
        lines.append(f"- Low disagreement: 50% majority | High disagreement: 75% majority")
    elif voting_method == 'reliability_weighted':
        lines.append(f"- Weights votes based on each model's estimated reliability")
        lines.append(f"- Gives more weight to historically accurate models")
    elif voting_method == 'bayesian':
        lines.append(f"- Uses Bayesian inference to combine predictions probabilistically")
        lines.append(f"- Accounts for model confidence and prior probabilities")
    elif voting_method == 'entropy_weighted':
        lines.append(f"- Weights predictions by certainty (low entropy = high weight)")
        lines.append(f"- More certain predictions get higher influence")
    elif voting_method == 'consensus_threshold':
        lines.append(f"- Adapts threshold based on model consensus strength")
        lines.append(f"- High consensus → lower threshold needed")
    
    lines.append("")
    lines.append(f"**Threshold**: {threshold}")
    lines.append("")
    lines.append(f"**Overall Accuracy**: {accuracy:.2f}% ({correct}/{total})")
    lines.append(f"**Precision**: {precision:.2f}% (of predicted fakes, how many are actually fake)")
    lines.append(f"**Recall**: {recall:.2f}% (of actual fakes, how many were detected)")
    lines.append(f"**F1-Score**: {f1_score:.2f}%")
    lines.append("")
    
    lines.append("### Confusion Matrix")
    lines.append("```")
    lines.append("                    Predicted")
    lines.append("                FAKE      REAL")
    lines.append(f"Actual FAKE     {true_positives:4d}      {false_negatives:4d}")
    lines.append(f"       REAL     {false_positives:4d}      {true_negatives:4d}")
    lines.append("```")
    lines.append("")
    
    # Overall Results Table
    lines.append("## [TARGET] Detection Results Summary")
    lines.append("")
    lines.append("### Overall Results")
    lines.append("")
    lines.append("| Image | Ground Truth | Prediction | [OK]/[FAIL] | Confidence | Avg Fake | Avg Real | Voting |")
    lines.append("|-------|--------------|------------|-----|------------|----------|----------|--------|")
    
    for data in results_data:
        correctness = "[OK]" if data['is_correct'] else "[FAIL]"
        lines.append(f"| {data['image']} | {data['ground_truth']} | {data['prediction']} | {correctness} | "
                    f"{data['confidence']:.4f} ({data['confidence']*100:.1f}%) | "
                    f"{data['avg_fake_score']:.4f} | {data['avg_real_score']:.4f} | {data['voting_summary']} |")
    
    lines.append("")
    
    # C2PA Summary
    images_with_c2pa = sum(1 for d in results_data if d.get('has_c2pa', False))
    if images_with_c2pa > 0:
        lines.append("### 🔐 C2PA Content Credentials Summary")
        lines.append("")
        lines.append(f"**Images with C2PA data**: {images_with_c2pa}/{total}")
        lines.append("")
        lines.append("| Image | Has C2PA | Validation State | Claim Generator |")
        lines.append("|-------|----------|------------------|-----------------|")
        
        for data in results_data:
            if data.get('has_c2pa', False):
                c2pa_info = data.get('c2pa_info', {})
                validation = c2pa_info.get('validationState', 'N/A')
                claim_gen = 'N/A'
                if c2pa_info.get('activeManifest'):
                    claim_gen = c2pa_info['activeManifest'].get('claimGenerator', 'N/A')
                lines.append(f"| {data['image']} | [OK] | {validation} | {claim_gen} |")
        
        lines.append("")
    
    # Individual Model Results
    lines.append("### 📋 Individual Model Results")
    lines.append("")
    
    for data in results_data:
        if data['individual_results']:
            correctness_icon = "[OK]" if data['is_correct'] else "[FAIL]"
            lines.append(f"#### {correctness_icon} {data['image']} (Ground Truth: {data['ground_truth']})")
            lines.append("")
            lines.append("| Model | Prediction | Confidence | Fake Score | Real Score |")
            lines.append("|-------|------------|------------|------------|------------|")
            
            for individual in data['individual_results']:
                pred = 'FAKE' if individual.is_deepfake else 'REAL'
                
                # Extract fake and real scores from the model's scores dictionary
                fake_score = 0.0
                real_score = 0.0
                
                # Check for various fake label formats
                for fake_key in ['artificial', 'Artificial', 'ARTIFICIAL',
                                'Fake', 'fake', 'FAKE', 'deepfake', 'Deepfake', 'DEEPFAKE', 
                                'generated', 'Generated', 'synthetic', 'Synthetic', 
                                'AI', 'ai', 'LABEL_1', '1']:
                    if fake_key in individual.scores:
                        fake_score = individual.scores[fake_key]
                        break
                
                # Check for various real label formats
                for real_key in ['human', 'Human', 'HUMAN',
                                'Real', 'real', 'REAL', 'authentic', 'Authentic', 'AUTHENTIC',
                                'genuine', 'Genuine', 'LABEL_0', '0']:
                    if real_key in individual.scores:
                        real_score = individual.scores[real_key]
                        break
                
                # If still no scores found, try to infer from available scores
                if fake_score == 0.0 and real_score == 0.0 and individual.scores:
                    scores_list = list(individual.scores.values())
                    if len(scores_list) >= 2:
                        real_score = scores_list[0]
                        fake_score = scores_list[1]
                
                lines.append(f"| {individual.model_name} | {pred} | "
                           f"{individual.confidence:.4f} ({individual.confidence*100:.1f}%) | "
                           f"{fake_score:.4f} | {real_score:.4f} |")
            
            lines.append("")
    
    return "\n".join(lines)


def display_overall_results_table(results_data):
    """Display overall results in a markdown table format
    
    Args:
        results_data: List of result dictionaries
    """
    print("| Image | Ground Truth | Prediction | [OK]/[FAIL] | Confidence | Avg Fake | Avg Real | Voting |")
    print("|-------|--------------|------------|-----|------------|----------|----------|--------|")
    
    for data in results_data:
        correctness = "[OK]" if data['is_correct'] else "[FAIL]"
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
            correctness_icon = "[OK]" if data['is_correct'] else "[FAIL]"
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
    
    print("\n\n## [STATS] Performance Metrics")
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
    elif voting_method == 'adaptive_threshold':
        print(f"- Adjusts threshold based on prediction variance (disagreement between models)")
        print(f"- Higher variance → higher threshold needed")
    elif voting_method == 'median':
        print(f"- Uses **median fake score** instead of average (more robust to outliers)")
    elif voting_method == 'robust_majority':
        print(f"- Excludes extreme outliers before majority voting using IQR method")
        print(f"- Best for handling models with very different sensitivities")
    elif voting_method == 'dynamic_consensus':
        print(f"- Requires higher agreement when models disagree strongly")
        print(f"- Low disagreement: 50% majority | High disagreement: 75% majority")
    elif voting_method == 'reliability_weighted':
        print(f"- Weights votes based on each model's estimated reliability")
        print(f"- Gives more weight to historically accurate models")
    elif voting_method == 'bayesian':
        print(f"- Uses Bayesian inference to combine predictions probabilistically")
        print(f"- Accounts for model confidence and prior probabilities")
    elif voting_method == 'entropy_weighted':
        print(f"- Weights predictions by certainty (low entropy = high weight)")
        print(f"- More certain predictions get higher influence")
    elif voting_method == 'consensus_threshold':
        print(f"- Adapts threshold based on model consensus strength")
        print(f"- High consensus → lower threshold needed")
    
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


def compare_voting_methods(image_path, ensemble, threshold=0.26):
    """Compare all voting methods on a single image"""
    print(f"\n\n## 🔬 Voting Method Comparison for: {os.path.basename(image_path)}")
    print("=" * 80)
    
    # All available voting methods
    voting_methods = [
        'majority', 'weighted', 'confidence_weighted', 'unanimous', 'maximum',
        'adaptive_threshold', 'median', 'robust_majority', 'dynamic_consensus',
        'reliability_weighted', 'bayesian', 'entropy_weighted', 'consensus_threshold'
    ]
    
    # Ground truth from filename
    image_file = os.path.basename(image_path)
    ground_truth_is_fake = 'fake' in image_file.lower()
    ground_truth = 'FAKE' if ground_truth_is_fake else 'REAL'
    
    print(f"**Ground Truth**: {ground_truth}")
    print(f"**Threshold**: {threshold}")
    
    print("\n| Voting Method | Prediction | Confidence | [OK]/[FAIL] | Notes |")
    print("|---------------|------------|------------|-----|-------|")
    
    results = {}
    for method in voting_methods:
        try:
            result = ensemble.detect(image_path, threshold=threshold, voting_method=method)
            prediction = 'FAKE' if result.is_deepfake else 'REAL'
            is_correct = result.is_deepfake == ground_truth_is_fake
            correctness = "[OK]" if is_correct else "[FAIL]"
            
            # Add method-specific notes
            notes = ""
            if method == 'majority':
                votes = result.voting_summary
                notes = f"Votes: {votes['fake']}F-{votes['real']}R"
            elif method in ['weighted', 'average']:
                notes = f"Avg fake: {result.average_fake_score:.3f}"
            elif method == 'maximum':
                max_conf = max(r.confidence for r in result.individual_results)
                notes = f"Max conf: {max_conf:.3f}"
            elif method == 'bayesian':
                notes = f"Posterior: {result.confidence:.3f}"
            elif method == 'reliability_weighted':
                notes = "Model weights applied"
            elif method == 'robust_majority':
                notes = "Outliers filtered"
            
            print(f"| {method} | {prediction} | {result.confidence:.3f} ({result.confidence*100:.1f}%) | {correctness} | {notes} |")
            
            results[method] = {
                'prediction': prediction,
                'confidence': result.confidence,
                'is_correct': is_correct,
                'result': result
            }
            
        except Exception as e:
            print(f"| {method} | ERROR | 0.000 (0.0%) | [FAIL] | {str(e)[:30]}... |")
            results[method] = {'prediction': 'ERROR', 'confidence': 0.0, 'is_correct': False}
    
    # Summary statistics
    correct_predictions = sum(1 for r in results.values() if r['is_correct'])
    total_methods = len([r for r in results.values() if r['prediction'] != 'ERROR'])
    
    print(f"\n**Consensus**: {correct_predictions}/{total_methods} methods correct ({correct_predictions/total_methods*100:.1f}%)")
    
    # Find best and worst performing methods
    if results:
        best_methods = [method for method, data in results.items() 
                       if data['is_correct'] and data['prediction'] != 'ERROR']
        worst_methods = [method for method, data in results.items() 
                        if not data['is_correct'] and data['prediction'] != 'ERROR']
        
        if best_methods:
            print(f"**Correct Methods**: {', '.join(best_methods)}")
        if worst_methods:
            print(f"**Incorrect Methods**: {', '.join(worst_methods)}")
    
    return results


def process_path_with_ensemble(path, ensemble, threshold=0.26, voting_method='weighted', debug=False):
    """Process a single image or directory with pre-loaded ensemble
    
    Args:
        path: Path to an image file or directory
        ensemble: Pre-loaded EnsembleDetector instance
        threshold: Detection threshold
        voting_method: Voting method for ensemble
        debug: Enable debug logging (default: False)
    """
    try:
        results_data = []
        
        if os.path.isfile(path):
            # Single image
            data = process_image_data(path, ensemble, threshold=threshold, voting_method=voting_method)
            if data:
                results_data.append(data)
                
        elif os.path.isdir(path):
            # Collect results from all images
            for image_file in os.listdir(path):
                image_path = os.path.join(path, image_file)
                if os.path.isfile(image_path):
                    # Process each image
                    data = process_image_data(image_path, ensemble, threshold=threshold, voting_method=voting_method)
                    if data:
                        results_data.append(data)
        else:
            print(json.dumps({"error": f"Path '{path}' does not exist or is not accessible."}))
            return
        
        # Convert to Pydantic model and output JSON
        if results_data:
            output_model = convert_to_pydantic_model(results_data, voting_method=voting_method)
            print(output_model.to_json())
        else:
            print(json.dumps({"error": "No images processed", "totalImages": 0, "images": []}))
            
    except Exception as e:
        print(json.dumps({"error": f"Error processing path: {str(e)}"}))



def main(path=None, debug=False):
    """Run deepfake detection on an image or directory
    
    Args:
        path: Path to an image file or directory. If None, enters interactive mode.
        debug: Enable debug logging (default: False)
    """
    
    try:
        # Configuration
        threshold = 0.26
        voting_method = 'weighted'
        
        # Load ensemble models into memory once at startup
        if debug:
            print("Loading ensemble models into memory...", file=sys.stderr)
        ensemble = create_ensemble_detector(debug=debug)
        if debug:
            print("Models loaded successfully!", file=sys.stderr)
        
        if path is not None:
            # Single execution mode with provided path
            process_path_with_ensemble(path, ensemble, threshold, voting_method, debug)
        else:
            # Interactive loop mode
            if debug:
                print("\nInteractive Mode", file=sys.stderr)
                print("Enter path to an image file or directory to analyze", file=sys.stderr)
                print("Enter 'exit' or press Enter to quit", file=sys.stderr)
            
            while True:
                user_input = input("Enter image/directory path (or 'exit'): ").strip()
                
                # Check for exit condition
                if user_input.lower() == 'exit' or user_input == '':
                    if debug:
                        print("Exiting deepfake detection. Goodbye!", file=sys.stderr)
                    break
                
                # Strip quotes from the path (handles both single and double quotes)
                user_input = user_input.strip('"').strip("'")
                
                # Process the provided path
                process_path_with_ensemble(user_input, ensemble, threshold, voting_method, debug)
        
    except Exception as e:
        print(json.dumps({"error": f"Error running detection: {str(e)}"}))



def demo_advanced_voting_methods():
    """Demonstrate all the new advanced voting methods"""
    print("\n\n[START] ADVANCED DEEPFAKE DETECTION DEMO")
    print("=" * 80)
    print("This demo showcases the new advanced voting methods for improved accuracy!")
    
    # Show all available voting methods
    list_voting_methods()
    
    # Example image path - update this to your test image
    test_image = "C:\\Kidambi\\WorkRelated\\DeepFake\\sample_images\\real_2.avif"
    
    if os.path.exists(test_image):
        print(f"\n\n[SEARCH] Testing all voting methods on: {os.path.basename(test_image)}")
        
        # Create ensemble
        ensemble = create_ensemble_detector()
        
        # Compare all voting methods
        results = compare_voting_methods(test_image, ensemble, threshold=0.26)
        
        print(f"\n\n[TIP] **Key Insights:**")
        print(f"- The new **reliability_weighted** method accounts for each model's strengths")
        print(f"- **Bayesian** method provides the most sophisticated probabilistic reasoning") 
        print(f"- **Robust_majority** handles outlier models (like overly aggressive ones)")
        print(f"- **Dynamic_consensus** requires stronger agreement when models disagree")
        print(f"- **Adaptive_threshold** adjusts sensitivity based on model uncertainty")
        
    else:
        print(f"\n[WARNING] Test image not found at: {test_image}")
        print(f"Please update the path in the demo function or run with your own image!")


if __name__ == "__main__":
    # Check if argument is provided via command line
    if len(sys.argv) > 1:
        path_arg = sys.argv[1]
        
       
        # Special commands
        if path_arg.lower() == 'demo':
            demo_advanced_voting_methods()
        elif path_arg.lower() == 'methods':
            list_voting_methods()
        elif path_arg.lower() == 'models':
            list_available_models_example()
        else:
            main(path_arg)
    else:
        main()