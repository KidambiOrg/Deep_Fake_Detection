import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from imagedetectionlib.detect_deepfake_image import DeepfakeImageDetector, EnsembleDetector
from imagedetectionlib.model_registry import list_available_models
from outputModel.image_detection_output_model import (
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
from utils.blob_storage_helper import get_blob_loader
import json
import threading
import time
import logging
from io import BytesIO
from PIL import Image
from typing import List

logger = logging.getLogger(__name__)

# Global ensemble detector - initialized lazily to avoid cold start timeout
_ensemble_detector = None
_ensemble_loading = False
_ensemble_lock = threading.Lock()

def download_models_from_blob_storage():
    """Download models from Azure Blob Storage if configured.
    Updates the MODEL_REGISTRY with local paths to downloaded models.
    """
    blob_loader = get_blob_loader()
    if not blob_loader or not blob_loader.blob_service_client:
        logger.info("Blob storage not configured, using local models")
        return
    
    logger.info("Checking for models in Azure Blob Storage...")
    
    # Download models that have blob_storage configuration
    from imagedetectionlib.model_registry import MODEL_REGISTRY
    for model_key, model_info in MODEL_REGISTRY.items():
        if 'blob_storage' not in model_info:
            continue
            
        blob_config = model_info['blob_storage']
        
        # Determine if it's a single file or directory
        if 'blob' in blob_config:
            # Single file (e.g., .pth, .safetensors)
            local_path = blob_loader.download_model(
                container_name=blob_config['container'],
                blob_name=blob_config['blob']
            )
            
            if local_path and 'model_path' in model_info:
                model_info['model_path'] = local_path
                logger.info(f"Updated {model_key} model path to: {local_path}")
        
        elif 'path' in blob_config:
            # Directory (e.g., HuggingFace model with multiple files)
            local_dir = blob_loader.download_directory(
                container_name=blob_config['container'],
                blob_prefix=blob_config['path']
            )
            
            if local_dir:
                # Update model path to use downloaded directory
                if 'local_path' in model_info:
                    model_info['local_path'] = str(local_dir)
                    logger.info(f"Updated {model_key} model directory to: {local_dir}")
                else:
                    # For HuggingFace models, update the name to point to local directory
                    model_info['local_path'] = str(local_dir)
                    logger.info(f"Set {model_key} local path to: {local_dir}")


def get_ensemble_detector(debug=False, timeout=300):
    """
    Get or initialize the global ensemble detector.
    Uses lazy loading to avoid cold start timeouts.
    Downloads models from Azure Blob Storage if configured.
    
    Args:
        debug: Enable debug logging
        timeout: Maximum time to wait for loading (seconds)
    
    Returns:
        EnsembleDetector instance or None if still loading
    """
    global _ensemble_detector, _ensemble_loading, _ensemble_lock
    
    # If already loaded, return immediately
    if _ensemble_detector is not None:
        return _ensemble_detector
    
    # Prevent multiple simultaneous initialization attempts
    with _ensemble_lock:
        # Double-check after acquiring lock
        if _ensemble_detector is not None:
            return _ensemble_detector
        
        # Check if already loading
        if _ensemble_loading:
            logger.info("Models are already being loaded by another request")
            return None
        
        # Mark as loading
        _ensemble_loading = True
    
    try:
        logger.info("Starting ensemble detector initialization...")
        
        # Download models from blob storage if configured
        download_models_from_blob_storage()
        
        # Initialize ensemble detector
        start_time = time.time()
        _ensemble_detector = EnsembleDetector(
            model_keys=[
                # HuggingFace models
                'prithiv-v1',
                'dima-vit',
                'organika-sdxl',           
                'nonescape-full',                              
                # SEResNet-50 models (excellent architecture for deepfake detection)            
                'seresnet50'              # Fine-tuned model (if available)            
            ],
            debug=debug
        )
        elapsed = time.time() - start_time
        
        logger.info(f"Ensemble detector initialized successfully in {elapsed:.2f} seconds")
        return _ensemble_detector
        
    except Exception as e:
        logger.error(f"Failed to initialize ensemble detector: {e}")
        logger.exception(e)
        return None
    finally:
        # Mark loading as complete
        with _ensemble_lock:
            _ensemble_loading = False


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
            threshold = 0.20
            voting_method = 'weighted'
            
            result = ensemble.detect(temp_path, threshold=threshold, voting_method=voting_method)
            
            # Extract EXIF data
            try:
                exif_dict = ExifExtractor.extract_exif(temp_path)
                result.exifData = ExifData(**exif_dict)
            except Exception as e:
                logger.warning(f"EXIF extraction failed: {e}")
                result.exifData = ExifData(hasExif=False, additionalData={'error': str(e)})
            
            # Extract C2PA data
            try:
                c2pa_extractor = C2paExtractor()
                c2pa_raw = c2pa_extractor.extract_c2pa_data(temp_path)
                
                # Create C2paManifestSummary if active manifest exists
                manifest_summary = None
                active_manifest_label = c2pa_raw.get('activeManifest')
                if active_manifest_label and isinstance(active_manifest_label, str):
                    manifests_dict = c2pa_raw.get('manifests', {})
                    if active_manifest_label in manifests_dict:
                        manifest_summary = C2paManifestSummary(**manifests_dict[active_manifest_label])
                
                # Create C2paData model
                result.c2paData = C2paData(
                    hasC2pa=c2pa_raw.get('hasC2pa', False),
                    validationState=c2pa_raw.get('validationState'),
                    activeManifest=manifest_summary,
                    manifests=c2pa_raw.get('manifests'),
                    validationResults=c2pa_raw.get('validationResults'),
                    validationStatus=c2pa_raw.get('validationStatus'),
                    isEmbedded=c2pa_raw.get('isEmbedded'),
                    remoteUrl=c2pa_raw.get('remoteUrl'),
                    error=c2pa_raw.get('error')
                )
            except Exception as e:
                logger.warning(f"C2PA extraction failed: {e}")
                result.c2paData = C2paData(hasC2pa=False, error=f"C2PA extraction failed: {str(e)}")
            
            # result is now an ImageDetectionResult Pydantic model with EXIF and C2PA data
            # Wrap in DetectionOutput with markdown report
            output_model = create_detection_output([result])
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


def process_image_data(image_path, ensemble, threshold=0.26, voting_method='weighted'):
    """Process a single image and return ImageDetectionResult
    
    Args:
        image_path: Path to the image file
        ensemble: EnsembleDetector instance
        threshold: Detection threshold
        voting_method: Voting method for ensemble
        
    Returns:
        ImageDetectionResult Pydantic model with detection results
    """
    try:
        result = ensemble.detect(image_path, threshold=threshold, voting_method=voting_method)
        
        # Add EXIF data if not already present
        if result.exifData is None and os.path.exists(image_path):
            try:
                exif_dict = ExifExtractor.extract_exif(image_path)
                result.exifData = ExifData(**exif_dict)
            except Exception:
                result.exifData = ExifData(hasExif=False)
        
        # Add C2PA data if not already present
        if result.c2paData is None and os.path.exists(image_path):
            try:
                c2pa_extractor = C2paExtractor()
                if c2pa_extractor.available:
                    c2pa_raw = c2pa_extractor.extract_c2pa_data(image_path)
                    
                    # Create C2paManifestSummary if active manifest exists
                    manifest_summary = None
                    active_manifest_label = c2pa_raw.get('activeManifest')
                    if active_manifest_label and isinstance(active_manifest_label, str):
                        manifests_dict = c2pa_raw.get('manifests', {})
                        if active_manifest_label in manifests_dict:
                            manifest_summary = C2paManifestSummary(**manifests_dict[active_manifest_label])
                    
                    result.c2paData = C2paData(
                        hasC2pa=c2pa_raw.get('hasC2pa', False),
                        validationState=c2pa_raw.get('validationState'),
                        activeManifest=manifest_summary,
                        manifests=c2pa_raw.get('manifests'),
                        validationResults=c2pa_raw.get('validationResults'),
                        validationStatus=c2pa_raw.get('validationStatus'),
                        isEmbedded=c2pa_raw.get('isEmbedded'),
                        remoteUrl=c2pa_raw.get('remoteUrl'),
                        error=c2pa_raw.get('error')
                    )
            except Exception as e:
                result.c2paData = C2paData(hasC2pa=False, error=f"C2PA extraction failed: {str(e)}")
        
        return result
        
    except Exception as e:
        image_file = os.path.basename(image_path)
        logger.error(f"Error processing image '{image_file}': {e}")
        # Return a minimal result on error
        return ImageDetectionResult(
            imageFile=image_file,
            prediction='ERROR',
            confidence=0.0,
            averageFake=0.0,
            averageReal=0.0,
            votingMethod=voting_method,
            voting=VotingResult(fake=0, real=0, total=0),
            individualModelResults=[]
        )


def create_detection_output(image_results: List[ImageDetectionResult]) -> DetectionOutput:
    """Wrap ImageDetectionResult list in DetectionOutput
    
    Args:
        image_results: List of ImageDetectionResult instances
        
    Returns:
        DetectionOutput Pydantic model
    """
    # Generate markdown content
    markdown_content = generate_markdown_report_from_pydantic(image_results)
    
    # Create final output
    output = DetectionOutput(
        markdown=markdown_content,
        totalImages=len(image_results),
        images=image_results
    )
    
    return output


def generate_markdown_report_from_pydantic(image_results: List[ImageDetectionResult]) -> str:
    """Generate markdown formatted report from Pydantic models
    
    Args:
        image_results: List of ImageDetectionResult instances
        
    Returns:
        Markdown formatted string
    """
    if not image_results:
        return ""
    
    lines = []
    
    # Header
    lines.append("# Deepfake Detection Results")
    lines.append("")
    
    # Settings
  
    
    # Results for each image
    for idx, result in enumerate(image_results, 1):
        lines.append(f"## Image {idx}: {result.imageFile} - `{result.prediction}`")
        lines.append("")       
        lines.append(f"**Confidence**: `{result.confidence:.2%}`")
        lines.append(f"**Average Fake Score**: `{result.averageFake:.4f}`")
        lines.append(f"**Average Real Score**: `{result.averageReal:.4f}`")
        lines.append("")
        
        lines.append("### Detection Settings")
        lines.append("")
        if image_results:
            lines.append(f"**Voting Method**: `{image_results[0].votingMethod}`")
        lines.append(f"**Total Images Processed**: `{len(image_results)}`")
        lines.append("")
        
        # Voting breakdown
        lines.append("### Voting Results")
        lines.append("")
        lines.append(f"- **Fake votes**: {result.voting.fake}")
        lines.append(f"- **Real votes**: {result.voting.real}")
        lines.append(f"- **Total models**: {result.voting.total}")
        lines.append("")
        lines.append("---")
        lines.append("")
    
    return "\n".join(lines)


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
    
    # # Performance Metrics
    # total = len(results_data)
    # correct = sum(1 for d in results_data if d['is_correct'])
    # accuracy = (correct / total) * 100 if total > 0 else 0
    
    # # Confusion matrix
    # true_positives = sum(1 for d in results_data if d['ground_truth'] == 'FAKE' and d['prediction'] == 'FAKE')
    # false_positives = sum(1 for d in results_data if d['ground_truth'] == 'REAL' and d['prediction'] == 'FAKE')
    # true_negatives = sum(1 for d in results_data if d['ground_truth'] == 'REAL' and d['prediction'] == 'REAL')
    # false_negatives = sum(1 for d in results_data if d['ground_truth'] == 'FAKE' and d['prediction'] == 'REAL')
    
    # # Calculate precision, recall, F1
    # precision = (true_positives / (true_positives + false_positives)) * 100 if (true_positives + false_positives) > 0 else 0
    # recall = (true_positives / (true_positives + false_negatives)) * 100 if (true_positives + false_negatives) > 0 else 0
    # f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    
    # lines.append("## [STATS] Performance Metrics")
    # lines.append("")
    # lines.append(f"**Voting Method**: `{voting_method}`")
    # lines.append("")
    
    # # Explain voting method
    # if voting_method == 'weighted' or voting_method == 'average':
    #     lines.append(f"- Compares **average fake score vs average real score** (avg fake > avg real → FAKE)")
    #     lines.append(f"- More robust for noisy images than simple majority voting")
    # elif voting_method == 'majority':
    #     lines.append(f"- Uses **simple majority** vote (most models agree)")
    # elif voting_method == 'confidence_weighted':
    #     lines.append(f"- Weights fake scores by each model's confidence")
    # elif voting_method == 'maximum':
    #     lines.append(f"- Uses prediction from model with highest confidence")
    # elif voting_method == 'unanimous':
    #     lines.append(f"- All models must agree")
    # elif voting_method == 'adaptive_threshold':
    #     lines.append(f"- Adjusts threshold based on prediction variance (disagreement between models)")
    #     lines.append(f"- Higher variance → higher threshold needed")
    # elif voting_method == 'median':
    #     lines.append(f"- Uses **median fake score** instead of average (more robust to outliers)")
    # elif voting_method == 'robust_majority':
    #     lines.append(f"- Excludes extreme outliers before majority voting using IQR method")
    #     lines.append(f"- Best for handling models with very different sensitivities")
    # elif voting_method == 'dynamic_consensus':
    #     lines.append(f"- Requires higher agreement when models disagree strongly")
    #     lines.append(f"- Low disagreement: 50% majority | High disagreement: 75% majority")
    # elif voting_method == 'reliability_weighted':
    #     lines.append(f"- Weights votes based on each model's estimated reliability")
    #     lines.append(f"- Gives more weight to historically accurate models")
    # elif voting_method == 'bayesian':
    #     lines.append(f"- Uses Bayesian inference to combine predictions probabilistically")
    #     lines.append(f"- Accounts for model confidence and prior probabilities")
    # elif voting_method == 'entropy_weighted':
    #     lines.append(f"- Weights predictions by certainty (low entropy = high weight)")
    #     lines.append(f"- More certain predictions get higher influence")
    # elif voting_method == 'consensus_threshold':
    #     lines.append(f"- Adapts threshold based on model consensus strength")
    #     lines.append(f"- High consensus → lower threshold needed")
    
    # lines.append("")
    # lines.append(f"**Threshold**: {threshold}")
    # lines.append("")
    # lines.append(f"**Overall Accuracy**: {accuracy:.2f}% ({correct}/{total})")
    # lines.append(f"**Precision**: {precision:.2f}% (of predicted fakes, how many are actually fake)")
    # lines.append(f"**Recall**: {recall:.2f}% (of actual fakes, how many were detected)")
    # lines.append(f"**F1-Score**: {f1_score:.2f}%")
    # lines.append("")
    
    # lines.append("### Confusion Matrix")
    # lines.append("```")
    # lines.append("                    Predicted")
    # lines.append("                FAKE      REAL")
    # lines.append(f"Actual FAKE     {true_positives:4d}      {false_negatives:4d}")
    # lines.append(f"       REAL     {false_positives:4d}      {true_negatives:4d}")
    # lines.append("```")
    # lines.append("")
    
    # Overall Results Table   
    lines.append("")
    lines.append("| Image | Prediction | Confidence | Avg Fake | Avg Real | Voting |")
    lines.append("|-------|------------|-------------|----------|----------|--------|")

    for data in results_data:
        correctness = "[OK]" if data['is_correct'] else "[FAIL]"
        lines.append(f"| {data['image']} | {data['prediction']} | "
                    f"{data['confidence']:.4f} ({data['confidence']*100:.1f}%) | "
                    f"{data['avg_fake_score']:.4f} | {data['avg_real_score']:.4f} | {data['voting_summary']} |")
    
    lines.append("")
    
    # # C2PA Summary
    # images_with_c2pa = sum(1 for d in results_data if d.get('has_c2pa', False))
    # if images_with_c2pa > 0:
    #     lines.append("### 🔐 C2PA Content Credentials Summary")
    #     lines.append("")
    #     lines.append(f"**Images with C2PA data**: {images_with_c2pa}/{total}")
    #     lines.append("")
    #     lines.append("| Image | Has C2PA | Validation State | Claim Generator |")
    #     lines.append("|-------|----------|------------------|-----------------|")
        
    #     for data in results_data:
    #         if data.get('has_c2pa', False):
    #             c2pa_info = data.get('c2pa_info', {})
    #             validation = c2pa_info.get('validationState', 'N/A')
    #             claim_gen = 'N/A'
    #             # activeManifest is now a String (label), get actual manifest from manifests dict
    #             active_manifest_label = c2pa_info.get('activeManifest')
    #             if active_manifest_label and isinstance(active_manifest_label, str):
    #                 manifests = c2pa_info.get('manifests', {})
    #                 if active_manifest_label in manifests:
    #                     active_manifest_data = manifests[active_manifest_label]
    #                     claim_gen = active_manifest_data.get('claimGenerator', 'N/A')
    #             lines.append(f"| {data['image']} | [OK] | {validation} | {claim_gen} |")
        
    #     lines.append("")
    
    # # Individual Model Results
    # lines.append("### 📋 Individual Model Results")
    # lines.append("")
    
    # for data in results_data:
    #     if data['individual_results']:
    #         correctness_icon = "[OK]" if data['is_correct'] else "[FAIL]"
    #         lines.append(f"#### {correctness_icon} {data['image']} (Ground Truth: {data['ground_truth']})")
    #         lines.append("")
    #         lines.append("| Model | Prediction | Confidence | Fake Score | Real Score |")
    #         lines.append("|-------|------------|------------|------------|------------|")
            
    #         for individual in data['individual_results']:
    #             pred = 'FAKE' if individual.is_deepfake else 'REAL'
                
    #             # Extract fake and real scores from the model's scores dictionary
    #             fake_score = 0.0
    #             real_score = 0.0
                
    #             # Check for various fake label formats
    #             for fake_key in ['artificial', 'Artificial', 'ARTIFICIAL',
    #                             'Fake', 'fake', 'FAKE', 'deepfake', 'Deepfake', 'DEEPFAKE', 
    #                             'generated', 'Generated', 'synthetic', 'Synthetic', 
    #                             'AI', 'ai', 'LABEL_1', '1']:
    #                 if fake_key in individual.scores:
    #                     fake_score = individual.scores[fake_key]
    #                     break
                
    #             # Check for various real label formats
    #             for real_key in ['human', 'Human', 'HUMAN',
    #                             'Real', 'real', 'REAL', 'authentic', 'Authentic', 'AUTHENTIC',
    #                             'genuine', 'Genuine', 'LABEL_0', '0']:
    #                 if real_key in individual.scores:
    #                     real_score = individual.scores[real_key]
    #                     break
                
    #             # If still no scores found, try to infer from available scores
    #             if fake_score == 0.0 and real_score == 0.0 and individual.scores:
    #                 scores_list = list(individual.scores.values())
    #                 if len(scores_list) >= 2:
    #                     real_score = scores_list[0]
    #                     fake_score = scores_list[1]
                
    #             lines.append(f"| {individual.model_name} | {pred} | "
    #                        f"{individual.confidence:.4f} ({individual.confidence*100:.1f}%) | "
    #                        f"{fake_score:.4f} | {real_score:.4f} |")
            
    #         lines.append("")
    
    return "\n".join(lines)
 


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
                    # Process each image - returns ImageDetectionResult
                    result = process_image_data(image_path, ensemble, threshold=threshold, voting_method=voting_method)
                    if result:
                        results_data.append(result)
        else:
            print(json.dumps({"error": f"Path '{path}' does not exist or is not accessible."}))
            return
        
        # Wrap results in DetectionOutput and output JSON
        if results_data:
            output_model = create_detection_output(results_data)
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
        threshold = 0.20
        voting_method = 'weighted'
        
        # Load ensemble models using the global detector (lazy loaded on first call)
        if debug:
            print("Loading ensemble models into memory...", file=sys.stderr)
        
        ensemble = get_ensemble_detector(debug=debug)
        
        if ensemble is None:
            if debug:
                print("Models are still loading. Please try again in a moment.", file=sys.stderr)
            print(json.dumps({
                "error": "Models are still loading. Please try again in a moment.",
                "status": "loading"
            }))
            return
        
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



if __name__ == "__main__":
    # Check if argument is provided via command line
    if len(sys.argv) > 1:
        path_arg = sys.argv[1]
        
       
        # Special commands
        if path_arg.lower() == 'methods':
            list_voting_methods()
        elif path_arg.lower() == 'models':
            list_available_models_example()
        else:
            main(path_arg)
    else:
        main()