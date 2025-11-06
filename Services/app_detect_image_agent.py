"""
Image Deepfake Detection Service using Agent Framework.

This service accepts an image and processes it using the Agent Framework's
concurrent workflow pattern with 5 detection agents:
- NoEscape, SEResNet50, Prithiv V1, DimaVit, Organika SDXL

Uses Logistic Regression meta-classifier (stacking ensemble) for optimal accuracy.
The meta-classifier learns to combine model predictions from labeled validation data.
Reference: https://www.geeksforgeeks.org/machine-learning/voting-classifier/

Supports 15+ ensemble voting methods including:
majority, weighted, bayesian, entropy_weighted, optimized_weighted, logistic_regression (default), etc.

Returns ImageDetectionResult in JSON format with markdown report.
"""

import sys
import os
import logging
import json
import base64
import io
from PIL import Image

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def analyze_image_with_agent(image_data, original_filename=None):
    """
    Analyze an image for deepfake detection using Agent Framework workflows.
    
    Args:
        image_data: Image data as bytes, file path, or base64 string
        original_filename: Original filename of the uploaded image (optional)
    
    Returns:
        dict: Detection results as dictionary (from Pydantic model)
    """
    try:
        # Import Agent Framework workflow
        from Agents import run_image_detection_workflow
        from utils.image_file_helper import load_and_save_to_temp
        
        logger.info('Starting image analysis with Agent Framework.')
        
        # Track if we need to clean up temp file
        temp_file_to_delete = None
        
        # Handle different input types
        if isinstance(image_data, str):
            # Check if it's a file path
            if os.path.exists(image_data):
                # IMPORTANT: Copy the image to temp folder to avoid deleting original
                logger.info(f"Copying image from path: {image_data}")
                
                # Read the image and convert to base64
                image = Image.open(image_data)
                image.load()
                
                # Ensure RGB format
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Save to base64
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                # Save to temp folder using helper - this creates a NEW file
                saved_image_path = load_and_save_to_temp(
                    base64_image=image_base64,
                    original_filename=original_filename or os.path.basename(image_data)
                )
                temp_file_to_delete = saved_image_path
                logger.info(f"Image copied to temp folder: {saved_image_path}")
                logger.info(f"Original file preserved at: {image_data}")
            else:
                # Assume it's base64 encoded
                logger.info("Processing base64 encoded image")
                saved_image_path = load_and_save_to_temp(
                    base64_image=image_data,
                    original_filename=original_filename or "uploaded_image"
                )
                temp_file_to_delete = saved_image_path
        elif isinstance(image_data, bytes):
            # Convert bytes to PIL Image
            logger.info(f"Processing image bytes, size: {len(image_data)} bytes")
            
            try:
                image = Image.open(io.BytesIO(image_data))
                image.load()  # Force loading the image data to detect truncation early
                
                # Ensure RGB format
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Save to base64 temporarily for the helper function
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                # Save image locally using helper - helper will generate unique filename with datetime and GUID
                saved_image_path = load_and_save_to_temp(
                    base64_image=image_base64,
                    original_filename=original_filename or "uploaded_image"
                )
                
                logger.info(f"Successfully loaded and saved image: {image.size}, mode: {image.mode}, path: {saved_image_path}")
                temp_file_to_delete = saved_image_path
                
            except Exception as img_err:
                logger.error(f"Failed to load/validate image: {img_err}")
                raise ValueError(f"Invalid or corrupted image file: {str(img_err)}")
        else:
            raise ValueError("image_data must be bytes, file path string, or base64 string")
        
        # Run Agent Framework concurrent workflow with file reference
        # Workflow will extract filename from path for output model
        # NOTE: Logistic Regression requires 100+ training samples for reliable weights
        # Currently trained on only 17 samples (76.5% accuracy) - insufficient for production
        # Using optimized_weighted (Bayesian soft voting) until more training data collected
        threshold = 0.20
        voting_method = 'optimized_weighted'  # Bayesian-optimized soft voting
        
        logger.info(f"Running detection with threshold={threshold}, voting_method={voting_method}")
        
        # Workflow will handle cleanup of temp file (cleanup_file=True)
        result = await run_image_detection_workflow(
            image_path=saved_image_path,
            threshold=threshold,
            voting_method=voting_method
        )
        
        # Convert Pydantic model to dictionary
        result_dict = result.model_dump()
        
        logger.info(f"Detection complete. Prediction: {result_dict.get('prediction', 'N/A')}")
        return result_dict
        
    except Exception as e:
        logger.error(f"Error in Agent Framework image analysis: {str(e)}")
        raise


def analyze_image_sync(image_data, original_filename=None):
    """
    Synchronous wrapper for analyze_image_with_agent.
    
    Args:
        image_data: Image data as bytes, file path, or base64 string
        original_filename: Original filename of the uploaded image (optional)
    
    Returns:
        dict: Detection results as dictionary
    """
    import asyncio
    
    try:
        # Create and run event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                analyze_image_with_agent(image_data, original_filename)
            )
            return result
        finally:
            loop.close()
    except Exception as e:
        logger.error(f"Error in synchronous wrapper: {str(e)}")
        raise


def main():
    """
    Main function for command-line usage.
    
    Usage:
        python app_detect_image_agent.py <image_path>
        python app_detect_image_agent.py <image_path> --output results.json
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze image for deepfake detection using Agent Framework')
    parser.add_argument('image_path', help='Path to image file')
    parser.add_argument('--output', type=str, help='Output JSON file path (optional)')
    
    args = parser.parse_args()
    
    # Validate image path
    if not os.path.exists(args.image_path):
        print(json.dumps({"error": f"Image file not found: {args.image_path}"}))
        sys.exit(1)
    
    try:
        # Run analysis
        print(f"Analyzing image: {args.image_path}", file=sys.stderr)
        print(f"Using settings: threshold=0.20, voting=optimized_weighted (Bayesian-optimized)", file=sys.stderr)
        
        result = analyze_image_sync(
            image_data=args.image_path,
            original_filename=os.path.basename(args.image_path)
        )
        
        # Output results
        result_json = json.dumps(result, indent=2)
        
        if args.output:
            # Write to file
            with open(args.output, 'w') as f:
                f.write(result_json)
            print(f"Results written to: {args.output}", file=sys.stderr)
        else:
            # Print to stdout
            print(result_json)
        
        print("\nAnalysis complete!", file=sys.stderr)
        
    except Exception as e:
        print(json.dumps({"error": f"Analysis failed: {str(e)}"}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
