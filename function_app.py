import azure.functions as func
import logging
import json
import sys
import warnings

# Suppress protobuf module cache warnings
warnings.filterwarnings('ignore')

# Workaround for Azure Functions protobuf namespace package issue
# Ensure google package is properly initialized as a namespace package
try:
    import google
    if not hasattr(google, '__path__'):
        import pkgutil
        google.__path__ = pkgutil.extend_path([], 'google')
except Exception:
    pass

# Initialize the Function App
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# ============================================================================
# HEALTH CHECK ENDPOINTS
# ============================================================================

@app.route(route="healthImageModel", methods=["GET"])
def health_check_image(req: func.HttpRequest) -> func.HttpResponse:
    """Health check endpoint for image detection (doesn't load models)"""
    logging.info('Health check endpoint for image detection called.')
    
    return func.HttpResponse(
        json.dumps({
            "status": "healthy",
            "service": "image_detection",
            "message": "Deepfake Image Detection API is running"
        }),
        status_code=200,
        mimetype="application/json"
    )


@app.route(route="healthTextModel", methods=["GET"])
def health_check_text(req: func.HttpRequest) -> func.HttpResponse:
    """Health check endpoint for text detection (doesn't load models)"""
    logging.info('Health check endpoint for text detection called.')
    
    return func.HttpResponse(
        json.dumps({
            "status": "healthy",
            "service": "text_detection",
            "message": "Deepfake Text Detection API is running"
        }),
        status_code=200,
        mimetype="application/json"
    )


# ============================================================================
# IMAGE DETECTION ENDPOINTS
# ============================================================================

 
@app.route(route="preloadImageModels", methods=["GET"])
async def preload_image_models(req: func.HttpRequest) -> func.HttpResponse:
    """Preload image detection models from blob storage to cache"""
    logging.info('Preload endpoint for image models called - downloading from blob storage.')
    
    try:
        from Agents.ImageAgents.predownload_from_blob_agent import run_preload_image_models
        
        # Run the preload workflow
        result = await run_preload_image_models()
        
        # Determine HTTP status based on result
        if result['status'] == 'success':
            status_code = 200
        elif result['status'] == 'partial':
            status_code = 207  # Multi-Status
        else:
            status_code = 500
        
        return func.HttpResponse(
            json.dumps(result),
            status_code=status_code,
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Error during image model preload: {str(e)}")
        return func.HttpResponse(
            json.dumps({
                "status": "error",
                "service": "image_detection",
                "message": f"Error preloading image models: {str(e)}",
                "total_models": 0,
                "successful": 0,
                "failed": 0
            }),
            status_code=500,
            mimetype="application/json"
        )


@app.route(route="analyzeImage", methods=["POST"])
async def analyze_image(req: func.HttpRequest) -> func.HttpResponse:
    """
    Analyze an image for deepfake detection using Agent Framework workflows.
    
    Uses concurrent workflow pattern with 5 detection agents:
    - NoEscape, SEResNet50, Prithiv V1, DimaVit, Organika SDXL
    
    Supports 14+ ensemble voting methods including:
    majority, weighted, bayesian, entropy_weighted, etc.
    """
    logging.info('Image analysis endpoint (Agent Framework) called.')

    try:
        # Import Agent Framework workflow
        from Agents import run_image_detection_workflow
        from utils.image_file_helper import load_and_save_to_temp
        import base64
        import io
        from PIL import Image
        
        # Get the uploaded image file
        image_file = req.files.get('image')
        
        if not image_file:
            return func.HttpResponse(
                json.dumps({"error": "Please provide an image file with key 'image'."}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Extract original filename
        original_filename = image_file.filename if image_file.filename else "uploaded_image"
        
    
        image_data = image_file.read()
        logging.info(f"Received image: {original_filename}, size: {len(image_data)} bytes")
        
        # Load, validate, and save the image with unique filename
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
            saved_image_path = load_and_save_to_temp(base64_image=image_base64, original_filename=original_filename)
            
            logging.info(f"Successfully loaded and saved image: {image.size}, mode: {image.mode}, path: {saved_image_path}")
            
        except Exception as img_err:
            logging.error(f"Failed to load/validate image: {img_err}")
            return func.HttpResponse(
                json.dumps({"error": f"Invalid or corrupted image file: {str(img_err)}"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Use fixed values matching original implementation
        threshold = 0.20
        voting_method = 'weighted'
        
        # Run Agent Framework concurrent workflow with file reference
        # Workflow will extract filename from path for output model
        result = await run_image_detection_workflow(
            image_path=saved_image_path,
            threshold=threshold,
            voting_method=voting_method
        )
        
        # Convert Pydantic model to JSON
        result_json = result.model_dump_json(indent=2)
        
        return func.HttpResponse(
            result_json,
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Error in Agent Framework image analysis: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": f"Analysis error: {str(e)}"}),
            status_code=500,
            mimetype="application/json"
        )


@app.route(route="getImageModels", methods=["GET"])
def get_image_models(req: func.HttpRequest) -> func.HttpResponse:
    """Get list of available image detection models"""
    logging.info('Fetching list of available image detection models.')

    try:
        # Import lightweight model registry (no heavy ML libraries loaded)
        from imagedetectionlib.model_registry import list_available_models
        
        # Get the list of available models
        models = list_available_models()
        
        # Convert to JSON
        models_json = json.dumps({
            "service": "image_detection",
            "total_models": len(models),
            "models": models
        }, indent=2)
        
        return func.HttpResponse(
            models_json,
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Error fetching image models: {str(e)}")
        return func.HttpResponse(
            f"Error fetching image models: {str(e)}",
            status_code=500
        )


# ============================================================================
# TEXT DETECTION ENDPOINTS
# ============================================================================



@app.route(route="preloadTextModels", methods=["GET"])
async def preload_text_models(req: func.HttpRequest) -> func.HttpResponse:
    """Preload text detection models from blob storage to cache"""
    logging.info('Preload endpoint for text models called - downloading from blob storage.')
    
    try:
        from Agents.TextAgents.predownload_from_blob_agent import run_preload_text_models
        
        # Run the preload workflow
        result = await run_preload_text_models()
        
        # Determine HTTP status based on result
        if result['status'] == 'success':
            status_code = 200
        elif result['status'] == 'partial':
            status_code = 207  # Multi-Status
        else:
            status_code = 500
        
        return func.HttpResponse(
            json.dumps(result),
            status_code=status_code,
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Error during text model preload: {str(e)}")
        return func.HttpResponse(
            json.dumps({
                "status": "error",
                "service": "text_detection",
                "message": f"Error preloading text models: {str(e)}",
                "total_models": 0,
                "successful": 0,
                "failed": 0
            }),
            status_code=500,
            mimetype="application/json"
        )


@app.route(route="analyzeText", methods=["POST"])
async def analyze_text(req: func.HttpRequest) -> func.HttpResponse:
    """
    Analyze text for AI-generated content detection using Agent Framework workflows.
    
    Uses concurrent workflow pattern with 4 detection agents:
    - BERT-ML, BoW-ML, CharCNN, DistilBERT
    """
    logging.info('Text analysis endpoint (Agent Framework) called.')

    try:
        # Import Agent Framework workflow
        from Agents import run_text_detection_workflow
        
        # Get request body
        try:
            req_body = req.get_json()
        except ValueError:
            return func.HttpResponse(
                json.dumps({
                    "error": "Invalid JSON",
                    "message": "Request body must be valid JSON"
                }),
                status_code=400,
                mimetype="application/json"
            )
        
        # Extract text only - no threshold parameter
        text = req_body.get('text')
        
        if not text:
            return func.HttpResponse(
                json.dumps({
                    "error": "Missing parameter",
                    "message": "Please provide 'text' parameter in request body"
                }),
                status_code=400,
                mimetype="application/json"
            )
        
        if not text.strip():
            return func.HttpResponse(
                json.dumps({
                    "error": "Invalid input",
                    "message": "Text cannot be empty"
                }),
                status_code=400,
                mimetype="application/json"
            )
        
        logging.info(f"Analyzing text (length: {len(text)} chars)")
        
        # Run Agent Framework concurrent workflow
        result = await run_text_detection_workflow(text=text)
        
        # Convert Pydantic model to JSON
        result_json = result.model_dump_json(indent=2)
        
        return func.HttpResponse(
            result_json,
            status_code=200,
            mimetype="application/json"
        )
        
    except ValueError as e:
        logging.error(f"Validation error: {str(e)}")
        return func.HttpResponse(
            json.dumps({
                "error": "Validation error",
                "message": str(e)
            }),
            status_code=400,
            mimetype="application/json"
        )
    except Exception as e:
        logging.error(f"Error in Agent Framework text analysis: {str(e)}")
        return func.HttpResponse(
            json.dumps({
                "error": "Analysis error",
                "message": f"Error analyzing text: {str(e)}"
            }),
            status_code=500,
            mimetype="application/json"
        )


@app.route(route="getTextModels", methods=["GET"])
def get_text_models(req: func.HttpRequest) -> func.HttpResponse:
    """Get list of available text detection models"""
    logging.info('Fetching list of available text detection models.')

    try:
        # Import lightweight model registry (no heavy ML libraries loaded)
        from textdetectionlib.model_registry import list_available_text_models
        
        # Get the list of available models
        models = list_available_text_models()
        
        # Convert to JSON
        models_json = json.dumps({
            "service": "text_detection",
            "total_models": len(models),
            "models": models
        }, indent=2)
        
        return func.HttpResponse(
            models_json,
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Error fetching text models: {str(e)}")
        return func.HttpResponse(
            f"Error fetching text models: {str(e)}",
            status_code=500
        )

 