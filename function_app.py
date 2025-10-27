import azure.functions as func
import logging
import json

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

@app.route(route="warmupImageModels", methods=["GET"])
def warmup_image_models(req: func.HttpRequest) -> func.HttpResponse:
    """Warm up endpoint to pre-load image detection models"""
    logging.info('Warmup endpoint for image models called - loading models.')
    
    try:
        from Services.app_detect_image import get_ensemble_detector
        
        # Try to load the ensemble
        ensemble = get_ensemble_detector(debug=True, timeout=600)
        
        if ensemble is not None:
            return func.HttpResponse(
                json.dumps({
                    "status": "ready",
                    "service": "image_detection",
                    "message": "Image detection models loaded successfully and ready for analysis"
                }),
                status_code=200,
                mimetype="application/json"
            )
        else:
            return func.HttpResponse(
                json.dumps({
                    "status": "loading",
                    "service": "image_detection",
                    "message": "Image detection models are still loading"
                }),
                status_code=202,
                mimetype="application/json"
            )
    except Exception as e:
        logging.error(f"Error during image model warmup: {str(e)}")
        return func.HttpResponse(
            json.dumps({
                "status": "error",
                "service": "image_detection",
                "message": f"Error loading image models: {str(e)}"
            }),
            status_code=500,
            mimetype="application/json"
        )


@app.route(route="analyzeImage", methods=["POST"])
def analyze_image(req: func.HttpRequest) -> func.HttpResponse:
    """Analyze an image for deepfake detection"""
    logging.info('Image analysis endpoint called.')

    try:
        # Lazy import to avoid loading models at startup
        from Services.app_detect_image import process_image
        
        # Get the uploaded image file
        image_file = req.files.get('image')
        
        if not image_file:
            return func.HttpResponse(
                "Please provide an image file in the request body with key 'image'.",
                status_code=400
            )
        
        # Read the image data
        image_data = image_file.read()
        
        logging.info(f"Received image file: {image_file.filename}, size: {len(image_data)} bytes")
        
        # Call the process_image function with filename
        result_json = process_image(image_data, filename=image_file.filename)
        
        return func.HttpResponse(
            result_json,
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        return func.HttpResponse(
            f"Error processing image: {str(e)}",
            status_code=500
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

@app.route(route="warmupTextModels", methods=["GET"])
def warmup_text_models(req: func.HttpRequest) -> func.HttpResponse:
    """Warm up endpoint to pre-load text detection models"""
    logging.info('Warmup endpoint for text models called - loading models.')
    
    try:
        from Services.app_detect_text import TextDetectionApp
        
        # Try to initialize the text detection app
        app_instance = TextDetectionApp(use_blob_storage=True)
        
        if app_instance.ensemble is not None:
            return func.HttpResponse(
                json.dumps({
                    "status": "ready",
                    "service": "text_detection",
                    "message": "Text detection models loaded successfully and ready for analysis"
                }),
                status_code=200,
                mimetype="application/json"
            )
        else:
            return func.HttpResponse(
                json.dumps({
                    "status": "loading",
                    "service": "text_detection",
                    "message": "Text detection models are still loading"
                }),
                status_code=202,
                mimetype="application/json"
            )
    except Exception as e:
        logging.error(f"Error during text model warmup: {str(e)}")
        return func.HttpResponse(
            json.dumps({
                "status": "error",
                "service": "text_detection",
                "message": f"Error loading text models: {str(e)}"
            }),
            status_code=500,
            mimetype="application/json"
        )


@app.route(route="analyzeText", methods=["POST"])
def analyze_text(req: func.HttpRequest) -> func.HttpResponse:
    """Analyze text for AI-generated content detection"""
    logging.info('Text analysis endpoint called.')

    try:
        # Lazy import to avoid loading models at startup
        from Services.app_detect_text import TextDetectionApp
        
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
        
        # Extract text only
        text = req_body.get('text')
        
        # Use default voting method
        voting_method = 'weighted'
        
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
        
        logging.info(f"Analyzing text (length: {len(text)} chars) with voting method: {voting_method}")
        
        # Initialize the text detection app
        app_instance = TextDetectionApp(voting_method=voting_method, use_blob_storage=True)
        
        # Analyze the text
        result_json = app_instance.detect_text_json(text, indent=2)
        
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
        logging.error(f"Error analyzing text: {str(e)}")
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

# ============================================================================
# REGISTER MCP ENDPOINTS
# ============================================================================

# Register MCP image detection endpoints
from MCP.mcp_deepfake_image import register_mcp_image_endpoints
register_mcp_image_endpoints(app)

# Register MCP text detection endpoints
from MCP.mcp_deepfake_text import register_mcp_text_endpoints
register_mcp_text_endpoints(app)
