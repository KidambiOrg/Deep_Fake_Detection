import azure.functions as func
import logging
import json

# Initialize the Function App
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="health", methods=["GET"])
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """Simple health check endpoint that doesn't load models"""
    logging.info('Health check endpoint called.')
    
    return func.HttpResponse(
        json.dumps({
            "status": "healthy",
            "message": "Deepfake Detection API is running"
        }),
        status_code=200,
        mimetype="application/json"
    )

@app.route(route="warmup", methods=["GET", "POST"])
def warmup(req: func.HttpRequest) -> func.HttpResponse:
    """Warm up endpoint to pre-load models"""
    logging.info('Warmup endpoint called - loading models.')
    
    try:
        from Services.app_detect_image import get_ensemble_detector
        
        # Try to load the ensemble
        ensemble = get_ensemble_detector(debug=True, timeout=600)
        
        if ensemble is not None:
            return func.HttpResponse(
                json.dumps({
                    "status": "ready",
                    "message": "Models loaded successfully and ready for analysis"
                }),
                status_code=200,
                mimetype="application/json"
            )
        else:
            return func.HttpResponse(
                json.dumps({
                    "status": "loading",
                    "message": "Models are still loading"
                }),
                status_code=202,
                mimetype="application/json"
            )
    except Exception as e:
        logging.error(f"Error during warmup: {str(e)}")
        return func.HttpResponse(
            json.dumps({
                "status": "error",
                "message": f"Error loading models: {str(e)}"
            }),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="analyzeImage", methods=["POST"])
def analyzeImage(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

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




@app.route(route="models", methods=["GET"])
def get_models(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Fetching list of available models.')

    try:
        # Import lightweight model registry (no heavy ML libraries loaded)
        from imagedetectionlib.model_registry import list_available_models
        
        # Get the list of available models
        models = list_available_models()
        
        # Convert to JSON
        models_json = json.dumps(models, indent=2)
        
        return func.HttpResponse(
            models_json,
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Error fetching models: {str(e)}")
        return func.HttpResponse(
            f"Error fetching models: {str(e)}",
            status_code=500
        )


# Register MCP image detection endpoints
from MCP.mcp_deepfake_image import register_mcp_image_endpoints
register_mcp_image_endpoints(app)
