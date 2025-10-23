import azure.functions as func
import logging
import json

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# Tool properties for MCP endpoints
class ToolProperty:
    def __init__(self, property_name: str, property_type: str, description: str):
        self.propertyName = property_name
        self.propertyType = property_type
        self.description = description
    
    def to_dict(self):
        return {
            "propertyName": self.propertyName,
            "propertyType": self.propertyType,
            "description": self.description,
        }

# Define tool properties for get_models MCP tool
tool_properties_get_models = [
    ToolProperty("filter_by_source", "string", "Optional: Filter models by source (e.g., 'huggingface', 'timm', 'torchvision')"),
    ToolProperty("filter_by_architecture", "string", "Optional: Filter models by architecture (e.g., 'EfficientNet', 'ViT', 'Xception')")
]

# Define tool properties for analyze_image MCP tool
tool_properties_analyze_image = [
    ToolProperty("image_url", "string", "Optional: URL of the image to analyze for deepfake detection"),
    ToolProperty("image_data", "string", "Optional: Base64-encoded image data. For local files, read the file and encode to base64 first"),
    ToolProperty("image_path", "string", "Optional: Local file path (will be read and converted to base64 automatically)")
]

# Convert tool properties to JSON
tool_properties_get_models_json = json.dumps([prop.to_dict() for prop in tool_properties_get_models])
tool_properties_analyze_image_json = json.dumps([prop.to_dict() for prop in tool_properties_analyze_image])

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

@app.route(route="analzeImage", methods=["POST"])
def analzeImage(req: func.HttpRequest) -> func.HttpResponse:
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
        
        # Call the process_image function
        result_json = process_image(image_data)
        
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

@app.generic_trigger(
    arg_name="context",
    type="mcpToolTrigger",
    toolName="get_deepfake_models",
    description="Get a list of available deepfake detection models with filtering options.",
    toolProperties="[]",
)
def get_deepfake_models_mcp(context) -> str:
    """
    MCP tool endpoint to retrieve available deepfake detection models.
    Supports filtering by source and architecture.
    
    Args:
        context: The MCP tool trigger context containing input arguments.
        
    Returns:
        str: JSON string with the list of available models or filtered results.
    """
    logging.info('MCP tool get_deepfake_models called.')
    
    try:
        # Import model registry functions
        from imagedetectionlib.model_registry import (
            list_available_models,
            get_models_by_source,
            get_models_by_architecture
        )
        
        # Parse the context to get arguments
        if context:
            content = json.loads(context)
            arguments = content.get("arguments", {})
            
            filter_by_source = arguments.get("filter_by_source")
            filter_by_architecture = arguments.get("filter_by_architecture")
            
            # Apply filters if provided
            if filter_by_source:
                logging.info(f"Filtering models by source: {filter_by_source}")
                models = get_models_by_source(filter_by_source)
            elif filter_by_architecture:
                logging.info(f"Filtering models by architecture: {filter_by_architecture}")
                models = get_models_by_architecture(filter_by_architecture)
            else:
                models = list_available_models()
        else:
            models = list_available_models()
        
        # Convert to formatted JSON string
        result = {
            "total_models": len(models),
            "models": models
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logging.error(f"Error in MCP get_deepfake_models: {str(e)}")
        error_result = {
            "error": str(e),
            "message": "Failed to retrieve deepfake detection models"
        }
        return json.dumps(error_result)


@app.generic_trigger(
    arg_name="context",
    type="mcpToolTrigger",
    toolName="check_models_status",
    description="Check if the deepfake detection models are loaded and ready for analysis.",
    toolProperties="[]",
)
def check_models_status_mcp(context) -> str:
    """
    MCP tool endpoint to check if models are loaded and ready.
    Useful to call before analyze_deepfake_image to avoid timeouts.
    
    Args:
        context: The MCP tool trigger context (not used).
        
    Returns:
        str: JSON string with model status information.
    """
    logging.info('MCP tool check_models_status called.')
    
    try:
        from Services.app_detect_image import get_ensemble_detector
        
        # Check if ensemble is loaded without waiting
        try:
            ensemble = get_ensemble_detector(debug=False, timeout=1)
            if ensemble is not None:
                return json.dumps({
                    "status": "ready",
                    "message": "Models are loaded and ready for image analysis",
                    "ready": True
                })
        except:
            pass
        
        return json.dumps({
            "status": "loading",
            "message": "Models are still loading. First-time initialization takes 2-5 minutes.",
            "ready": False,
            "tip": "Call the warmup endpoint (/v1.0/warmup) or wait a few minutes before analyzing images."
        })
        
    except Exception as e:
        logging.error(f"Error checking model status: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": f"Error checking model status: {str(e)}",
            "ready": False
        })


@app.generic_trigger(
    arg_name="context",
    type="mcpToolTrigger",
    toolName="analyze_deepfake_image",
    description="Analyze an image to detect if it's a deepfake or AI-generated. Accepts image URL, local file path, or base64-encoded data.",
    toolProperties=tool_properties_analyze_image_json,
)
def analyze_deepfake_image_mcp(context) -> str:
    """
    MCP tool endpoint to analyze an image for deepfake detection.
    Accepts URL, local file path, or base64-encoded image data.
    Uses default threshold (0.20) and voting method (median).
    
    Args:
        context: The MCP tool trigger context containing input arguments:
                 - image_url (optional): URL of the image to analyze
                 - image_path (optional): Local file path to the image
                 - image_data (optional): Base64-encoded image data
        
    Returns:
        str: JSON string with detection results including prediction, confidence, and individual model results.
    """
    logging.info('MCP tool analyze_deepfake_image called.')
    
    try:
        import requests
        import base64
        from io import BytesIO
        
        # Parse the context to get arguments
        if not context:
            return json.dumps({
                "error": "No context provided",
                "message": "Please provide either image_url or image_data parameter"
            })
        
        content = json.loads(context)
        arguments = content.get("arguments", {})
        
        image_url = arguments.get("image_url")
        image_data_b64 = arguments.get("image_data")
        image_path = arguments.get("image_path")
        
        # Use fixed default values
        threshold = 0.20
        voting_method = "median"
        
        # Validate that at least one image source is provided
        if not image_url and not image_data_b64 and not image_path:
            return json.dumps({
                "error": "Missing required parameter",
                "message": "Please provide either image_url, image_data, or image_path parameter"
            })
        
        # Get image data from the appropriate source
        image_data = None
        image_source = None
        
        if image_path:
            # Read local file and convert to bytes
            try:
                import os
                logging.info(f"Reading local file: {image_path}")
                
                if not os.path.exists(image_path):
                    return json.dumps({
                        "error": "File not found",
                        "message": f"The specified file does not exist: {image_path}"
                    })
                
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                
                image_source = f"local_file:{os.path.basename(image_path)}"
                logging.info(f"Read local file: {len(image_data)} bytes")
            except Exception as e:
                return json.dumps({
                    "error": f"Failed to read local file: {str(e)}",
                    "message": "Unable to read the specified file"
                })
        
        elif image_data_b64:
            # Decode base64 image data
            try:
                logging.info("Processing base64-encoded image data")
                # Remove data URL prefix if present (e.g., "data:image/png;base64,")
                if ',' in image_data_b64 and image_data_b64.startswith('data:'):
                    image_data_b64 = image_data_b64.split(',', 1)[1]
                
                image_data = base64.b64decode(image_data_b64)
                image_source = "base64_data"
                logging.info(f"Decoded base64 image: {len(image_data)} bytes")
            except Exception as e:
                return json.dumps({
                    "error": f"Failed to decode base64 image data: {str(e)}",
                    "message": "Invalid base64 image data provided"
                })
        
        elif image_url:
            # Download the image from URL
            try:
                logging.info(f"Downloading image from URL: {image_url}")
                response = requests.get(image_url, timeout=30)
                response.raise_for_status()
                image_data = response.content
                image_source = image_url
                logging.info(f"Downloaded image: {len(image_data)} bytes")
            except requests.RequestException as e:
                return json.dumps({
                    "error": f"Failed to download image: {str(e)}",
                    "message": "Unable to fetch image from the provided URL"
                })
        
        logging.info(f"Using default threshold: {threshold}, voting method: {voting_method}")
        
        # Lazy import to avoid loading models at startup
        from Services.app_detect_image import process_image, get_ensemble_detector
        
        # Check if ensemble is already loaded
        try:
            # Try to get the ensemble with a very short timeout to check if it's ready
            ensemble = get_ensemble_detector(debug=False, timeout=5)
            if ensemble is None:
                return json.dumps({
                    "status": "loading",
                    "message": "Models are still loading. This is the first request and model initialization takes 2-5 minutes. Please try again in a few moments.",
                    "estimated_wait_time": "2-5 minutes",
                    "tip": "Subsequent requests will be much faster once models are loaded."
                })
        except Exception as e:
            logging.warning(f"Models not ready yet: {str(e)}")
            return json.dumps({
                "status": "loading",
                "message": "Models are being initialized. This happens on first request and takes 2-5 minutes. Please try again shortly.",
                "estimated_wait_time": "2-5 minutes"
            })
        
        # Process the image
        result_json = process_image(image_data)
        
        # Parse and enhance the result
        result = json.loads(result_json)
        
        # Add metadata
        result["mcp_metadata"] = {
            "image_source": image_source,
            "threshold": threshold,
            "voting_method": voting_method,
            "analysis_timestamp": "2025-10-23"
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logging.error(f"Error in MCP analyze_deepfake_image: {str(e)}")
        error_result = {
            "error": str(e),
            "message": "Failed to analyze image for deepfake detection"
        }
        return json.dumps(error_result)