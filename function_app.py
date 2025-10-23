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

# Convert tool properties to JSON
tool_properties_get_models_json = json.dumps([prop.to_dict() for prop in tool_properties_get_models])

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