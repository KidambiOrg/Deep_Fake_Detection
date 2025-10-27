"""
MCP (Model Context Protocol) Endpoints for Deepfake Image Detection
This module contains MCP tool trigger endpoints for image-based deepfake detection.
"""
import azure.functions as func
import logging
import json


def register_mcp_image_endpoints(app: func.FunctionApp):
    """
    Register all MCP image detection endpoints with the provided FunctionApp instance.
    
    Args:
        app: The Azure Functions FunctionApp instance to register endpoints with.
    """
    
    # Tool properties for get_models MCP tool
    tool_properties_get_models = []
    
    # Tool properties for analyze_image MCP tool
    tool_properties_analyze_image = [
        {
            "propertyName": "image_url",
            "propertyType": "string",
            "description": "Optional: URL of the image to analyze for deepfake detection"
        },
        {
            "propertyName": "image_data",
            "propertyType": "string",
            "description": "Optional: Base64-encoded image data. For local files, read the file and encode to base64 first"
        },
        {
            "propertyName": "image_path",
            "propertyType": "string",
            "description": "Optional: Local file path (will be read and converted to base64 automatically)"
        }
    ]
    
    tool_properties_analyze_image_json = json.dumps(tool_properties_analyze_image)
    
    @app.generic_trigger(
        arg_name="context",
        type="mcpToolTrigger",
        toolName="get_image_detection_models",
        description="Get a list of available image deepfake detection models with filtering options.",
        toolProperties="[]",
    )
    def get_image_detection_models_mcp(context) -> str:
        """
        MCP tool endpoint to retrieve available image deepfake detection models.
        Supports filtering by source and architecture.
        
        Args:
            context: The MCP tool trigger context containing input arguments.
            
        Returns:
            str: JSON string with the list of available models or filtered results.
        """
        logging.info('MCP tool get_image_detection_models called.')
        
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
            logging.error(f"Error in MCP get_image_detection_models: {str(e)}")
            error_result = {
                "error": str(e),
                "message": "Failed to retrieve image deepfake detection models"
            }
            return json.dumps(error_result)
    
    
    @app.generic_trigger(
        arg_name="context",
        type="mcpToolTrigger",
        toolName="check_image_models_status",
        description="Check if the image deepfake detection models are loaded and ready for analysis.",
        toolProperties="[]",
    )
    def check_image_models_status_mcp(context) -> str:
        """
        MCP tool endpoint to check if image models are loaded and ready.
        Useful to call before analyze_image_deepfake to avoid timeouts.
        
        Args:
            context: The MCP tool trigger context (not used).
            
        Returns:
            str: JSON string with model status information.
        """
        logging.info('MCP tool check_image_models_status called.')
        
        try:
            from Services.app_detect_image import get_ensemble_detector
            
            # Check if ensemble is loaded without waiting
            try:
                ensemble = get_ensemble_detector(debug=False, timeout=1)
                if ensemble is not None:
                    return json.dumps({
                        "status": "ready",
                        "message": "Image models are loaded and ready for image analysis",
                        "ready": True
                    })
            except:
                pass
            
            return json.dumps({
                "status": "loading",
                "message": "Image models are still loading. First-time initialization takes 2-5 minutes.",
                "ready": False,
                "tip": "Call the warmup endpoint (/v1.0/warmupImageModels) or wait a few minutes before analyzing images."
            })
            
        except Exception as e:
            logging.error(f"Error checking image model status: {str(e)}")
            return json.dumps({
                "status": "error",
                "message": f"Error checking image model status: {str(e)}",
                "ready": False
            })
    
    
    @app.generic_trigger(
        arg_name="context",
        type="mcpToolTrigger",
        toolName="analyze_image_deepfake",
        description="Analyze an image to detect if it's a deepfake or AI-generated. Accepts image URL, local file path, or base64-encoded data.",
        toolProperties=tool_properties_analyze_image_json,
    )
    def analyze_image_deepfake_mcp(context) -> str:
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
        logging.info('MCP tool analyze_image_deepfake called.')
        
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
                "analysis_timestamp": "2025-10-27"
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logging.error(f"Error in MCP analyze_image_deepfake: {str(e)}")
            error_result = {
                "error": str(e),
                "message": "Failed to analyze image for deepfake detection"
            }
            return json.dumps(error_result)
