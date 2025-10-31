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
        from Services.app_detect_text import get_text_ensemble
        
        # Try to load the text ensemble
        ensemble = get_text_ensemble(debug=True, timeout=600)
        
        if ensemble is not None:
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
        from Services.app_detect_text import detect_text_json
        
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
        
        # Extract text and voting method
        text = req_body.get('text')
        voting_method = req_body.get('voting_method', 'weighted')
        
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
        
        # Analyze the text using module-level function
        result_json = detect_text_json(text, voting_method=voting_method, indent=2)
        
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
# MCP TOOL TRIGGER ENDPOINTS - IMAGE DETECTION
# ============================================================================

# Tool properties for image detection MCP tools
_tool_properties_analyze_image = json.dumps([
    {
        "propertyName": "image_url",
        "propertyType": "string",
        "description": "Optional: URL of the image to analyze for deepfake detection",
        "isRequired": False
    },
    {
        "propertyName": "image_data",
        "propertyType": "string",
        "description": "Optional: Base64-encoded image data. For local files, read the file and encode to base64 first",
        "isRequired": False
    },
    {
        "propertyName": "image_path",
        "propertyType": "string",
        "description": "Optional: Local file path (will be read and converted to base64 automatically)",
        "isRequired": False
    }
])


@app.mcp_tool_trigger(
    arg_name="context",
    tool_name="get_image_detection_models",
    description="Get a list of available image deepfake detection models with filtering options.",
    tool_properties="[]",
)
def get_image_detection_models_mcp(context) -> str:
    """MCP tool endpoint to retrieve available image deepfake detection models."""
    logging.info('MCP tool get_image_detection_models called.')
    
    try:
        from imagedetectionlib.model_registry import (
            list_available_models,
            get_models_by_source,
            get_models_by_architecture
        )
        
        if context:
            content = json.loads(context)
            arguments = content.get("arguments", {})
            
            filter_by_source = arguments.get("filter_by_source")
            filter_by_architecture = arguments.get("filter_by_architecture")
            
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
        
        result = {
            "total_models": len(models),
            "models": models
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logging.error(f"Error in MCP get_image_detection_models: {str(e)}")
        return json.dumps({
            "error": str(e),
            "message": "Failed to retrieve image deepfake detection models"
        })


@app.mcp_tool_trigger(
    arg_name="context",
    tool_name="check_image_models_status",
    description="Check if the image deepfake detection models are loaded and ready for analysis.",
    tool_properties="[]",
)
def check_image_models_status_mcp(context) -> str:
    """MCP tool endpoint to check if image models are loaded and ready."""
    logging.info('MCP tool check_image_models_status called.')
    
    try:
        from Services.app_detect_image import get_ensemble_detector
        
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


@app.mcp_tool_trigger(
    arg_name="context",
    tool_name="analyze_image_deepfake",
    description="Analyze an image to detect if it's a deepfake or AI-generated. Accepts image URL, local file path, or base64-encoded data.",
    tool_properties=_tool_properties_analyze_image,
)
def analyze_image_deepfake_mcp(context) -> str:
    """MCP tool endpoint to analyze an image for deepfake detection."""
    logging.info('MCP tool analyze_image_deepfake called.')
    
    try:
        import requests
        import base64
        
        if not context:
            return json.dumps({
                "error": "No context provided",
                "message": "Please provide either image_url, image_data, or image_path parameter"
            })
        
        content = json.loads(context)
        arguments = content.get("arguments", {})
        
        image_url = arguments.get("image_url")
        image_data_b64 = arguments.get("image_data")
        image_path = arguments.get("image_path")
        
        threshold = 0.20
        voting_method = "median"
        
        if not image_url and not image_data_b64 and not image_path:
            return json.dumps({
                "error": "Missing required parameter",
                "message": "Please provide either image_url, image_data, or image_path parameter"
            })
        
        image_data = None
        image_source = None
        
        if image_path:
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
            try:
                logging.info("Processing base64-encoded image data")
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
        
        from Services.app_detect_image import process_image, get_ensemble_detector
        
        try:
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
        
        result_json = process_image(image_data)
        result = json.loads(result_json)
        
        result["mcp_metadata"] = {
            "image_source": image_source,
            "threshold": threshold,
            "voting_method": voting_method,
            "analysis_timestamp": "2025-10-30"
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logging.error(f"Error in MCP analyze_image_deepfake: {str(e)}")
        return json.dumps({
            "error": str(e),
            "message": "Failed to analyze image for deepfake detection"
        })


# ============================================================================
# MCP TOOL TRIGGER ENDPOINTS - TEXT DETECTION
# ============================================================================

# Tool properties for text detection MCP tools
_tool_properties_analyze_text = json.dumps([
    {
        "propertyName": "text",
        "propertyType": "string",
        "description": "The text to analyze for AI-generated content detection",
        "isRequired": True
    }
])


@app.mcp_tool_trigger(
    arg_name="context",
    tool_name="get_text_detection_models",
    description="Get a list of available text detection models with filtering options.",
    tool_properties="[]",
)
def get_text_detection_models_mcp(context) -> str:
    """MCP tool endpoint to retrieve available text detection models."""
    logging.info('MCP tool get_text_detection_models called.')
    
    try:
        from textdetectionlib.model_registry import (
            list_available_text_models,
            get_text_models_by_architecture
        )
        
        models = None
        if context:
            content = json.loads(context)
            arguments = content.get("arguments", {})
            
            filter_by_architecture = arguments.get("filter_by_architecture")
            
            if filter_by_architecture:
                logging.info(f"Filtering models by architecture: {filter_by_architecture}")
                models = get_text_models_by_architecture(filter_by_architecture)
            else:
                models = list_available_text_models()
        else:
            models = list_available_text_models()
        
        result = {
            "total_models": len(models),
            "models": models
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logging.error(f"Error in MCP get_text_detection_models: {str(e)}")
        return json.dumps({
            "error": str(e),
            "message": "Failed to retrieve text detection models"
        })


@app.mcp_tool_trigger(
    arg_name="context",
    tool_name="check_text_models_status",
    description="Check if the text detection models are loaded and ready for analysis.",
    tool_properties="[]",
)
def check_text_models_status_mcp(context) -> str:
    """MCP tool endpoint to check if text models are loaded and ready."""
    logging.info('MCP tool check_text_models_status called.')
    
    try:
        from Services.app_detect_text import get_text_ensemble
        
        ensemble = get_text_ensemble()
        
        if ensemble is not None:
            return json.dumps({
                "status": "ready",
                "message": "Text detection models are loaded and ready for analysis",
                "ready": True
            })
        else:
            return json.dumps({
                "status": "loading",
                "message": "Text models are still loading. First-time initialization takes 1-3 minutes.",
                "ready": False,
                "tip": "Call the warmup endpoint (/v1.0/warmupTextModels) or wait a few minutes before analyzing text."
            })
        
    except Exception as e:
        logging.error(f"Error checking text model status: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": f"Error checking text model status: {str(e)}",
            "ready": False
        })


@app.mcp_tool_trigger(
    arg_name="context",
    tool_name="analyze_text_deepfake",
    description="Analyze text to detect if it's AI-generated or human-written using an ensemble of ML models.",
    tool_properties=_tool_properties_analyze_text,
)
def analyze_text_deepfake_mcp(context) -> str:
    """MCP tool endpoint to analyze text for AI-generated content detection."""
    logging.info('MCP tool analyze_text_deepfake called.')
    
    try:
        if not context:
            return json.dumps({
                "error": "No context provided",
                "message": "Please provide the 'text' parameter"
            })
        
        content = json.loads(context)
        arguments = content.get("arguments", {})
        
        text = arguments.get("text")
        voting_method = "weighted"
        
        if not text:
            return json.dumps({
                "error": "Missing required parameter",
                "message": "Please provide the 'text' parameter"
            })
        
        if not text.strip():
            return json.dumps({
                "error": "Invalid input",
                "message": "Text cannot be empty"
            })
        
        logging.info(f"Analyzing text (length: {len(text)} chars) with voting method: {voting_method}")
        
        from Services.app_detect_text import detect_text_json
        
        result_json = detect_text_json(text, voting_method=voting_method, indent=2)
        result = json.loads(result_json)
        
        result["mcp_metadata"] = {
            "text_length": len(text),
            "voting_method": voting_method,
            "analysis_timestamp": "2025-10-30"
        }
        
        return json.dumps(result, indent=2)
        
    except ValueError as e:
        logging.error(f"Validation error in MCP analyze_text_deepfake: {str(e)}")
        return json.dumps({
            "error": "Validation error",
            "message": str(e)
        })
    except Exception as e:
        logging.error(f"Error in MCP analyze_text_deepfake: {str(e)}")
        return json.dumps({
            "error": str(e),
            "message": "Failed to analyze text for AI-generated content detection"
        })
