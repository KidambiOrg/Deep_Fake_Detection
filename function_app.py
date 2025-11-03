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

# # ============================================================================
# # MCP TOOL TRIGGER ENDPOINTS - IMAGE DETECTION
# # ============================================================================

# # Tool properties for image detection MCP tools
# _tool_properties_analyze_image = json.dumps([
#     {
#         "propertyName": "image_url",
#         "propertyType": "string",
#         "description": "Optional: URL of the image to analyze for deepfake detection",
#         "isRequired": False
#     },
#     {
#         "propertyName": "image_data",
#         "propertyType": "string",
#         "description": "Optional: Base64-encoded image data. For local files, read the file and encode to base64 first",
#         "isRequired": False
#     },
#     {
#         "propertyName": "image_path",
#         "propertyType": "string",
#         "description": "Optional: Local file path (will be read and converted to base64 automatically)",
#         "isRequired": False
#     }
# ])


# @app.mcp_tool_trigger(
#     arg_name="context",
#     tool_name="get_image_detection_models",
#     description="Get a list of available image deepfake detection models with filtering options.",
#     tool_properties="[]",
# )
# def get_image_detection_models_mcp(context) -> str:
#     """MCP tool endpoint to retrieve available image deepfake detection models."""
#     logging.info('MCP tool get_image_detection_models called.')
    
#     try:
#         from imagedetectionlib.model_registry import (
#             list_available_models,
#             get_models_by_source,
#             get_models_by_architecture
#         )
        
#         if context:
#             content = json.loads(context)
#             arguments = content.get("arguments", {})
            
#             filter_by_source = arguments.get("filter_by_source")
#             filter_by_architecture = arguments.get("filter_by_architecture")
            
#             if filter_by_source:
#                 logging.info(f"Filtering models by source: {filter_by_source}")
#                 models = get_models_by_source(filter_by_source)
#             elif filter_by_architecture:
#                 logging.info(f"Filtering models by architecture: {filter_by_architecture}")
#                 models = get_models_by_architecture(filter_by_architecture)
#             else:
#                 models = list_available_models()
#         else:
#             models = list_available_models()
        
#         result = {
#             "total_models": len(models),
#             "models": models
#         }
        
#         return json.dumps(result, indent=2)
        
#     except Exception as e:
#         logging.error(f"Error in MCP get_image_detection_models: {str(e)}")
#         return json.dumps({
#             "error": str(e),
#             "message": "Failed to retrieve image deepfake detection models"
#         })


# @app.mcp_tool_trigger(
#     arg_name="context",
#     tool_name="check_image_models_status",
#     description="Check if the image deepfake detection models are loaded and ready for analysis.",
#     tool_properties="[]",
# )
# def check_image_models_status_mcp(context) -> str:
#     """MCP tool endpoint to check if image models are loaded and ready."""
#     logging.info('MCP tool check_image_models_status called.')
    
#     try:
#         from Services.app_detect_image import get_ensemble_detector
        
#         try:
#             ensemble = get_ensemble_detector(debug=False, timeout=1)
#             if ensemble is not None:
#                 return json.dumps({
#                     "status": "ready",
#                     "message": "Image models are loaded and ready for image analysis",
#                     "ready": True
#                 })
#         except:
#             pass
        
#         return json.dumps({
#             "status": "loading",
#             "message": "Image models are still loading. First-time initialization takes 2-5 minutes.",
#             "ready": False,
#             "tip": "Call the warmup endpoint (/v1.0/warmupImageModels) or wait a few minutes before analyzing images."
#         })
        
#     except Exception as e:
#         logging.error(f"Error checking image model status: {str(e)}")
#         return json.dumps({
#             "status": "error",
#             "message": f"Error checking image model status: {str(e)}",
#             "ready": False
#         })


# @app.mcp_tool_trigger(
#     arg_name="context",
#     tool_name="analyze_image_deepfake",
#     description="Analyze an image to detect if it's a deepfake or AI-generated. Accepts image URL, local file path, or base64-encoded data. Uses Agent Framework with concurrent workflow.",
#     tool_properties=_tool_properties_analyze_image,
# )
# async def analyze_image_deepfake_mcp(context) -> str:
#     """MCP tool endpoint to analyze an image for deepfake detection using Agent Framework."""
#     logging.info('MCP tool analyze_image_deepfake (Agent Framework) called.')
    
#     try:
#         import requests
#         import base64
#         import asyncio
#         import io
        
#         if not context:
#             return json.dumps({
#                 "error": "No context provided",
#                 "message": "Please provide either image_url, image_data, or image_path parameter"
#             })
        
#         content = json.loads(context)
#         arguments = content.get("arguments", {})
        
#         image_url = arguments.get("image_url")
#         image_data_b64 = arguments.get("image_data")
#         image_path = arguments.get("image_path")
        
#         # Use fixed values matching original implementation
#         threshold = 0.20
#         voting_method = 'weighted'
        
#         if not image_url and not image_data_b64 and not image_path:
#             return json.dumps({
#                 "error": "Missing required parameter",
#                 "message": "Please provide either image_url, image_data, or image_path parameter"
#             })
        
#         image_data = None
#         image_source = None
#         filename = None
        
#         if image_path:
#             try:
#                 import os
#                 logging.info(f"Reading local file: {image_path}")
                
#                 if not os.path.exists(image_path):
#                     return json.dumps({
#                         "error": "File not found",
#                         "message": f"The specified file does not exist: {image_path}"
#                     })
                
#                 with open(image_path, 'rb') as f:
#                     image_data = f.read()
                
#                 filename = os.path.basename(image_path)
#                 image_source = f"local_file:{filename}"
#                 logging.info(f"Read local file: {len(image_data)} bytes")
#             except Exception as e:
#                 return json.dumps({
#                     "error": f"Failed to read local file: {str(e)}",
#                     "message": "Unable to read the specified file"
#                 })
        
#         elif image_data_b64:
#             try:
#                 logging.info("Processing base64-encoded image data")
#                 if ',' in image_data_b64 and image_data_b64.startswith('data:'):
#                     image_data_b64 = image_data_b64.split(',', 1)[1]
                
#                 image_data = base64.b64decode(image_data_b64)
#                 filename = "base64_image"
#                 image_source = "base64_data"
#                 logging.info(f"Decoded base64 image: {len(image_data)} bytes")
#             except Exception as e:
#                 return json.dumps({
#                     "error": f"Failed to decode base64 image data: {str(e)}",
#                     "message": "Invalid base64 image data provided"
#                 })
        
#         elif image_url:
#             try:
#                 import os
#                 logging.info(f"Downloading image from URL: {image_url}")
#                 response = requests.get(image_url, timeout=30)
#                 response.raise_for_status()
#                 image_data = response.content
#                 # Extract filename from URL
#                 from urllib.parse import urlparse
#                 parsed_url = urlparse(image_url)
#                 filename = os.path.basename(parsed_url.path) if parsed_url.path else "downloaded_image"
#                 if not filename or filename == '/':
#                     filename = "downloaded_image"
#                 image_source = image_url
#                 logging.info(f"Downloaded image: {len(image_data)} bytes")
#             except requests.RequestException as e:
#                 return json.dumps({
#                     "error": f"Failed to download image: {str(e)}",
#                     "message": "Unable to fetch image from the provided URL"
#                 })
        
#         logging.info(f"Using threshold: {threshold}, voting method: {voting_method}")
        
#         # Import Agent Framework workflow and helper
#         from Agents import run_image_detection_workflow
#         from utils.image_file_helper import load_and_save_to_temp
#         from PIL import Image, ImageFile
        
#         # For local file paths, use the path directly
#         # For URLs or base64, validate and save to temp file first
#         saved_file_path = None
        
#         if image_path and os.path.exists(image_path):
#             # Use the local file path directly (no need to copy)
#             saved_file_path = image_path
#             logging.info(f"Using local file path directly: {image_path}")
#         else:
#             # Validate and save image to temp file
#             try:
#                 ImageFile.LOAD_TRUNCATED_IMAGES = True  # Allow loading truncated images
                
#                 image = Image.open(io.BytesIO(image_data))
#                 image.load()  # Force loading the image data to detect truncation early
                
#                 # Ensure RGB format
#                 if image.mode != 'RGB':
#                     image = image.convert('RGB')
                
#                 # Re-encode to base64 temporarily
#                 buffered = io.BytesIO()
#                 image.save(buffered, format="PNG")
#                 image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
#                 # Save to temp file using helper with original filename for unique naming
#                 saved_file_path = load_and_save_to_temp(base64_image=image_base64, original_filename=filename)
                
#                 logging.info(f"Successfully validated and saved image: {image.size}, mode: {image.mode}, path: {saved_file_path}")
                
#             except Exception as img_err:
#                 logging.error(f"Failed to load/validate image: {img_err}")
#                 return json.dumps({
#                     "error": f"Invalid or corrupted image: {str(img_err)}",
#                     "message": "The provided image is incomplete or corrupted"
#                 })
        
#         # Run Agent Framework concurrent workflow with file path
#         # Workflow will extract filename from path
#         result = await run_image_detection_workflow(
#             image_path=saved_file_path,
#             threshold=threshold,
#             voting_method=voting_method
#         )
        
#         # Convert Pydantic model to dict
#         result_dict = result.model_dump()
        
#         # Add MCP metadata
#         result_dict["mcp_metadata"] = {
#             "image_source": image_source,
#             "threshold": threshold,
#             "voting_method": voting_method,
#             "framework": "agent-framework-sdk",
#             "concurrent_agents": 5,
#             "analysis_timestamp": "2025-10-31"
#         }
        
#         return json.dumps(result_dict, indent=2)
        
#     except Exception as e:
#         logging.error(f"Error in MCP analyze_image_deepfake: {str(e)}")
#         return json.dumps({
#             "error": str(e),
#             "message": "Failed to analyze image for deepfake detection"
#         })


# # ============================================================================
# # MCP TOOL TRIGGER ENDPOINTS - TEXT DETECTION
# # ============================================================================

# # Tool properties for text detection MCP tools
# _tool_properties_analyze_text = json.dumps([
#     {
#         "propertyName": "text",
#         "propertyType": "string",
#         "description": "The text to analyze for AI-generated content detection",
#         "isRequired": True
#     }
# ])


# @app.mcp_tool_trigger(
#     arg_name="context",
#     tool_name="get_text_detection_models",
#     description="Get a list of available text detection models with filtering options.",
#     tool_properties="[]",
# )
# def get_text_detection_models_mcp(context) -> str:
#     """MCP tool endpoint to retrieve available text detection models."""
#     logging.info('MCP tool get_text_detection_models called.')
    
#     try:
#         from textdetectionlib.model_registry import (
#             list_available_text_models,
#             get_text_models_by_architecture
#         )
        
#         models = None
#         if context:
#             content = json.loads(context)
#             arguments = content.get("arguments", {})
            
#             filter_by_architecture = arguments.get("filter_by_architecture")
            
#             if filter_by_architecture:
#                 logging.info(f"Filtering models by architecture: {filter_by_architecture}")
#                 models = get_text_models_by_architecture(filter_by_architecture)
#             else:
#                 models = list_available_text_models()
#         else:
#             models = list_available_text_models()
        
#         result = {
#             "total_models": len(models),
#             "models": models
#         }
        
#         return json.dumps(result, indent=2)
        
#     except Exception as e:
#         logging.error(f"Error in MCP get_text_detection_models: {str(e)}")
#         return json.dumps({
#             "error": str(e),
#             "message": "Failed to retrieve text detection models"
#         })


# @app.mcp_tool_trigger(
#     arg_name="context",
#     tool_name="check_text_models_status",
#     description="Check if the text detection models are loaded and ready for analysis.",
#     tool_properties="[]",
# )
# def check_text_models_status_mcp(context) -> str:
#     """MCP tool endpoint to check if text models are loaded and ready."""
#     logging.info('MCP tool check_text_models_status called.')
    
#     try:
#         from Services.app_detect_text import get_text_ensemble
        
#         ensemble = get_text_ensemble()
        
#         if ensemble is not None:
#             return json.dumps({
#                 "status": "ready",
#                 "message": "Text detection models are loaded and ready for analysis",
#                 "ready": True
#             })
#         else:
#             return json.dumps({
#                 "status": "loading",
#                 "message": "Text models are still loading. First-time initialization takes 1-3 minutes.",
#                 "ready": False,
#                 "tip": "Call the warmup endpoint (/v1.0/warmupTextModels) or wait a few minutes before analyzing text."
#             })
        
#     except Exception as e:
#         logging.error(f"Error checking text model status: {str(e)}")
#         return json.dumps({
#             "status": "error",
#             "message": f"Error checking text model status: {str(e)}",
#             "ready": False
#         })


# @app.mcp_tool_trigger(
#     arg_name="context",
#     tool_name="analyze_text_deepfake",
#     description="Analyze text to detect if it's AI-generated or human-written using Agent Framework with concurrent ML models.",
#     tool_properties=_tool_properties_analyze_text,
# )
# async def analyze_text_deepfake_mcp(context) -> str:
#     """MCP tool endpoint to analyze text for AI-generated content detection using Agent Framework."""
#     logging.info('MCP tool analyze_text_deepfake (Agent Framework) called.')
    
#     try:
#         if not context:
#             return json.dumps({
#                 "error": "No context provided",
#                 "message": "Please provide the 'text' parameter"
#             })
        
#         content = json.loads(context)
#         arguments = content.get("arguments", {})
        
#         text = arguments.get("text")
        
#         if not text:
#             return json.dumps({
#                 "error": "Missing required parameter",
#                 "message": "Please provide the 'text' parameter"
#             })
        
#         if not text.strip():
#             return json.dumps({
#                 "error": "Invalid input",
#                 "message": "Text cannot be empty"
#             })
        
#         logging.info(f"Analyzing text (length: {len(text)} chars)")
        
#         # Import Agent Framework workflow
#         from Agents import run_text_detection_workflow
        
#         # Create a text identifier (truncated text preview)
#         text_identifier = text[:50].replace('\n', ' ').strip()
#         if len(text) > 50:
#             text_identifier += "..."
        
#         # Run Agent Framework concurrent workflow
#         result = await run_text_detection_workflow(text=text, text_identifier=text_identifier)
        
#         # Convert Pydantic model to dict
#         result_dict = result.model_dump()
        
#         # Add MCP metadata
#         result_dict["mcp_metadata"] = {
#             "text_length": len(text),
#             "framework": "agent-framework-sdk",
#             "concurrent_agents": 4,
#             "analysis_timestamp": "2025-10-31"
#         }
        
#         return json.dumps(result_dict, indent=2)
        
#     except ValueError as e:
#         logging.error(f"Validation error in MCP analyze_text_deepfake: {str(e)}")
#         return json.dumps({
#             "error": "Validation error",
#             "message": str(e)
#         })
#     except Exception as e:
#         logging.error(f"Error in MCP analyze_text_deepfake: {str(e)}")
#         return json.dumps({
#             "error": str(e),
#             "message": "Failed to analyze text for AI-generated content detection"
#         })
