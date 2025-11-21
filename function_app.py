import azure.functions as func
import logging
import json
import sys
import warnings
import asyncio

# Fix protobuf import issue with Azure Functions Core Tools
# The bundled protobuf in core tools is old and missing runtime_version
try:
    import google.protobuf
    # Add runtime_version if missing (workaround for old bundled protobuf)
    if not hasattr(google.protobuf, 'runtime_version'):
        google.protobuf.runtime_version = lambda: "5.27.2"
except Exception:
    pass

# Suppress protobuf module cache warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Suppress aiohttp ResourceWarning about unclosed sessions
# These are managed by the Azure SDK and will be cleaned up
warnings.filterwarnings('ignore', category=ResourceWarning, message='unclosed.*<ssl.SSLSocket.*')
warnings.filterwarnings('ignore', category=ResourceWarning, message='unclosed.*<TCPTransport.*')
warnings.filterwarnings('ignore', category=ResourceWarning, message='unclosed.*<aiohttp.*')

# Suppress asyncio warnings about destroyed tasks
warnings.filterwarnings('ignore', message='.*Task.*was destroyed.*')

# Workaround for Azure Functions protobuf namespace package issue
# Ensure google package is properly initialized as a namespace package
try:
    import google
    if not hasattr(google, '__path__'):
        import pkgutil
        google.__path__ = pkgutil.extend_path([], 'google')
except Exception:
    pass

# Configure logging to reduce noise from Azure SDK
logging.getLogger('azure').setLevel(logging.WARNING)
logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.WARNING)
logging.getLogger('azure.core.pipeline.transport').setLevel(logging.WARNING)
logging.getLogger('aiohttp').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.ERROR)  # Only show errors from asyncio

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
    """Preload image detection models from local Models folder to memory"""
    logging.info('Preload endpoint for image models called - loading from Models folder.')
    
    try:
        from Agents.ImageAgents.image_workflow import get_or_create_agents
        
        # Create agents - this will cache them and trigger model loading
        logging.info("Initializing agents and loading models...")
        agents = get_or_create_agents()
        
        # Force model loading by calling each tool's load function
        from Agents.ImageAgents.tools import nonescape_detect_tool
        from Agents.ImageAgents.tools import prithiv_v1_detect_tool
        from Agents.ImageAgents.tools import dima_vit_detect_tool
        from Agents.ImageAgents.tools import dimavit_v2_finetuned_detect_tool
        from Agents.ImageAgents.tools import organika_sdxl_detect_tool
        from Agents.ImageAgents.tools import mobilenetv2_detect_tool
        from Agents.ImageAgents.tools import mobilenetv2_cnn_detect_tool
        from Agents.ImageAgents.tools import cnn_v3_detect_tool
        from Agents.ImageAgents.tools import prithiv_open_deep_fake_detect_tool
        
        models_loaded = []
        models_failed = []
        
        # Preload each model
        try:
            nonescape_detect_tool.load_nonescape_model()
            models_loaded.append("nonescape")
        except Exception as e:
            models_failed.append(f"nonescape: {str(e)}")
        
        try:
            prithiv_v1_detect_tool.load_prithiv_model()
            models_loaded.append("prithiv_v1")
        except Exception as e:
            models_failed.append(f"prithiv_v1: {str(e)}")
        
        try:
            dima_vit_detect_tool.load_dima_model()
            models_loaded.append("dima_vit")
        except Exception as e:
            models_failed.append(f"dima_vit: {str(e)}")
        
        try:
            dimavit_v2_finetuned_detect_tool.load_dimavit_v2_model()
            models_loaded.append("dimavit_v2_finetuned")
        except Exception as e:
            models_failed.append(f"dimavit_v2_finetuned: {str(e)}")
        
        try:
            organika_sdxl_detect_tool.load_organika_model()
            models_loaded.append("organika_sdxl")
        except Exception as e:
            models_failed.append(f"organika_sdxl: {str(e)}")
        
        try:
            mobilenetv2_detect_tool.load_mobilenetv2_model()
            models_loaded.append("mobilenetv2")
        except Exception as e:
            models_failed.append(f"mobilenetv2: {str(e)}")
        
        try:
            mobilenetv2_cnn_detect_tool.load_mobilenetv2_cnn_model()
            models_loaded.append("mobilenetv2_cnn")
        except Exception as e:
            models_failed.append(f"mobilenetv2_cnn: {str(e)}")
        
        try:
            cnn_v3_detect_tool.load_cnn_v3_model()
            models_loaded.append("cnn_v3")
        except Exception as e:
            models_failed.append(f"cnn_v3: {str(e)}")
        
        try:
            prithiv_open_deep_fake_detect_tool.load_prithiv_open_model()
            models_loaded.append("prithiv_open_deep_fake")
        except Exception as e:
            models_failed.append(f"prithiv_open_deep_fake: {str(e)}")
        
        # Determine status
        total = len(models_loaded) + len(models_failed)
        successful = len(models_loaded)
        failed = len(models_failed)
        
        if failed == 0:
            status = "success"
            status_code = 200
        elif successful > 0:
            status = "partial"
            status_code = 207
        else:
            status = "error"
            status_code = 500
        
        result = {
            "status": status,
            "service": "image_detection",
            "message": f"Loaded {successful}/{total} models from local Models folder",
            "total_models": total,
            "successful": successful,
            "failed": failed,
            "models_loaded": models_loaded,
            "models_failed": models_failed
        }
        
        logging.info(f"Model preload complete: {successful}/{total} models loaded")
        
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
    
    Uses concurrent workflow pattern with 7 detection agents:
    - NoEscape, SEResNet50, Prithiv V1, DimaVit, Organika SDXL, MobileNetV2, MobileNetV2-CNN
    
    Supports 14+ ensemble voting methods including:
    majority, weighted, bayesian, entropy_weighted, etc.
    """
    logging.info('Image analysis endpoint (Agent Framework) called.')

    try:
        # Import Agent Framework workflow
        from Agents import run_image_detection_workflow
        from utils.image_file_helper import load_and_save_to_temp
        import base64
        import binascii
        import io
        from PIL import Image
        
        image_data = None
        original_filename = "uploaded_image"
        
        # Check content type to determine how to parse the request
        content_type = req.headers.get('Content-Type', '').lower()
        logging.info(f"Request Content-Type: {content_type}")
        
        # Try to get image from req.files first (standard multipart file upload)
        image_file = req.files.get('image')
        
        if image_file:
            # Handle standard multipart file upload (from tools like Postman, curl, etc.)
            original_filename = image_file.filename if image_file.filename else "uploaded_image"
            image_data = image_file.read()
            logging.info(f"Received image from req.files: {original_filename}, size: {len(image_data)} bytes")
        
        # Try JSON body with imageUrl (base64 data URL) or base64 image
        if not image_data:
            try:
                req_body = req.get_json()
                
                if isinstance(req_body, dict):
                    # Check for imageUrl field (base64 data URL format)
                    if 'imageUrl' in req_body:
                        image_url = req_body['imageUrl']
                        logging.info(f"Processing imageUrl field (base64 data URL)")
                        
                        # Handle data URL format: data:image/jpeg;base64,/9j/4AAQ...
                        if isinstance(image_url, str):
                            # Remove data URI prefix if present
                            if ',' in image_url:
                                base64_data = image_url.split(',', 1)[1]
                            else:
                                base64_data = image_url
                            
                            image_data = base64.b64decode(base64_data)
                            logging.info(f"Decoded image from base64 data URL, size: {len(image_data)} bytes")
                    
                    # Check for base64 image data (alternative field name)
                    elif 'image' in req_body:
                        base64_data = req_body['image']
                        # Remove data URI prefix if present
                        if isinstance(base64_data, str) and ',' in base64_data:
                            base64_data = base64_data.split(',', 1)[1]
                        image_data = base64.b64decode(base64_data)
                        logging.info(f"Received image from JSON body (base64), size: {len(image_data)} bytes")
                    
            except (ValueError, binascii.Error) as decode_err:
                logging.error(f"Failed to decode base64 image: {decode_err}")
                return func.HttpResponse(
                    json.dumps({
                        "error": "Failed to decode base64 image",
                        "details": str(decode_err)
                    }),
                    status_code=400,
                    mimetype="application/json"
                )
            except Exception as json_err:
                logging.warning(f"Failed to parse JSON from body: {json_err}")
        
        # Final check
        if not image_data:
            return func.HttpResponse(
                json.dumps({
                    "error": "Please provide an image file with key 'image' or 'imageUrl' in JSON body.",
                    "details": "Supported formats: multipart/form-data file upload, JSON body with 'imageUrl' (base64 data URL), or JSON body with base64 'image'",
                    "received_content_type": content_type
                }),
                status_code=400,
                mimetype="application/json"
            )
        
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
        
        # Use Logistic Regression meta-classifier voting
        # This uses a trained LR model from Models/lr_meta_classifier.pkl
        # The meta-classifier learns optimal combination weights from training data
        threshold = 0.20
        voting_method = 'logistic_regression'  # LR meta-classifier ensemble
       
        
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

 