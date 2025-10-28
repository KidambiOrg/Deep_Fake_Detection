"""
MCP (Model Context Protocol) Endpoints for Deepfake Text Detection
This module contains MCP tool trigger endpoints for text-based deepfake detection.
"""
import azure.functions as func
import logging
import json


def register_mcp_text_endpoints(app: func.FunctionApp):
    """
    Register all MCP text detection endpoints with the provided FunctionApp instance.
    
    Args:
        app: The Azure Functions FunctionApp instance to register endpoints with.
    """
    
    # Tool properties for get_text_models MCP tool
    tool_properties_get_text_models = []
    
    # Tool properties for analyze_text MCP tool - only accepts text input
    tool_properties_analyze_text = [
        {
            "propertyName": "text",
            "propertyType": "string",
            "description": "Required: The text to analyze for AI-generated content detection"
        }
    ]
    
    tool_properties_analyze_text_json = json.dumps(tool_properties_analyze_text)
    
    
    @app.generic_trigger(
        arg_name="context",
        type="mcpToolTrigger",
        toolName="get_text_detection_models",
        description="Get a list of available text detection models with filtering options.",
        toolProperties="[]",
    )
    def get_text_detection_models_mcp(context) -> str:
        """
        MCP tool endpoint to retrieve available text detection models.
        Supports filtering by architecture.
        
        Args:
            context: The MCP tool trigger context containing input arguments.
            
        Returns:
            str: JSON string with the list of available models or filtered results.
        """
        logging.info('MCP tool get_text_detection_models called.')
        
        try:
            # Import model registry functions
            from textdetectionlib.model_registry import (
                list_available_text_models,
                get_text_models_by_architecture
            )
            
            # Parse the context to get arguments
            models = None
            if context:
                content = json.loads(context)
                arguments = content.get("arguments", {})
                
                filter_by_architecture = arguments.get("filter_by_architecture")
                
                # Apply filters if provided
                if filter_by_architecture:
                    logging.info(f"Filtering models by architecture: {filter_by_architecture}")
                    models = get_text_models_by_architecture(filter_by_architecture)
                else:
                    models = list_available_text_models()
            else:
                models = list_available_text_models()
            
            # Convert to formatted JSON string
            result = {
                "total_models": len(models),
                "models": models
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logging.error(f"Error in MCP get_text_detection_models: {str(e)}")
            error_result = {
                "error": str(e),
                "message": "Failed to retrieve text detection models"
            }
            return json.dumps(error_result)
    
    
    @app.generic_trigger(
        arg_name="context",
        type="mcpToolTrigger",
        toolName="check_text_models_status",
        description="Check if the text detection models are loaded and ready for analysis.",
        toolProperties="[]",
    )
    def check_text_models_status_mcp(context) -> str:
        """
        MCP tool endpoint to check if text models are loaded and ready.
        Useful to call before analyze_text to avoid timeouts.
        
        Args:
            context: The MCP tool trigger context (not used).
            
        Returns:
            str: JSON string with model status information.
        """
        logging.info('MCP tool check_text_models_status called.')
        
        try:
            from Services.app_detect_text import get_text_ensemble
            
            # Try to get the ensemble to check if models are loaded
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
    
    
    @app.generic_trigger(
        arg_name="context",
        type="mcpToolTrigger",
        toolName="analyze_text_deepfake",
        description="Analyze text to detect if it's AI-generated or human-written using an ensemble of ML models.",
        toolProperties=tool_properties_analyze_text_json,
    )
    def analyze_text_deepfake_mcp(context) -> str:
        """
        MCP tool endpoint to analyze text for AI-generated content detection.
        Uses an ensemble of BERT, DistilBERT, BoW, and CharCNN models.
        Uses default voting method 'weighted'.
        
        Args:
            context: The MCP tool trigger context containing input arguments:
                     - text (required): The text to analyze
            
        Returns:
            str: JSON string with detection results including prediction, confidence, and individual model scores.
        """
        logging.info('MCP tool analyze_text_deepfake called.')
        
        try:
            # Parse the context to get arguments
            if not context:
                return json.dumps({
                    "error": "No context provided",
                    "message": "Please provide the 'text' parameter"
                })
            
            content = json.loads(context)
            arguments = content.get("arguments", {})
            
            text = arguments.get("text")
            
            # Use default voting method
            voting_method = "weighted"
            
            # Validate input
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
            
            # Lazy import to avoid loading models at startup
            from Services.app_detect_text import detect_text_json
            
            # Analyze the text using module-level function
            result_json = detect_text_json(text, voting_method=voting_method, indent=2)
            
            # Parse and enhance the result
            result = json.loads(result_json)
            
            # Add metadata
            result["mcp_metadata"] = {
                "text_length": len(text),
                "voting_method": voting_method,
                "analysis_timestamp": "2025-10-27"
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
            error_result = {
                "error": str(e),
                "message": "Failed to analyze text for AI-generated content detection"
            }
            return json.dumps(error_result)
