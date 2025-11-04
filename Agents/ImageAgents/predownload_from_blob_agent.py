"""Agent for preloading image detection models from Azure Blob Storage."""

from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential
from Agents.ImageAgents.tools.download_from_blob_tool import download_multiple_models
from imagedetectionlib.model_registry import list_available_models
import os
import logging
import json
import threading

logger = logging.getLogger(__name__)

# Global agent cache - initialized once and reused
_cached_preload_agent = None
_agent_lock = threading.Lock()


def preload_image_models_tool() -> str:
    """
    Agent tool: Preload all image detection models from blob storage to cache.
    
    Downloads all image models defined in the model registry from Azure Blob Storage
    and caches them locally for faster loading during detection workflows.
    
    Returns:
        JSON string with preload results including status and download details
    """
    try:
        logger.info("Starting image model preload from blob storage...")
        
        # Get all available image models from registry
        models = list_available_models()
        
        # Build list of model configurations to download
        model_configs = []
        
        for model_key, model_info in models.items():
            blob_config = model_info.get('blob_storage')
            
            if not blob_config:
                logger.warning(f"Model {model_key} does not have blob_storage configuration")
                continue
            
            container = blob_config.get('container')
            
            # Check if this is a directory download or single file download
            if 'path' in blob_config:
                # Directory download (e.g., HuggingFace models)
                model_configs.append({
                    'container_name': container,
                    'blob_prefix': blob_config['path'],
                    'model_key': model_key,
                    'model_name': model_info.get('display_name', model_key)
                })
            elif 'blob' in blob_config:
                # Single file download (e.g., safetensors files)
                model_configs.append({
                    'container_name': container,
                    'blob_name': blob_config['blob'],
                    'model_key': model_key,
                    'model_name': model_info.get('display_name', model_key)
                })
            else:
                logger.warning(f"Model {model_key} has invalid blob_storage configuration")
        
        logger.info(f"Found {len(model_configs)} image models to preload")
        
        if not model_configs:
            return json.dumps({
                "status": "error",
                "message": "No models found to preload - check model registry configuration",
                "total_models": 0,
                "successful": 0,
                "failed": 0
            })
        
        # Download all models (force_download=True by default to overwrite cache)
        result = download_multiple_models(
            model_configs=model_configs,
            force_download=True  # Always overwrite cache
        )
        
        # Build clean model details (no raw results)
        models_output = [
            {
                'model_key': config.get('model_key'),
                'model_name': config.get('model_name'),
                'status': result['results'][i]['status'],
                'message': result['results'][i].get('message', '')
            }
            for i, config in enumerate(model_configs)
        ]
        
        logger.info(f"Image model preload complete: {result['successful']}/{result['total_models']} successful")
        
        # Return clean output without raw results
        return json.dumps({
            'status': result['status'],
            'total_models': result['total_models'],
            'successful': result['successful'],
            'failed': result['failed'],
            'models': models_output
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Error preloading image models: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Exception during preload: {str(e)}",
            "total_models": 0,
            "successful": 0,
            "failed": 0
        })


def get_or_create_preload_image_agent():
    """
    Get or create preload agent with singleton pattern.
    Agent is loaded only once and cached globally.
    Thread-safe initialization prevents duplicate loading.
    
    Returns:
        Cached preload agent instance
    """
    global _cached_preload_agent
    
    # Fast path: if agent already exists, return immediately
    if _cached_preload_agent is not None:
        logger.info("Reusing cached image preload agent")
        return _cached_preload_agent
    
    # Slow path: need to create agent (with lock to prevent concurrent initialization)
    with _agent_lock:
        # Double-check after acquiring lock
        if _cached_preload_agent is not None:
            logger.info("Reusing cached image preload agent (after lock)")
            return _cached_preload_agent
        
        logger.info("Initializing image preload agent for the first time...")
        
        agent_name = "Image Model Preloader"
        instructions = """You are a specialized agent for preloading image detection models from Azure Blob Storage.

Your task is to download all image detection models and cache them locally for faster loading.

When activated:
1. Call preload_image_models_tool() immediately
2. The tool will download all models defined in the model registry
3. Output the JSON results returned by the tool

Do not add any commentary or explanation - just return the JSON output from the tool."""
        
        # Initialize Azure AI Foundry Agent Client
        project_endpoint = os.getenv('AZURE_AI_PROJECT_ENDPOINT')
        deployment_name = os.getenv('AZURE_AI_MODEL_DEPLOYMENT_NAME')

        chat_client = AzureAIAgentClient(
            project_endpoint=project_endpoint,
            model_deployment_name=deployment_name,
            async_credential=AzureCliCredential()
        )
        
        # Create agent
        logger.info(f"Creating {agent_name} agent...")
        agent = chat_client.create_agent(
            name=agent_name,
            instructions=instructions,
            tools=[preload_image_models_tool],
            seed=42
        )
        
        # Cache the agent globally
        _cached_preload_agent = agent
        
        logger.info(f"Created and cached {agent_name} agent successfully")
        return agent


async def run_preload_image_models():
    """
    Run the image model preload workflow.
    
    Returns:
        Dictionary with preload results
    """
    try:
        logger.info("Starting image model preload workflow...")
        
        # Get or create cached agent
        agent = get_or_create_preload_image_agent()
        
        # Run agent with simple trigger message
        result = await agent.run("Preload all image detection models")
        
        # Parse JSON result (result is string from agent)
        result_dict = json.loads(str(result))
        
        logger.info("Image model preload workflow completed")
        return result_dict
        
    except Exception as e:
        logger.error(f"Error in preload workflow: {e}")
        return {
            "status": "error",
            "message": f"Workflow error: {str(e)}",
            "total_models": 0,
            "successful": 0,
            "failed": 0
        }
