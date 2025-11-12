"""CNN V3 FineTuned Agent for deepfake detection using Agent Framework."""

from Agents.ImageAgents.tools.cnn_v3_detect_tool import detect_with_cnn_v3_tool
from imagedetectionlib.model_registry import get_model_info
from utils.azure_client_pool import get_or_create_azure_client
import logging

logger = logging.getLogger(__name__)

def create_cnn_v3_agent():
    """
    Create or reuse CNN V3 FineTuned detection agent from Azure AI Foundry.
    
    Uses shared Azure AI client pool to reduce initialization overhead.
    
    Returns:
        Agent configured with CNN V3 FineTuned detection tool
    """
    # Get unique agent name from model registry (use model key for uniqueness)
    model_info = get_model_info('cnn_v3_finetuned')
    agent_name = f"cnn-v3-{model_info['display_name']}" if model_info else 'cnn-v3-CNN V3 FineTuned Detector'
    instructions = """You are a specialized deepfake detection agent using the CNN V3 FineTuned model.
Your task is to analyze images and determine if they are real or AI-generated/manipulated.

IMPORTANT: You will receive an image file path as input (a string path to the image file).

When you receive an image:
1. Call detect_with_cnn_v3_tool(image_path=<the file path provided>)
2. The tool will return a dictionary with the detection results
3. Output ONLY the JSON dictionary returned by the tool - do not add any commentary or explanation

Example response format:
{"model": "cnn_v3_finetuned", "prediction": "FAKE", "confidence": 0.89, "fakeScore": 0.89, "realScore": 0.11}

Do not include any text before or after the JSON output."""
    
    # Create dedicated Azure AI client for this agent (avoids agent pooling/caching issues)
    import os
    from agent_framework.azure import AzureAIAgentClient
    from azure.identity.aio import AzureCliCredential
    
    project_endpoint = os.getenv('AZURE_AI_PROJECT_ENDPOINT')
    deployment_name = os.getenv('AZURE_AI_MODEL_DEPLOYMENT_NAME')
    
    chat_client = AzureAIAgentClient(
        project_endpoint=project_endpoint,
        model_deployment_name=deployment_name,
        async_credential=AzureCliCredential()
    )
    
    # Create agent (Azure AI Foundry handles agent lifecycle)
    logger.info(f"Creating {agent_name} agent...")
    agent = chat_client.create_agent(
        name=agent_name,
        instructions=instructions,
        tools=[detect_with_cnn_v3_tool],
        seed=42
    )
    
    logger.info(f"Created {agent_name} agent successfully")
    return agent
