"""MobileNetV2 Agent for deepfake detection using Agent Framework."""

from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential
from Agents.ImageAgents.tools.mobilenetv2_detect_tool import detect_with_mobilenetv2_tool
from imagedetectionlib.model_registry import get_model_info
import os
import logging

logger = logging.getLogger(__name__)

def create_mobilenetv2_agent():
    """
    Create or reuse MobileNetV2 detection agent from Azure AI Foundry.
    
    Returns:
        Agent configured with MobileNetV2 detection tool
    """
    # Get agent name from model registry
    agent_name = get_model_info('mobilenetv2')['display_name']
    instructions = """You are a specialized deepfake detection agent using the MobileNetV2 model.
Your task is to analyze images and determine if they are real or AI-generated/manipulated.

IMPORTANT: You will receive an image file path as input.

When you receive an image path:
1. Call detect_with_mobilenetv2_tool(image_path=<the file path provided>)
2. The tool will return a dictionary with the detection results
3. Output ONLY the JSON dictionary returned by the tool - do not add any commentary or explanation

Example response format:
{"model": "mobilenetv2", "prediction": "FAKE", "confidence": 0.89, "fakeScore": 0.89, "realScore": 0.11}

Do not include any text before or after the JSON output."""
    
    
    # Initialize Azure AI Foundry Agent Client
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
        tools=[detect_with_mobilenetv2_tool],
        seed=42
    )
    
    logger.info(f"Created {agent_name} agent successfully")
    return agent
