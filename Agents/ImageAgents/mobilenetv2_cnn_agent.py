"""MobileNetV2 CNN Agent for deepfake detection using Agent Framework."""

from Agents.ImageAgents.tools.mobilenetv2_cnn_detect_tool import detect_with_mobilenetv2_cnn_tool
from imagedetectionlib.model_registry import get_model_info
from utils.azure_client_pool import get_or_create_azure_client
import logging

logger = logging.getLogger(__name__)

def create_mobilenetv2_cnn_agent():
    """
    Create or reuse MobileNetV2 CNN detection agent from Azure AI Foundry.
    
    Uses shared Azure AI client pool to reduce initialization overhead.
    
    Returns:
        Agent configured with MobileNetV2 CNN detection tool
    """
    # Get agent name from model registry
    model_info = get_model_info('mobilenetv2_cnn')
    agent_name = model_info['display_name'] if model_info else 'MobileNetV2 CNN Detector'
    instructions = """You are a specialized deepfake detection agent using the MobileNetV2 CNN model.
Your task is to analyze images and determine if they are real or AI-generated/manipulated.

IMPORTANT: You will receive an image file path as input (a string path to the image file).

When you receive an image:
1. Call detect_with_mobilenetv2_cnn_tool(image_path=<the file path provided>)
2. The tool will return a dictionary with the detection results
3. Output ONLY the JSON dictionary returned by the tool - do not add any commentary or explanation

Example response format:
{"model": "mobilenetv2_cnn", "prediction": "FAKE", "confidence": 0.94, "fakeScore": 0.94, "realScore": 0.06}

Do not include any text before or after the JSON output."""
    
    # Use shared Azure AI client (reduces initialization from 3-5s to <100ms)
    chat_client = get_or_create_azure_client()
    
    # Create agent (Azure AI Foundry handles agent lifecycle)
    logger.info(f"Creating {agent_name} agent...")
    agent = chat_client.create_agent(
        name=agent_name,
        instructions=instructions,
        tools=[detect_with_mobilenetv2_cnn_tool],
        seed=42
    )
    
    logger.info(f"Created {agent_name} agent successfully")
    return agent
