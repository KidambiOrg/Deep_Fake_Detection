"""CharCNN Agent for text deepfake detection."""

from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential
from Agents.TextAgents.tools.charcnn_detect_tool import detect_with_charcnn_tool
from textdetectionlib.model_registry import get_text_model_info
import os
import logging

logger = logging.getLogger(__name__)

def create_charcnn_agent():
    """
    Create or reuse CharCNN detection agent from Azure AI Foundry.
    """
    # Get agent name from model registry
    agent_name = get_text_model_info('charcnn')['display_name']
    instructions = """You are a specialized AI-generated text detection agent using the CharCNN model.
Analyze text and determine if it is human-written or AI-generated.

Return IndividualModelScore with fakeScore and realScore."""
    
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
        tools=[detect_with_charcnn_tool]
    )
    
    logger.info(f"Created {agent_name} agent successfully")
    return agent
