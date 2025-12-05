"""Face Deduplicator Agent for removing duplicate faces."""

import logging
from utils.azure_client_pool import get_or_create_azure_client

logger = logging.getLogger(__name__)


def create_face_deduplicator_agent():
    """
    Create Face Deduplicator agent.
    
    Removes duplicate faces using DeepFace embeddings.
    
    Returns:
        Agent configured with deduplication tool
    """
    agent_name = "face-deduplicator"
    instructions = """You are a specialized face deduplication agent that removes duplicate faces.

Your task is to identify and remove duplicate faces (same person) from extracted face images.

IMPORTANT: You will receive face detection results with an output_folder path.

When you receive the data:
1. Extract the output_folder from the input
2. Call deduplicate_faces_tool with faces_folder=output_folder
3. The tool will return a dictionary with deduplication statistics
4. Output ONLY the JSON dictionary returned by the tool

Do not add any commentary or explanation. Return only the tool's output."""
    
    # Import tool
    from Agents.VideoAgents.tools.face_deduplicator_tool import deduplicate_faces_tool
    
    # Get shared Azure AI client
    chat_client = get_or_create_azure_client()
    
    logger.info(f"Creating {agent_name} agent...")
    agent = chat_client.create_agent(
        name=agent_name,
        instructions=instructions,
        tools=[deduplicate_faces_tool]
    )
    
    logger.info(f"✓ Created {agent_name} agent (ID: {agent.id})")
    return agent
