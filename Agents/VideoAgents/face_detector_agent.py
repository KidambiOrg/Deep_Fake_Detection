"""Face Detector Agent for extracting faces from frames."""

import logging
from utils.azure_client_pool import get_or_create_azure_client

logger = logging.getLogger(__name__)


def create_face_detector_agent():
    """
    Create Face Detector agent.
    
    Detects and extracts faces from video frames.
    
    Returns:
        Agent configured with face detection tool
    """
    agent_name = "face-detector"
    instructions = """You are a specialized face detection agent that extracts faces from key frame images.

Your task is to detect faces in saved key frame images and extract them with proper cropping.

IMPORTANT: You will receive key frame metadata from the previous agent.

When you receive input:
1. Extract keyframes_folder and output_folder from the input JSON
2. Call detect_faces_in_frames_tool with keyframes_folder and output_folder parameters
3. Include min_confidence, min_face_size, and max_workers if present in input
4. The tool will load frames from disk, detect faces, and save extracted faces
5. Output ONLY the JSON dictionary returned by the tool

Do not add any commentary or explanation. Return only the tool's output."""
    
    # Import tool
    from Agents.VideoAgents.tools.face_detector_tool import detect_faces_in_frames_tool
    
    # Get shared Azure AI client
    chat_client = get_or_create_azure_client()
    
    logger.info(f"Creating {agent_name} agent...")
    agent = chat_client.create_agent(
        name=agent_name,
        instructions=instructions,
        tools=[detect_faces_in_frames_tool]
    )
    
    logger.info(f"✓ Created {agent_name} agent (ID: {agent.id})")
    return agent
