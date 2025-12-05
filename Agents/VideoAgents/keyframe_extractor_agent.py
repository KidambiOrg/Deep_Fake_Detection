"""Key Frame Extractor Agent for video processing."""

import logging
from utils.azure_client_pool import get_or_create_azure_client

logger = logging.getLogger(__name__)


def create_keyframe_extractor_agent():
    """
    Create Key Frame Extractor agent.
    
    Extracts key frames from video based on histogram differences.
    
    Returns:
        Agent configured with key frame extraction tool
    """
    agent_name = "keyframe-extractor"
    instructions = """You are a specialized video processing agent that extracts key frames from videos.

Your task is to analyze a video, extract key frames, and save them to disk.

IMPORTANT: You will receive a JSON string containing video processing parameters.

When you receive the input:
1. Parse the JSON to extract video_path, output_folder, and other parameters (max_frames, method, threshold, min_gap_sec)
2. Call extract_key_frames_tool with these parameters
3. The tool will save key frames to disk and return metadata including keyframes_folder path
4. Output ONLY the JSON dictionary returned by the tool - it includes keyframes_folder for the next agent

Do not add any commentary or explanation. Return only the tool's output."""
    
    # Import tool
    from Agents.VideoAgents.tools.keyframe_extractor_tool import extract_key_frames_tool
    
    # Get shared Azure AI client
    chat_client = get_or_create_azure_client()
    
    logger.info(f"Creating {agent_name} agent...")
    agent = chat_client.create_agent(
        name=agent_name,
        instructions=instructions,
        tools=[extract_key_frames_tool]
    )
    
    logger.info(f"✓ Created {agent_name} agent (ID: {agent.id})")
    return agent
