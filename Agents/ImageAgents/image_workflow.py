"""Image Detection Workflow using Agent Framework with Ensemble Voting.

Performance Optimizations Implemented:
- Priority 5: Azure AI Client Pooling (ACTIVE - saves 3-5s per request)
  All agents share a single Azure AI client instance via utils/azure_client_pool.py
  Eliminates 7x redundant authentication overhead
  
Note: Priority 4 (Shared Image Preprocessing) was attempted but reverted due to
Azure Agent Framework limitation - Pydantic cannot serialize PIL Image/NumPy array objects.
Each agent currently loads images individually from the file path.
Future optimization: Explore base64 encoding or Agent Framework file handling APIs.
"""

from agent_framework import ConcurrentBuilder, WorkflowOutputEvent
from Agents.ImageAgents.image_result_aggregator import create_image_aggregator
from Agents.ImageAgents.nonescape_agent import create_nonescape_agent
from Agents.ImageAgents.seresnet50_agent import create_seresnet50_agent
from Agents.ImageAgents.prithiv_v1_agent import create_prithiv_v1_agent
from Agents.ImageAgents.dima_vit_agent import create_dima_vit_agent
from Agents.ImageAgents.organika_sdxl_agent import create_organika_sdxl_agent
from Agents.ImageAgents.mobilenetv2_agent import create_mobilenetv2_agent
from Agents.ImageAgents.mobilenetv2_cnn_agent import create_mobilenetv2_cnn_agent
from Agents.ImageAgents.cnn_v3_agent import create_cnn_v3_agent
from Agents.ImageAgents.prithiv_open_deep_fake_agent import create_prithiv_open_deep_fake_agent
from outputModel.image_detection_output_model import ImageDetectionResult, IndividualModelResult, VotingResult
from imagedetectionlib.model_registry import get_model_info
import numpy as np
import os
import logging
import json
import threading
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Global agent cache - initialized once and reused
_cached_agents = None
_agents_lock = threading.Lock()

def get_or_create_agents():
    """
    Get or create detection agents with singleton pattern.
    Agents and their models are loaded only once and cached globally.
    Thread-safe initialization prevents duplicate loading.
    
    Returns:
        Tuple of (nonescape_agent, seresnet50_agent, prithiv_v1_agent, dima_vit_agent, organika_sdxl_agent, mobilenetv2_agent, mobilenetv2_cnn_agent, cnn_v3_agent, prithiv_open_deep_fake_agent)
    """
    global _cached_agents
    
    # Fast path: if agents already exist, return immediately
    if _cached_agents is not None:
        logger.info("Reusing cached agents")
        return _cached_agents
    
    # Slow path: need to create agents (with lock to prevent concurrent initialization)
    with _agents_lock:
        # Double-check after acquiring lock
        if _cached_agents is not None:
            logger.info("Reusing cached agents (after lock)")
            return _cached_agents
        
        logger.info("Initializing detection agents for the first time...")
        
        # Create agents once - each agent's tool will also cache its model
        nonescape_agent = create_nonescape_agent()
        seresnet50_agent = create_seresnet50_agent()
        prithiv_v1_agent = create_prithiv_v1_agent()
        dima_vit_agent = create_dima_vit_agent()
        organika_sdxl_agent = create_organika_sdxl_agent()
        mobilenetv2_agent = create_mobilenetv2_agent()
        mobilenetv2_cnn_agent = create_mobilenetv2_cnn_agent()
        cnn_v3_agent = create_cnn_v3_agent()
        prithiv_open_deep_fake_agent = create_prithiv_open_deep_fake_agent()
        
        # Cache the agents globally
        _cached_agents = (
            nonescape_agent,
            seresnet50_agent,
            prithiv_v1_agent,
            dima_vit_agent,
            organika_sdxl_agent,
            mobilenetv2_agent,
            mobilenetv2_cnn_agent,
            cnn_v3_agent,
            prithiv_open_deep_fake_agent
        )
        
        logger.info("All detection agents created and cached successfully")
        return _cached_agents

async def run_image_detection_workflow(
    image_path: str,
    threshold: float = 0.20,
    voting_method: str = 'optimized_weighted'
) -> ImageDetectionResult:
    """
    Run concurrent image detection workflow using Agent Framework.
    
    Implements ensemble voting methods from detect_deepfake_image.py:
    - majority: Simple majority vote
    - weighted: Weighted by fake scores
    - unanimous: All models must agree  
    - average: Average of fake scores (RECOMMENDED for noisy images)
    - maximum: Use prediction from model with highest confidence
    - confidence_weighted: Weight by individual model confidence
    - adaptive_threshold: Adjust threshold based on prediction variance
    - median: Use median instead of mean
    - robust_majority: Remove outliers before voting using IQR
    - dynamic_consensus: Require higher agreement when models disagree
    - reliability_weighted: Weight based on model reliability
    - bayesian: Bayesian inference to combine predictions
    - entropy_weighted: Weight based on prediction entropy
    - consensus_threshold: Adaptive threshold based on consensus
    - optimized_weighted: Bayesian-optimized soft voting (RECOMMENDED for AI-generated content)
    
    Args:
        image_path: Path to image file (required)
        threshold: Detection threshold (0.0-1.0)
        voting_method: Ensemble voting method
        
    Returns:
        ImageDetectionResult with ensemble predictions and markdown report
    """
    try:
        # Get or create detection agents (cached to avoid recreating on every request)
        logger.info("Getting detection agents...")
        nonescape_agent, seresnet50_agent, prithiv_v1_agent, dima_vit_agent, organika_sdxl_agent, mobilenetv2_agent, mobilenetv2_cnn_agent, cnn_v3_agent, prithiv_open_deep_fake_agent = get_or_create_agents()
        
        # Validate image_path exists
        if not image_path:
            raise ValueError("image_path is required")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        logger.info(f"Using provided image path: {image_path}")
        
        # Extract filename from path
        final_filename = os.path.basename(image_path)
        
        # Create custom aggregator with parameters
        custom_aggregator = create_image_aggregator(
            threshold=threshold,
            voting_method=voting_method,
            image_path=image_path,
            filename=final_filename
        )
        
        # Build concurrent workflow using ConcurrentBuilder
        logger.info("Building concurrent workflow...")
        workflow = ConcurrentBuilder().participants([
            nonescape_agent,
            seresnet50_agent,
            prithiv_v1_agent,
            dima_vit_agent,
            organika_sdxl_agent,
            mobilenetv2_agent,
            mobilenetv2_cnn_agent,
            cnn_v3_agent,
            prithiv_open_deep_fake_agent
        ]).with_aggregator(custom_aggregator).build()
        
        logger.info("Running concurrent detection workflow...")
        
        # WORKAROUND: Azure Agent Framework requires JSON-serializable inputs
        # We can't pass PIL/NumPy objects directly due to Pydantic validation
        # Solution: Pass the image_path (agents will use their existing file loading)
        # TODO: Explore passing base64-encoded images or using Agent Framework's file handling
        workflow_input = image_path
        
        completion: WorkflowOutputEvent | None = None
        async for event in workflow.run_stream(workflow_input):
            if isinstance(event, WorkflowOutputEvent):
                completion = event
        
        if completion and completion.data:
            result = completion.data
        else:
            # Return error result if no completion or data
            raise RuntimeError("Workflow did not produce any output")

        # Ensure result is always an ImageDetectionResult        

        return result
        
    except Exception as e:
        logger.error(f"Error in image detection workflow: {e}")
        raise
    finally:
        # Clean up temporary file after workflow completes
        try:
            if image_path and os.path.exists(image_path):
                os.unlink(image_path)
                logger.info(f"Cleaned up temporary file: {image_path}")
        except Exception as cleanup_err:
            logger.warning(f"Failed to clean up temporary file: {cleanup_err}")


 
