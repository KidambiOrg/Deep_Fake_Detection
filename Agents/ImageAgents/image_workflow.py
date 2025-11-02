"""Image Detection Workflow using Agent Framework with Ensemble Voting."""

from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential
from agent_framework import ConcurrentBuilder
from Agents.ImageAgents.image_result_aggregator import create_image_aggregator
from Agents.ImageAgents.nonescape_agent import create_nonescape_agent
from Agents.ImageAgents.seresnet50_agent import create_seresnet50_agent
from Agents.ImageAgents.prithiv_v1_agent import create_prithiv_v1_agent
from Agents.ImageAgents.dima_vit_agent import create_dima_vit_agent
from Agents.ImageAgents.organika_sdxl_agent import create_organika_sdxl_agent
from outputModel.image_detection_output_model import ImageDetectionResult, IndividualModelResult, VotingResult
from imagedetectionlib.model_registry import get_model_info
from utils.image_file_helper import load_and_save_to_temp
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
        Tuple of (nonescape_agent, seresnet50_agent, prithiv_v1_agent, dima_vit_agent, organika_sdxl_agent)
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
        
        # Cache the agents globally
        _cached_agents = (
            nonescape_agent,
            seresnet50_agent,
            prithiv_v1_agent,
            dima_vit_agent,
            organika_sdxl_agent
        )
        
        logger.info("All detection agents created and cached successfully")
        return _cached_agents

async def run_image_detection_workflow(
    image_path: Optional[str] = None,
    base64_image: Optional[str] = None,
    threshold: float = 0.20,
    voting_method: str = 'weighted'
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
    
    Args:
        image_path: Path to image file (optional if base64_image provided)
        base64_image: Base64-encoded image (optional if image_path provided)
        threshold: Detection threshold (0.0-1.0)
        voting_method: Ensemble voting method
        
    Returns:
        ImageDetectionResult with ensemble predictions and markdown report
    """
    # Initialize temp_image_path early to avoid "possibly unbound" errors in exception handlers
    temp_image_path = None
    
    try:
        # Get or create detection agents (cached to avoid recreating on every request)
        logger.info("Getting detection agents...")
        nonescape_agent, seresnet50_agent, prithiv_v1_agent, dima_vit_agent, organika_sdxl_agent = get_or_create_agents()
        
        # Load image and save to temporary file
        temp_image_path = load_and_save_to_temp(
            image_path=image_path,
            base64_image=base64_image
        )
        
        # Create custom aggregator with parameters
        img_path = temp_image_path if temp_image_path else (image_path or "")
        custom_aggregator = create_image_aggregator(
            threshold=threshold,
            voting_method=voting_method,
            image_path=img_path
        )
        
        # Build concurrent workflow using ConcurrentBuilder
        logger.info("Building concurrent workflow...")
        workflow = ConcurrentBuilder().participants([
            nonescape_agent,
            seresnet50_agent,
            prithiv_v1_agent,
            dima_vit_agent,
            organika_sdxl_agent
        ]).with_aggregator(custom_aggregator).build()
        
        logger.info("Running concurrent detection workflow...")
        
        # Run the workflow - pass file path reference instead of base64
        events = await workflow.run(temp_image_path)
        
        # Get the aggregated output from the custom aggregator
        # The aggregator returns a single ImageDetectionResult, but get_outputs() returns it in a list
        outputs = events.get_outputs()
        
        if not outputs:
            raise RuntimeError("No outputs returned from workflow")
        
        # The aggregator returns ImageDetectionResult directly
        result = outputs[0]
        
        logger.info(f"Workflow completed: {result.prediction} with {result.confidence:.2%} confidence")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in image detection workflow: {e}")
        # Clean up temporary file on error
        try:
            import os
            if temp_image_path and os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
                logger.info(f"Cleaned up temporary file after error: {temp_image_path}")
        except Exception as cleanup_err:
            logger.warning(f"Failed to clean up temporary file: {cleanup_err}")
        raise


def generate_markdown_report(result: ImageDetectionResult) -> str:
    """Generate comprehensive markdown report for image detection results."""
    
    markdown = f"""# Deepfake Detection Report

## Overall Result
- **Image**: {result.imageFile}
- **Prediction**: **{result.prediction}**
- **Confidence**: {result.confidence:.2%}
- **Voting Method**: {result.votingMethod}

## Ensemble Voting
- **Fake Votes**: {result.voting.fake}
- **Real Votes**: {result.voting.real}
- **Total Models**: {result.voting.total}

## Average Scores
- **Average Fake Score**: {result.averageFake:.2%}
- **Average Real Score**: {result.averageReal:.2%}

## Individual Model Results

| Model | Prediction | Confidence | Fake Score | Real Score |
|-------|------------|------------|------------|------------|
"""
    
    for model_result in result.individualModelResults:
        markdown += f"| {model_result.model} | {model_result.prediction} | {model_result.confidence:.2%} | {model_result.fakeScore:.2%} | {model_result.realScore:.2%} |\n"
    
    # Add recommendation
    markdown += f"\n## Recommendation\n\n"
    
    if result.prediction == 'FAKE':
        if result.confidence > 0.8:
            markdown += "⚠️ **High confidence deepfake detection.** Very likely to be fake.\n"
        elif result.confidence > 0.6:
            markdown += "⚠️ **Moderate confidence deepfake detection.** Likely to be fake.\n"
        else:
            markdown += "⚠️ **Low confidence deepfake detection.** Consider manual review.\n"
    else:
        if result.confidence > 0.8:
            markdown += "✅ **High confidence real image.** Very likely to be authentic.\n"
        elif result.confidence > 0.6:
            markdown += "✅ **Moderate confidence real image.** Likely to be authentic.\n"
        else:
            markdown += "⚠️ **Borderline case.** Consider manual review or trying different voting methods.\n"
    
    return markdown
