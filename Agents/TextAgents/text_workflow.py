"""Text Detection Workflow using Agent Framework with Ensemble Voting."""

from agent_framework import ConcurrentBuilder, WorkflowOutputEvent
from Agents.TextAgents.text_result_aggregator import create_text_aggregator
from Agents.TextAgents.bert_ml_agent import create_bert_ml_agent
from Agents.TextAgents.bow_ml_agent import create_bow_ml_agent
from Agents.TextAgents.charcnn_agent import create_charcnn_agent
from Agents.TextAgents.distilbert_agent import create_distilbert_agent
from outputModel.text_detection_output_model import TextDetectionResult
import os
import logging
import threading
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Global agent cache - initialized once and reused
_cached_text_agents = None
_text_agents_lock = threading.Lock()

def get_or_create_text_agents():
    """
    Get or create text detection agents with singleton pattern.
    Agents and their models are loaded only once and cached globally.
    Thread-safe initialization prevents duplicate loading.
        
    Returns:
        Tuple of (bert_ml_agent, bow_ml_agent, charcnn_agent, distilbert_agent)
    """
    global _cached_text_agents
    
    # Fast path: if agents already exist, return immediately
    if _cached_text_agents is not None:
        logger.info("Reusing cached text agents")
        return _cached_text_agents
    
    # Slow path: need to create agents (with lock to prevent concurrent initialization)
    with _text_agents_lock:
        # Double-check after acquiring lock
        if _cached_text_agents is not None:
            logger.info("Reusing cached text agents (after lock)")
            return _cached_text_agents
        
        logger.info("Initializing text detection agents for the first time...")
        
        try:
            # Create agents once - each agent's tool will also cache its model
            logger.info("Creating BERT-ML agent...")
            bert_ml_agent = create_bert_ml_agent()
            
            logger.info("Creating BoW-ML agent...")
            bow_ml_agent = create_bow_ml_agent()
            
            logger.info("Creating CharCNN agent...")
            charcnn_agent = create_charcnn_agent()
            
            logger.info("Creating DistilBERT agent...")
            distilbert_agent = create_distilbert_agent()
            
            # Cache the agents globally
            _cached_text_agents = (
                bert_ml_agent,
                bow_ml_agent,
                charcnn_agent,
                distilbert_agent
            )
            
            logger.info("All text agents created and cached successfully")
            return _cached_text_agents
            
        except Exception as e:
            logger.error(f"Failed to create text agents: {e}")
            logger.exception(e)
            raise RuntimeError(f"Failed to initialize text detection agents: {e}") from e

async def run_text_detection_workflow(text: str) -> TextDetectionResult:
    """Run concurrent text detection workflow using Agent Framework."""
    try:
        # Get or create detection agents (cached to avoid recreating on every request)
        logger.info("Getting text detection agents...")
        bert_ml_agent, bow_ml_agent, charcnn_agent, distilbert_agent = get_or_create_text_agents()
        
        # Create custom aggregator with parameters
        custom_aggregator = create_text_aggregator(text=text)
        
        # Build concurrent workflow using ConcurrentBuilder
        logger.info("Building concurrent text workflow...")
        workflow = ConcurrentBuilder().participants([
            bert_ml_agent,
            bow_ml_agent,
            charcnn_agent,
            distilbert_agent
        ]).with_aggregator(custom_aggregator).build()
        
        logger.info(f"Running concurrent text detection workflow with text: {text[:100]}...")
        
        # Run the workflow - pass text to stream
        completion: WorkflowOutputEvent | None = None
        async for event in workflow.run_stream(text):
            if isinstance(event, WorkflowOutputEvent):
                completion = event
        
        if completion and completion.data:
            result = completion.data
        else:
            # Return error result if no completion or data
            raise RuntimeError("Workflow did not produce any output")
        
        logger.info(f"Workflow completed: {result.prediction} with {result.confidence:.2%} confidence")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in text detection workflow: {e}")
        raise
