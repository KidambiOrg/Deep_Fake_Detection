"""Text Detection Workflow using Agent Framework with Ensemble Voting."""

from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential
from agent_framework import ConcurrentBuilder
from Agents.TextAgents.bert_ml_agent import create_bert_ml_agent
from Agents.TextAgents.bow_ml_agent import create_bow_ml_agent
from Agents.TextAgents.charcnn_agent import create_charcnn_agent
from Agents.TextAgents.distilbert_agent import create_distilbert_agent
from outputModel.text_detection_output_model import TextDetectionResult, IndividualModelScore
from textdetectionlib.model_registry import get_text_model_info
import numpy as np
import os
import logging
import json
import threading
from typing import List
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
        
        # Create agents once - each agent's tool will also cache its model
        bert_ml_agent = create_bert_ml_agent()
        bow_ml_agent = create_bow_ml_agent()
        charcnn_agent = create_charcnn_agent()
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

async def run_text_detection_workflow(text: str) -> TextDetectionResult:
    """Run concurrent text detection workflow using Agent Framework."""
    try:
        # Get or create detection agents (cached to avoid recreating on every request)
        logger.info("Getting text detection agents...")
        bert_ml_agent, bow_ml_agent, charcnn_agent, distilbert_agent = get_or_create_text_agents()
        
        # Build concurrent workflow using ConcurrentBuilder
        logger.info("Building concurrent text workflow...")
        workflow = ConcurrentBuilder().participants([
            bert_ml_agent,
            bow_ml_agent,
            charcnn_agent,
            distilbert_agent
        ]).build()
        
        # Prepare input for agents
        input_data = json.dumps({
            'text': text
        })
        
        logger.info("Running concurrent text detection workflow...")
        
        # Run the workflow - ConcurrentBuilder handles fan-out/fan-in automatically
        events = await workflow.run(input_data)
        
        # Get the aggregated outputs from all agents
        outputs = events.get_outputs()
        
        # Each agent's tools return IndividualModelScore objects
        # The workflow collects them into a list
        individual_scores: List[IndividualModelScore] = []
        
        for output in outputs:
            # Output should be a list of messages from each agent
            if isinstance(output, list):
                # Extract the tool results from the agent messages
                for msg in output:
                    if hasattr(msg, 'text'):
                        try:
                            # Try parsing as JSON first (if agent returns JSON string)
                            score_data = json.loads(msg.text)
                            score = IndividualModelScore(**score_data)
                            individual_scores.append(score)
                        except (json.JSONDecodeError, TypeError):
                            # If not JSON, check if it's already the model object
                            if isinstance(msg.text, IndividualModelScore):
                                individual_scores.append(msg.text)
            elif isinstance(output, IndividualModelScore):
                individual_scores.append(output)
        
        if not individual_scores:
            raise RuntimeError("No detection results returned from any model")
        
        # Calculate ensemble - using correct property names
        bot_scores = [s.bot_score for s in individual_scores]
        human_scores = [s.human_score for s in individual_scores]
        
        avg_bot = np.mean(bot_scores)
        avg_human = np.mean(human_scores)
        
        # Count votes
        bot_votes = sum(1 for s in individual_scores if s.prediction.lower() == 'bot')
        human_votes = len(individual_scores) - bot_votes
        
        # Determine prediction using weighted voting
        prediction = 'AI-generated' if avg_bot > avg_human else 'human-generated'
        confidence = max(avg_bot, avg_human)
        
        # Generate markdown report
        markdown = generate_text_markdown(
            individual_scores, 
            prediction, 
            float(confidence), 
            float(avg_bot), 
            float(avg_human),
            bot_votes,
            human_votes
        )
        
        result = TextDetectionResult(
            markdown=markdown,
            text=text,
            prediction=prediction,
            confidence=float(confidence),
            average_bot_score=float(avg_bot),
            average_human_score=float(avg_human),
            voting_method='weighted',
            individual_scores=individual_scores,
            voting_breakdown={'bot': bot_votes, 'human': human_votes}
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in text detection workflow: {e}")
        raise

def generate_text_markdown(
    individual_scores: List[IndividualModelScore],
    prediction: str,
    confidence: float,
    avg_bot: float,
    avg_human: float,
    bot_votes: int,
    human_votes: int
) -> str:
    """Generate markdown report for text detection."""
    markdown = f"""# AI-Generated Text Detection Report

## Overall Result
- **Prediction**: **{prediction}**
- **Confidence**: {confidence:.2%}

## Voting Breakdown
- **Bot/AI Votes**: {bot_votes}
- **Human Votes**: {human_votes}
- **Total Models**: {len(individual_scores)}

## Average Scores
- **Average Bot Score**: {avg_bot:.2%}
- **Average Human Score**: {avg_human:.2%}

## Individual Model Results

| Model | Prediction | Confidence | Bot Score | Human Score |
|-------|------------|------------|-----------|-------------|
"""
    
    for score in individual_scores:
        markdown += f"| {score.model_name} | {score.prediction} | {score.confidence:.2%} | {score.bot_score:.2%} | {score.human_score:.2%} |\n"
    
    # Add recommendation based on prediction
    markdown += f"\n## Recommendation\n\n"
    if 'AI' in prediction or 'bot' in prediction.lower():
        markdown += f"⚠️ **Text appears to be AI-generated** (confidence: {confidence:.2%})\n"
    else:
        markdown += f"✅ **Text appears to be human-written** (confidence: {confidence:.2%})\n"
    
    return markdown
