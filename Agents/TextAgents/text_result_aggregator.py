"""
Custom aggregator for text detection workflow results.
Implements ensemble voting and result summarization based on agent-framework patterns.
"""

import logging
import re
from typing import Any, Callable, List
import numpy as np
from outputModel.text_detection_output_model import (
    TextDetectionResult,
    IndividualModelScore,
    TextMetadata
)

logger = logging.getLogger(__name__)


def create_text_aggregator(
    text: str
) -> Callable:
    """
    Create a custom aggregator function with specific parameters.
    
    This factory function creates an aggregator callback that can be used with
    ConcurrentBuilder().with_aggregator(). The returned function captures the
    text parameter via closure.
    
    Args:
        text: The text being analyzed
        
    Returns:
        Async aggregator function that accepts results and returns TextDetectionResult
    """
    async def custom_aggregator(results: List[Any]) -> TextDetectionResult:
        return await summarize_results(
            results=results,
            text=text
        )
    
    return custom_aggregator


async def summarize_results(
    results: list[Any],
    text: str
) -> TextDetectionResult:
    """
    Custom aggregator callback that consolidates individual model detection results
    into a single ensemble prediction with voting.
    
    Based on the pattern from agent-framework's concurrent_custom_aggregator.py,
    this function receives results from parallel agent executions and synthesizes
    them into a unified output.
    
    Args:
        results: List of AgentExecutorResponse objects from workflow
        text: The text being analyzed
        
    Returns:
        TextDetectionResult with ensemble predictions
    """
    logger.info(f"Aggregating {len(results)} text detection results")
    
    # Extract IndividualModelScore objects from agent responses
    individual_scores: List[IndividualModelScore] = []
    
    for idx, result in enumerate(results):
        logger.info(f"Processing result {idx}: type={type(result)}")
        
        try:
            # Get messages from agent_run_response
            messages = getattr(result.agent_run_response, "messages", [])
            
            if messages:
                # Extract all messages and look for tool outputs
                for msg in messages:
                    if hasattr(msg, 'text'):
                        msg_text = msg.text
                        
                        # Skip if it's the original input text
                        if isinstance(msg_text, str) and msg_text.strip() == text.strip():
                            logger.debug(f"Skipping initial input message")
                            continue
                        
                        # Check if it's already an IndividualModelScore object
                        if isinstance(msg_text, IndividualModelScore):
                            logger.info(f"  Found IndividualModelScore object")
                            individual_scores.append(msg_text)
                        # Parse from dictionary
                        elif isinstance(msg_text, dict):
                            logger.info(f"  Parsing as dict")
                            individual_scores.append(IndividualModelScore(**msg_text))
                        # Parse from JSON string
                        elif isinstance(msg_text, str):
                            import json
                            stripped_text = msg_text.strip()
                            if stripped_text.startswith('{'):
                                logger.info(f"  Parsing as JSON string")
                                score_data = json.loads(stripped_text)
                                individual_scores.append(IndividualModelScore(**score_data))
                            else:
                                logger.debug(f"  Skipping non-JSON text: {msg_text[:100]}")
        except Exception as e:
            executor_id = getattr(result, 'executor_id', 'unknown')
            logger.warning(f"Failed to parse result from {executor_id}: {e}")
    
    if not individual_scores:
        raise RuntimeError("No detection results returned from any model")
    
    logger.info(f"Extracted {len(individual_scores)} individual model scores")
    
    # Apply ensemble voting logic
    ensemble_result = apply_ensemble_voting(
        individual_scores=individual_scores,
        text=text
    )
    
    return ensemble_result


def apply_ensemble_voting(
    individual_scores: List[IndividualModelScore],
    text: str
) -> TextDetectionResult:
    """
    Apply ensemble voting logic for text detection.
    
    Uses weighted voting based on average bot/human scores.
    """
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
    
    # Calculate text metadata
    metadata = calculate_text_metadata(text)
    
    # Create voting breakdown
    voting_breakdown = {'bot': bot_votes, 'human': human_votes}
    
    # Create TextDetectionResult
    result = TextDetectionResult(
        text=text,
        prediction=prediction,
        confidence=float(confidence),
        average_bot_score=float(avg_bot),
        average_human_score=float(avg_human),
        voting_method='weighted',
        metadata=metadata,
        individual_scores=individual_scores,
        voting_breakdown=voting_breakdown,
        markdown=""  # Will be populated next
    )
    
    # Generate and populate markdown report
    result.markdown = generate_text_markdown(result)
    
    return result


def calculate_text_metadata(text: str) -> TextMetadata:
    """Calculate metadata for the analyzed text."""
    # Count words (split by whitespace)
    words = text.split()
    word_count = len(words)
    
    # Count characters (including spaces)
    char_count = len(text)
    
    # Count sentences (split by sentence-ending punctuation)
    # Handle common sentence endings: . ! ?
    sentences = re.split(r'[.!?]+', text)
    # Filter out empty strings
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)
    
    return TextMetadata(
        wordCount=word_count,
        totalSize=char_count,
        characterCount=char_count,
        sentenceCount=sentence_count
    )


def generate_text_markdown(result: TextDetectionResult) -> str:
    """Generate markdown report for text detection."""
    markdown = f"""# AI-Generated Text Detection Report

## Overall Result
- **Prediction**: **{result.prediction}**
- **Confidence**: {result.confidence:.2%}

## Text Metadata
- **Word Count**: {result.metadata.wordCount}
- **Character Count**: {result.metadata.characterCount}
- **Sentence Count**: {result.metadata.sentenceCount}
- **Total Size**: {result.metadata.totalSize} characters

## Voting Breakdown
- **Bot/AI Votes**: {result.voting_breakdown['bot']}
- **Human Votes**: {result.voting_breakdown['human']}
- **Total Models**: {len(result.individual_scores)}

## Average Scores
- **Average Bot Score**: {result.average_bot_score:.2%}
- **Average Human Score**: {result.average_human_score:.2%}

## Individual Model Results

| Model | Prediction | Confidence | Bot Score | Human Score |
|-------|------------|------------|-----------|-------------|
"""
    
    for score in result.individual_scores:
        markdown += f"| {score.model_name} | {score.prediction} | {score.confidence:.2%} | {score.bot_score:.2%} | {score.human_score:.2%} |\n"
    
    # Add recommendation based on prediction
    markdown += f"\n## Recommendation\n\n"
    if 'AI' in result.prediction or 'bot' in result.prediction.lower():
        markdown += f"⚠️ **Text appears to be AI-generated** (confidence: {result.confidence:.2%})\n"
    else:
        markdown += f"✅ **Text appears to be human-written** (confidence: {result.confidence:.2%})\n"
    
    return markdown
