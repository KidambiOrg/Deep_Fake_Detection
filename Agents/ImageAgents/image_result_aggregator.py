"""
Custom aggregator for image detection workflow results.
Implements ensemble voting and result summarization based on agent-framework patterns.
"""

import logging
import os
from typing import Any, Callable, List
import numpy as np
from outputModel.image_detection_output_model import (
    ImageDetectionResult,
    IndividualModelResult,
    VotingResult
)
from utils.bayesian_weight_optimizer import (
    apply_weighted_soft_voting,
    get_optimized_weights
)
from utils.logistic_regression_voter import apply_logistic_regression_voting

logger = logging.getLogger(__name__)


def create_image_aggregator(
    threshold: float,
    voting_method: str,
    image_path: str = "",
    filename: str = ""
) -> Callable:
    """
    Create a custom aggregator function with specific parameters.
    
    This factory function creates an aggregator callback that can be used with
    ConcurrentBuilder().with_aggregator(). The returned function captures the
    threshold, voting_method, image_path, and filename parameters via closure.
    
    Args:
        threshold: Detection threshold (0.0-1.0)
        voting_method: Ensemble voting method
        image_path: Path to the image being analyzed
        filename: Name of the image file for tracking
        
    Returns:
        Async aggregator function that accepts results and returns ImageDetectionResult
    """
    async def custom_aggregator(results: List[Any]) -> ImageDetectionResult:
        return await summarize_results(
            results=results,
            threshold=threshold,
            voting_method=voting_method,
            image_path=image_path,
            filename=filename
        )
    
    return custom_aggregator


async def summarize_results(
    results: list[Any],
    threshold: float = 0.20,
    voting_method: str = 'weighted',
    image_path: str = "",
    filename: str = ""
) -> ImageDetectionResult:
    """
    Custom aggregator callback that consolidates individual model detection results
    into a single ensemble prediction with voting.
    
    Based on the pattern from agent-framework's concurrent_custom_aggregator.py,
    this function receives results from parallel agent executions and synthesizes
    them into a unified output.
    
    Args:
        results: List of AgentExecutorResponse objects from workflow
        threshold: Detection threshold (0.0-1.0)
        voting_method: Ensemble voting method
        image_path: Optional path to the analyzed image
        filename: Name of the image file for tracking
        
    Returns:
        ImageDetectionResult with ensemble predictions
    """
    logger.info(f"Aggregating {len(results)} detection results")
    
    # Extract IndividualModelResult objects from agent responses
    individual_results: List[IndividualModelResult] = []
    
    for idx, result in enumerate(results):
        logger.info(f"Processing result {idx}: type={type(result)}")
        
        try:
            # Get messages from agent_run_response
            messages = getattr(result.agent_run_response, "messages", [])
            
            if messages:
                # Extract the final assistant message
                final_msg = messages[-1]
                
                if hasattr(final_msg, 'text'):
                    msg_text = final_msg.text
                    
                    # Check if it's already an IndividualModelResult object
                    if isinstance(msg_text, IndividualModelResult):
                        logger.info(f"  Found IndividualModelResult object")
                        individual_results.append(msg_text)
                    # Parse from dictionary
                    elif isinstance(msg_text, dict):
                        logger.info(f"  Parsing as dict")
                        individual_results.append(IndividualModelResult(**msg_text))
                    # Parse from JSON string
                    elif isinstance(msg_text, str):
                        import json
                        stripped_text = msg_text.strip()
                        if stripped_text.startswith('{') or stripped_text.startswith('['):
                            logger.info(f"  Parsing as JSON string")
                            result_data = json.loads(stripped_text)
                            individual_results.append(IndividualModelResult(**result_data))
                        else:
                            logger.info(f"  Skipping non-JSON text: {msg_text[:200]}")
        except Exception as e:
            executor_id = getattr(result, 'executor_id', 'unknown')
            logger.warning(f"Failed to parse result from {executor_id}: {e}")
    
    if not individual_results:
        raise RuntimeError("No detection results returned from any model")
    
    logger.info(f"Extracted {len(individual_results)} individual model results")
    
    # Apply ensemble voting logic
    ensemble_result = apply_ensemble_voting(
        individual_results=individual_results,
        threshold=threshold,
        voting_method=voting_method,
        image_path=image_path,
        filename=filename
    )
    
    return ensemble_result


def apply_ensemble_voting(
    individual_results: List[IndividualModelResult],
    threshold: float,
    voting_method: str,
    image_path: str = "",
    filename: str = ""
) -> ImageDetectionResult:
    """
    Apply ensemble voting logic from detect_deepfake_image.py.
    
    This implements the exact voting methods from EnsembleDetector.detect().
    
    Voting methods:
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
    """
    # Calculate voting summary
    fake_votes = sum(1 for r in individual_results if r.prediction == 'FAKE')
    real_votes = len(individual_results) - fake_votes
    
    voting_summary = {
        'fake': fake_votes,
        'real': real_votes,
        'total': len(individual_results)
    }
    
    # Extract scores - ONLY from models that returned valid predictions
    # Exclude models with zero confidence (indicates error or no prediction)
    valid_results = [r for r in individual_results if r.confidence > 0.0]
    
    if valid_results:
        fake_scores = [r.fakeScore for r in valid_results]
        real_scores = [r.realScore for r in valid_results]
        
        avg_fake_score = sum(fake_scores) / len(fake_scores)
        avg_real_score = sum(real_scores) / len(real_scores)
    else:
        # Fallback if all models failed
        fake_scores = [r.fakeScore for r in individual_results]
        real_scores = [r.realScore for r in individual_results]
        
        avg_fake_score = sum(fake_scores) / len(fake_scores) if fake_scores else 0.0
        avg_real_score = sum(real_scores) / len(real_scores) if real_scores else 0.0
    
    # Determine final prediction based on voting method
    if voting_method == 'majority':
        is_deepfake = fake_votes > real_votes
        
        if is_deepfake:
            fake_model_scores = [fake_scores[i] for i in range(len(individual_results)) 
                                if individual_results[i].prediction == 'FAKE']
            confidence = np.mean(fake_model_scores) if fake_model_scores else avg_fake_score
        else:
            real_model_scores = [real_scores[i] for i in range(len(individual_results)) 
                                if individual_results[i].prediction == 'REAL']
            confidence = np.mean(real_model_scores) if real_model_scores else avg_real_score
    
    elif voting_method == 'weighted':
        # Enhanced weighted voting: Weight by confidence AND actual scores
        # Each model contributes its fake/real scores weighted by its confidence
        weighted_fake_sum = 0.0
        weighted_real_sum = 0.0
        total_weight = 0.0
        
        for i, result in enumerate(individual_results):
            # Use the model's confidence as the weight
            weight = result.confidence
            
            # Weight the actual fake/real scores by confidence
            weighted_fake_sum += fake_scores[i] * weight
            weighted_real_sum += real_scores[i] * weight
            
            total_weight += weight
        
        # Calculate confidence-weighted averages
        if total_weight > 0:
            weighted_avg_fake = weighted_fake_sum / total_weight
            weighted_avg_real = weighted_real_sum / total_weight
        else:
            weighted_avg_fake = avg_fake_score
            weighted_avg_real = avg_real_score
        
        # Make prediction based on which weighted average is higher
        is_deepfake = weighted_avg_fake > weighted_avg_real
        
        if is_deepfake:
            fake_model_scores = [fake_scores[i] for i in range(len(individual_results)) 
                                if individual_results[i].prediction == 'FAKE']
            confidence = np.mean(fake_model_scores) if fake_model_scores else weighted_avg_fake
        else:
            real_model_scores = [real_scores[i] for i in range(len(individual_results)) 
                                if individual_results[i].prediction == 'REAL']
            confidence = np.mean(real_model_scores) if real_model_scores else weighted_avg_real
    
    elif voting_method == 'unanimous':
        is_deepfake = fake_votes == len(individual_results)
        confidence = avg_fake_score if is_deepfake else avg_real_score
    
    elif voting_method == 'average':
        is_deepfake = avg_fake_score >= threshold
        
        if is_deepfake:
            fake_model_scores = [fake_scores[i] for i in range(len(individual_results)) 
                                if individual_results[i].prediction == 'FAKE']
            confidence = np.mean(fake_model_scores) if fake_model_scores else avg_fake_score
        else:
            real_model_scores = [real_scores[i] for i in range(len(individual_results)) 
                                if individual_results[i].prediction == 'REAL']
            confidence = np.mean(real_model_scores) if real_model_scores else avg_real_score
    
    elif voting_method == 'maximum':
        max_confidence_result = max(individual_results, key=lambda r: r.confidence)
        is_deepfake = max_confidence_result.prediction == 'FAKE'
        confidence = max_confidence_result.confidence
    
    elif voting_method == 'confidence_weighted':
        weighted_fake_sum = sum(fake_scores[i] * individual_results[i].confidence 
                               for i in range(len(individual_results)))
        total_confidence = sum(r.confidence for r in individual_results)
        weighted_avg_fake = weighted_fake_sum / total_confidence if total_confidence > 0 else 0.0
        
        is_deepfake = weighted_avg_fake >= threshold
        confidence = weighted_avg_fake
    
    elif voting_method == 'adaptive_threshold':
        score_variance = np.var(fake_scores) if len(fake_scores) > 1 else 0.0
        adjusted_threshold = threshold * (1 + score_variance)
        is_deepfake = avg_fake_score >= adjusted_threshold
        
        if is_deepfake:
            fake_model_scores = [fake_scores[i] for i in range(len(individual_results)) 
                                if individual_results[i].prediction == 'FAKE']
            confidence = np.mean(fake_model_scores) if fake_model_scores else avg_fake_score
        else:
            real_model_scores = [real_scores[i] for i in range(len(individual_results)) 
                                if individual_results[i].prediction == 'REAL']
            confidence = np.mean(real_model_scores) if real_model_scores else avg_real_score
    
    elif voting_method == 'median':
        median_fake_score = np.median(fake_scores) if fake_scores else 0.0
        median_real_score = np.median(real_scores) if real_scores else 0.0
        is_deepfake = median_fake_score >= threshold
        confidence = median_fake_score if is_deepfake else median_real_score
    
    elif voting_method == 'robust_majority':
        if len(fake_scores) >= 4:
            q1_fake, q3_fake = np.percentile(fake_scores, [25, 75])
            iqr_fake = q3_fake - q1_fake
            lower_bound = q1_fake - 1.5 * iqr_fake
            upper_bound = q3_fake + 1.5 * iqr_fake
            
            filtered_results = []
            for i, result in enumerate(individual_results):
                if lower_bound <= fake_scores[i] <= upper_bound:
                    filtered_results.append(result)
            
            if filtered_results:
                filtered_fake_votes = sum(1 for r in filtered_results if r.prediction == 'FAKE')
                filtered_real_votes = len(filtered_results) - filtered_fake_votes
                is_deepfake = filtered_fake_votes > filtered_real_votes
                
                filtered_fake_scores = [r.fakeScore for r in filtered_results]
                filtered_real_scores = [r.realScore for r in filtered_results]
                confidence = max(np.mean(filtered_fake_scores), np.mean(filtered_real_scores))
            else:
                is_deepfake = fake_votes > real_votes
                confidence = max(avg_fake_score, avg_real_score)
        else:
            is_deepfake = fake_votes > real_votes
            confidence = max(avg_fake_score, avg_real_score)
    
    elif voting_method == 'dynamic_consensus':
        total_models = len(individual_results)
        disagreement_ratio = abs(fake_votes - real_votes) / total_models
        
        if disagreement_ratio <= 0.2:
            required_majority = 0.5
        elif disagreement_ratio <= 0.4:
            required_majority = 0.6
        else:
            required_majority = 0.75
        
        fake_ratio = fake_votes / total_models
        is_deepfake = fake_ratio >= required_majority
        confidence = max(avg_fake_score, avg_real_score)
    
    elif voting_method == 'reliability_weighted':
        model_weights = {
            'nonescape-full': 0.95,  # Increased: Best for AI-generated content like DALL-E
            'seresnet50': 0.82,      # Increased: Good general detector
            'prithiv-v1': 0.72,      # Decreased: Lower reliability
            'dima-vit': 0.78,        # Decreased: Can miss AI-generated content
            'organika-sdxl': 0.68    # Decreased: Can miss AI-generated content
        }
        
        weighted_fake_votes = 0.0
        weighted_total_votes = 0.0
        weighted_fake_sum = 0.0
        weighted_real_sum = 0.0
        
        for i, result in enumerate(individual_results):
            weight = model_weights.get(result.model, 0.75)
            
            weighted_total_votes += weight
            if result.prediction == 'FAKE':
                weighted_fake_votes += weight
            
            weighted_fake_sum += fake_scores[i] * weight
            weighted_real_sum += real_scores[i] * weight
        
        if weighted_total_votes > 0:
            weighted_fake_ratio = weighted_fake_votes / weighted_total_votes
            weighted_avg_fake = weighted_fake_sum / weighted_total_votes
            weighted_avg_real = weighted_real_sum / weighted_total_votes
            
            is_deepfake = weighted_fake_ratio > 0.5
            confidence = weighted_avg_fake if is_deepfake else weighted_avg_real
        else:
            is_deepfake = fake_votes > real_votes
            confidence = max(avg_fake_score, avg_real_score)
    
    elif voting_method == 'bayesian':
        prior_fake = 0.5
        log_likelihood_fake = 0.0
        log_likelihood_real = 0.0
        
        for result in individual_results:
            model_confidence = result.confidence
            if result.prediction == 'FAKE':
                log_likelihood_fake += np.log(model_confidence + 1e-10)
                log_likelihood_real += np.log(1 - model_confidence + 1e-10)
            else:
                log_likelihood_fake += np.log(1 - model_confidence + 1e-10)
                log_likelihood_real += np.log(model_confidence + 1e-10)
        
        likelihood_fake = np.exp(log_likelihood_fake)
        likelihood_real = np.exp(log_likelihood_real)
        
        evidence = likelihood_fake * prior_fake + likelihood_real * (1 - prior_fake)
        if evidence > 0:
            posterior_fake = (likelihood_fake * prior_fake) / evidence
        else:
            posterior_fake = prior_fake
        
        is_deepfake = posterior_fake >= threshold
        confidence = posterior_fake
    
    elif voting_method == 'entropy_weighted':
        weighted_fake_sum = 0.0
        total_weight = 0.0
        
        for i, result in enumerate(individual_results):
            fake_score = fake_scores[i]
            real_score = real_scores[i]
            
            p_fake = max(fake_score, 1e-10)
            p_real = max(real_score, 1e-10)
            entropy = -(p_fake * np.log2(p_fake) + p_real * np.log2(p_real))
            
            weight = 1.0 / (1.0 + entropy)
            
            weighted_fake_sum += fake_score * weight
            total_weight += weight
        
        if total_weight > 0:
            entropy_weighted_fake = weighted_fake_sum / total_weight
            is_deepfake = entropy_weighted_fake >= threshold
            confidence = entropy_weighted_fake if is_deepfake else (1 - entropy_weighted_fake)
        else:
            is_deepfake = avg_fake_score >= threshold
            confidence = avg_fake_score if is_deepfake else avg_real_score
    
    elif voting_method == 'consensus_threshold':
        max_votes = max(fake_votes, real_votes)
        consensus_strength = max_votes / len(individual_results)
        adjusted_threshold = threshold * (2 - consensus_strength)
        
        is_deepfake = avg_fake_score >= adjusted_threshold
        confidence = avg_fake_score if is_deepfake else avg_real_score
    
    elif voting_method == 'optimized_weighted':
        # Weighted soft voting with Bayesian-optimized model weights
        # Uses pre-computed optimal weights learned from validation data
        # Combines probability scores with adaptive confidence weighting
        
        # Get optimized weights for all models
        optimized_weights = get_optimized_weights()
        
        # Apply weighted soft voting with adaptive weighting
        is_deepfake, confidence = apply_weighted_soft_voting(
            individual_results=[
                {
                    'model': r.model,
                    'fakeScore': r.fakeScore,
                    'realScore': r.realScore,
                    'confidence': r.confidence
                }
                for r in individual_results
            ],
            weights=optimized_weights,
            adaptive=True  # Enable adaptive weighting based on uncertainty
        )
    
    elif voting_method == 'logistic_regression':
        # Logistic Regression meta-classifier (stacking approach)
        # Trains a meta-model to learn optimal combination of base models
        # Based on sklearn VotingClassifier approach
        # Reference: https://www.geeksforgeeks.org/machine-learning/voting-classifier/
        
        prediction, confidence, avg_fake, avg_real = apply_logistic_regression_voting(
            results=[
                {
                    'model': r.model,
                    'fakeScore': r.fakeScore,
                    'realScore': r.realScore,
                    'confidence': r.confidence
                }
                for r in individual_results
            ]
        )
        
        is_deepfake = (prediction == 'FAKE')
    
    else:
        raise ValueError(f"Unknown voting method: {voting_method}")
    
    # Determine image filename - use provided filename first, fallback to path extraction
    if filename:
        image_file = filename
    elif image_path:
        image_file = os.path.basename(image_path)
    else:
        image_file = "uploaded_image"
    
    # Create VotingResult
    voting_result = VotingResult(
        fake=voting_summary.get('fake', 0),
        real=voting_summary.get('real', 0),
        total=voting_summary.get('total', len(individual_results))
    )
    
    # Create ImageDetectionResult
    result = ImageDetectionResult(
        imageFile=image_file,
        prediction='FAKE' if is_deepfake else 'REAL',
        confidence=float(confidence),
        averageFake=avg_fake_score,
        averageReal=avg_real_score,
        votingMethod=voting_method,
        voting=voting_result,
        individualModelResults=individual_results,
        markdown=""  # Will be populated next
    )
    
    # Generate and populate markdown report
    result.markdown = generate_markdown_report(result)
    
    return result


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
    
    
    return markdown
