"""Image Detection Workflow using Agent Framework with Ensemble Voting."""

from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential
from agent_framework import ConcurrentBuilder
from Agents.ImageAgents.nonescape_agent import create_nonescape_agent
from Agents.ImageAgents.seresnet50_agent import create_seresnet50_agent
from Agents.ImageAgents.prithiv_v1_agent import create_prithiv_v1_agent
from Agents.ImageAgents.dima_vit_agent import create_dima_vit_agent
from Agents.ImageAgents.organika_sdxl_agent import create_organika_sdxl_agent
from outputModel.image_detection_output_model import ImageDetectionResult, IndividualModelResult, VotingResult
from imagedetectionlib.model_registry import get_model_info
import numpy as np
import os
import logging
import json
import base64
import io
from typing import List, Optional
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

def get_or_create_agents():
    """
    Get or create detection agents. Each agent manages its own lifecycle.
    
    Returns:
        Tuple of (nonescape_agent, seresnet50_agent, prithiv_v1_agent, dima_vit_agent, organika_sdxl_agent)
    """
    logger.info("Initializing detection agents...")
    
    # Each agent internally gets its display name from registry and checks for existing agent
    nonescape_agent = create_nonescape_agent()
    seresnet50_agent = create_seresnet50_agent()
    prithiv_v1_agent = create_prithiv_v1_agent()
    dima_vit_agent = create_dima_vit_agent()
    organika_sdxl_agent = create_organika_sdxl_agent()
    
    logger.info("All detection agents created successfully")
    
    return (
        nonescape_agent,
        seresnet50_agent,
        prithiv_v1_agent,
        dima_vit_agent,
        organika_sdxl_agent
    )

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
        
        # Build concurrent workflow using ConcurrentBuilder
        logger.info("Building concurrent workflow...")
        workflow = ConcurrentBuilder().participants([
            nonescape_agent,
            seresnet50_agent,
            prithiv_v1_agent,
            dima_vit_agent,
            organika_sdxl_agent
        ]).build()
        
        # Load and convert PIL Image to base64 for serialization
        logger.info("Loading and encoding image...")
        image = None
        image_base64 = None
        
        if base64_image:
            # Already have base64
            image_base64 = base64_image
            # Decode to verify and get size
            image_data = base64.b64decode(base64_image)
            image = Image.open(io.BytesIO(image_data))
        elif image_path:
            # Load from file path and convert to base64
            image = Image.open(image_path)
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            # Encode to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        else:
            raise ValueError("Either image_path or base64_image must be provided")
        
        logger.info(f"Loaded image: {image.size}, mode: {image.mode}")
        
        # Save image to temporary file and pass reference instead of base64
        # This avoids corruption issues when passing large strings through LLM
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(mode='wb', suffix='.png', delete=False)
        try:
            image.save(temp_file, format='PNG')
            temp_image_path = temp_file.name
            temp_file.close()
            logger.info(f"Saved image to temporary file: {temp_image_path}")
        except Exception as e:
            logger.error(f"Failed to save temporary image: {e}")
            temp_file.close()
            raise
        
        logger.info("Running concurrent detection workflow...")
        
        # Run the workflow - pass file path reference instead of base64
        events = await workflow.run(temp_image_path)
        
        # Get the aggregated outputs from all agents
        outputs = events.get_outputs()
        
        logger.info(f"Received {len(outputs)} outputs from workflow")
        
        # Each agent's tools return IndividualModelResult objects
        # The workflow collects them into a list
        individual_results: List[IndividualModelResult] = []
        
        for idx, output in enumerate(outputs):
            logger.info(f"Processing output {idx}: type={type(output)}")
            
            # Output should be a list of messages from each agent
            if isinstance(output, list):
                logger.info(f"  Output is list with {len(output)} items")
                # Extract the tool results from the agent messages
                for msg_idx, msg in enumerate(output):
                    logger.info(f"  Message {msg_idx}: type={type(msg)}, has_text={hasattr(msg, 'text')}")
                    if hasattr(msg, 'text'):
                        logger.info(f"    msg.text type={type(msg.text)}, value={str(msg.text)[:200] if msg.text else 'None'}")
                        try:
                            # Check for IndividualModelResult first (response_format returns this directly)
                            if isinstance(msg.text, IndividualModelResult):
                                logger.info(f"    Already IndividualModelResult object")
                                individual_results.append(msg.text)
                            # Tools return dictionaries, parse them
                            elif isinstance(msg.text, dict):
                                # Already a dictionary
                                logger.info(f"    Parsing as dict: {msg.text}")
                                result = IndividualModelResult(**msg.text)
                                individual_results.append(result)
                            elif isinstance(msg.text, str):
                                # Skip messages that are clearly not JSON (e.g., file paths, plain text)
                                stripped_text = msg.text.strip()
                                
                                # Only try to parse if it looks like JSON (starts with { or [)
                                if stripped_text and (stripped_text.startswith('{') or stripped_text.startswith('[')):
                                    logger.info(f"    Parsing as JSON string: {msg.text[:200]}")
                                    result_data = json.loads(msg.text)
                                    result = IndividualModelResult(**result_data)
                                    individual_results.append(result)
                                else:
                                    logger.info(f"    Skipping non-JSON text: {msg.text[:200]}")
                        except (json.JSONDecodeError, TypeError, ValueError) as e:
                            logger.warning(f"Failed to parse result: {e}")
                            logger.warning(f"  Problematic text: {msg.text}")
            elif isinstance(output, dict):
                # Direct dictionary output
                logger.info(f"  Output is dict: {output}")
                result = IndividualModelResult(**output)
                individual_results.append(result)
            elif isinstance(output, IndividualModelResult):
                logger.info(f"  Output is IndividualModelResult")
                individual_results.append(output)
        
        if not individual_results:
            raise RuntimeError("No detection results returned from any model")
        
        logger.info(f"Received {len(individual_results)} individual model results")
        
        # Apply ensemble voting logic (from detect_deepfake_image.py)
        ensemble_result = apply_ensemble_voting(
            individual_results=individual_results,
            threshold=threshold,
            voting_method=voting_method,
            image_path=image_path
        )
        
        # Clean up temporary file
        try:
            import os
            if temp_image_path and os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
                logger.info(f"Cleaned up temporary file: {temp_image_path}")
        except Exception as cleanup_err:
            logger.warning(f"Failed to clean up temporary file: {cleanup_err}")
        
        return ensemble_result
        
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


def apply_ensemble_voting(
    individual_results: List[IndividualModelResult],
    threshold: float,
    voting_method: str,
    image_path: Optional[str]
) -> ImageDetectionResult:
    """
    Apply ensemble voting logic from detect_deepfake_image.py.
    
    This implements the exact voting methods from EnsembleDetector.detect().
    """
    # Calculate voting summary
    fake_votes = sum(1 for r in individual_results if r.prediction == 'FAKE')
    real_votes = len(individual_results) - fake_votes
    
    voting_summary = {
        'fake': fake_votes,
        'real': real_votes,
        'total': len(individual_results)
    }
    
    # Extract scores
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
        is_deepfake = avg_fake_score > avg_real_score
        
        if is_deepfake:
            fake_model_scores = [fake_scores[i] for i in range(len(individual_results)) 
                                if individual_results[i].prediction == 'FAKE']
            confidence = np.mean(fake_model_scores) if fake_model_scores else avg_fake_score
        else:
            real_model_scores = [real_scores[i] for i in range(len(individual_results)) 
                                if individual_results[i].prediction == 'REAL']
            confidence = np.mean(real_model_scores) if real_model_scores else avg_real_score
    
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
            'nonescape-full': 0.92,
            'seresnet50': 0.80,
            'prithiv-v1': 0.75,
            'dima-vit': 0.85,
            'organika-sdxl': 0.70
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
    
    else:
        raise ValueError(f"Unknown voting method: {voting_method}")
    
    # Determine image filename
    if image_path:
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
        individualModelResults=individual_results
    )
    
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
