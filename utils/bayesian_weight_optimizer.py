"""
Bayesian optimization for learning optimal model weights in ensemble voting.

This module provides tools to optimize model weights using Bayesian optimization
based on a labeled validation dataset.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class BayesianWeightOptimizer:
    """
    Optimize ensemble model weights using Bayesian optimization.
    
    This class learns optimal weights for each model in the ensemble
    to maximize detection accuracy on a validation dataset.
    """
    
    def __init__(self, model_names: List[str]):
        """
        Initialize the optimizer.
        
        Args:
            model_names: List of model names in the ensemble
        """
        self.model_names = model_names
        self.n_models = len(model_names)
        self.optimal_weights = None
        self.optimization_history = []
        
    def optimize(
        self,
        validation_predictions: List[Dict[str, float]],
        validation_labels: List[bool],
        n_calls: int = 50,
        random_state: int = 42
    ) -> Dict[str, float]:
        """
        Find optimal weights using Bayesian optimization.
        
        Args:
            validation_predictions: List of prediction dictionaries, each containing
                                   fake_scores for each model
            validation_labels: Ground truth labels (True = Fake, False = Real)
            n_calls: Number of optimization iterations
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary mapping model names to optimal weights
        """
        try:
            from skopt import gp_minimize
            from skopt.space import Real
            from skopt.utils import use_named_args
        except ImportError:
            logger.error("scikit-optimize not installed. Run: pip install scikit-optimize")
            return self._get_default_weights()
        
        # Define search space: weights between 0.5 and 1.0 for each model
        search_space = [Real(0.5, 1.0, name=f'weight_{name}') 
                       for name in self.model_names]
        
        # Define objective function
        @use_named_args(search_space)
        def objective(**weights):
            """Calculate negative accuracy (to minimize)."""
            weight_dict = {name: weights[f'weight_{name}'] 
                          for name in self.model_names}
            
            accuracy = self._evaluate_weights(
                weight_dict,
                validation_predictions,
                validation_labels
            )
            
            # Return negative accuracy (we minimize)
            return -accuracy
        
        logger.info(f"Starting Bayesian optimization with {n_calls} iterations...")
        
        # Run Bayesian optimization
        result = gp_minimize(
            objective,
            search_space,
            n_calls=n_calls,
            random_state=random_state,
            verbose=False
        )
        
        # Extract optimal weights
        optimal_weights = {
            name: result.x[i] 
            for i, name in enumerate(self.model_names)
        }
        
        self.optimal_weights = optimal_weights
        self.optimization_history = result.func_vals
        
        best_accuracy = -result.fun
        logger.info(f"Optimization complete. Best accuracy: {best_accuracy:.4f}")
        logger.info(f"Optimal weights: {optimal_weights}")
        
        return optimal_weights
    
    def _evaluate_weights(
        self,
        weights: Dict[str, float],
        predictions: List[Dict[str, float]],
        labels: List[bool]
    ) -> float:
        """
        Evaluate ensemble accuracy with given weights.
        
        Args:
            weights: Dictionary mapping model names to weights
            predictions: List of prediction dictionaries
            labels: Ground truth labels
            
        Returns:
            Accuracy score (0.0 to 1.0)
        """
        correct = 0
        total = len(labels)
        
        for pred_dict, true_label in zip(predictions, labels):
            # Calculate weighted average fake score
            weighted_fake_score = 0.0
            total_weight = 0.0
            
            for model_name in self.model_names:
                if model_name in pred_dict:
                    weight = weights.get(model_name, 0.75)
                    fake_score = pred_dict[model_name]
                    
                    weighted_fake_score += fake_score * weight
                    total_weight += weight
            
            if total_weight > 0:
                avg_fake_score = weighted_fake_score / total_weight
                predicted_fake = avg_fake_score > 0.5
                
                if predicted_fake == true_label:
                    correct += 1
        
        return correct / total if total > 0 else 0.0
    
    def _get_default_weights(self) -> Dict[str, float]:
        """Return default weights if optimization fails."""
        return {
            'nonescape-full': 0.95,
            'seresnet50': 0.82,
            'prithiv-v1': 0.72,
            'dima-vit': 0.78,
            'organika-sdxl': 0.68
        }
    
    def save_weights(self, filepath: str):
        """Save optimized weights to file."""
        import json
        
        if self.optimal_weights is None:
            raise ValueError("No optimized weights available. Run optimize() first.")
        
        with open(filepath, 'w') as f:
            json.dump(self.optimal_weights, f, indent=2)
        
        logger.info(f"Saved optimal weights to {filepath}")
    
    def load_weights(self, filepath: str) -> Dict[str, float]:
        """Load pre-computed optimal weights from file."""
        import json
        
        with open(filepath, 'r') as f:
            weights = json.load(f)
        
        self.optimal_weights = weights
        logger.info(f"Loaded optimal weights from {filepath}")
        
        return weights


# Pre-computed optimal weights (can be updated via optimization)
OPTIMIZED_MODEL_WEIGHTS = {
    'nonescape-full': 0.95,  # Best for AI-generated content (DALL-E, Stable Diffusion, etc.)
    'seresnet50': 0.85,      # Strong general-purpose detector
    'prithiv-v1': 0.70,      # Moderate reliability, can be inconsistent
    'dima-vit': 0.75,        # Good but can miss AI-generated images
    'organika-sdxl': 0.65    # Specialized for SDXL, lower general reliability
}


def get_optimized_weights(
    model_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Get pre-computed optimal weights for ensemble models.
    
    Args:
        model_names: List of model names to get weights for.
                    If None, returns all weights.
    
    Returns:
        Dictionary mapping model names to their optimal weights
    """
    if model_names is None:
        return OPTIMIZED_MODEL_WEIGHTS.copy()
    
    return {
        name: OPTIMIZED_MODEL_WEIGHTS.get(name, 0.75)
        for name in model_names
    }


def apply_weighted_soft_voting(
    individual_results: List[Dict],
    weights: Optional[Dict[str, float]] = None,
    adaptive: bool = True
) -> Tuple[bool, float]:
    """
    Apply weighted soft voting with optional adaptive weighting.
    
    Soft voting uses the probability scores (fake_score, real_score) rather than
    hard predictions, weighted by model-specific learned weights.
    
    Args:
        individual_results: List of individual model results with fakeScore, realScore
        weights: Dictionary mapping model names to weights. If None, uses optimized defaults.
        adaptive: If True, adjusts weights based on prediction uncertainty
        
    Returns:
        Tuple of (is_fake: bool, confidence: float)
    """
    if weights is None:
        weights = OPTIMIZED_MODEL_WEIGHTS
    
    weighted_fake_sum = 0.0
    weighted_real_sum = 0.0
    total_weight = 0.0
    
    # Calculate base weighted sums
    for result in individual_results:
        model_name = result.get('model', '')
        fake_score = result.get('fakeScore', 0.0)
        real_score = result.get('realScore', 0.0)
        
        # Get base weight for this model
        base_weight = weights.get(model_name, 0.75)
        
        # Apply adaptive weighting if enabled
        if adaptive:
            # Boost weight for high-confidence predictions
            confidence = result.get('confidence', 0.5)
            uncertainty = 1.0 - abs(fake_score - real_score)
            
            # Higher confidence and lower uncertainty = higher weight
            adaptive_factor = confidence * (1.0 - uncertainty * 0.3)
            weight = base_weight * adaptive_factor
        else:
            weight = base_weight
        
        weighted_fake_sum += fake_score * weight
        weighted_real_sum += real_score * weight
        total_weight += weight
    
    # Calculate final weighted averages
    if total_weight > 0:
        avg_fake_score = weighted_fake_sum / total_weight
        avg_real_score = weighted_real_sum / total_weight
    else:
        # Fallback to simple average
        avg_fake_score = np.mean([r.get('fakeScore', 0.0) for r in individual_results])
        avg_real_score = np.mean([r.get('realScore', 0.0) for r in individual_results])
    
    # Determine prediction
    is_fake = avg_fake_score > avg_real_score
    confidence = max(avg_fake_score, avg_real_score)
    
    return is_fake, confidence
