"""
DetectLLM-NPR: Normalized Perturbation Rank for Zero-Shot AI Text Detection

This module implements the DetectLLM-NPR (Normalized Perturbation Rank) algorithm
for detecting AI-generated text based on the research paper:
"DetectLLM: Leveraging Normalized Log Rank Information for Zero-Shot Detection of Machine-Generated Text"

The NPR method analyzes how log-rank scores change when text is perturbed,
leveraging the principle that machine-generated text is more susceptible to
perturbations than human-written text.

Paper: https://arxiv.org/pdf/2306.05540
Repository: https://github.com/mbzuai-nlp/DetectLLM

Author: Based on the work by Jinyan Su, Terry Yue Zhuo, Di Wang, Preslav Nakov
Implementation: Deep Fake Detection Library
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    T5Tokenizer, 
    T5ForConditionalGeneration,
    AutoConfig
)
from typing import List, Tuple, Dict, Optional, Union, Any
import re
import logging
import warnings
from dataclasses import dataclass
import hashlib
import time
from functools import lru_cache
import threading
import math
from scipy import stats

# Suppress warnings
warnings.filterwarnings("ignore", message=".*loss_type.*unrecognized.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model cache
_MODEL_CACHE = {}
_CACHE_LOCK = threading.Lock()


@dataclass
class NPRDetectionResult:
    """Result of NPR-based AI text detection"""
    text: str
    is_ai_generated: bool
    confidence: float
    npr_score: float
    original_log_rank: float
    perturbed_log_rank_mean: float
    original_log_prob: float
    perturbed_log_prob_mean: float
    num_perturbations_used: int
    processing_time: float


class DetectLLMNPR:
    """
    DetectLLM-NPR: Normalized Perturbation Rank for AI Text Detection
    
    This implementation follows the DetectLLM paper methodology, specifically
    the NPR (Normalized Perturbation Rank) approach which:
    
    1. Computes log-rank scores for original text
    2. Generates perturbations of the text
    3. Computes log-rank scores for perturbed texts
    4. Calculates NPR score based on rank changes
    5. Classifies text based on NPR threshold
    
    The key insight is that machine-generated text is more susceptible to
    perturbations, resulting in higher NPR scores.
    """
    
    def __init__(self, 
                 scoring_model: str = "gpt2-xl",
                 perturbation_model: str = "t5-large",
                 device: Optional[str] = None,
                 cache_models: bool = True,
                 random_seed: Optional[int] = None):
        """
        Initialize DetectLLM-NPR detector.
        
        Args:
            scoring_model: Model for computing log probabilities and ranks
                          Options: gpt2, gpt2-medium, gpt2-large, gpt2-xl
            perturbation_model: Model for generating text perturbations
                               Options: t5-small, t5-base, t5-large, t5-3b
            device: Device to run models on ('cpu', 'cuda', or None for auto)
            cache_models: Whether to cache models for faster repeated use
            random_seed: Fixed seed for reproducible results
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.scoring_model_name = scoring_model
        self.perturbation_model_name = perturbation_model
        self.cache_models = cache_models
        self.random_seed = random_seed
        
        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_seed)
        
        logger.info(f"Initializing DetectLLM-NPR on device: {self.device}")
        logger.info(f"Scoring model: {scoring_model}")
        logger.info(f"Perturbation model: {perturbation_model}")
        
        # Load models
        self._load_models()
        
        # Probability and rank caches
        self._prob_cache = {} if cache_models else None
        self._rank_cache = {} if cache_models else None
        
    def _load_models(self):
        """Load the scoring and perturbation models."""
        # Load scoring model (for log probabilities and ranking)
        logger.info(f"Loading scoring model: {self.scoring_model_name}")
        self.scoring_tokenizer = AutoTokenizer.from_pretrained(self.scoring_model_name)
        
        # Handle configuration to avoid warnings
        config = AutoConfig.from_pretrained(self.scoring_model_name)
        if hasattr(config, 'loss_type'):
            delattr(config, 'loss_type')
        
        self.scoring_model = AutoModelForCausalLM.from_pretrained(
            self.scoring_model_name, 
            config=config,
            dtype=torch.float16 if self.device == 'cuda' else torch.float32
        )
        self.scoring_model = self.scoring_model.to(self.device)
        self.scoring_model.eval()
        
        # Add pad token if needed
        if self.scoring_tokenizer.pad_token is None:
            self.scoring_tokenizer.pad_token = self.scoring_tokenizer.eos_token
        
        # Load perturbation model (T5 for mask-filling)
        logger.info(f"Loading perturbation model: {self.perturbation_model_name}")
        self.perturbation_tokenizer = T5Tokenizer.from_pretrained(
            self.perturbation_model_name, 
            legacy=False
        )
        self.perturbation_model = T5ForConditionalGeneration.from_pretrained(
            self.perturbation_model_name,
            dtype=torch.float16 if self.device == 'cuda' else torch.float32
        )
        self.perturbation_model = self.perturbation_model.to(self.device)
        self.perturbation_model.eval()
        
        logger.info("Models loaded successfully")
        
    def compute_log_probability(self, text: str) -> float:
        """
        Compute log probability of text under the scoring model.
        
        Args:
            text: Input text
            
        Returns:
            Average log probability per token
        """
        # Check cache
        if self._prob_cache is not None:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in self._prob_cache:
                return self._prob_cache[text_hash]
        
        # Tokenize
        inputs = self.scoring_tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.scoring_model(**inputs, labels=inputs["input_ids"])
            # Convert negative log likelihood to log probability
            log_prob = -outputs.loss.item()
        
        # Cache result
        if self._prob_cache is not None:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            self._prob_cache[text_hash] = log_prob
            
        return log_prob
    
    def compute_log_rank(self, text: str, reference_texts: Optional[List[str]] = None) -> float:
        """
        Compute log-rank score for the text.
        
        According to the DetectLLM paper, log-rank is computed based on
        token-level ranking within the model's vocabulary. For simplicity,
        we use the negative log probability as a proxy for rank.
        
        Args:
            text: Input text to rank
            reference_texts: Optional reference texts for ranking context (unused in current impl)
        
        Returns:
            Log-rank score (negative log probability)
        """
        # Check cache
        if self._rank_cache is not None:
            cache_key = hashlib.md5(text.encode()).hexdigest()
            if cache_key in self._rank_cache:
                return self._rank_cache[cache_key]
        
        # Compute log probability and convert to rank-like score
        log_prob = self.compute_log_probability(text)
        
        # Use negative log probability as log-rank proxy
        # Higher probability = lower rank = lower log-rank
        log_rank_score = -log_prob
        
        # Cache result
        if self._rank_cache is not None:
            cache_key = hashlib.md5(text.encode()).hexdigest()
            self._rank_cache[cache_key] = log_rank_score
            
        return log_rank_score
    
    def generate_perturbations(self, 
                             text: str, 
                             num_perturbations: int = 25,
                             mask_ratio: float = 0.15,
                             perturbation_strategy: str = "random") -> List[str]:
        """
        Generate perturbations of the input text using T5 mask-filling.
        
        Args:
            text: Original text to perturb
            num_perturbations: Number of perturbations to generate
            mask_ratio: Fraction of tokens to mask in each perturbation
            perturbation_strategy: Strategy for selecting tokens to mask
                                 ('random', 'important', 'mixed')
        
        Returns:
            List of perturbed texts
        """
        words = text.split()
        if len(words) < 3:
            logger.warning("Text too short for meaningful perturbations")
            return [text] * num_perturbations
        
        perturbations = []
        max_attempts = num_perturbations * 2  # Safety limit
        attempts = 0
        
        while len(perturbations) < num_perturbations and attempts < max_attempts:
            attempts += 1
            
            try:
                # Determine tokens to mask
                num_to_mask = max(1, int(len(words) * mask_ratio))
                
                if perturbation_strategy == "random":
                    mask_indices = np.random.choice(
                        len(words), 
                        size=min(num_to_mask, len(words)), 
                        replace=False
                    )
                elif perturbation_strategy == "important":
                    # Prefer masking content words (longer, non-function words)
                    function_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'was', 'are', 'were'}
                    important_indices = [i for i, word in enumerate(words) 
                                       if len(word) > 3 and word.lower() not in function_words]
                    
                    if len(important_indices) >= num_to_mask:
                        mask_indices = np.random.choice(
                            important_indices, 
                            size=num_to_mask, 
                            replace=False
                        )
                    else:
                        mask_indices = np.random.choice(
                            len(words), 
                            size=num_to_mask, 
                            replace=False
                        )
                else:  # mixed strategy
                    # Mix of random and important tokens
                    function_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'was', 'are', 'were'}
                    important_indices = [i for i, word in enumerate(words) 
                                       if len(word) > 3 and word.lower() not in function_words]
                    
                    num_important = min(num_to_mask // 2, len(important_indices))
                    num_random = num_to_mask - num_important
                    
                    mask_indices = []
                    if num_important > 0:
                        mask_indices.extend(np.random.choice(
                            important_indices, 
                            size=num_important, 
                            replace=False
                        ))
                    
                    remaining_indices = [i for i in range(len(words)) if i not in mask_indices]
                    if num_random > 0 and len(remaining_indices) > 0:
                        mask_indices.extend(np.random.choice(
                            remaining_indices, 
                            size=min(num_random, len(remaining_indices)), 
                            replace=False
                        ))
                    
                    mask_indices = np.array(mask_indices)
                
                # Create masked text for T5
                masked_text = self._create_t5_masked_text(words, mask_indices)
                
                # Generate replacement with T5
                perturbed_text = self._generate_with_t5(masked_text, text, mask_indices)
                
                # Quality checks
                if (perturbed_text and 
                    perturbed_text != text and 
                    len(perturbed_text.split()) == len(words) and
                    perturbed_text not in perturbations):
                    perturbations.append(perturbed_text)
                    
            except Exception as e:
                logger.warning(f"Failed to generate perturbation {attempts}: {e}")
                continue
        
        # Fill remaining slots with simple word replacements if needed
        while len(perturbations) < num_perturbations:
            simple_perturbation = self._simple_word_replacement(text)
            if simple_perturbation not in perturbations:
                perturbations.append(simple_perturbation)
            else:
                break
        
        logger.info(f"Generated {len(perturbations)} perturbations in {attempts} attempts")
        return perturbations[:num_perturbations]
    
    def _create_t5_masked_text(self, words: List[str], mask_indices: np.ndarray) -> str:
        """Create T5-compatible masked text."""
        masked_words = words.copy()
        mask_count = 0
        
        # Sort mask indices to maintain order
        for idx in sorted(mask_indices):
            if idx < len(masked_words):
                masked_words[idx] = f"<extra_id_{mask_count}>"
                mask_count += 1
        
        # Add final mask token
        masked_text = " ".join(masked_words) + f" <extra_id_{mask_count}>"
        return masked_text
    
    def _generate_with_t5(self, masked_text: str, original_text: str, 
                         mask_indices: np.ndarray) -> str:
        """Generate text using T5 mask-filling."""
        inputs = self.perturbation_tokenizer(
            masked_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generation_kwargs = {
                "max_length": 256,
                "num_return_sequences": 1,
                "pad_token_id": self.perturbation_tokenizer.pad_token_id,
            }
            
            if self.random_seed is not None:
                generation_kwargs.update({
                    "do_sample": False,
                    "num_beams": 1,
                })
            else:
                generation_kwargs.update({
                    "do_sample": True,
                    "temperature": 1.0,
                    "top_p": 0.9,
                })
            
            outputs = self.perturbation_model.generate(**inputs, **generation_kwargs)
        
        # Decode generated text
        generated = self.perturbation_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Replace masks in original text
        return self._replace_masks_in_text(original_text, generated, mask_indices)
    
    def _replace_masks_in_text(self, original_text: str, generated_text: str, 
                              mask_indices: np.ndarray) -> str:
        """Replace masked portions with generated content."""
        words = original_text.split()
        generated_words = generated_text.split()
        
        # Simple replacement strategy
        gen_idx = 0
        for mask_idx in sorted(mask_indices):
            if mask_idx < len(words) and gen_idx < len(generated_words):
                words[mask_idx] = generated_words[gen_idx]
                gen_idx += 1
        
        return " ".join(words)
    
    def _simple_word_replacement(self, text: str) -> str:
        """Simple fallback perturbation method."""
        words = text.split()
        if len(words) < 2:
            return text
        
        # Replace one random word with a common word
        idx = np.random.randint(len(words))
        replacements = ['the', 'and', 'to', 'of', 'a', 'in', 'is', 'that', 'for', 'with']
        words[idx] = np.random.choice(replacements)
        
        return " ".join(words)
    
    def compute_npr_score(self, 
                         text: str, 
                         num_perturbations: int = 25,
                         perturbation_strategy: str = "mixed",
                         reference_texts: Optional[List[str]] = None) -> Tuple[float, Dict[str, Any]]:
        """
        Compute the NPR (Normalized Perturbation Rank) score.
        
        The NPR score measures how much the log-rank increases after perturbations.
        Higher NPR scores indicate machine-generated text.
        
        Args:
            text: Input text to analyze
            num_perturbations: Number of perturbations to generate
            perturbation_strategy: Strategy for perturbation generation
            reference_texts: Optional reference texts for ranking context
        
        Returns:
            Tuple of (npr_score, detailed_metrics)
        """
        # Compute original log-rank and log-probability
        original_log_rank = self.compute_log_rank(text, reference_texts)
        original_log_prob = self.compute_log_probability(text)
        
        # Generate perturbations
        perturbations = self.generate_perturbations(
            text, 
            num_perturbations, 
            perturbation_strategy=perturbation_strategy
        )
        
        # Compute log-ranks for perturbations
        perturbed_log_ranks = []
        perturbed_log_probs = []
        
        for perturbed_text in perturbations:
            try:
                log_rank = self.compute_log_rank(perturbed_text, reference_texts)
                log_prob = self.compute_log_probability(perturbed_text)
                perturbed_log_ranks.append(log_rank)
                perturbed_log_probs.append(log_prob)
            except Exception as e:
                logger.warning(f"Failed to compute metrics for perturbation: {e}")
                continue
        
        if not perturbed_log_ranks:
            raise ValueError("Failed to compute metrics for any perturbations")
        
        # Compute statistics
        perturbed_log_rank_mean = np.mean(perturbed_log_ranks)
        perturbed_log_rank_std = np.std(perturbed_log_ranks)
        perturbed_log_prob_mean = np.mean(perturbed_log_probs)
        perturbed_log_prob_std = np.std(perturbed_log_probs)
        
        # Improved NPR score calculation with strong variance penalty for human text
        rank_increase = perturbed_log_rank_mean - original_log_rank
        
        # Calculate coefficient of variation (CV) - ratio of std to mean
        # Higher CV indicates more variance (typical of human text)
        if abs(perturbed_log_rank_mean) > 1e-6:
            cv_rank = perturbed_log_rank_std / abs(perturbed_log_rank_mean)
        else:
            cv_rank = 0.0
        
        # Enhanced normalization with stronger variance penalty
        # Human text: high variance → much lower NPR score
        # AI text: low variance → higher NPR score
        if perturbed_log_rank_std > 1e-6:
            # Use a stronger variance factor that heavily penalizes high variance
            # This makes human text (with natural variance) score much lower
            variance_factor = perturbed_log_rank_std * (1.5 + 2.0 * cv_rank)
            npr_score = rank_increase / (abs(original_log_rank) + variance_factor)
        elif abs(original_log_rank) > 1e-6:
            # Fallback to original normalization
            npr_score = rank_increase / abs(original_log_rank)
        else:
            # Fallback: use raw difference when original rank is near zero
            npr_score = rank_increase
        
        # Apply stronger penalty if variance is high (strong human signal)
        if cv_rank > 0.25:  # Very high variance - strongly human-like
            npr_score *= 0.6  # Reduce score by 40%
        elif cv_rank > 0.15:  # High variance - moderately human-like
            npr_score *= 0.75  # Reduce score by 25%
        
        # Additional penalty for high probability variance (another human signal)
        if abs(perturbed_log_prob_mean) > 1e-6:
            cv_prob = perturbed_log_prob_std / abs(perturbed_log_prob_mean)
            if cv_prob > 0.2:
                npr_score *= 0.85  # Reduce score by 15%
        
        # Calculate probability drop (how much probability decreases after perturbation)
        prob_drop = original_log_prob - perturbed_log_prob_mean
        
        # Detailed metrics
        detailed_metrics = {
            'original_log_rank': original_log_rank,
            'original_log_prob': original_log_prob,
            'perturbed_log_rank_mean': perturbed_log_rank_mean,
            'perturbed_log_rank_std': perturbed_log_rank_std,
            'perturbed_log_prob_mean': perturbed_log_prob_mean,
            'perturbed_log_prob_std': perturbed_log_prob_std,
            'num_perturbations_used': len(perturbed_log_ranks),
            'rank_difference': rank_increase,
            'prob_difference': perturbed_log_prob_mean - original_log_prob,
            'prob_drop': prob_drop,
            'coefficient_of_variation': cv_rank,
            'rank_variance_ratio': perturbed_log_rank_std / (abs(original_log_rank) + 1e-6)
        }
        
        return float(npr_score), detailed_metrics
    
    def detect(self, 
               text: str, 
               threshold: float = 0.20,
               num_perturbations: int = 25,
               perturbation_strategy: str = "mixed",
               reference_texts: Optional[List[str]] = None) -> NPRDetectionResult:
        """
        Detect if text is AI-generated using NPR method.
        
        Args:
            text: Text to analyze
            threshold: NPR threshold for classification (default 0.20, higher = more AI-like)
                      Recommended: 0.15-0.20 for balanced detection
                      Lower values (0.10) = more sensitive, higher false positives
                      Higher values (0.25+) = more conservative, may miss AI text
            num_perturbations: Number of perturbations to generate
            perturbation_strategy: Strategy for generating perturbations
            reference_texts: Optional reference texts for ranking context
        
        Returns:
            NPRDetectionResult with detection results
        """
        start_time = time.time()
        
        logger.info(f"Analyzing text of length {len(text)} characters")
        
        # Compute NPR score
        npr_score, metrics = self.compute_npr_score(
            text, 
            num_perturbations, 
            perturbation_strategy, 
            reference_texts
        )
        
        # Enhanced classification using multiple signals
        # Primary signal: NPR score
        is_ai_generated = npr_score > threshold
        
        # Secondary signal: coefficient of variation (CV)
        # Lower CV (< 0.1) suggests more consistent/AI-like behavior
        cv = metrics.get('coefficient_of_variation', 0)
        
        # Confidence calculation considering multiple factors:
        # 1. Distance from threshold
        # 2. Coefficient of variation (consistency of perturbations)
        # 3. Magnitude of rank difference
        
        distance_from_threshold = abs(npr_score - threshold)
        
        # Base confidence on distance from threshold
        confidence = min(distance_from_threshold / threshold, 1.0) if threshold > 0 else abs(npr_score)
        
        # Adjust confidence based on CV - low variance increases confidence for AI detection
        if is_ai_generated and cv < 0.1:
            confidence = min(confidence * 1.2, 1.0)  # Boost confidence
        elif not is_ai_generated and cv > 0.15:
            confidence = min(confidence * 1.1, 1.0)  # Boost confidence for human
        
        # Ensure confidence is in [0, 1] range
        confidence = max(0.0, min(1.0, confidence))
        
        processing_time = time.time() - start_time
        
        logger.info(f"NPR Score: {npr_score:.4f}")
        logger.info(f"Coefficient of Variation: {cv:.4f}")
        logger.info(f"Classification: {'AI-generated' if is_ai_generated else 'Human-written'}")
        logger.info(f"Confidence: {confidence:.4f}")
        logger.info(f"Processing time: {processing_time:.2f}s")
        
        return NPRDetectionResult(
            text=text,
            is_ai_generated=is_ai_generated,
            confidence=confidence,
            npr_score=npr_score,
            original_log_rank=metrics['original_log_rank'],
            perturbed_log_rank_mean=metrics['perturbed_log_rank_mean'],
            original_log_prob=metrics['original_log_prob'], 
            perturbed_log_prob_mean=metrics['perturbed_log_prob_mean'],
            num_perturbations_used=metrics['num_perturbations_used'],
            processing_time=processing_time
        )
    
    def detect_batch(self, 
                    texts: List[str], 
                    **kwargs) -> List[NPRDetectionResult]:
        """
        Detect AI-generated text for multiple texts.
        
        Args:
            texts: List of texts to analyze
            **kwargs: Arguments passed to detect()
        
        Returns:
            List of NPRDetectionResult objects
        """
        results = []
        for i, text in enumerate(texts):
            logger.info(f"Processing text {i+1}/{len(texts)}")
            result = self.detect(text, **kwargs)
            results.append(result)
        return results
    
    @classmethod
    def get_model_recommendations(cls) -> Dict[str, Dict]:
        """Get model recommendations for different use cases."""
        return {
            "fast": {
                "scoring_model": "gpt2",
                "perturbation_model": "t5-small",
                "description": "Fastest option, good for testing",
                "accuracy": "⭐⭐⭐",
                "speed": "⚡⚡⚡⚡⚡"
            },
            "balanced": {
                "scoring_model": "gpt2-large",
                "perturbation_model": "t5-base", 
                "description": "Best balance of accuracy and speed",
                "accuracy": "⭐⭐⭐⭐",
                "speed": "⚡⚡⚡"
            },
            "accurate": {
                "scoring_model": "gpt2-xl",
                "perturbation_model": "t5-3b",
                "description": "Highest accuracy (recommended by paper)",
                "accuracy": "⭐⭐⭐⭐⭐",
                "speed": "⚡"
            }
        }


def main():
    """
    Example usage of DetectLLM-NPR for AI text detection.
    """
    # Sample texts for testing
    human_text = """
    The old lighthouse stood majestically on the rocky cliff, its white tower 
    gleaming in the afternoon sun. For over a century, it had guided ships 
    safely through the treacherous waters below. Sarah climbed the winding 
    stairs inside, each step echoing in the hollow chamber. From the top, 
    she could see for miles across the endless blue ocean.
    """
    
    ai_text = """
    Artificial intelligence represents a significant technological advancement 
    that has the potential to transform various industries and aspects of 
    human society. Machine learning algorithms can process large datasets 
    to identify patterns and make predictions with remarkable accuracy. 
    Furthermore, natural language processing enables more intuitive 
    human-computer interactions through sophisticated text understanding.
    """
    
    # Initialize DetectLLM-NPR (using balanced configuration)
    print("🔬 DetectLLM-NPR: AI Text Detection")
    print("=" * 50)
    print("Initializing detector...")
    
    detector = DetectLLMNPR(
        scoring_model="gpt2-large",
        perturbation_model="t5-base",
        random_seed=42  # For reproducible results
    )
    
    # Test on sample texts
    texts = [human_text.strip(), ai_text.strip()]
    labels = ["Human-written", "AI-generated (typical)"]
    
    for text, label in zip(texts, labels):
        print(f"\n{'='*60}")
        print(f"Testing {label} text:")
        print(f"Text preview: {text[:100]}...")
        
        # Detect using NPR method
        result = detector.detect(
            text,
            threshold=0.1,  # Conservative threshold
            num_perturbations=15,  # Reduced for demo speed
            perturbation_strategy="mixed"
        )
        
        print(f"\n📊 NPR Detection Results:")
        print(f"   Prediction: {'🤖 AI-generated' if result.is_ai_generated else '👤 Human-written'}")
        print(f"   NPR Score: {result.npr_score:+.4f}")
        print(f"   Confidence: {result.confidence:.4f}")
        print(f"   Original Log-Rank: {result.original_log_rank:.4f}")
        print(f"   Perturbed Log-Rank: {result.perturbed_log_rank_mean:.4f}")
        print(f"   Perturbations Used: {result.num_perturbations_used}")
        print(f"   Processing Time: {result.processing_time:.2f}s")
        
        # Interpretation
        if result.npr_score > 0.2:
            interpretation = "Strong indication of AI generation"
        elif result.npr_score > 0.1:
            interpretation = "Moderate indication of AI generation"
        elif result.npr_score > 0.0:
            interpretation = "Weak indication of AI generation"
        elif result.npr_score > -0.1:
            interpretation = "Uncertain - could be either"
        else:
            interpretation = "Strong indication of human writing"
        
        print(f"   Interpretation: {interpretation}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()