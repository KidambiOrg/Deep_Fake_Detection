"""
DetectGPT Implementation for Deep Fake Text Detection

This module implements the DetectGPT algorithm for detecting AI-generated text.
DetectGPT uses the curvature of the model's log probability to distinguish
between human-written and AI-generated text.

Reference: "DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature"
"""


import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple, Dict, Optional, Union, Any
import re
import logging
import warnings
from dataclasses import dataclass
import hashlib
import time
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Suppress specific transformers warnings
warnings.filterwarnings("ignore", message=".*loss_type.*unrecognized.*")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model cache to avoid reloading models
_MODEL_CACHE = {}
_CACHE_LOCK = threading.Lock()

def _get_cached_model(model_name: str, model_type: str, device: str):
    """
    Get or create cached model to avoid reloading.
    
    Args:
        model_name: Name of the model
        model_type: Type of model ('scoring' or 'mask')
        device: Device to load model on
        
    Returns:
        Tuple of (tokenizer, model)
    """
    cache_key = f"{model_name}_{model_type}_{device}"
    
    with _CACHE_LOCK:
        if cache_key in _MODEL_CACHE:
            logger.info(f"Using cached {model_type} model: {model_name}")
            return _MODEL_CACHE[cache_key]
        
        logger.info(f"Loading {model_type} model: {model_name}")
        start_time = time.time()
        
        tokenizer = None
        model = None
        
        if model_type == 'scoring':
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model with explicit configuration to avoid warnings
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name)
            if hasattr(config, 'loss_type'):
                delattr(config, 'loss_type')
            
            model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
            model = model.to(device)  # type: ignore
            model.eval()
            
            # Add pad token if it doesn't exist
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
        elif model_type == 'mask':
            from transformers import T5Tokenizer, T5ForConditionalGeneration
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(model_name)
            model = model.to(device)  # type: ignore
            model.eval()
        
        if tokenizer is None or model is None:
            raise ValueError(f"Unknown model type: {model_type}")
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.1f}s")
        
        # Cache the models
        _MODEL_CACHE[cache_key] = (tokenizer, model)
        return tokenizer, model

@dataclass
class DetectionResult:
    """Result of text detection analysis"""
    text: str
    is_ai_generated: bool
    confidence: float
    original_logprob: float
    perturbed_logprob_mean: float
    criterion_score: float

class DetectGPT:
    """
    Fast DetectGPT implementation for detecting AI-generated text.
    
    Optimizations included:
    - Model caching to avoid reloading
    - Batch processing of perturbations
    - Intelligent perturbation generation
    - Early stopping mechanisms
    - Fast detection mode
    """
    
    def __init__(self, 
                 model_name: str = "gpt2-xl",
                 mask_model_name: str = "t5-base",
                 device: Optional[str] = None,
                 cache_models: bool = True,
                 random_seed: Optional[int] = None):
        """
        Initialize DetectGPT with specified models.
        
        Args:
            model_name: Name of the scoring model (for log probability computation)
                       Options: distilgpt2, gpt2, gpt2-medium, gpt2-large, gpt2-xl
                       Alternative: microsoft/DialoGPT-medium (for conversational text)
            mask_model_name: Name of the mask-filling model (for perturbations)
                            Options: t5-small, t5-base, t5-large, google/flan-t5-base
            device: Device to run models on ('cuda' or 'cpu')
            cache_models: Whether to use model caching for speed
            random_seed: Fixed seed for reproducible results (None for random behavior)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.mask_model_name = mask_model_name
        self.cache_models = cache_models
        self.random_seed = random_seed
        
        # Set random seeds for reproducibility if specified
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(random_seed)
                torch.cuda.manual_seed_all(random_seed)
        
        logger.info(f"Using device: {self.device}")
        
        # Load models (with caching if enabled)
        if cache_models:
            self.tokenizer, self.model = _get_cached_model(model_name, 'scoring', self.device)
            self.mask_tokenizer, self.mask_model = _get_cached_model(mask_model_name, 'mask', self.device)
        else:
            self._load_models_directly()
        
        # Cache for computed probabilities
        self._prob_cache = {} if cache_models else None
        
    def _reset_random_state(self):
        """Reset random state for reproducible results."""
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.random_seed)
    
    def _load_models_directly(self):
        """Load models directly without caching (fallback method)."""
        # Load scoring model
        logger.info(f"Loading scoring model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(self.model_name)
        if hasattr(config, 'loss_type'):
            delattr(config, 'loss_type')
        
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, config=config)
        self.model = self.model.to(self.device)  # type: ignore
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load mask-filling model
        logger.info(f"Loading mask-filling model: {self.mask_model_name}")
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        self.mask_tokenizer = T5Tokenizer.from_pretrained(self.mask_model_name, legacy=False)       
        self.mask_model = T5ForConditionalGeneration.from_pretrained(self.mask_model_name)
        self.mask_model = self.mask_model.to(self.device)  # type: ignore
        self.mask_model.eval()
        
    def compute_log_probability(self, text: str) -> float:
        """
        Compute the log probability of text under the scoring model with caching.
        
        Args:
            text: Input text to score
            
        Returns:
            Average log probability per token
        """
        # Check cache first
        if self._prob_cache is not None:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in self._prob_cache:
                return self._prob_cache[text_hash]
        
        # Tokenize text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            # Negative log likelihood
            neg_log_likelihood = outputs.loss.item()
            # Convert to log probability per token
            log_prob = -neg_log_likelihood
        
        # Cache the result
        if self._prob_cache is not None:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            self._prob_cache[text_hash] = log_prob
            
        return log_prob
    
    def compute_log_probabilities_batch(self, texts: List[str], batch_size: int = 8) -> List[float]:
        """
        Compute log probabilities for multiple texts in batches for speed.
        
        Args:
            texts: List of texts to score
            batch_size: Number of texts to process at once
            
        Returns:
            List of log probabilities
        """
        if not texts:
            return []
        
        log_probs = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Check cache first
            batch_results = []
            uncached_texts = []
            uncached_indices = []
            
            for j, text in enumerate(batch_texts):
                if self._prob_cache is not None:
                    text_hash = hashlib.md5(text.encode()).hexdigest()
                    if text_hash in self._prob_cache:
                        batch_results.append((j, self._prob_cache[text_hash]))
                        continue
                
                uncached_texts.append(text)
                uncached_indices.append(j)
            
            # Process uncached texts in batch
            if uncached_texts:
                # Tokenize all texts in batch
                inputs = self.tokenizer(
                    uncached_texts, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=512,
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                    
                    # Get individual losses
                    if hasattr(outputs, 'losses') and outputs.losses is not None:
                        losses = outputs.losses.cpu().numpy()
                    else:
                        # Fallback: compute individual losses
                        losses = []
                        for k, text in enumerate(uncached_texts):
                            single_input = {key: val[k:k+1] for key, val in inputs.items()}
                            single_output = self.model(**single_input, labels=single_input["input_ids"])
                            losses.append(single_output.loss.item())
                        losses = np.array(losses)
                    
                    # Convert to log probabilities
                    batch_log_probs = -losses
                
                # Cache and store results
                for k, (text_idx, text) in enumerate(zip(uncached_indices, uncached_texts)):
                    log_prob = float(batch_log_probs[k])
                    batch_results.append((text_idx, log_prob))
                    
                    # Cache the result
                    if self._prob_cache is not None:
                        text_hash = hashlib.md5(text.encode()).hexdigest()
                        self._prob_cache[text_hash] = log_prob
            
            # Sort results by original order and extract log probs
            batch_results.sort(key=lambda x: x[0])
            log_probs.extend([result[1] for result in batch_results])
        
        return log_probs
    
    def generate_perturbations(self, text: str, num_perturbations: int = 100, 
                             mask_ratio: float = 0.15, intelligent: bool = True) -> List[str]:
        """
        Generate perturbations of the input text using mask-filling with intelligent optimization.
        
        Args:
            text: Input text to perturb
            num_perturbations: Number of perturbations to generate
            mask_ratio: Ratio of tokens to mask
            intelligent: Whether to use intelligent perturbation strategies
            
        Returns:
            List of perturbed texts
        """
        if intelligent:
            return self._generate_intelligent_perturbations(text, num_perturbations, mask_ratio)
        else:
            return self._generate_basic_perturbations(text, num_perturbations, mask_ratio)
    
    def _generate_intelligent_perturbations(self, text: str, num_perturbations: int, 
                                          mask_ratio: float) -> List[str]:
        """
        Generate perturbations using intelligent strategies with optimized performance.
        
        Performance optimizations:
        1. Batch T5 generation for multiple perturbations at once
        2. Pre-compute word sets for faster similarity calculations
        3. Vectorized operations where possible
        4. Reduced redundant string operations
        5. More aggressive early stopping
        """
        perturbations = []
        words = text.split()
        text_word_set = set(text.lower().split())  # Pre-compute for similarity
        
        # Pre-compute static values
        important_indices = self._identify_important_words(words)
        adaptive_mask_ratio = self._calculate_adaptive_mask_ratio(len(words), mask_ratio)
        temperatures = [0.7, 1.0, 1.3]
        
        # Optimized batch processing
        batch_size = min(16, num_perturbations)  # Larger batches for T5
        target_diversity = 0.75  # Slightly lower threshold for faster stopping
        max_attempts = int(num_perturbations * 1.5)  # Reduced safety margin
        
        attempts = 0
        last_diversity_check = 0
        
        while len(perturbations) < num_perturbations and attempts < max_attempts:
            remaining = num_perturbations - len(perturbations)
            current_batch_size = min(batch_size, remaining)
            
            # Generate batch of mask configurations
            batch_configs = self._generate_mask_batch_configs(
                words, important_indices, adaptive_mask_ratio, 
                current_batch_size, temperatures
            )
            
            # Batch generate perturbations using T5
            batch_perturbations = self._generate_perturbations_batch(
                text, words, batch_configs
            )
            
            # Fast quality filtering
            valid_perturbations = self._fast_filter_perturbations(
                text, text_word_set, batch_perturbations, perturbations
            )
            
            perturbations.extend(valid_perturbations)
            attempts += len(batch_perturbations)
            
            # Less frequent diversity checking for performance
            if (len(perturbations) > 10 and 
                len(perturbations) - last_diversity_check >= 8):
                
                current_diversity = self._fast_calculate_diversity(text_word_set, perturbations)
                last_diversity_check = len(perturbations)
                
                # More aggressive early stopping
                if (current_diversity >= target_diversity and 
                    len(perturbations) >= num_perturbations // 3):  # Stop at 1/3 instead of 1/2
                    logger.info(f"Early stopping: diversity {current_diversity:.2f} with {len(perturbations)} perturbations")
                    break
        
        logger.info(f"Generated {len(perturbations)} perturbations in {attempts} attempts")
        return perturbations[:num_perturbations]
    
    def _identify_important_words(self, words: List[str]) -> List[int]:
        """
        Identify important words that should be prioritized for masking.
        Optimized version with pre-compiled patterns and vectorized operations.
        """
        # Pre-compiled function words set (moved to class level for caching)
        if not hasattr(self, '_function_words'):
            self._function_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                'of', 'with', 'by', 'is', 'was', 'are', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 
                'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
            }
        
        # Pre-compiled punctuation for faster stripping
        if not hasattr(self, '_punctuation'):
            self._punctuation = str.maketrans('', '', '.,!?;:"()[]{}')
        
        important_indices = []
        
        for i, word in enumerate(words):
            # Fast word processing
            word_clean = word.lower().translate(self._punctuation)
            
            # Vectorized conditions for speed
            if (len(word_clean) > 4 and  # Longer threshold for efficiency
                word_clean not in self._function_words and
                word_clean.isalpha() and
                not word_clean.isupper()):
                important_indices.append(i)
        
        # Return at least some indices if none found
        if not important_indices and len(words) > 5:
            # Fallback: select longer words
            word_lengths = [(i, len(word)) for i, word in enumerate(words)]
            word_lengths.sort(key=lambda x: x[1], reverse=True)
            important_indices = [i for i, _ in word_lengths[:len(words)//3]]
        
        return important_indices
    
    def _calculate_adaptive_mask_ratio(self, text_length: int, base_ratio: float) -> float:
        """
        Calculate adaptive mask ratio based on text length.
        Shorter texts need higher ratios for effective perturbation.
        """
        if text_length < 20:
            return min(0.25, base_ratio * 1.5)  # More aggressive for short texts
        elif text_length > 100:
            return max(0.1, base_ratio * 0.8)   # More conservative for long texts
        else:
            return base_ratio
    
    def _generate_single_perturbation(self, original_text: str, words: List[str], 
                                    mask_indices: np.ndarray, temperature: float) -> str:
        """
        Generate a single perturbation with specified temperature.
        """
        try:
            # Create masked text for T5
            masked_text = ""
            mask_count = 0
            for i, word in enumerate(words):
                if i in mask_indices:
                    masked_text += f"<extra_id_{mask_count}> "
                    mask_count += 1
                else:
                    masked_text += word + " "
            
            # Add final mask token
            masked_text += f"<extra_id_{mask_count}>"
            
            # Generate text with T5
            inputs = self.mask_tokenizer(masked_text, return_tensors="pt", 
                                       truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generation_kwargs = {
                    "max_length": 512,
                    "num_return_sequences": 1,
                    "pad_token_id": self.mask_tokenizer.pad_token_id
                }
                
                # Use deterministic settings if seed is set
                if self.random_seed is not None:
                    generation_kwargs.update({
                        "do_sample": False,  # Deterministic
                        "num_beams": 1,      # Greedy decoding
                    })
                else:
                    generation_kwargs.update({
                        "do_sample": True,   # Stochastic
                        "temperature": temperature,
                        "top_p": 0.9,
                    })
                
                outputs = self.mask_model.generate(**inputs, **generation_kwargs)
            
            # Decode the generated text
            generated = self.mask_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Replace masks in original text with generated content
            return self._replace_masks(original_text, generated, mask_indices)
            
        except Exception as e:
            logger.warning(f"Failed to generate perturbation: {e}")
            return self._fallback_perturbation(words, mask_indices)
    
    def _fallback_perturbation(self, words: List[str], mask_indices: np.ndarray) -> str:
        """
        Generate a fallback perturbation when T5 generation fails.
        """
        perturbed_words = words.copy()
        
        # Better fallback replacements based on context
        fallback_replacements = {
            'noun': ['thing', 'item', 'object', 'element', 'part'],
            'verb': ['does', 'makes', 'gets', 'gives', 'takes'],
            'adj': ['good', 'new', 'big', 'important', 'different'],
            'default': ['the', 'and', 'to', 'of', 'a']
        }
        
        for idx in mask_indices:
            if idx < len(perturbed_words):
                original_word = perturbed_words[idx].lower()
                
                # Simple heuristic to determine word type
                if original_word.endswith(('ing', 'ed', 'es')):
                    replacements = fallback_replacements['verb']
                elif original_word.endswith(('ly', 'ful', 'ous')):
                    replacements = fallback_replacements['adj']
                elif len(original_word) > 4:
                    replacements = fallback_replacements['noun']
                else:
                    replacements = fallback_replacements['default']
                
                perturbed_words[idx] = np.random.choice(replacements)
        
        return " ".join(perturbed_words)
    
    def _filter_perturbations_by_quality(self, original_text: str, 
                                       new_perturbations: List[str], 
                                       existing_perturbations: List[str]) -> List[str]:
        """
        Filter perturbations based on quality criteria.
        """
        filtered = []
        
        for perturb in new_perturbations:
            # Quality checks
            if (len(perturb.split()) == len(original_text.split()) and  # Same word count
                perturb != original_text and  # Not identical to original
                perturb not in existing_perturbations and  # Not duplicate
                len(perturb.strip()) > 0):  # Not empty
                
                # Check diversity (Jaccard similarity)
                similarity = self._calculate_jaccard_similarity(original_text, perturb)
                if 0.3 <= similarity <= 0.9:  # Good diversity range
                    filtered.append(perturb)
        
        return filtered
    
    def _calculate_diversity_score(self, original_text: str, perturbations: List[str]) -> float:
        """
        Calculate average diversity score of perturbations.
        """
        if not perturbations:
            return 0.0
        
        similarities = []
        for perturb in perturbations:
            sim = self._calculate_jaccard_similarity(original_text, perturb)
            similarities.append(1.0 - sim)  # Convert similarity to diversity
        
        return float(np.mean(similarities))
    
    def _calculate_jaccard_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate Jaccard similarity between two texts.
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _generate_mask_batch_configs(self, words: List[str], important_indices: List[int], 
                                   adaptive_mask_ratio: float, batch_size: int, 
                                   temperatures: List[float]) -> List[Tuple]:
        """
        Generate batch of mask configurations for efficient processing.
        """
        configs = []
        for i in range(batch_size):
            temp = temperatures[i % len(temperatures)]
            
            # Choose indices to mask
            if important_indices and np.random.random() < 0.7:
                available_indices = important_indices
            else:
                available_indices = list(range(len(words)))
            
            num_to_mask = max(1, int(len(words) * adaptive_mask_ratio))
            num_to_mask = min(num_to_mask, len(available_indices))
            
            mask_indices = np.random.choice(available_indices, size=num_to_mask, replace=False)
            configs.append((mask_indices, temp))
        
        return configs
    
    def _generate_perturbations_batch(self, original_text: str, words: List[str], 
                                    batch_configs: List[Tuple]) -> List[str]:
        """
        Generate multiple perturbations in a single T5 batch call for maximum performance.
        """
        if not batch_configs:
            return []
        
        # Prepare batch of masked texts
        masked_texts = []
        for mask_indices, temp in batch_configs:
            masked_text = self._create_masked_text(words, mask_indices)
            masked_texts.append(masked_text)
        
        # Batch tokenization
        try:
            inputs = self.mask_tokenizer(
                masked_texts, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Batch generation
            with torch.no_grad():
                generation_kwargs = {
                    "max_length": 256,
                    "num_return_sequences": 1,
                    "pad_token_id": self.mask_tokenizer.pad_token_id,
                }
                
                # Use deterministic settings if seed is set
                if self.random_seed is not None:
                    generation_kwargs.update({
                        "do_sample": False,  # Deterministic
                        "num_beams": 1,      # Greedy decoding
                    })
                else:
                    generation_kwargs.update({
                        "do_sample": True,   # Stochastic
                        "temperature": 1.0,
                        "top_p": 0.9,
                        "num_beams": 1,
                    })
                
                outputs = self.mask_model.generate(**inputs, **generation_kwargs)
            
            # Process batch results
            perturbations = []
            for i, (mask_indices, temp) in enumerate(batch_configs):
                try:
                    generated = self.mask_tokenizer.decode(outputs[i], skip_special_tokens=True)
                    perturbed_text = self._replace_masks(original_text, generated, mask_indices)
                    if perturbed_text and perturbed_text != original_text:
                        perturbations.append(perturbed_text)
                except Exception as e:
                    # Fallback to simple replacement
                    fallback = self._fallback_perturbation(words, mask_indices)
                    if fallback != original_text:
                        perturbations.append(fallback)
            
            return perturbations
            
        except Exception as e:
            logger.warning(f"Batch generation failed, falling back to individual: {e}")
            # Fallback to individual generation
            perturbations = []
            for mask_indices, temp in batch_configs:
                try:
                    perturbed = self._generate_single_perturbation(original_text, words, mask_indices, temp)
                    if perturbed and perturbed != original_text:
                        perturbations.append(perturbed)
                except:
                    fallback = self._fallback_perturbation(words, mask_indices)
                    if fallback != original_text:
                        perturbations.append(fallback)
            return perturbations
    
    def _create_masked_text(self, words: List[str], mask_indices: np.ndarray) -> str:
        """
        Create masked text efficiently using string building.
        """
        result_parts = []
        mask_count = 0
        
        for i, word in enumerate(words):
            if i in mask_indices:
                result_parts.append(f"<extra_id_{mask_count}>")
                mask_count += 1
            else:
                result_parts.append(word)
        
        result_parts.append(f"<extra_id_{mask_count}>")
        return " ".join(result_parts)
    
    def _fast_filter_perturbations(self, original_text: str, original_word_set: set,
                                 new_perturbations: List[str], 
                                 existing_perturbations: List[str]) -> List[str]:
        """
        Fast quality filtering using pre-computed word sets and vectorized operations.
        """
        if not new_perturbations:
            return []
        
        original_word_count = len(original_text.split())
        existing_set = set(existing_perturbations)  # Convert to set for O(1) lookup
        filtered = []
        
        for perturb in new_perturbations:
            # Quick basic checks
            if (perturb in existing_set or 
                perturb == original_text or 
                len(perturb.strip()) == 0):
                continue
            
            perturb_words = perturb.split()
            if len(perturb_words) != original_word_count:
                continue
            
            # Fast diversity check using pre-computed word set
            perturb_word_set = set(w.lower() for w in perturb_words)
            intersection = len(original_word_set.intersection(perturb_word_set))
            union = len(original_word_set.union(perturb_word_set))
            
            if union > 0:
                similarity = intersection / union
                if 0.3 <= similarity <= 0.9:  # Good diversity range
                    filtered.append(perturb)
        
        return filtered
    
    def _fast_calculate_diversity(self, original_word_set: set, perturbations: List[str]) -> float:
        """
        Fast diversity calculation using pre-computed word sets.
        """
        if not perturbations:
            return 0.0
        
        # Sample only recent perturbations for speed
        sample_size = min(10, len(perturbations))
        sample_perturbations = perturbations[-sample_size:]
        
        similarities = []
        for perturb in sample_perturbations:
            perturb_word_set = set(perturb.lower().split())
            intersection = len(original_word_set.intersection(perturb_word_set))
            union = len(original_word_set.union(perturb_word_set))
            
            if union > 0:
                similarity = intersection / union
                similarities.append(1.0 - similarity)  # Convert to diversity
        
        return float(np.mean(similarities)) if similarities else 0.0
    
    def _generate_basic_perturbations(self, text: str, num_perturbations: int, 
                                    mask_ratio: float) -> List[str]:
        """
        Generate perturbations using the basic strategy (original implementation).
        """
        perturbations = []
        words = text.split()
        
        for _ in range(num_perturbations):
            # Create a copy of the text
            perturbed_words = words.copy()
            
            # Randomly select words to mask
            num_to_mask = max(1, int(len(words) * mask_ratio))
            mask_indices = np.random.choice(len(words), size=num_to_mask, replace=False)
            
            # Create masked text for T5
            masked_text = ""
            mask_count = 0
            for i, word in enumerate(words):
                if i in mask_indices:
                    masked_text += f"<extra_id_{mask_count}> "
                    mask_count += 1
                else:
                    masked_text += word + " "
            
            # Add final mask token
            masked_text += f"<extra_id_{mask_count}>"
            
            # Generate text with T5
            try:
                inputs = self.mask_tokenizer(masked_text, return_tensors="pt", 
                                           truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    generation_kwargs = {
                        "max_length": 512,
                        "num_return_sequences": 1,
                        "pad_token_id": self.mask_tokenizer.pad_token_id
                    }
                    
                    # Use deterministic settings if seed is set
                    if self.random_seed is not None:
                        generation_kwargs.update({
                            "do_sample": False,  # Deterministic
                            "num_beams": 1,      # Greedy decoding
                        })
                    else:
                        generation_kwargs.update({
                            "do_sample": True,   # Stochastic
                            "temperature": 1.0,
                        })
                    
                    outputs = self.mask_model.generate(**inputs, **generation_kwargs)
                
                # Decode the generated text
                generated = self.mask_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Replace masks in original text with generated content
                perturbed_text = self._replace_masks(text, generated, mask_indices)
                perturbations.append(perturbed_text)
                
            except Exception as e:
                logger.warning(f"Failed to generate perturbation: {e}")
                # Fall back to simple word replacement
                perturbed_words = words.copy()
                for idx in mask_indices:
                    if idx < len(perturbed_words):
                        # Simple replacement with a common word
                        perturbed_words[idx] = np.random.choice(['the', 'and', 'to', 'of', 'a'])
                perturbations.append(" ".join(perturbed_words))
        
        return perturbations
    
    def _replace_masks(self, original_text: str, generated_text: str, 
                      mask_indices: np.ndarray) -> str:
        """
        Replace masked portions in original text with generated content.
        
        Args:
            original_text: Original text with masks
            generated_text: Generated text from T5
            mask_indices: Indices of masked words
            
        Returns:
            Text with masks replaced
        """
        words = original_text.split()
        generated_parts = generated_text.split()
        
        # Simple replacement strategy
        gen_idx = 0
        for mask_idx in sorted(mask_indices):
            if mask_idx < len(words) and gen_idx < len(generated_parts):
                words[mask_idx] = generated_parts[gen_idx]
                gen_idx += 1
        
        return " ".join(words)
    
    def detect(self, text: str, num_perturbations: int = 100, 
               threshold: float = 0.0, early_stopping: bool = True,
               confidence_threshold: float = 2.0, min_perturbations: int = 20) -> DetectionResult:
        """
        Detect if text is AI-generated using DetectGPT algorithm with early stopping.
        
        Args:
            text: Input text to analyze
            num_perturbations: Maximum number of perturbations to generate
            threshold: Threshold for classification (positive = AI-generated)
            early_stopping: Whether to use early stopping when confidence is high
            confidence_threshold: Stop early if confidence exceeds this value
            min_perturbations: Minimum perturbations before early stopping
            
        Returns:
            DetectionResult with analysis results
        """
        # Reset random state for reproducible results
        self._reset_random_state()
        
        logger.info(f"Analyzing text of length {len(text)} characters")
        
        # Compute log probability of original text
        original_logprob = self.compute_log_probability(text)
        logger.info(f"Original log probability: {original_logprob:.4f}")
        
        # Generate perturbations with early stopping
        logger.info(f"Generating up to {num_perturbations} perturbations...")
        perturbations = self.generate_perturbations(text, num_perturbations,intelligent=True)
        
        # Compute log probabilities with early stopping
        perturbed_logprobs = []
        criterion_scores = []
        
        for i, perturbed_text in enumerate(perturbations):
            if i % 20 == 0:
                logger.info(f"Processing perturbation {i+1}/{len(perturbations)}")
            
            try:
                logprob = self.compute_log_probability(perturbed_text)
                perturbed_logprobs.append(logprob)
                
                # Early stopping check
                if early_stopping and i >= min_perturbations - 1:
                    current_mean = np.mean(perturbed_logprobs)
                    current_criterion = original_logprob - current_mean
                    criterion_scores.append(abs(current_criterion))
                    
                    # Check if we have stable confidence
                    if len(criterion_scores) >= 5:
                        recent_scores = criterion_scores[-5:]
                        score_std = np.std(recent_scores)
                        mean_confidence = np.mean(recent_scores)
                        
                        # Early stopping conditions:
                        # 1. High confidence and low variance
                        # 2. Very high confidence regardless of variance
                        if ((mean_confidence > confidence_threshold and score_std < 0.2) or
                            mean_confidence > confidence_threshold * 1.5):
                            logger.info(f"Early stopping at {i+1} perturbations - "
                                      f"confidence: {mean_confidence:.2f}, std: {score_std:.3f}")
                            break
                            
            except Exception as e:
                logger.warning(f"Failed to compute log probability for perturbation {i}: {e}")
        
        if not perturbed_logprobs:
            raise ValueError("Failed to compute any perturbation log probabilities")
        
        # Compute final DetectGPT criterion
        perturbed_logprob_mean = np.mean(perturbed_logprobs)
        criterion_score = original_logprob - perturbed_logprob_mean
        
        logger.info(f"Perturbed log probability mean: {perturbed_logprob_mean:.4f}")
        logger.info(f"DetectGPT criterion score: {criterion_score:.4f}")
        logger.info(f"Used {len(perturbed_logprobs)} perturbations")
        
        # Classification
        is_ai_generated = criterion_score > threshold
        confidence = abs(criterion_score)  # Distance from threshold
        
        return DetectionResult(
            text=text,
            is_ai_generated=bool(is_ai_generated),
            confidence=float(confidence),
            original_logprob=float(original_logprob),
            perturbed_logprob_mean=float(perturbed_logprob_mean),
            criterion_score=float(criterion_score)
        )
    
    def detect_fast(self, text: str, num_perturbations: int = 25,
                   threshold: float = 0.0) -> DetectionResult:
        """
        Fast detection mode with reduced accuracy but much faster processing.
        
        This mode uses:
        - Fewer perturbations (default 25 vs 100)
        - Smaller mask ratios for faster generation
        - Batch processing for all perturbations
        - Simplified perturbation strategies
        
        Args:
            text: Input text to analyze
            num_perturbations: Number of perturbations (reduced default)
            threshold: Threshold for classification
            
        Returns:
            DetectionResult with analysis results
        """
        # Reset random state for reproducible results
        self._reset_random_state()
        
        logger.info(f"Fast mode: Analyzing text of length {len(text)} characters")
        
        # Compute original log probability
        original_logprob = self.compute_log_probability(text)
        
        # Generate fewer, simpler perturbations
        logger.info(f"Fast mode: Generating {num_perturbations} perturbations...")
        perturbations = self.generate_perturbations(
            text, 
            num_perturbations, 
            mask_ratio=0.1,  # Smaller mask ratio for speed
            intelligent=False  # Use basic strategy for speed
        )
        
        # Batch process all perturbations for maximum speed
        logger.info("Fast mode: Computing perturbation probabilities in batch...")
        perturbed_logprobs = self.compute_log_probabilities_batch(
            perturbations, 
            batch_size=min(8, len(perturbations))
        )
        
        if not perturbed_logprobs:
            raise ValueError("Failed to compute any perturbation log probabilities")
        
        # Compute criterion
        perturbed_logprob_mean = np.mean(perturbed_logprobs)
        criterion_score = original_logprob - perturbed_logprob_mean
        
        # Classification
        is_ai_generated = criterion_score > threshold
        confidence = abs(criterion_score)
        
        logger.info(f"Fast mode: criterion score: {criterion_score:.4f}, "
                   f"confidence: {confidence:.4f}")
        
        return DetectionResult(
            text=text,
            is_ai_generated=bool(is_ai_generated),
            confidence=float(confidence),
            original_logprob=float(original_logprob),
            perturbed_logprob_mean=float(perturbed_logprob_mean),
            criterion_score=float(criterion_score)
        )
    
    def batch_detect(self, texts: List[str], **kwargs) -> List[DetectionResult]:
        """
        Detect AI-generated text for multiple texts.
        
        Args:
            texts: List of texts to analyze
            **kwargs: Additional arguments passed to detect()
            
        Returns:
            List of DetectionResult objects
        """
        results = []
        for i, text in enumerate(texts):
            logger.info(f"Processing text {i+1}/{len(texts)}")
            result = self.detect(text, **kwargs)
            results.append(result)
        return results
    
    @classmethod
    def get_model_recommendations(cls) -> Dict[str, Dict]:
        """
        Get model recommendations for different use cases.
        
        Returns:
            Dictionary with recommended model configurations
        """
        return {
            "fast": {
                "model_name": "distilgpt2",
                "mask_model_name": "t5-small",
                "description": "Fastest option, good for testing",
                "accuracy": "⭐⭐⭐",
                "speed": "⚡⚡⚡⚡⚡"
            },
            "balanced": {
                "model_name": "gpt2-large", 
                "mask_model_name": "t5-base",
                "description": "Best balance of accuracy and speed",
                "accuracy": "⭐⭐⭐⭐",
                "speed": "⚡⚡⚡"
            },
            "accurate": {
                "model_name": "gpt2-xl",
                "mask_model_name": "t5-large",
                "description": "Highest accuracy, requires more resources",
                "accuracy": "⭐⭐⭐⭐⭐",
                "speed": "⚡"
            },
            "conversational": {
                "model_name": "microsoft/DialoGPT-medium",
                "mask_model_name": "t5-base",
                "description": "Best for chat/dialogue text",
                "accuracy": "⭐⭐⭐⭐",
                "speed": "⚡⚡⚡"
            }
        }
    
    @classmethod
    def create_enhanced(cls, use_case: str = "balanced", device: Optional[str] = None):
        """
        Create DetectGPT with enhanced model configuration.
        
        Args:
            use_case: 'fast', 'balanced', 'accurate', or 'conversational'
            device: Device to use ('cuda' or 'cpu')
            
        Returns:
            DetectGPT instance with enhanced configuration
        """
        recommendations = cls.get_model_recommendations()
        
        if use_case not in recommendations:
            raise ValueError(f"Unknown use case. Choose from: {list(recommendations.keys())}")
        
        config = recommendations[use_case]
        logger.info(f"Creating enhanced DetectGPT for '{use_case}' use case")
        logger.info(f"Using: {config['model_name']} + {config['mask_model_name']}")
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        return cls(
            model_name=config['model_name'],
            mask_model_name=config['mask_model_name'],
            device=device
        )


def main():
    """
    Example usage of DetectGPT for text detection.
    """
    # Sample texts for testing
    human_text = """
    The sun was setting over the mountains, casting long shadows across the valley below. 
    Sarah had been hiking for hours, and her legs were beginning to feel the strain. 
    She paused to catch her breath and take in the breathtaking view before her.
    """
    
    ai_text = """
    As an AI language model, I can help you understand the concept of machine learning. 
    Machine learning is a subset of artificial intelligence that enables computers to learn 
    and improve from experience without being explicitly programmed for every task.
    """
    
    # Initialize DetectGPT
    print("Initializing DetectGPT...")
    detector = DetectGPT(model_name="gpt2-medium", mask_model_name="t5-small")
    
    # Test on sample texts
    texts = [human_text.strip(), ai_text.strip()]
    labels = ["Human-written", "AI-generated"]
    
    for text, label in zip(texts, labels):
        print(f"\n{'='*50}")
        print(f"Testing {label} text:")
        print(f"Text: {text[:100]}...")
        
        result = detector.detect(text, num_perturbations=20)  # Reduced for demo
        
        print(f"\nResults:")
        print(f"Prediction: {'AI-generated' if result.is_ai_generated else 'Human-written'}")
        print(f"Confidence: {result.confidence:.4f}")
        print(f"Criterion Score: {result.criterion_score:.4f}")
        print(f"Original Log Prob: {result.original_logprob:.4f}")
        print(f"Perturbed Log Prob Mean: {result.perturbed_logprob_mean:.4f}")


if __name__ == "__main__":
    main()