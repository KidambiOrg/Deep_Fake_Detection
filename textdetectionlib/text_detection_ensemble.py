"""
Ensemble module for combining multiple text detection models
"""
import numpy as np
from typing import List, Dict, Optional, Literal
import logging

from textdetectionlib.detector_bert_ml import DetectorBertML, BertMLResult
from textdetectionlib.detector_bow_ml import DetectorBowML, BowMLResult
from textdetectionlib.detector_charcnn import DetectorCharCNN, CharCNNResult
from textdetectionlib.detector_distilbert import DetectorDistilBertML, DistilBertMLResult
from outputModel.text_detection_output_model import TextDetectionResult, IndividualModelScore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextDetectionEnsemble:
    """
    Ensemble combining multiple text detection models.
    Supports multiple voting strategies: majority, weighted, unanimous.
    """
    
    def __init__(
        self,
        bert_config: Optional[Dict] = None,
        bow_config: Optional[Dict] = None,
        charcnn_config: Optional[Dict] = None,
        distilbert_config: Optional[Dict] = None,
        voting_method: Literal["majority", "weighted", "unanimous"] = "majority",
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize ensemble with multiple detectors
        
        Args:
            bert_config: Configuration dict for BERT detector
            bow_config: Configuration dict for BoW detector
            charcnn_config: Configuration dict for CharCNN detector
            distilbert_config: Configuration dict for DistilBERT detector
            voting_method: Voting strategy ("majority", "weighted", "unanimous")
            weights: Model weights for weighted voting (must sum to 1.0)
        """
        self.voting_method = voting_method
        self.weights = weights or {
            "BERT_ML": 0.3,
            "BoW_ML": 0.2,
            "CharCNN": 0.2,
            "DistilBERT": 0.3
        }
        
        # Validate weights
        if voting_method == "weighted":
            total_weight = sum(self.weights.values())
            if abs(total_weight - 1.0) > 0.01:
                raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        
        # Initialize detectors
        self.detectors = {}
        
        logger.info("Initializing ensemble detectors...")
        
        if bert_config:
            try:
                logger.info("Initializing BERT_ML detector...")
                self.detectors["BERT_ML"] = DetectorBertML(**bert_config)
            except Exception as e:
                logger.warning(f"Failed to initialize BERT_ML: {e}")
        
        if bow_config:
            try:
                logger.info("Initializing BoW_ML detector...")
                self.detectors["BoW_ML"] = DetectorBowML(**bow_config)
            except Exception as e:
                logger.warning(f"Failed to initialize BoW_ML: {e}")
        
        if charcnn_config:
            try:
                logger.info("Initializing CharCNN detector...")
                self.detectors["CharCNN"] = DetectorCharCNN(**charcnn_config)
            except Exception as e:
                logger.warning(f"Failed to initialize CharCNN: {e}")
        
        if distilbert_config:
            try:
                logger.info("Initializing DistilBERT detector...")
                self.detectors["DistilBERT"] = DetectorDistilBertML(**distilbert_config)
            except Exception as e:
                logger.warning(f"Failed to initialize DistilBERT: {e}")
        
        if not self.detectors:
            raise ValueError("No detectors were successfully initialized")
        
        logger.info(f"Ensemble initialized with {len(self.detectors)} detectors: {list(self.detectors.keys())}")
    
    def _convert_to_ai_human_labels(self, prediction: str) -> str:
        """Convert bot/human labels to AI-generated/human-generated"""
        if prediction.lower() == "bot":
            return "AI-generated"
        elif prediction.lower() == "human":
            return "human-generated"
        else:
            return prediction
    
    def _majority_voting(
        self,
        individual_scores: List[IndividualModelScore]
    ) -> tuple[str, float, Dict[str, int]]:
        """
        Majority voting: prediction with most votes wins
        
        Returns:
            (final_prediction, confidence, voting_breakdown)
        """
        bot_votes = sum(1 for s in individual_scores if s.prediction == "bot")
        human_votes = sum(1 for s in individual_scores if s.prediction == "human")
        
        total_votes = len(individual_scores)
        voting_breakdown = {"bot": bot_votes, "human": human_votes}
        
        if bot_votes > human_votes:
            final_prediction = "AI-generated"
            confidence = bot_votes / total_votes
        else:
            final_prediction = "human-generated"
            confidence = human_votes / total_votes
        
        return final_prediction, confidence, voting_breakdown
    
    def _weighted_voting(
        self,
        individual_scores: List[IndividualModelScore]
    ) -> tuple[str, float, Dict[str, int]]:
        """
        Weighted voting: each model's vote weighted by its confidence and model weight
        
        Returns:
            (final_prediction, confidence, voting_breakdown)
        """
        weighted_bot_score = 0.0
        weighted_human_score = 0.0
        
        for score in individual_scores:
            model_weight = self.weights.get(score.model_name, 1.0 / len(individual_scores))
            
            if score.prediction == "bot":
                weighted_bot_score += model_weight * score.confidence
            else:
                weighted_human_score += model_weight * score.confidence
        
        # Count raw votes for breakdown
        bot_votes = sum(1 for s in individual_scores if s.prediction == "bot")
        human_votes = sum(1 for s in individual_scores if s.prediction == "human")
        voting_breakdown = {"bot": bot_votes, "human": human_votes}
        
        if weighted_bot_score > weighted_human_score:
            final_prediction = "AI-generated"
            confidence = weighted_bot_score / (weighted_bot_score + weighted_human_score)
        else:
            final_prediction = "human-generated"
            confidence = weighted_human_score / (weighted_bot_score + weighted_human_score)
        
        return final_prediction, confidence, voting_breakdown
    
    def _unanimous_voting(
        self,
        individual_scores: List[IndividualModelScore]
    ) -> tuple[str, float, Dict[str, int]]:
        """
        Unanimous voting: all models must agree
        
        Returns:
            (final_prediction, confidence, voting_breakdown)
        """
        predictions = [s.prediction for s in individual_scores]
        
        bot_votes = sum(1 for p in predictions if p == "bot")
        human_votes = sum(1 for p in predictions if p == "human")
        voting_breakdown = {"bot": bot_votes, "human": human_votes}
        
        # Check if all agree
        if len(set(predictions)) == 1:
            # Unanimous
            prediction = predictions[0]
            final_prediction = self._convert_to_ai_human_labels(prediction)
            # Average confidence when unanimous
            avg_confidence = np.mean([s.confidence for s in individual_scores])
            return final_prediction, float(avg_confidence), voting_breakdown
        else:
            # No unanimous agreement - fall back to majority
            return self._majority_voting(individual_scores)
    
    def detect(self, text: str) -> TextDetectionResult:
        """
        Detect text using ensemble of models
        
        Args:
            text: Input text to classify
            
        Returns:
            TextDetectionResult with aggregated prediction
        """
        individual_scores = []
        
        # Run each detector
        for model_name, detector in self.detectors.items():
            try:
                if model_name == "BERT_ML":
                    result: BertMLResult = detector.detect(text)
                    individual_scores.append(IndividualModelScore(
                        model_name=model_name,
                        prediction=result.prediction,
                        confidence=result.confidence,
                        bot_score=result.bot_score,
                        human_score=result.human_score
                    ))
                
                elif model_name == "BoW_ML":
                    result: BowMLResult = detector.detect(text)
                    individual_scores.append(IndividualModelScore(
                        model_name=model_name,
                        prediction=result.prediction,
                        confidence=result.confidence,
                        bot_score=result.bot_score,
                        human_score=result.human_score
                    ))
                
                elif model_name == "CharCNN":
                    result: CharCNNResult = detector.detect(text)
                    individual_scores.append(IndividualModelScore(
                        model_name=model_name,
                        prediction=result.prediction,
                        confidence=result.confidence,
                        bot_score=result.bot_score,
                        human_score=result.human_score
                    ))
                
                elif model_name == "DistilBERT":
                    result: DistilBertMLResult = detector.detect(text)
                    individual_scores.append(IndividualModelScore(
                        model_name=model_name,
                        prediction=result.prediction,
                        confidence=result.confidence,
                        bot_score=result.bot_score,
                        human_score=result.human_score
                    ))
            
            except Exception as e:
                logger.warning(f"Detector {model_name} failed: {e}")
                continue
        
        if not individual_scores:
            raise ValueError("All detectors failed")
        
        # Aggregate predictions based on voting method
        if self.voting_method == "majority":
            final_prediction, confidence, voting_breakdown = self._majority_voting(individual_scores)
        elif self.voting_method == "weighted":
            final_prediction, confidence, voting_breakdown = self._weighted_voting(individual_scores)
        elif self.voting_method == "unanimous":
            final_prediction, confidence, voting_breakdown = self._unanimous_voting(individual_scores)
        else:
            raise ValueError(f"Unknown voting method: {self.voting_method}")
        
        # Calculate average scores
        avg_bot_score = float(np.mean([s.bot_score for s in individual_scores]))
        avg_human_score = float(np.mean([s.human_score for s in individual_scores]))
        
        return TextDetectionResult(
            text=text,
            prediction=final_prediction,
            confidence=confidence,
            average_bot_score=avg_bot_score,
            average_human_score=avg_human_score,
            voting_method=self.voting_method,
            individual_scores=individual_scores,
            voting_breakdown=voting_breakdown
        )
    
    def detect_batch(self, texts: List[str]) -> List[TextDetectionResult]:
        """
        Detect multiple texts in batch
        
        Args:
            texts: List of input texts
            
        Returns:
            List of TextDetectionResult objects
        """
        results = []
        for i, text in enumerate(texts):
            if i % 10 == 0:
                logger.info(f"Processing ensemble batch {i}/{len(texts)}")
            result = self.detect(text)
            results.append(result)
        return results
