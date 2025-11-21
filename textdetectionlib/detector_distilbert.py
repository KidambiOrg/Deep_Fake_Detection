"""
DistilBERT-based ML classifier for deepfake text detection
Based on TweepFake repository implementation (efficient version of BERT)
"""
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass
import pickle
import os
import logging

try:
    import torch
    from transformers import DistilBertTokenizer, DistilBertModel, pipeline
    from sklearn.linear_model import LogisticRegression
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DistilBertMLResult:
    """Result from DistilBERT ML detection"""
    text: str
    prediction: str  # "bot" or "human"
    prediction_label: int  # 0 for human, 1 for bot
    confidence: float
    bot_score: float
    human_score: float
    embedding: Optional[np.ndarray] = None


class DetectorDistilBertML:
    """
    DistilBERT-based ML classifier for detecting AI-generated text.
    Uses DistilBERT for feature extraction and LogisticRegression for classification.
    More efficient than full BERT (40% smaller, 60% faster).
    
    Labels: {"human": 0, "bot": 1}
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        pretrained_weights: str = "distilbert-base-cased",
        device: Optional[str] = None
    ):
        """
        Initialize DistilBERT ML detector
        
        Args:
            model_path: Path to trained sklearn model (LogisticRegression)
            pretrained_weights: DistilBERT pretrained model name
            device: Device to run on ('cuda' or 'cpu')
        """
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError(
                "Required dependencies not available. "
                "Please install: pip install torch transformers scikit-learn"
            )
        
        self.pretrained_weights = pretrained_weights
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        
        # Label mappings (human=0, bot=1)
        self.label_dict = {"human": 0, "bot": 1}
        self.label_dict_reverse = {0: "human", 1: "bot"}
        
        # Initialize DistilBERT components
        logger.info(f"Loading DistilBERT model: {pretrained_weights}")
        self.tokenizer = DistilBertTokenizer.from_pretrained(pretrained_weights)
        self.distilbert_model = DistilBertModel.from_pretrained(pretrained_weights)
        self.distilbert_model.to(self.device)
        self.distilbert_model.eval()
        
        # Initialize feature extraction pipeline
        self.nlp = pipeline(
            "feature-extraction",
            tokenizer=self.tokenizer,
            model=self.distilbert_model,
            device=0 if self.device == 'cuda' else -1
        )
        
        # Initialize classifier
        self.classifier: Optional[LogisticRegression] = None
        if self.model_path and os.path.exists(self.model_path):
            self.load_model(self.model_path)
        else:
            logger.warning("No trained model loaded. Train a model before prediction.")
    
    def load_model(self, model_path: str):
        """Load trained sklearn model"""
        logger.info(f"Loading trained model from: {model_path}")
        with open(model_path, 'rb') as f:
            self.classifier = pickle.load(f)
    
    def save_model(self, model_path: str):
        """Save trained sklearn model"""
        if self.classifier is None:
            raise ValueError("No model to save. Train a model first.")
        logger.info(f"Saving model to: {model_path}")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(self.classifier, f)
    
    def extract_distilbert_features(self, text: str) -> np.ndarray:
        """
        Extract DistilBERT embeddings from text
        
        Args:
            text: Input text
            
        Returns:
            Mean pooled DistilBERT embedding vector (768-dim)
        """
        # Truncate to max DistilBERT length (512 tokens)
        text_truncated = text[:512]
        
        # Extract features using pipeline
        vec = np.array(self.nlp(text_truncated))
        
        # Mean pool across sequence dimension
        # vec shape: (1, seq_len, hidden_size)
        mean_vec = vec.reshape((vec.shape[1], vec.shape[2])).mean(axis=0)
        
        return mean_vec
    
    def train(
        self,
        train_texts: list,
        train_labels: list,
        val_texts: Optional[list] = None,
        val_labels: Optional[list] = None,
        C: float = 1.0
    ):
        """
        Train the LogisticRegression classifier on DistilBERT features
        
        Args:
            train_texts: List of training texts
            train_labels: List of training labels (0=human, 1=bot)
            val_texts: Optional validation texts
            val_labels: Optional validation labels
            C: Regularization parameter for LogisticRegression
        """
        logger.info(f"Extracting DistilBERT features for {len(train_texts)} training samples...")
        
        # Extract features
        train_features = []
        for i, text in enumerate(train_texts):
            if i % 100 == 0:
                logger.info(f"Processing training sample {i}/{len(train_texts)}")
            features = self.extract_distilbert_features(text)
            train_features.append(features)
        
        train_features = np.array(train_features)
        
        # Train classifier
        logger.info("Training LogisticRegression classifier...")
        self.classifier = LogisticRegression(solver='liblinear', C=C, random_state=523)
        self.classifier.fit(train_features, train_labels)
        
        # Evaluate on training set
        train_score = self.classifier.score(train_features, train_labels)
        logger.info(f"Training accuracy: {train_score:.4f}")
        
        # Evaluate on validation set if provided
        if val_texts and val_labels:
            logger.info(f"Extracting DistilBERT features for {len(val_texts)} validation samples...")
            val_features = []
            for i, text in enumerate(val_texts):
                if i % 100 == 0:
                    logger.info(f"Processing validation sample {i}/{len(val_texts)}")
                features = self.extract_distilbert_features(text)
                val_features.append(features)
            
            val_features = np.array(val_features)
            val_score = self.classifier.score(val_features, val_labels)
            logger.info(f"Validation accuracy: {val_score:.4f}")
    
    def detect(self, text: str, return_embedding: bool = False) -> DistilBertMLResult:
        """
        Detect if text is AI-generated or human-written
        
        Args:
            text: Input text to classify
            return_embedding: Whether to return DistilBERT embedding in result
            
        Returns:
            DistilBertMLResult with prediction and confidence scores
        """
        if self.classifier is None:
            raise ValueError("No trained model available. Load or train a model first.")
        
        # Extract features
        features = self.extract_distilbert_features(text)
        features_reshaped = features.reshape(1, -1)
        
        # Predict
        prediction_label = self.classifier.predict(features_reshaped)[0]
        prediction = self.label_dict_reverse[prediction_label]
        
        # Get probability scores
        proba = self.classifier.predict_proba(features_reshaped)[0]
        human_score = float(proba[0])  # class 0
        bot_score = float(proba[1])     # class 1
        
        confidence = max(human_score, bot_score)
        
        return DistilBertMLResult(
            text=text,
            prediction=prediction,
            prediction_label=int(prediction_label),
            confidence=confidence,
            bot_score=bot_score,
            human_score=human_score,
            embedding=features if return_embedding else None
        )
    
    def detect_batch(self, texts: list, return_embeddings: bool = False) -> list:
        """
        Detect multiple texts in batch
        
        Args:
            texts: List of input texts
            return_embeddings: Whether to return embeddings
            
        Returns:
            List of DistilBertMLResult objects
        """
        results = []
        for i, text in enumerate(texts):
            if i % 10 == 0:
                logger.info(f"Processing batch sample {i}/{len(texts)}")
            result = self.detect(text, return_embedding=return_embeddings)
            results.append(result)
        return results
