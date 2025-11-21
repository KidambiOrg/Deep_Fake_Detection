"""
Bag of Words (BoW) ML classifier for deepfake text detection
Based on TweepFake repository implementation
"""
import numpy as np
from typing import Optional, List
from dataclasses import dataclass
import pickle
import os
import logging
import re

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from nltk.tokenize import TweetTokenizer
    import nltk
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BowMLResult:
    """Result from BoW ML detection"""
    text: str
    prediction: str  # "bot" or "human"
    prediction_label: int  # 0 for human, 1 for bot
    confidence: float
    bot_score: float
    human_score: float
    tokenized_text: Optional[List[str]] = None


class DetectorBowML:
    """
    Bag of Words ML classifier for detecting AI-generated text.
    Uses TF-IDF for feature extraction and LogisticRegression for classification.
    
    Labels: {"human": 0, "bot": 1}
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        vectorizer_path: Optional[str] = None,
        max_features: int = 10000,
        ngram_range: tuple = (1, 2)
    ):
        """
        Initialize BoW ML detector
        
        Args:
            model_path: Path to trained sklearn model (LogisticRegression)
            vectorizer_path: Path to trained TF-IDF vectorizer
            max_features: Maximum number of features for TF-IDF
            ngram_range: N-gram range for TF-IDF (default unigrams and bigrams)
        """
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError(
                "Required dependencies not available. "
                "Please install: pip install scikit-learn nltk"
            )
        
        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
        
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.max_features = max_features
        self.ngram_range = ngram_range
        
        # Label mappings (human=0, bot=1)
        self.label_dict = {"human": 0, "bot": 1}
        self.label_dict_reverse = {0: "human", 1: "bot"}
        
        # Download from blob storage if requested
        if load_from_blob:
            downloaded_paths = self._load_from_blob_storage()
            if downloaded_paths:
                if 'classifier' in downloaded_paths:
                    self.model_path = downloaded_paths['classifier']
                if 'vectorizer' in downloaded_paths:
                    self.vectorizer_path = downloaded_paths['vectorizer']
        
        # Initialize tokenizer
        self.tweet_tokenizer = TweetTokenizer()
        
        # Initialize TF-IDF vectorizer
        self.vectorizer: Optional[TfidfVectorizer] = None
        if self.vectorizer_path and os.path.exists(self.vectorizer_path):
            self.load_vectorizer(self.vectorizer_path)
        else:
            logger.info(f"Initializing new TF-IDF vectorizer (max_features={max_features})")
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                lowercase=True,
                tokenizer=self._identity_tokenizer,  # Pre-tokenized
                preprocessor=self._identity_preprocessor  # Pre-preprocessed
            )
        
        # Initialize classifier
        self.classifier: Optional[LogisticRegression] = None
        if self.model_path and os.path.exists(self.model_path):
            self.load_model(self.model_path)
        else:
            logger.warning("No trained model loaded. Train a model before prediction.")
        
        # Initialize vectorizer
        if self.vectorizer_path and os.path.exists(self.vectorizer_path):
            self.load_vectorizer(self.vectorizer_path)
        else:
            logger.warning("No trained vectorizer loaded.")
    
    @staticmethod
    def _identity_tokenizer(x):
        """Identity function for pre-tokenized text"""
        return x
    
    @staticmethod
    def _identity_preprocessor(x):
        """Identity function for pre-preprocessed text"""
        return x
    
    def load_model(self, model_path: str):
        """Load trained sklearn model"""
        logger.info(f"Loading model from: {model_path}")
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
    
    def load_vectorizer(self, vectorizer_path: str):
        """Load trained TF-IDF vectorizer"""
        logger.info(f"Loading vectorizer from: {vectorizer_path}")
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
    
    def save_vectorizer(self, vectorizer_path: str):
        """Save trained TF-IDF vectorizer"""
        if self.vectorizer is None:
            raise ValueError("No vectorizer to save.")
        logger.info(f"Saving vectorizer to: {vectorizer_path}")
        os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess and tokenize text following TweepFake approach
        
        Args:
            text: Raw input text
            
        Returns:
            List of preprocessed tokens
        """
        # Tokenize using TweetTokenizer
        tokenized = self.tweet_tokenizer.tokenize(text)
        
        # Convert to lowercase
        tokenized = [t.lower() for t in tokenized]
        
        # Remove hashtags
        tokenized = [t for t in tokenized if not t.startswith("#")]
        
        # Process mentions and URLs
        tokenized_mod = []
        for t in tokenized:
            if t.startswith("@"):
                if len(t) > 1:
                    tokenized_mod.append("__user_mention__")
                else:
                    tokenized_mod.append("__invalid_user_mention__")
            else:
                tokenized_mod.append(t)
        
        # Replace URLs
        allText = " ".join(tokenized_mod)
        allText = re.sub(r'http\S+', '__url__', allText)
        tokenized = allText.split()
        
        return tokenized
    
    def train(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
        C: float = 1.0
    ):
        """
        Train the BoW classifier
        
        Args:
            train_texts: List of training texts
            train_labels: List of training labels (0=human, 1=bot)
            val_texts: Optional validation texts
            val_labels: Optional validation labels
            C: Regularization parameter for LogisticRegression
        """
        logger.info(f"Preprocessing {len(train_texts)} training samples...")
        
        # Preprocess training texts
        train_tokenized = []
        for i, text in enumerate(train_texts):
            if i % 100 == 0:
                logger.info(f"Preprocessing training sample {i}/{len(train_texts)}")
            tokens = self.preprocess_text(text)
            train_tokenized.append(tokens)
        
        # Fit TF-IDF vectorizer and transform
        logger.info("Fitting TF-IDF vectorizer...")
        train_features = self.vectorizer.fit_transform(train_tokenized)
        
        # Train classifier
        logger.info("Training LogisticRegression classifier...")
        self.classifier = LogisticRegression(solver='liblinear', C=C, random_state=523)
        self.classifier.fit(train_features, train_labels)
        
        # Evaluate on training set
        train_score = self.classifier.score(train_features, train_labels)
        logger.info(f"Training accuracy: {train_score:.4f}")
        
        # Evaluate on validation set if provided
        if val_texts and val_labels:
            logger.info(f"Preprocessing {len(val_texts)} validation samples...")
            val_tokenized = []
            for i, text in enumerate(val_texts):
                if i % 100 == 0:
                    logger.info(f"Preprocessing validation sample {i}/{len(val_texts)}")
                tokens = self.preprocess_text(text)
                val_tokenized.append(tokens)
            
            val_features = self.vectorizer.transform(val_tokenized)
            val_score = self.classifier.score(val_features, val_labels)
            logger.info(f"Validation accuracy: {val_score:.4f}")
    
    def detect(self, text: str, return_tokens: bool = False) -> BowMLResult:
        """
        Detect if text is AI-generated or human-written
        
        Args:
            text: Input text to classify
            return_tokens: Whether to return tokenized text in result
            
        Returns:
            BowMLResult with prediction and confidence scores
        """
        if self.classifier is None:
            raise ValueError("No trained model available. Load or train a model first.")
        if self.vectorizer is None:
            raise ValueError("No vectorizer available. Load or train a vectorizer first.")
        
        # Preprocess text
        tokens = self.preprocess_text(text)
        
        # Extract TF-IDF features
        features = self.vectorizer.transform([tokens])
        
        # Predict
        prediction_label = self.classifier.predict(features)[0]
        prediction = self.label_dict_reverse[prediction_label]
        
        # Get probability scores
        proba = self.classifier.predict_proba(features)[0]
        human_score = float(proba[0])  # class 0
        bot_score = float(proba[1])     # class 1
        
        confidence = max(human_score, bot_score)
        
        return BowMLResult(
            text=text,
            prediction=prediction,
            prediction_label=int(prediction_label),
            confidence=confidence,
            bot_score=bot_score,
            human_score=human_score,
            tokenized_text=tokens if return_tokens else None
        )
    
    def detect_batch(self, texts: List[str], return_tokens: bool = False) -> List[BowMLResult]:
        """
        Detect multiple texts in batch
        
        Args:
            texts: List of input texts
            return_tokens: Whether to return tokenized texts
            
        Returns:
            List of BowMLResult objects
        """
        results = []
        for i, text in enumerate(texts):
            if i % 10 == 0:
                logger.info(f"Processing batch sample {i}/{len(texts)}")
            result = self.detect(text, return_tokens=return_tokens)
            results.append(result)
        return results
