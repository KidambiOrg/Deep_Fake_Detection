"""
Character-level CNN classifier for deepfake text detection
Based on TweepFake repository implementation
"""
from __future__ import annotations
import numpy as np
from typing import Optional, Any
from dataclasses import dataclass
import pickle
import os
import logging

try:
    from tensorflow import keras
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dropout, Dense, concatenate
    from keras.models import Model
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    Model = Any  # type: ignore
    Tokenizer = Any  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CharCNNResult:
    """Result from CharCNN detection"""
    text: str
    prediction: str  # "bot" or "human"
    prediction_label: int  # 0 for human, 1 for bot
    confidence: float
    bot_score: float
    human_score: float


class DetectorCharCNN:
    """
    Character-level CNN classifier for detecting AI-generated text.
    Uses Conv1D layers with multiple filter sizes for character-level feature extraction.
    
    Labels: {"human": 0, "bot": 1}
    Architecture: Embedding → Conv1D(3,4,5) → GlobalMaxPooling → Dropout → Dense
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        vocab_size: Optional[int] = None,
        emb_size: int = 32,
        input_size: int = 320,
        num_filters: int = 128,
        dropout: float = 0.2
    ):
        """
        Initialize CharCNN detector
        
        Args:
            model_path: Path to trained Keras model
            tokenizer_path: Path to trained character tokenizer
            vocab_size: Vocabulary size (set after tokenizer is fitted)
            emb_size: Embedding dimension
            input_size: Maximum sequence length
            num_filters: Number of filters for each Conv1D layer
            dropout: Dropout rate
        """
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError(
                "Required dependencies not available. "
                "Please install: pip install tensorflow keras"
            )
        
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.input_size = input_size
        self.num_filters = num_filters
        self.dropout = dropout
        
        # Label mappings (human=0, bot=1)
        self.label_dict = {"human": 0, "bot": 1}
        self.label_dict_reverse = {0: "human", 1: "bot"}
        
        # Initialize tokenizer
        self.tokenizer: Optional[Tokenizer] = None
        if tokenizer_path and os.path.exists(tokenizer_path):
            self.load_tokenizer(tokenizer_path)
        else:
            logger.info("Initializing new character-level tokenizer")
            self.tokenizer = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
        
        # Initialize model
        self.model: Optional[Model] = None
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            if vocab_size is not None:
                self.model = self.build_model()
            else:
                logger.warning("No trained model loaded and vocab_size not specified. Train a model before prediction.")
    
    def build_model(self) -> Model:
        """
        Build CharCNN model with multiple filter sizes
        
        Returns:
            Compiled Keras model
        """
        if self.vocab_size is None:
            raise ValueError("vocab_size must be set before building model")
        
        logger.info(f"Building CharCNN model (vocab_size={self.vocab_size}, input_size={self.input_size})")
        
        # Input layer
        input_layer = Input(shape=(self.input_size,))
        
        # Embedding layer
        x = Embedding(self.vocab_size + 1, self.emb_size)(input_layer)
        
        # Conv1D with filter size 3
        x3 = Conv1D(self.num_filters, 3, activation="tanh")(x)
        x3 = GlobalMaxPooling1D()(x3)
        
        # Conv1D with filter size 4
        x4 = Conv1D(self.num_filters, 4, activation="tanh")(x)
        x4 = GlobalMaxPooling1D()(x4)
        
        # Conv1D with filter size 5
        x5 = Conv1D(self.num_filters, 5, activation="tanh")(x)
        x5 = GlobalMaxPooling1D()(x5)
        
        # Concatenate all conv outputs
        conc = concatenate([x3, x4, x5])
        conc = Dropout(self.dropout)(conc)
        
        # Output layer (binary classification)
        final_out = Dense(1, activation="sigmoid")(conc)
        
        # Build model
        model = Model(inputs=input_layer, outputs=final_out)
        model.compile(
            loss=['binary_crossentropy'],
            optimizer='adam',
            metrics=["accuracy"]
        )
        
        logger.info(model.summary())
        
        return model
    
    def load_model(self, model_path: str):
        """Load trained Keras model"""
        logger.info(f"Loading trained model from: {model_path}")
        self.model = keras.models.load_model(model_path)
        # Extract vocab_size from model
        if self.model:
            # Get vocab size from embedding layer
            embedding_layer = self.model.layers[1]  # Embedding is the second layer
            self.vocab_size = embedding_layer.input_dim - 1
    
    def save_model(self, model_path: str):
        """Save trained Keras model"""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        logger.info(f"Saving model to: {model_path}")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
    
    def load_tokenizer(self, tokenizer_path: str):
        """Load trained tokenizer"""
        logger.info(f"Loading tokenizer from: {tokenizer_path}")
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        # Set vocab size
        self.vocab_size = len(self.tokenizer.word_index)
    
    def save_tokenizer(self, tokenizer_path: str):
        """Save trained tokenizer"""
        if self.tokenizer is None:
            raise ValueError("No tokenizer to save.")
        logger.info(f"Saving tokenizer to: {tokenizer_path}")
        os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
    
    def preprocess_text(self, text: str) -> np.ndarray:
        """
        Preprocess text to character sequences
        
        Args:
            text: Raw input text
            
        Returns:
            Padded sequence array
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")
        
        # Convert to lowercase
        text_lower = text.lower()
        
        # Convert to sequences
        sequences = self.tokenizer.texts_to_sequences([text_lower])
        
        # Pad sequences
        padded = pad_sequences(sequences, maxlen=self.input_size, padding='post')
        
        return padded
    
    def train(
        self,
        train_texts: list,
        train_labels: list,
        val_texts: Optional[list] = None,
        val_labels: Optional[list] = None,
        epochs: int = 10,
        batch_size: int = 32
    ):
        """
        Train the CharCNN classifier
        
        Args:
            train_texts: List of training texts
            train_labels: List of training labels (0=human, 1=bot)
            val_texts: Optional validation texts
            val_labels: Optional validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        logger.info(f"Preprocessing {len(train_texts)} training samples...")
        
        # Convert texts to lowercase
        train_data = [text.lower() for text in train_texts]
        
        # Fit tokenizer on training data
        logger.info("Fitting character tokenizer...")
        self.tokenizer.fit_on_texts(train_data)
        self.vocab_size = len(self.tokenizer.word_index)
        logger.info(f"Vocabulary size: {self.vocab_size}")
        
        # Convert to sequences and pad
        train_sequences = self.tokenizer.texts_to_sequences(train_data)
        train_features = pad_sequences(train_sequences, maxlen=self.input_size, padding='post')
        train_features = np.array(train_features, dtype='float32')
        train_labels_array = np.array(train_labels, dtype='float32')
        
        # Build model if not already built
        if self.model is None:
            self.model = self.build_model()
        
        # Prepare validation data if provided
        validation_data = None
        if val_texts and val_labels:
            logger.info(f"Preprocessing {len(val_texts)} validation samples...")
            val_data = [text.lower() for text in val_texts]
            val_sequences = self.tokenizer.texts_to_sequences(val_data)
            val_features = pad_sequences(val_sequences, maxlen=self.input_size, padding='post')
            val_features = np.array(val_features, dtype='float32')
            val_labels_array = np.array(val_labels, dtype='float32')
            validation_data = (val_features, val_labels_array)
        
        # Train model
        logger.info(f"Training CharCNN for {epochs} epochs...")
        history = self.model.fit(
            train_features,
            train_labels_array,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            verbose=1
        )
        
        return history
    
    def detect(self, text: str) -> CharCNNResult:
        """
        Detect if text is AI-generated or human-written
        
        Args:
            text: Input text to classify
            
        Returns:
            CharCNNResult with prediction and confidence scores
        """
        if self.model is None:
            raise ValueError("No trained model available. Load or train a model first.")
        if self.tokenizer is None:
            raise ValueError("No tokenizer available. Load or train a tokenizer first.")
        
        # Preprocess text
        features = self.preprocess_text(text)
        
        # Predict
        bot_score_raw = self.model.predict(features, verbose=0)[0][0]
        bot_score = float(bot_score_raw)
        human_score = 1.0 - bot_score
        
        # Determine prediction
        prediction_label = 1 if bot_score > 0.5 else 0
        prediction = self.label_dict_reverse[prediction_label]
        
        confidence = max(human_score, bot_score)
        
        return CharCNNResult(
            text=text,
            prediction=prediction,
            prediction_label=prediction_label,
            confidence=confidence,
            bot_score=bot_score,
            human_score=human_score
        )
    
    def detect_batch(self, texts: list) -> list:
        """
        Detect multiple texts in batch
        
        Args:
            texts: List of input texts
            
        Returns:
            List of CharCNNResult objects
        """
        if self.model is None:
            raise ValueError("No trained model available. Load or train a model first.")
        if self.tokenizer is None:
            raise ValueError("No tokenizer available. Load or train a tokenizer first.")
        
        # Preprocess all texts
        features_list = [self.preprocess_text(text) for text in texts]
        features_batch = np.vstack(features_list)
        
        # Predict batch
        bot_scores_raw = self.model.predict(features_batch, verbose=0)
        
        # Build results
        results = []
        for i, text in enumerate(texts):
            bot_score = float(bot_scores_raw[i][0])
            human_score = 1.0 - bot_score
            prediction_label = 1 if bot_score > 0.5 else 0
            prediction = self.label_dict_reverse[prediction_label]
            confidence = max(human_score, bot_score)
            
            results.append(CharCNNResult(
                text=text,
                prediction=prediction,
                prediction_label=prediction_label,
                confidence=confidence,
                bot_score=bot_score,
                human_score=human_score
            ))
        
        return results
