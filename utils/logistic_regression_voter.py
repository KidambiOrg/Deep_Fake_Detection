"""
Logistic Regression Meta-Classifier for Ensemble Voting.

This implements a stacking approach where a Logistic Regression model
learns to optimally combine predictions from multiple deepfake detection models.

Based on sklearn VotingClassifier approach from:
https://www.geeksforgeeks.org/machine-learning/voting-classifier/
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class LogisticRegressionVoter:
    """
    Meta-classifier that learns to combine deepfake detection model predictions
    using Logistic Regression.
    
    This is a stacking ensemble approach where:
    1. Base models (5 deepfake detectors) make predictions
    2. LR meta-classifier learns optimal combination weights
    3. Final prediction is made by the trained LR model
    
    Attributes:
        model: Trained LogisticRegression classifier
        scaler: StandardScaler for feature normalization
        model_order: List of model names in feature order
        is_trained: Whether the meta-classifier has been trained
    """
    
    def __init__(self):
        """Initialize the Logistic Regression voter."""
        self.model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced',  # Handle imbalanced data
            solver='lbfgs'
        )
        self.scaler = StandardScaler()
        self.model_order = ['nonescape-full', 'seresnet50', 'prithiv-v1', 
                           'dima-vit', 'organika-sdxl']
        self.is_trained = False
    
    def _extract_features(self, results: List[Dict]) -> np.ndarray:
        """
        Extract features from model results for LR input.
        
        Creates feature vector with 10 features per image:
        - 5 fake scores (one per model)
        - 5 real scores (one per model)
        
        Args:
            results: List of detection results from base models
            
        Returns:
            Feature array of shape (n_samples, 10)
        """
        features = []
        
        for result in results:
            model_name = result.get('model', '')
            fake_score = result.get('fakeScore', 0.5)
            real_score = result.get('realScore', 0.5)
            
            # Store in consistent order
            if model_name in self.model_order:
                features.extend([fake_score, real_score])
        
        return np.array(features).reshape(1, -1)
    
    def train(self, 
              training_data: List[Tuple[List[Dict], bool]],
              save_path: Optional[str] = None) -> Dict[str, float]:
        """
        Train the Logistic Regression meta-classifier.
        
        Args:
            training_data: List of (results, is_fake) tuples
                - results: List of model detection results
                - is_fake: Ground truth label (True=fake, False=real)
            save_path: Optional path to save trained model
            
        Returns:
            Training metrics (accuracy, precision, recall)
        """
        X_train = []
        y_train = []
        
        # Extract features and labels
        for results, is_fake in training_data:
            features = self._extract_features(results)
            X_train.append(features[0])
            y_train.append(1 if is_fake else 0)  # 1=FAKE, 0=REAL
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Calculate training metrics
        y_pred = self.model.predict(X_train_scaled)
        accuracy = np.mean(y_pred == y_train)
        
        # Precision and recall
        true_positives = np.sum((y_pred == 1) & (y_train == 1))
        false_positives = np.sum((y_pred == 1) & (y_train == 0))
        false_negatives = np.sum((y_pred == 0) & (y_train == 1))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'n_samples': len(y_train),
            'n_fake': np.sum(y_train),
            'n_real': len(y_train) - np.sum(y_train)
        }
        
        # Save model if path provided
        if save_path:
            self.save(save_path)
        
        return metrics
    
    def predict(self, results: List[Dict]) -> Tuple[bool, float, Dict[str, float]]:
        """
        Make prediction using trained LR meta-classifier.
        
        Args:
            results: List of detection results from base models
            
        Returns:
            Tuple of (is_fake, confidence, details)
            - is_fake: Final prediction (True=fake, False=real)
            - confidence: Confidence score [0-1]
            - details: Additional information (fake_prob, real_prob, coefficients)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction. Call train() first.")
        
        # Extract features
        features = self._extract_features(results)
        features_scaled = self.scaler.transform(features)
        
        # Get prediction and probabilities
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        is_fake = bool(prediction == 1)
        fake_prob = probabilities[1]
        real_prob = probabilities[0]
        confidence = max(fake_prob, real_prob)
        
        # Get model coefficients (learned weights)
        coefficients = {}
        for i, model_name in enumerate(self.model_order):
            fake_coef = self.model.coef_[0][i * 2]
            real_coef = self.model.coef_[0][i * 2 + 1]
            coefficients[model_name] = {
                'fake_weight': float(fake_coef),
                'real_weight': float(real_coef)
            }
        
        details = {
            'fake_probability': float(fake_prob),
            'real_probability': float(real_prob),
            'model_coefficients': coefficients,
            'intercept': float(self.model.intercept_[0])
        }
        
        return is_fake, confidence, details
    
    def save(self, path: str):
        """Save trained model to disk."""
        save_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_order': self.model_order,
            'is_trained': self.is_trained
        }
        
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
    
    def load(self, path: str):
        """Load trained model from disk."""
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.model = save_data['model']
        self.scaler = save_data['scaler']
        self.model_order = save_data['model_order']
        self.is_trained = save_data['is_trained']


# Global instance for production use
_lr_voter = None
_lr_voter_path = Path(__file__).parent.parent / 'Models' / 'lr_meta_classifier.pkl'


def get_lr_voter() -> LogisticRegressionVoter:
    """
    Get or create the global LR voter instance.
    Loads from disk if available, otherwise returns untrained instance.
    """
    global _lr_voter
    
    if _lr_voter is None:
        _lr_voter = LogisticRegressionVoter()
        
        # Try to load pre-trained model
        if _lr_voter_path.exists():
            try:
                _lr_voter.load(str(_lr_voter_path))
            except Exception as e:
                print(f"Warning: Could not load pre-trained LR voter: {e}")
    
    return _lr_voter


def apply_logistic_regression_voting(results: List[Dict]) -> Tuple[str, float, float, float]:
    """
    Apply logistic regression meta-classifier voting.
    
    This is a stacking approach where a trained Logistic Regression model
    learns to optimally combine the predictions from all base models.
    
    Args:
        results: List of detection results from base models
        
    Returns:
        Tuple of (prediction, confidence, fake_score, real_score)
    """
    voter = get_lr_voter()
    
    if not voter.is_trained:
        # Fallback to simple averaging if not trained
        print("Warning: LR voter not trained. Using simple average as fallback.")
        fake_scores = [r.get('fakeScore', 0.5) for r in results]
        real_scores = [r.get('realScore', 0.5) for r in results]
        
        avg_fake = np.mean(fake_scores)
        avg_real = np.mean(real_scores)
        
        prediction = 'FAKE' if avg_fake > avg_real else 'REAL'
        confidence = max(avg_fake, avg_real)
        
        return prediction, confidence, avg_fake, avg_real
    
    # Use trained LR model
    is_fake, confidence, details = voter.predict(results)
    fake_prob = details['fake_probability']
    real_prob = details['real_probability']
    
    prediction = 'FAKE' if is_fake else 'REAL'
    
    return prediction, confidence, fake_prob, real_prob


# Pre-training data (if available) - can be populated with validation results
# Format: [(results_list, is_fake_bool), ...]
PRE_TRAINING_DATA: List[Tuple[List[Dict], bool]] = []
