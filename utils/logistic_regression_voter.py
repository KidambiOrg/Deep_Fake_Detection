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
    
    def __init__(self, use_tuning=True):
        """
        Initialize the Logistic Regression voter.
        
        Args:
            use_tuning: If True, uses optimized hyperparameters from grid search
        """
        if use_tuning:
            # Optimized hyperparameters (can be tuned via GridSearchCV)
            self.model = LogisticRegression(
                random_state=42,
                max_iter=2000,
                class_weight='balanced',
                solver='saga',  # Better for large datasets
                penalty='l1',   # L1 regularization for feature selection
                C=0.5,          # Regularization strength
                tol=1e-4
            )
        else:
            # Default configuration
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced',
                solver='lbfgs'
            )
        self.scaler = StandardScaler()
        # Model order for 8 models (16 features)
        self.model_order = ['nonescape-full', 'seresnet50', 'prithiv-v1', 
                           'dima-vit', 'organika-sdxl', 'mobilenetv2', 'mobilenetv2_cnn', 
                           'prithiv-open-deep-fake']
        self.is_trained = False
    
    def _extract_features(self, results: List[Dict]) -> np.ndarray:
        """
        Extract features from model results for LR input.
        
        Creates feature vector with features per image based on model_order:
        - N fake scores (one per model in model_order)
        - N real scores (one per model in model_order)
        - Additional engineered features for better classification
        
        If a model in model_order is missing from results, uses default 0.5 for both scores.
        
        Args:
            results: List of detection results from base models
            
        Returns:
            Feature array of shape (n_samples, features)
        """
        # Create lookup dictionary from results
        result_dict = {}
        for result in results:
            model_name = result.get('model', '')
            if model_name:
                result_dict[model_name] = {
                    'fake': result.get('fakeScore', 0.5),
                    'real': result.get('realScore', 0.5),
                    'confidence': result.get('confidence', 0.5)
                }
        
        # Extract base features in the exact order expected by the trained model
        features = []
        confidences = []
        fake_scores = []
        real_scores = []
        
        for model_name in self.model_order:
            if model_name in result_dict:
                fake = result_dict[model_name]['fake']
                real = result_dict[model_name]['real']
                conf = result_dict[model_name]['confidence']
                
                features.extend([fake, real])
                fake_scores.append(fake)
                real_scores.append(real)
                confidences.append(conf)
            else:
                # Model not present - use neutral scores (0.5, 0.5)
                features.extend([0.5, 0.5])
                fake_scores.append(0.5)
                real_scores.append(0.5)
                confidences.append(0.5)
        
        # Add engineered features for better discrimination
        # These help the LR model learn more complex patterns
        if fake_scores and real_scores:
            # Statistical features
            features.append(np.mean(fake_scores))  # Average fake score
            features.append(np.std(fake_scores))   # Variance in fake scores (agreement)
            features.append(np.max(fake_scores))   # Max fake score
            features.append(np.min(fake_scores))   # Min fake score
            features.append(np.mean(real_scores))  # Average real score
            features.append(np.std(real_scores))   # Variance in real scores
            features.append(np.mean(confidences))  # Average confidence
            features.append(np.max(confidences))   # Max confidence (strongest model)
            
            # Agreement features
            features.append(np.std(confidences))   # Model agreement/disagreement
            features.append(max(fake_scores) - min(fake_scores))  # Fake score range
            features.append(max(real_scores) - min(real_scores))  # Real score range
            
            # Ratio features
            avg_fake = np.mean(fake_scores)
            avg_real = np.mean(real_scores)
            features.append(avg_fake / (avg_real + 1e-10))  # Fake/Real ratio
            
            # Count-based features
            features.append(np.sum(np.array(fake_scores) > 0.7))  # High-confidence fake count
            features.append(np.sum(np.array(real_scores) > 0.7))  # High-confidence real count
            features.append(np.sum(np.array(confidences) > 0.8))  # Very confident models count
            
            # Quartile features for distribution understanding
            features.append(np.percentile(fake_scores, 25))  # 25th percentile fake
            features.append(np.percentile(fake_scores, 75))  # 75th percentile fake
            features.append(np.median(fake_scores))  # Median fake score
        
        return np.array(features).reshape(1, -1)
    
    def train(self, 
              training_data: List[Tuple[List[Dict], bool]],
              save_path: Optional[str] = None,
              use_cross_validation: bool = True) -> Dict[str, float]:
        """
        Train the Logistic Regression meta-classifier.
        
        Args:
            training_data: List of (results, is_fake) tuples
                - results: List of model detection results
                - is_fake: Ground truth label (True=fake, False=real)
            save_path: Optional path to save trained model
            use_cross_validation: If True, performs 5-fold CV for better evaluation
            
        Returns:
            Training metrics (accuracy, precision, recall, cv_score if enabled)
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
        
        # Perform cross-validation before final training
        cv_score = None
        if use_cross_validation and len(y_train) >= 10:
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='accuracy')
            cv_score = cv_scores.mean()
            print(f"Cross-validation accuracy: {cv_score:.2%} (+/- {cv_scores.std() * 2:.2%})")
        
        # Train the model on full dataset
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
        
        if cv_score is not None:
            metrics['cv_accuracy'] = cv_score
        
        # Save model if path provided
        if save_path:
            self.save(save_path)
        
        return metrics
    
    def predict(self, results: List[Dict]) -> Tuple[bool, float, Dict[str, float]]:
        """
        Make prediction using trained LR meta-classifier with calibrated confidence.
        
        Args:
            results: List of detection results from base models
            
        Returns:
            Tuple of (is_fake, confidence, details)
            - is_fake: Final prediction (True=fake, False=real)
            - confidence: Calibrated confidence score [0-1]
            - details: Additional information (fake_prob, real_prob, calibration metrics, coefficients)
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
        
        # Calculate base confidence from LR model
        base_confidence = max(fake_prob, real_prob)
        
        # === CONFIDENCE CALIBRATION ===
        # The LR model can be overconfident. We calibrate based on:
        # 1. Voting consensus (how many models agree?)
        # 2. Score separation (how different are avg fake vs real?)
        # 3. Model agreement (how much variance in predictions?)
        
        fake_scores = [r.get('fakeScore', 0.5) for r in results]
        real_scores = [r.get('realScore', 0.5) for r in results]
        
        # 1. Voting consensus factor
        predictions = ['FAKE' if r.get('fakeScore', 0.5) > r.get('realScore', 0.5) else 'REAL' 
                       for r in results]
        majority_votes = predictions.count('FAKE' if is_fake else 'REAL')
        consensus_ratio = majority_votes / len(results) if len(results) > 0 else 0.5
        
        # 2. Score separation factor
        avg_fake = np.mean(fake_scores) if len(fake_scores) > 0 else 0.5
        avg_real = np.mean(real_scores) if len(real_scores) > 0 else 0.5
        score_diff = abs(avg_fake - avg_real)
        
        # 3. Model agreement factor (inverse of variance)
        score_variance = np.std(fake_scores) if len(fake_scores) > 1 else 0.0
        agreement_factor = 1.0 - min(score_variance, 0.3) / 0.3  # Normalize to [0.7, 1.0]
        
        # Compute calibration score as weighted combination
        # - Consensus ratio (40%): Strong agreement increases confidence
        # - Score difference (40%): Large separation increases confidence
        # - Agreement factor (20%): Low variance increases confidence
        calibration = (
            0.40 * consensus_ratio +
            0.40 * score_diff +
            0.20 * agreement_factor
        )
        
        # Apply tiered calibration floor based on consensus strength
        # This ensures confidence reflects actual model agreement
        if consensus_ratio >= 0.875:  # Strong consensus (7-8 models agree)
            calibration = max(calibration, 0.70)  # 85% confidence floor
        elif consensus_ratio >= 0.75:  # Good consensus (6 models agree)
            calibration = max(calibration, 0.60)  # 80% confidence floor
        elif consensus_ratio >= 0.625:  # Weak majority (5 models agree)
            calibration = max(calibration, 0.50)  # 75% confidence floor
        # else: use natural calibration for very close votes (4-4 or similar)
        
        # Apply calibration to base confidence
        # Map base_confidence [0.5, 1.0] using calibration factor
        # Higher calibration = trust LR model more
        # Lower calibration = reduce confidence toward 50%
        if base_confidence > 0.5:
            calibrated_confidence = 0.5 + (base_confidence - 0.5) * calibration
        else:
            calibrated_confidence = base_confidence
        
        confidence = float(calibrated_confidence)
        
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
            'base_confidence': float(base_confidence),
            'calibrated_confidence': float(confidence),
            'calibration_metrics': {
                'consensus_ratio': float(consensus_ratio),
                'score_difference': float(score_diff),
                'agreement_factor': float(agreement_factor),
                'calibration_score': float(calibration)
            },
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
        # Fallback to Bayesian inference if not trained
        print("Warning: LR voter not trained. Using Bayesian inference as fallback.")
        
        # Bayesian inference: combine independent model predictions
        # Using log-sum-exp trick for numerical stability
        
        fake_scores = []
        real_scores = []
        
        for result in results:
            fake_scores.append(result.get('fakeScore', 0.5))
            real_scores.append(result.get('realScore', 0.5))
        
        # Convert to log probabilities for numerical stability
        log_fake_scores = [np.log(max(s, 1e-10)) for s in fake_scores]
        log_real_scores = [np.log(max(s, 1e-10)) for s in real_scores]
        
        # Sum log probabilities (equivalent to multiplying probabilities)
        sum_log_fake = np.sum(log_fake_scores)
        sum_log_real = np.sum(log_real_scores)
        
        # Use log-sum-exp trick to compute normalized probabilities
        # First, shift by the maximum to prevent overflow
        max_log = max(sum_log_fake, sum_log_real)
        
        exp_fake = np.exp(sum_log_fake - max_log)
        exp_real = np.exp(sum_log_real - max_log)
        
        # Normalize
        total = exp_fake + exp_real
        if total > 0:
            fake_prob = exp_fake / total
            real_prob = exp_real / total
        else:
            fake_prob = 0.5
            real_prob = 0.5
        
        # Make prediction based on which probability is higher
        base_confidence = max(fake_prob, real_prob)
        
        # === APPLY CALIBRATION (same as trained LR model) ===
        # Calculate calibration factors
        predictions = ['FAKE' if r.get('fakeScore', 0.5) > r.get('realScore', 0.5) else 'REAL' 
                       for r in results]
        is_fake = fake_prob > real_prob
        majority_votes = predictions.count('FAKE' if is_fake else 'REAL')
        consensus_ratio = majority_votes / len(results) if len(results) > 0 else 0.5
        
        avg_fake = np.mean(fake_scores) if len(fake_scores) > 0 else 0.5
        avg_real = np.mean(real_scores) if len(real_scores) > 0 else 0.5
        score_diff = abs(avg_fake - avg_real)
        
        score_variance = np.std(fake_scores) if len(fake_scores) > 1 else 0.0
        agreement_factor = 1.0 - min(score_variance, 0.3) / 0.3
        
        calibration = (
            0.40 * consensus_ratio +
            0.40 * score_diff +
            0.20 * agreement_factor
        )
        
        # Apply tiered calibration floor based on consensus strength
        if consensus_ratio >= 0.875:  # Strong consensus (7-8 models agree)
            calibration = max(calibration, 0.70)  # 85% confidence floor
        elif consensus_ratio >= 0.75:  # Good consensus (6 models agree)
            calibration = max(calibration, 0.60)  # 80% confidence floor
        elif consensus_ratio >= 0.625:  # Weak majority (5 models agree)
            calibration = max(calibration, 0.50)  # 75% confidence floor
        # else: use natural calibration for very close votes
        
        # Apply calibration to base confidence
        if base_confidence > 0.5:
            confidence = 0.5 + (base_confidence - 0.5) * calibration
        else:
            confidence = base_confidence
        
        prediction = 'FAKE' if fake_prob > real_prob else 'REAL'
        
        return prediction, float(confidence), float(fake_prob), float(real_prob)
    
    # Use trained LR model
    is_fake, confidence, details = voter.predict(results)
    fake_prob = details['fake_probability']
    real_prob = details['real_probability']
    
    # Sanity check: if LR model contradicts average scores significantly, use Bayesian instead
    # This protects against poorly trained models
    avg_fake = np.mean([r.get('fakeScore', 0.5) for r in results])
    avg_real = np.mean([r.get('realScore', 0.5) for r in results])
    
    # If average scores disagree with LR prediction (difference > 0.10), fall back to Bayesian
    avg_indicates_fake = avg_fake > avg_real
    lr_predicts_fake = is_fake
    
    if avg_indicates_fake != lr_predicts_fake:
        score_diff = abs(avg_fake - avg_real)
        if score_diff > 0.10:  # Disagreement threshold lowered to catch more edge cases
            lr_prediction_str = 'FAKE' if is_fake else 'REAL'
            print(f"Warning: LR model prediction ({lr_prediction_str}) contradicts average scores (avg_fake={avg_fake:.3f}, avg_real={avg_real:.3f}). Using Bayesian inference.")
            
            # Use Bayesian inference instead
            fake_scores = [r.get('fakeScore', 0.5) for r in results]
            real_scores = [r.get('realScore', 0.5) for r in results]
            
            # Convert to log probabilities for numerical stability
            log_fake_scores = [np.log(max(s, 1e-10)) for s in fake_scores]
            log_real_scores = [np.log(max(s, 1e-10)) for s in real_scores]
            
            # Sum log probabilities (equivalent to multiplying probabilities)
            sum_log_fake = np.sum(log_fake_scores)
            sum_log_real = np.sum(log_real_scores)
            
            # Use log-sum-exp trick to compute normalized probabilities
            max_log = max(sum_log_fake, sum_log_real)
            
            exp_fake = np.exp(sum_log_fake - max_log)
            exp_real = np.exp(sum_log_real - max_log)
            
            # Normalize
            total = exp_fake + exp_real
            if total > 0:
                fake_prob = exp_fake / total
                real_prob = exp_real / total
            else:
                fake_prob = 0.5
                real_prob = 0.5
            
            base_confidence = max(fake_prob, real_prob)
            
            # === APPLY CALIBRATION ===
            predictions = ['FAKE' if r.get('fakeScore', 0.5) > r.get('realScore', 0.5) else 'REAL' 
                           for r in results]
            is_fake_pred = fake_prob > real_prob
            majority_votes = predictions.count('FAKE' if is_fake_pred else 'REAL')
            consensus_ratio = majority_votes / len(results) if len(results) > 0 else 0.5
            
            avg_fake_sc = np.mean(fake_scores) if len(fake_scores) > 0 else 0.5
            avg_real_sc = np.mean(real_scores) if len(real_scores) > 0 else 0.5
            score_diff_sc = abs(avg_fake_sc - avg_real_sc)
            
            score_variance_sc = np.std(fake_scores) if len(fake_scores) > 1 else 0.0
            agreement_factor_sc = 1.0 - min(score_variance_sc, 0.3) / 0.3
            
            calibration_sc = (
                0.40 * consensus_ratio +
                0.40 * score_diff_sc +
                0.20 * agreement_factor_sc
            )
            
            # Apply tiered calibration floor based on consensus strength
            if consensus_ratio >= 0.875:  # Strong consensus (7-8 models agree)
                calibration_sc = max(calibration_sc, 0.70)  # 85% confidence floor
            elif consensus_ratio >= 0.75:  # Good consensus (6 models agree)
                calibration_sc = max(calibration_sc, 0.60)  # 80% confidence floor
            elif consensus_ratio >= 0.625:  # Weak majority (5 models agree)
                calibration_sc = max(calibration_sc, 0.50)  # 75% confidence floor
            # else: use natural calibration for very close votes
            
            # Apply calibration to base confidence
            if base_confidence > 0.5:
                confidence = 0.5 + (base_confidence - 0.5) * calibration_sc
            else:
                confidence = base_confidence
            
            prediction = 'FAKE' if fake_prob > real_prob else 'REAL'
            
            return prediction, float(confidence), float(fake_prob), float(real_prob)
    
    prediction = 'FAKE' if is_fake else 'REAL'
    
    return prediction, confidence, fake_prob, real_prob


# Pre-training data (if available) - can be populated with validation results
# Format: [(results_list, is_fake_bool), ...]
PRE_TRAINING_DATA: List[Tuple[List[Dict], bool]] = []
