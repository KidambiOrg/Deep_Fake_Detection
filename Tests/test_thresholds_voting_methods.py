#!/usr/bin/env python3
"""
Comprehensive Test Script for Deepfake Detection
Tests multiple thresholds and voting methods on sample images
Provides recommendations for optimal configuration

Author: Deep Fake Detection Team
Date: 2025-10-21
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

# Add parent directory to path to import modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from app_detect_image import create_ensemble_detector, process_image_data


class DeepfakeTestAnalyzer:
    """Comprehensive test analyzer for deepfake detection with multiple configurations"""
    
    def __init__(self, image_directory, debug=False):
        """
        Initialize the test analyzer
        
        Args:
            image_directory: Path to directory containing test images
            debug: Enable debug logging
        """
        self.image_directory = image_directory
        self.debug = debug
        self.ensemble = None
        self.results = {}
        
        # Test configurations
        self.thresholds = [0.1, 0.2, 0.26, 0.3, 0.4, 0.5, 0.6, 0.7]
        self.voting_methods = [
            'majority', 'weighted', 'confidence_weighted', 'unanimous', 'maximum',
            'adaptive_threshold', 'median', 'robust_majority', 'dynamic_consensus',
            'reliability_weighted', 'bayesian', 'entropy_weighted', 'consensus_threshold'
        ]
        
    def load_ensemble(self):
        """Load the ensemble detector"""
        print("\n[INIT] Loading ensemble detector...")
        print("=" * 80)
        
        try:
            self.ensemble = create_ensemble_detector(debug=self.debug)
            print("[OK] Ensemble detector loaded successfully!")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load ensemble: {e}")
            return False
    
    def get_test_images(self):
        """Get list of test images from directory"""
        print(f"\n[SEARCH] Scanning directory: {self.image_directory}")
        
        if not os.path.exists(self.image_directory):
            print(f"[ERROR] Directory not found: {self.image_directory}")
            return []
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.avif', '.webp', '.gif'}
        image_files = []
        
        for file in os.listdir(self.image_directory):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(self.image_directory, file)
                image_files.append(image_path)
        
        print(f"[OK] Found {len(image_files)} test images")
        
        if self.debug:
            for img in image_files:
                print(f"  - {os.path.basename(img)}")
        
        return image_files
    
    def determine_ground_truth(self, image_path):
        """
        Determine ground truth from filename
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (is_fake: bool, label: str)
        """
        filename = os.path.basename(image_path).lower()
        
        fake_indicators = ['fake', 'deepfake', 'ai', 'generated', 'synthetic']
        real_indicators = ['real', 'authentic', 'genuine']
        
        # Check for fake indicators
        for indicator in fake_indicators:
            if indicator in filename:
                return True, 'FAKE'
        
        # Check for real indicators
        for indicator in real_indicators:
            if indicator in filename:
                return False, 'REAL'
        
        # Default to unknown if can't determine
        return None, 'UNKNOWN'
    
    def run_comprehensive_test(self):
        """Run comprehensive test with all threshold and voting method combinations"""
        
        print("\n[TEST] Starting Comprehensive Test")
        print("=" * 80)
        
        # Get test images
        image_files = self.get_test_images()
        
        if not image_files:
            print("[ERROR] No test images found!")
            return None
        
        # Initialize results storage
        total_tests = len(self.thresholds) * len(self.voting_methods) * len(image_files)
        current_test = 0
        
        print(f"\n[INFO] Test Configuration:")
        print(f"  - Images: {len(image_files)}")
        print(f"  - Thresholds: {len(self.thresholds)} ({self.thresholds})")
        print(f"  - Voting Methods: {len(self.voting_methods)}")
        print(f"  - Total Tests: {total_tests}")
        print("\nThis may take several minutes. Please wait...\n")
        
        start_time = time.time()
        
        # Test each combination
        for threshold in self.thresholds:
            print(f"\n[PROGRESS] Testing Threshold: {threshold}")
            print("-" * 60)
            
            for voting_method in self.voting_methods:
                # Initialize storage for this configuration
                config_key = f"t{threshold}_v{voting_method}"
                self.results[config_key] = {
                    'threshold': threshold,
                    'voting_method': voting_method,
                    'images': [],
                    'correct': 0,
                    'total': 0,
                    'true_positives': 0,
                    'true_negatives': 0,
                    'false_positives': 0,
                    'false_negatives': 0,
                    'confidences': [],
                    'processing_times': [],
                    'unknown_ground_truth': 0,
                    'errors': 0
                }
                
                # Test each image
                for image_path in image_files:
                    current_test += 1
                    
                    try:
                        # Get ground truth
                        is_fake_gt, gt_label = self.determine_ground_truth(image_path)
                        
                        # Run detection
                        test_start = time.time()
                        result = self.ensemble.detect(
                            image_path, 
                            threshold=threshold, 
                            voting_method=voting_method
                        )
                        test_time = time.time() - test_start
                        
                        # Store results
                        image_result = {
                            'image': os.path.basename(image_path),
                            'ground_truth': gt_label,
                            'prediction': 'FAKE' if result.is_deepfake else 'REAL',
                            'confidence': result.confidence,
                            'avg_fake_score': result.average_fake_score,
                            'avg_real_score': result.average_real_score,
                            'processing_time': test_time
                        }
                        
                        self.results[config_key]['images'].append(image_result)
                        self.results[config_key]['confidences'].append(result.confidence)
                        self.results[config_key]['processing_times'].append(test_time)
                        
                        # Calculate metrics if ground truth is known
                        if is_fake_gt is not None:
                            self.results[config_key]['total'] += 1
                            
                            is_correct = result.is_deepfake == is_fake_gt
                            if is_correct:
                                self.results[config_key]['correct'] += 1
                            
                            # Confusion matrix
                            if is_fake_gt and result.is_deepfake:
                                self.results[config_key]['true_positives'] += 1
                            elif not is_fake_gt and not result.is_deepfake:
                                self.results[config_key]['true_negatives'] += 1
                            elif not is_fake_gt and result.is_deepfake:
                                self.results[config_key]['false_positives'] += 1
                            elif is_fake_gt and not result.is_deepfake:
                                self.results[config_key]['false_negatives'] += 1
                        else:
                            self.results[config_key]['unknown_ground_truth'] += 1
                        
                    except Exception as e:
                        self.results[config_key]['errors'] += 1
                        if self.debug:
                            print(f"[ERROR] Error processing {os.path.basename(image_path)}: {e}")
                
                # Progress update
                progress = (current_test / total_tests) * 100
                print(f"  [{voting_method:20s}] Progress: {progress:.1f}% ({current_test}/{total_tests})", end='\r')
        
        total_time = time.time() - start_time
        
        print(f"\n\n[SUCCESS] Testing complete!")
        print(f"Total Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
        return self.results
    
    def calculate_metrics(self):
        """Calculate comprehensive metrics for all configurations"""
        
        print("\n[ANALYSIS] Calculating Metrics...")
        
        metrics = {}
        
        for config_key, data in self.results.items():
            if data['total'] == 0:
                continue
            
            # Basic metrics
            accuracy = (data['correct'] / data['total']) * 100
            
            # Precision, Recall, F1
            tp = data['true_positives']
            tn = data['true_negatives']
            fp = data['false_positives']
            fn = data['false_negatives']
            
            precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0
            recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
            f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
            
            # Confidence metrics
            confidences = data['confidences']
            avg_confidence = np.mean(confidences) if confidences else 0
            std_confidence = np.std(confidences) if confidences else 0
            min_confidence = np.min(confidences) if confidences else 0
            max_confidence = np.max(confidences) if confidences else 0
            
            # Processing time
            avg_time = np.mean(data['processing_times']) if data['processing_times'] else 0
            
            metrics[config_key] = {
                'threshold': data['threshold'],
                'voting_method': data['voting_method'],
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'avg_confidence': avg_confidence,
                'std_confidence': std_confidence,
                'min_confidence': min_confidence,
                'max_confidence': max_confidence,
                'avg_time': avg_time,
                'true_positives': tp,
                'true_negatives': tn,
                'false_positives': fp,
                'false_negatives': fn,
                'errors': data['errors']
            }
        
        return metrics
    
    def display_results(self, metrics):
        """Display comprehensive results"""
        
        print("\n" + "=" * 100)
        print("COMPREHENSIVE TEST RESULTS")
        print("=" * 100)
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(metrics).T
        
        # Ensure numeric columns are properly typed
        numeric_columns = ['threshold', 'accuracy', 'precision', 'recall', 'f1_score', 
                          'avg_confidence', 'std_confidence', 'min_confidence', 'max_confidence',
                          'avg_time', 'true_positives', 'true_negatives', 
                          'false_positives', 'false_negatives', 'errors']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by accuracy first, then confidence
        df_sorted = df.sort_values(['accuracy', 'avg_confidence'], ascending=[False, False])
        
        print("\n### 🏆 TOP 10 CONFIGURATIONS (by Accuracy)")
        print("-" * 100)
        print(f"{'Rank':<5} {'Threshold':<10} {'Voting Method':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Confidence':<12}")
        print("-" * 100)
        
        for i, (config_key, row) in enumerate(df_sorted.head(10).iterrows(), 1):
            print(f"{i:<5} {row['threshold']:<10.2f} {row['voting_method']:<20} "
                  f"{row['accuracy']:<10.2f} {row['precision']:<10.2f} {row['recall']:<10.2f} "
                  f"{row['f1_score']:<10.2f} {row['avg_confidence']:<12.4f}")
        
        # Best by different metrics
        print("\n### 🥇 BEST CONFIGURATIONS BY METRIC")
        print("-" * 100)
        
        best_accuracy = df_sorted.iloc[0]
        best_precision = df.loc[df['precision'].idxmax()]
        best_recall = df.loc[df['recall'].idxmax()]
        best_f1 = df.loc[df['f1_score'].idxmax()]
        best_confidence = df.loc[df['avg_confidence'].idxmax()]
        best_speed = df.loc[df['avg_time'].idxmin()]
        
        print(f"\n✓ Best Accuracy: {best_accuracy['voting_method']} @ threshold {best_accuracy['threshold']:.2f}")
        print(f"  - Accuracy: {best_accuracy['accuracy']:.2f}%")
        print(f"  - Confidence: {best_accuracy['avg_confidence']:.4f}")
        
        print(f"\n✓ Best Precision: {best_precision['voting_method']} @ threshold {best_precision['threshold']:.2f}")
        print(f"  - Precision: {best_precision['precision']:.2f}%")
        print(f"  - Accuracy: {best_precision['accuracy']:.2f}%")
        
        print(f"\n✓ Best Recall: {best_recall['voting_method']} @ threshold {best_recall['threshold']:.2f}")
        print(f"  - Recall: {best_recall['recall']:.2f}%")
        print(f"  - Accuracy: {best_recall['accuracy']:.2f}%")
        
        print(f"\n✓ Best F1-Score: {best_f1['voting_method']} @ threshold {best_f1['threshold']:.2f}")
        print(f"  - F1-Score: {best_f1['f1_score']:.2f}%")
        print(f"  - Accuracy: {best_f1['accuracy']:.2f}%")
        
        print(f"\n✓ Highest Confidence: {best_confidence['voting_method']} @ threshold {best_confidence['threshold']:.2f}")
        print(f"  - Avg Confidence: {best_confidence['avg_confidence']:.4f}")
        print(f"  - Accuracy: {best_confidence['accuracy']:.2f}%")
        
        print(f"\n✓ Fastest: {best_speed['voting_method']} @ threshold {best_speed['threshold']:.2f}")
        print(f"  - Avg Time: {best_speed['avg_time']:.3f}s")
        
        # Threshold Analysis
        print("\n### 📊 THRESHOLD ANALYSIS")
        print("-" * 100)
        
        threshold_analysis = df.groupby('threshold').agg({
            'accuracy': 'mean',
            'precision': 'mean',
            'recall': 'mean',
            'f1_score': 'mean',
            'avg_confidence': 'mean'
        }).round(2)
        
        print("\nAverage Performance Across All Voting Methods by Threshold:")
        print(threshold_analysis.to_string())
        
        # Voting Method Analysis
        print("\n### 🗳️ VOTING METHOD ANALYSIS")
        print("-" * 100)
        
        voting_analysis = df.groupby('voting_method').agg({
            'accuracy': 'mean',
            'precision': 'mean',
            'recall': 'mean',
            'f1_score': 'mean',
            'avg_confidence': 'mean'
        }).round(2)
        
        voting_sorted = voting_analysis.sort_values('accuracy', ascending=False)
        print("\nAverage Performance Across All Thresholds by Voting Method:")
        print(voting_sorted.to_string())
        
        return df_sorted, best_accuracy, best_f1, best_confidence
    
    def generate_recommendations(self, df_sorted, best_accuracy, best_f1, best_confidence):
        """Generate final recommendations"""
        
        print("\n" + "=" * 100)
        print("🎯 FINAL RECOMMENDATIONS")
        print("=" * 100)
        
        print("\n### 🏆 PRIMARY RECOMMENDATION (Highest Accuracy with Good Confidence)")
        print("-" * 100)
        
        # Find config with best balance of accuracy and confidence
        df_sorted['score'] = df_sorted['accuracy'] * 0.7 + df_sorted['avg_confidence'] * 30
        best_overall = df_sorted.iloc[0]
        
        print(f"\n✅ RECOMMENDED CONFIGURATION:")
        print(f"   - Threshold: {best_overall['threshold']:.2f}")
        print(f"   - Voting Method: {best_overall['voting_method']}")
        print(f"\n   Performance Metrics:")
        print(f"   - Accuracy: {best_overall['accuracy']:.2f}%")
        print(f"   - Precision: {best_overall['precision']:.2f}%")
        print(f"   - Recall: {best_overall['recall']:.2f}%")
        print(f"   - F1-Score: {best_overall['f1_score']:.2f}%")
        print(f"   - Average Confidence: {best_overall['avg_confidence']:.4f}")
        print(f"   - Std Confidence: {best_overall['std_confidence']:.4f}")
        print(f"   - Average Processing Time: {best_overall['avg_time']:.3f}s")
        
        print(f"\n   Confusion Matrix:")
        print(f"   - True Positives: {int(best_overall['true_positives'])}")
        print(f"   - True Negatives: {int(best_overall['true_negatives'])}")
        print(f"   - False Positives: {int(best_overall['false_positives'])}")
        print(f"   - False Negatives: {int(best_overall['false_negatives'])}")
        
        # Alternative recommendations
        print("\n### 📋 ALTERNATIVE RECOMMENDATIONS")
        print("-" * 100)
        
        print(f"\n1. For Production (Best F1-Score):")
        print(f"   - Threshold: {best_f1['threshold']:.2f}")
        print(f"   - Voting Method: {best_f1['voting_method']}")
        print(f"   - F1-Score: {best_f1['f1_score']:.2f}%")
        print(f"   - Accuracy: {best_f1['accuracy']:.2f}%")
        
        print(f"\n2. For High Confidence:")
        print(f"   - Threshold: {best_confidence['threshold']:.2f}")
        print(f"   - Voting Method: {best_confidence['voting_method']}")
        print(f"   - Avg Confidence: {best_confidence['avg_confidence']:.4f}")
        print(f"   - Accuracy: {best_confidence['accuracy']:.2f}%")
        
        print(f"\n3. For Maximum Accuracy:")
        print(f"   - Threshold: {best_accuracy['threshold']:.2f}")
        print(f"   - Voting Method: {best_accuracy['voting_method']}")
        print(f"   - Accuracy: {best_accuracy['accuracy']:.2f}%")
        
        # Use case recommendations
        print("\n### 💡 USE CASE SPECIFIC RECOMMENDATIONS")
        print("-" * 100)
        
        print("\n✓ Real-time Applications:")
        fastest = df_sorted.nsmallest(1, 'avg_time').iloc[0]
        print(f"   - Use: {fastest['voting_method']} @ threshold {fastest['threshold']:.2f}")
        print(f"   - Speed: {fastest['avg_time']:.3f}s | Accuracy: {fastest['accuracy']:.2f}%")
        
        print("\n✓ Conservative Detection (Minimize False Positives):")
        conservative = df_sorted.nlargest(1, 'precision').iloc[0]
        print(f"   - Use: {conservative['voting_method']} @ threshold {conservative['threshold']:.2f}")
        print(f"   - Precision: {conservative['precision']:.2f}% | Accuracy: {conservative['accuracy']:.2f}%")
        
        print("\n✓ Sensitive Detection (Catch All Fakes):")
        sensitive = df_sorted.nlargest(1, 'recall').iloc[0]
        print(f"   - Use: {sensitive['voting_method']} @ threshold {sensitive['threshold']:.2f}")
        print(f"   - Recall: {sensitive['recall']:.2f}% | Accuracy: {sensitive['accuracy']:.2f}%")
        
        print("\n" + "=" * 100)
        print("END OF ANALYSIS")
        print("=" * 100)
        
        return {
            'primary': {
                'threshold': best_overall['threshold'],
                'voting_method': best_overall['voting_method'],
                'accuracy': best_overall['accuracy'],
                'confidence': best_overall['avg_confidence']
            },
            'production': {
                'threshold': best_f1['threshold'],
                'voting_method': best_f1['voting_method'],
                'f1_score': best_f1['f1_score']
            },
            'high_confidence': {
                'threshold': best_confidence['threshold'],
                'voting_method': best_confidence['voting_method'],
                'confidence': best_confidence['avg_confidence']
            }
        }
    
    def export_results(self, metrics, filename='test_results.csv'):
        """Export results to CSV file"""
        
        df = pd.DataFrame(metrics).T
        output_path = os.path.join(os.path.dirname(__file__), filename)
        df.to_csv(output_path)
        
        print(f"\n[EXPORT] Results exported to: {output_path}")
        
        return output_path


def main():
    """Main execution function"""
    
    print("\n" + "=" * 100)
    print("DEEPFAKE DETECTION: COMPREHENSIVE THRESHOLD & VOTING METHOD TEST")
    print("=" * 100)
    
    # Configuration
    default_image_dir = r"C:\Kidambi\WorkRelated\DeepFake\sample_images\testthresholdvoting"
    
    # Get debug flag first
    debug = '--debug' in sys.argv or '-d' in sys.argv
    
    # Get image directory from command line or use default (filter out flags)
    image_dir = default_image_dir
    for arg in sys.argv[1:]:
        if not arg.startswith('-'):
            image_dir = arg
            break
    
    print(f"\n[CONFIG] Configuration:")
    print(f"  - Image Directory: {image_dir}")
    print(f"  - Debug Mode: {debug}")
    
    # Create analyzer
    analyzer = DeepfakeTestAnalyzer(image_dir, debug=debug)
    
    # Load ensemble
    if not analyzer.load_ensemble():
        print("\n[ERROR] Failed to load ensemble detector. Exiting.")
        return 1
    
    # Run comprehensive test
    results = analyzer.run_comprehensive_test()
    
    if results is None:
        print("\n[ERROR] Test failed. Exiting.")
        return 1
    
    # Calculate metrics
    metrics = analyzer.calculate_metrics()
    
    # Display results
    df_sorted, best_accuracy, best_f1, best_confidence = analyzer.display_results(metrics)
    
    # Generate recommendations
    recommendations = analyzer.generate_recommendations(df_sorted, best_accuracy, best_f1, best_confidence)
    
    # Export results
    analyzer.export_results(metrics, filename='deepfake_test_results.csv')
    
    print("\n[SUCCESS] Test completed successfully!")
    
    return 0


if __name__ == "__main__":
    exit(main())
