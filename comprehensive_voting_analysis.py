#!/usr/bin/env python3
"""
Comprehensive Voting Method Analysis Script
Tests all voting methods against a dataset of images and provides recommendations
"""

import os
import sys
import time
import numpy as np
from collections import defaultdict, Counter
from app_detect_image import create_ensemble_detector, process_image_data

def analyze_all_voting_methods(directory_path, threshold=0.26):
    """
    Comprehensive analysis of all voting methods on a dataset
    
    Args:
        directory_path: Path to directory containing test images
        threshold: Detection threshold to use
    
    Returns:
        Dictionary with analysis results and recommendations
    """
    
    print("🔬 COMPREHENSIVE VOTING METHOD ANALYSIS")
    print("=" * 60)
    print(f"📁 Dataset: {directory_path}")
    print(f"🎯 Threshold: {threshold}")
    
    # All available voting methods
    voting_methods = [
        'majority', 'weighted', 'confidence_weighted', 'unanimous', 'maximum',
        'adaptive_threshold', 'median', 'robust_majority', 'dynamic_consensus',
        'reliability_weighted', 'bayesian', 'entropy_weighted', 'consensus_threshold'
    ]
    
    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"❌ Error: Directory '{directory_path}' does not exist!")
        return None
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.avif', '.webp'}
    image_files = []
    
    for file in os.listdir(directory_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(directory_path, file))
    
    if not image_files:
        print(f"❌ No image files found in {directory_path}")
        return None
    
    print(f"📸 Found {len(image_files)} images to analyze")
    
    # Create ensemble detector
    print("🔧 Creating ensemble detector...")
    try:
        ensemble = create_ensemble_detector()
        print("✅ Ensemble created successfully!")
    except Exception as e:
        print(f"❌ Error creating ensemble: {e}")
        return None
    
    # Results storage
    method_results = {method: {
        'correct': 0,
        'total': 0,
        'true_positives': 0,
        'true_negatives': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'confidences': [],
        'errors': 0,
        'processing_times': []
    } for method in voting_methods}
    
    image_results = {}  # Per-image results for detailed analysis
    
    print(f"\\n🧪 Testing {len(voting_methods)} voting methods on {len(image_files)} images...")
    print("This may take several minutes...\n")
    
    # Process each image with each voting method
    for i, image_path in enumerate(image_files, 1):
        image_name = os.path.basename(image_path)
        print(f"[{i}/{len(image_files)}] Processing: {image_name}")
        
        # Determine ground truth from filename
        ground_truth_is_fake = 'fake' in image_name.lower()
        ground_truth = 'FAKE' if ground_truth_is_fake else 'REAL'
        
        image_results[image_name] = {
            'ground_truth': ground_truth,
            'methods': {}
        }
        
        # Test each voting method
        for method in voting_methods:
            try:
                start_time = time.time()
                
                # Get prediction using this voting method
                result = ensemble.detect(image_path, threshold=threshold, voting_method=method)
                
                processing_time = time.time() - start_time
                method_results[method]['processing_times'].append(processing_time)
                
                # Analyze results
                predicted_fake = result.is_deepfake
                is_correct = predicted_fake == ground_truth_is_fake
                
                # Update statistics
                method_results[method]['total'] += 1
                method_results[method]['confidences'].append(result.confidence)
                
                if is_correct:
                    method_results[method]['correct'] += 1
                
                # Confusion matrix
                if ground_truth_is_fake and predicted_fake:
                    method_results[method]['true_positives'] += 1
                elif not ground_truth_is_fake and not predicted_fake:
                    method_results[method]['true_negatives'] += 1
                elif not ground_truth_is_fake and predicted_fake:
                    method_results[method]['false_positives'] += 1
                elif ground_truth_is_fake and not predicted_fake:
                    method_results[method]['false_negatives'] += 1
                
                # Store per-image result
                image_results[image_name]['methods'][method] = {
                    'prediction': 'FAKE' if predicted_fake else 'REAL',
                    'confidence': result.confidence,
                    'correct': is_correct,
                    'processing_time': processing_time
                }
                
            except Exception as e:
                method_results[method]['errors'] += 1
                image_results[image_name]['methods'][method] = {
                    'prediction': 'ERROR',
                    'confidence': 0.0,
                    'correct': False,
                    'error': str(e),
                    'processing_time': 0.0
                }
                print(f"    ❌ Error with {method}: {str(e)[:50]}...")
    
    print("\\n✅ Analysis complete! Generating results...")
    
    # Calculate comprehensive metrics
    analysis_results = calculate_comprehensive_metrics(method_results, image_results)
    
    # Display results
    display_analysis_results(analysis_results, method_results, len(image_files))
    
    # Generate recommendations
    recommendations = generate_recommendations(analysis_results, method_results)
    display_recommendations(recommendations)
    
    return {
        'analysis_results': analysis_results,
        'method_results': method_results,
        'image_results': image_results,
        'recommendations': recommendations
    }

def calculate_comprehensive_metrics(method_results, image_results):
    """Calculate comprehensive metrics for each voting method"""
    
    analysis_results = {}
    
    for method, results in method_results.items():
        if results['total'] == 0:
            continue
            
        # Basic metrics
        accuracy = results['correct'] / results['total'] * 100
        
        # Precision, Recall, F1
        tp = results['true_positives']
        tn = results['true_negatives'] 
        fp = results['false_positives']
        fn = results['false_negatives']
        
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Confidence statistics
        confidences = results['confidences']
        avg_confidence = np.mean(confidences) if confidences else 0
        confidence_std = np.std(confidences) if confidences else 0
        
        # Processing time statistics
        times = results['processing_times']
        avg_time = np.mean(times) if times else 0
        
        # Consistency score (how often this method agrees with others)
        consistency_scores = []
        for image_name, image_data in image_results.items():
            if method in image_data['methods']:
                method_pred = image_data['methods'][method]['prediction']
                other_preds = [data['prediction'] for m, data in image_data['methods'].items() 
                              if m != method and data['prediction'] != 'ERROR']
                if other_preds:
                    agreement_ratio = sum(1 for pred in other_preds if pred == method_pred) / len(other_preds)
                    consistency_scores.append(agreement_ratio)
        
        consistency = np.mean(consistency_scores) * 100 if consistency_scores else 0
        
        analysis_results[method] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'avg_confidence': avg_confidence,
            'confidence_std': confidence_std,
            'avg_processing_time': avg_time,
            'consistency': consistency,
            'error_rate': results['errors'] / (results['total'] + results['errors']) * 100 if results['total'] + results['errors'] > 0 else 0,
            'total_processed': results['total']
        }
    
    return analysis_results

def display_analysis_results(analysis_results, method_results, total_images):
    """Display comprehensive analysis results"""
    
    print("\\n\\n📊 COMPREHENSIVE RESULTS")
    print("=" * 80)
    
    # Sort by accuracy
    sorted_methods = sorted(analysis_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    print("\\n### 🏆 Overall Performance Ranking")
    print("| Rank | Method | Accuracy | Precision | Recall | F1-Score | Avg Time |")
    print("|------|--------|----------|-----------|--------|----------|----------|")
    
    for rank, (method, metrics) in enumerate(sorted_methods, 1):
        print(f"| {rank:2d} | {method:15s} | {metrics['accuracy']:6.2f}% | "
              f"{metrics['precision']:7.2f}% | {metrics['recall']:6.2f}% | "
              f"{metrics['f1_score']:6.2f}% | {metrics['avg_processing_time']:6.3f}s |")
    
    print("\\n### 📈 Detailed Metrics")
    print("| Method | Consistency | Confidence | Std Dev | Error Rate |")
    print("|--------|-------------|------------|---------|------------|")
    
    for method, metrics in sorted_methods:
        print(f"| {method:15s} | {metrics['consistency']:9.2f}% | "
              f"{metrics['avg_confidence']:8.3f} | {metrics['confidence_std']:6.3f} | "
              f"{metrics['error_rate']:8.2f}% |")
    
    # Category winners
    print("\\n### 🥇 Category Winners")
    
    best_accuracy = max(sorted_methods, key=lambda x: x[1]['accuracy'])
    best_precision = max(analysis_results.items(), key=lambda x: x[1]['precision'])
    best_recall = max(analysis_results.items(), key=lambda x: x[1]['recall'])
    best_f1 = max(analysis_results.items(), key=lambda x: x[1]['f1_score'])
    best_speed = min(analysis_results.items(), key=lambda x: x[1]['avg_processing_time'])
    best_consistency = max(analysis_results.items(), key=lambda x: x[1]['consistency'])
    
    print(f"- **Highest Accuracy**: {best_accuracy[0]} ({best_accuracy[1]['accuracy']:.2f}%)")
    print(f"- **Best Precision**: {best_precision[0]} ({best_precision[1]['precision']:.2f}%)")
    print(f"- **Best Recall**: {best_recall[0]} ({best_recall[1]['recall']:.2f}%)")
    print(f"- **Best F1-Score**: {best_f1[0]} ({best_f1[1]['f1_score']:.2f}%)")
    print(f"- **Fastest**: {best_speed[0]} ({best_speed[1]['avg_processing_time']:.3f}s)")
    print(f"- **Most Consistent**: {best_consistency[0]} ({best_consistency[1]['consistency']:.2f}%)")

def generate_recommendations(analysis_results, method_results):
    """Generate data-driven recommendations"""
    
    # Sort methods by different criteria
    by_accuracy = sorted(analysis_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    by_f1 = sorted(analysis_results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
    by_consistency = sorted(analysis_results.items(), key=lambda x: x[1]['consistency'], reverse=True)
    by_speed = sorted(analysis_results.items(), key=lambda x: x[1]['avg_processing_time'])
    
    recommendations = {
        'overall_best': by_accuracy[0][0],
        'balanced_performance': by_f1[0][0],
        'most_reliable': by_consistency[0][0],
        'fastest': by_speed[0][0],
        'top_3_accuracy': [method[0] for method in by_accuracy[:3]],
        'avoid': []
    }
    
    # Find methods to avoid (low accuracy or high error rate)
    for method, metrics in analysis_results.items():
        if metrics['accuracy'] < 70 or metrics['error_rate'] > 10:
            recommendations['avoid'].append(method)
    
    # Specific use case recommendations
    recommendations['use_cases'] = {
        'production': by_f1[0][0],  # Best F1 score for balanced performance
        'research': by_accuracy[0][0],  # Highest accuracy
        'real_time': by_speed[0][0],  # Fastest processing
        'conservative': None,  # Will be determined by precision
        'sensitive': None   # Will be determined by recall
    }
    
    # Find best for specific needs
    best_precision = max(analysis_results.items(), key=lambda x: x[1]['precision'])
    best_recall = max(analysis_results.items(), key=lambda x: x[1]['recall'])
    
    recommendations['use_cases']['conservative'] = best_precision[0]
    recommendations['use_cases']['sensitive'] = best_recall[0]
    
    return recommendations

def display_recommendations(recommendations):
    """Display final recommendations"""
    
    print("\\n\\n🎯 DATA-DRIVEN RECOMMENDATIONS")
    print("=" * 60)
    
    print(f"\\n### 🏆 **OVERALL WINNER**: `{recommendations['overall_best']}`")
    print("- Highest accuracy across your dataset")
    print("- Best general-purpose choice for your images")
    
    print(f"\\n### ⚖️ **BALANCED PERFORMANCE**: `{recommendations['balanced_performance']}`")
    print("- Best F1-score (balance of precision and recall)")
    print("- Recommended for production use")
    
    print(f"\\n### 🎯 **MOST RELIABLE**: `{recommendations['most_reliable']}`")
    print("- Highest consistency with other methods")
    print("- Most predictable behavior")
    
    print(f"\\n### ⚡ **FASTEST**: `{recommendations['fastest']}`")
    print("- Lowest processing time")
    print("- Best for real-time applications")
    
    print("\\n### 📋 **USE CASE RECOMMENDATIONS**")
    print("| Use Case | Recommended Method | Why |")
    print("|----------|-------------------|-----|")
    print(f"| Production Systems | `{recommendations['use_cases']['production']}` | Best balanced performance |")
    print(f"| Research/Academic | `{recommendations['use_cases']['research']}` | Highest accuracy |")
    print(f"| Real-time Detection | `{recommendations['use_cases']['real_time']}` | Fastest processing |")
    print(f"| Conservative (few false positives) | `{recommendations['use_cases']['conservative']}` | Highest precision |")
    print(f"| Sensitive (catch all fakes) | `{recommendations['use_cases']['sensitive']}` | Highest recall |")
    
    print(f"\\n### 🥉 **TOP 3 METHODS**:")
    for i, method in enumerate(recommendations['top_3_accuracy'], 1):
        print(f"{i}. `{method}`")
    
    if recommendations['avoid']:
        print(f"\\n### ⚠️ **METHODS TO AVOID**:")
        for method in recommendations['avoid']:
            print(f"- `{method}` (low accuracy or high error rate)")
    else:
        print(f"\\n### ✅ **ALL METHODS PERFORMED WELL!**")
        print("No methods showed significantly poor performance.")
    
    print(f"\\n### 💡 **FINAL RECOMMENDATION**")
    print(f"Based on your specific dataset, we recommend using:")
    print(f"1. **Primary**: `{recommendations['overall_best']}` for best accuracy")
    print(f"2. **Fallback**: `{recommendations['balanced_performance']}` for balanced performance")
    print(f"3. **Speed-critical**: `{recommendations['fastest']}` when speed matters")

def main():
    """Main execution function"""
    
    # Default directory - update this to your path
    default_directory = "C:\\Kidambi\\WorkRelated\\DeepFake\\sample_images"
    
    # Get directory from command line or use default
    if len(sys.argv) > 1:
        directory_path = sys.argv[1]
    else:
        directory_path = default_directory
    
    # Get threshold from command line or use default
    if len(sys.argv) > 2:
        try:
            threshold = float(sys.argv[2])
        except ValueError:
            threshold = 0.26
            print(f"Invalid threshold, using default: {threshold}")
    else:
        threshold = 0.26
    
    print(f"🚀 Starting comprehensive voting method analysis...")
    print(f"📁 Directory: {directory_path}")
    print(f"🎯 Threshold: {threshold}")
    
    # Run analysis
    results = analyze_all_voting_methods(directory_path, threshold)
    
    if results:
        print(f"\\n\\n✅ Analysis completed successfully!")
        print(f"Results and recommendations have been generated above.")
    else:
        print(f"\\n❌ Analysis failed. Please check the directory path and try again.")

if __name__ == "__main__":
    main()