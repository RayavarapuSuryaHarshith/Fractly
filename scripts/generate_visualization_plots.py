#!/usr/bin/env python3
"""
Generate visualization plots for the final hybrid model test results.
This script reads the test results JSON file and creates comprehensive visualization plots.
"""

import matplotlib.pyplot as plt
import json
import numpy as np
import seaborn as sns
from pathlib import Path

def load_results(results_file="results/final_hybrid_test_results.json"):
    """Load results from JSON file."""
    try:
        with open(results_file, "r") as f:
            results = json.load(f)
        return results
    except FileNotFoundError:
        print(f"Error: {results_file} not found!")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {results_file}!")
        return None

def create_confusion_matrix_plot(results):
    """Create a confusion matrix heatmap."""
    # Extract confusion matrix data
    confusion_matrix = np.array(results["confusion_matrix"])
    
    # Create labels
    class_names = ['No Fracture', 'Fracture']
    
    # Create the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Final Hybrid Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close to avoid display issues
    print("✅ Confusion matrix heatmap saved to results/confusion_matrix_heatmap.png")

def create_confusion_matrix_breakdown(results):
    """Create a bar plot breakdown of confusion matrix values."""
    confusion_matrix = results["confusion_matrix"]
    
    # Extract values correctly from the 2D list
    true_negatives = confusion_matrix[0][0]   # No fracture predicted as no fracture
    false_positives = confusion_matrix[0][1]  # No fracture predicted as fracture
    false_negatives = confusion_matrix[1][0]  # Fracture predicted as no fracture
    true_positives = confusion_matrix[1][1]   # Fracture predicted as fracture
    
    # Create bar plot
    values = [true_positives, false_negatives, true_negatives, false_positives]
    labels = ["True Positives", "False Negatives", "True Negatives", "False Positives"]
    colors = ['green', 'red', 'blue', 'orange']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=colors)
    plt.title("Confusion Matrix Breakdown")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(value), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/confusion_matrix_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close to avoid display issues
    print("✅ Confusion matrix breakdown saved to results/confusion_matrix_breakdown.png")

def create_accuracy_metrics_plot(results):
    """Create a comprehensive metrics visualization."""
    metrics = {
        'Overall Accuracy': results['overall_accuracy'],
        'Fracture Accuracy': results['fracture_accuracy'], 
        'No Fracture Accuracy': results['no_fracture_accuracy'],
        'Sensitivity (Recall)': results['sensitivity'],
        'Specificity': results['specificity']
    }
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(metrics.keys(), metrics.values(), 
                   color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'])
    
    plt.title("Model Performance Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, (metric, value) in zip(bars, metrics.items()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Add horizontal line for target accuracy if available
    if 'target_accuracy' in results:
        plt.axhline(y=results['target_accuracy'], color='red', linestyle='--', 
                   label=f'Target: {results["target_accuracy"]:.1f}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close to avoid display issues
    print("✅ Performance metrics plot saved to results/performance_metrics.png")

def create_decision_strategy_plot(results):
    """Create a pie chart showing decision strategy breakdown."""
    if 'decision_reasons' in results:
        decision_data = results['decision_reasons']
        
        plt.figure(figsize=(10, 8))
        labels = list(decision_data.keys())
        values = list(decision_data.values())
        
        # Create pie chart
        plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title("Decision Strategy Breakdown")
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('results/decision_strategy_breakdown.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close to avoid display issues
        print("✅ Decision strategy breakdown saved to results/decision_strategy_breakdown.png")

def create_comprehensive_summary(results):
    """Create a comprehensive summary plot."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Confusion Matrix
    confusion_matrix = np.array(results["confusion_matrix"])
    class_names = ['No Fracture', 'Fracture']
    
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0,0])
    axes[0,0].set_title('Confusion Matrix')
    axes[0,0].set_ylabel('True Label')
    axes[0,0].set_xlabel('Predicted Label')
    
    # 2. Performance Metrics
    metrics = {
        'Overall': results['overall_accuracy'],
        'Fracture': results['fracture_accuracy'], 
        'No Fracture': results['no_fracture_accuracy'],
        'Sensitivity': results['sensitivity'],
        'Specificity': results['specificity']
    }
    
    bars = axes[0,1].bar(metrics.keys(), metrics.values(), 
                        color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'])
    axes[0,1].set_title('Performance Metrics')
    axes[0,1].set_ylabel('Score')
    axes[0,1].set_ylim(0, 1.1)
    axes[0,1].tick_params(axis='x', rotation=45)
    
    for bar, (metric, value) in zip(bars, metrics.items()):
        axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                      f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Decision Strategy
    if 'decision_reasons' in results:
        decision_data = results['decision_reasons']
        axes[1,0].pie(decision_data.values(), labels=decision_data.keys(), 
                     autopct='%1.1f%%', startangle=90)
        axes[1,0].set_title('Decision Strategy Breakdown')
    
    # 4. Model Summary
    axes[1,1].text(0.1, 0.9, f"Model: {results['model_type']}", fontsize=12, transform=axes[1,1].transAxes)
    axes[1,1].text(0.1, 0.8, f"Total Samples: {results['total_samples']}", fontsize=12, transform=axes[1,1].transAxes)
    axes[1,1].text(0.1, 0.7, f"Overall Accuracy: {results['overall_accuracy']:.3f}", fontsize=12, transform=axes[1,1].transAxes)
    axes[1,1].text(0.1, 0.6, f"Mean Confidence: {results['mean_confidence']:.3f}", fontsize=12, transform=axes[1,1].transAxes)
    axes[1,1].text(0.1, 0.5, f"Agreement Rate: {results['agreement_rate']:.3f}", fontsize=12, transform=axes[1,1].transAxes)
    axes[1,1].text(0.1, 0.4, f"Target Achieved: {results['target_achieved']}", fontsize=12, transform=axes[1,1].transAxes)
    axes[1,1].set_title('Model Summary')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/comprehensive_model_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close to avoid display issues
    print("✅ Comprehensive dashboard saved to results/comprehensive_model_dashboard.png")

def main():
    """Main function to generate all plots."""
    print("Loading test results...")
    results = load_results()
    
    if results is None:
        return
    
    print("Creating visualizations...")
    
    # Create individual plots
    print("1. Creating confusion matrix heatmap...")
    create_confusion_matrix_plot(results)
    
    print("2. Creating confusion matrix breakdown...")
    create_confusion_matrix_breakdown(results)
    
    print("3. Creating performance metrics plot...")
    create_accuracy_metrics_plot(results)
    
    print("4. Creating decision strategy plot...")
    create_decision_strategy_plot(results)
    
    print("5. Creating comprehensive dashboard...")
    create_comprehensive_summary(results)
    
    print("\n🎉 All visualizations created successfully!")
    print("Generated files in results/ directory:")
    print("- confusion_matrix_heatmap.png")
    print("- confusion_matrix_breakdown.png") 
    print("- performance_metrics.png")
    print("- decision_strategy_breakdown.png")
    print("- comprehensive_model_dashboard.png")

if __name__ == "__main__":
    main()