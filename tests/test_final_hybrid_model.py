#!/usr/bin/env python3
"""
FINAL HYBRID SYSTEM TEST
====================================

Test script for the final hybrid system .
This script validates the performance of the extracted model.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

# Add src to path
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from final_hybrid_model import FinalHybridFractureDetector

def test_final_90plus_hybrid():
    """Test the final 90%+ hybrid system"""

    print("🎯 TESTING FINAL 90%+ ACCURACY HYBRID SYSTEM")
    print("=" * 60)
    
    # Initialize the final hybrid system with newly trained weights
    hybrid = FinalHybridFractureDetector(
        yolo_weights_path="weights/yolo_binary_high_accuracy.pt",
        fracnet_weights_path="weights/custom_fracnet2d_best.pth"  # Use newly trained balanced weights
    )
    
    print("✅ Final 90%+ hybrid system initialized with newly trained balanced weights")

    # Test dataset path - use the new balanced dataset
    test_data = r"dataset\yolo_classification\test"

    # Collect test images
    test_images = []
    for class_dir in ['fracture', 'no_fracture']:
        class_path = os.path.join(test_data, class_dir)
        if os.path.exists(class_path):
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    true_class = 0 if class_dir == 'fracture' else 1
                    test_images.append((os.path.join(class_path, img_file), true_class))

    print(f"📊 Testing on {len(test_images)} images")

    # Run predictions
    predictions = []
    ground_truth = []
    confidence_scores = []
    decision_reasons = {}
    agreement_count = 0

    print("\n🔍 Running predictions...")
    for img_path, true_label in tqdm(test_images, desc="Processing images"):
        try:
            # Use the new predict method
            result = hybrid.predict(img_path)
            
            # Extract results
            pred_label = 0 if result['is_fracture'] else 1
            confidence = result['confidence']
            decision_reason = result['decision_reason']
            models_agree = result['models_agree']
            
            predictions.append(pred_label)
            ground_truth.append(true_label)
            confidence_scores.append(confidence)
            
            # Track decision reasons
            if decision_reason not in decision_reasons:
                decision_reasons[decision_reason] = 0
            decision_reasons[decision_reason] += 1
            
            # Track model agreement
            if models_agree:
                agreement_count += 1
                
        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")
            predictions.append(1)  # Default to no fracture
            ground_truth.append(true_label)
            confidence_scores.append(0.5)

    # Convert to numpy arrays
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    confidence_scores = np.array(confidence_scores)

    # Calculate metrics
    overall_accuracy = accuracy_score(ground_truth, predictions)
    
    # Per-class accuracy
    fracture_mask = ground_truth == 0
    no_fracture_mask = ground_truth == 1
    
    fracture_accuracy = accuracy_score(ground_truth[fracture_mask], predictions[fracture_mask]) if np.sum(fracture_mask) > 0 else 0
    no_fracture_accuracy = accuracy_score(ground_truth[no_fracture_mask], predictions[no_fracture_mask]) if np.sum(no_fracture_mask) > 0 else 0
    
    # Sensitivity and Specificity
    sensitivity = fracture_accuracy  # True positive rate for fractures
    specificity = no_fracture_accuracy  # True negative rate for non-fractures
    
    # Calculate agreement rate
    agreement_rate = agreement_count / len(test_images) if len(test_images) > 0 else 0

    print("\n" + "="*60)
    print("🏆 FINAL HYBRID SYSTEM RESULTS")
    print("="*60)
    print(f"📊 Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    print(f"🦴 Fracture Detection Accuracy: {fracture_accuracy:.4f} ({fracture_accuracy*100:.2f}%)")
    print(f"✅ No-Fracture Detection Accuracy: {no_fracture_accuracy:.4f} ({no_fracture_accuracy*100:.2f}%)")
    print(f"🎯 Sensitivity (Fracture Detection): {sensitivity:.4f} ({sensitivity*100:.2f}%)")
    print(f"🔍 Specificity (Normal Classification): {specificity:.4f} ({specificity*100:.2f}%)")
    print(f"🤝 Model Agreement Rate: {agreement_rate:.4f} ({agreement_rate*100:.2f}%)")
    print(f"😊 Mean Confidence: {np.mean(confidence_scores):.4f}")

    print(f"\n📋 Decision Strategy Breakdown:")
    for reason, count in decision_reasons.items():
        percentage = (count / len(test_images)) * 100
        print(f"   {reason}: {count} cases ({percentage:.1f}%)")

    # Confusion Matrix
    print(f"\n🔢 Confusion Matrix:")
    cm = confusion_matrix(ground_truth, predictions)
    print("     Predicted")
    print("       0    1")
    print(f"Act 0 [{cm[0,0]:3d}  {cm[0,1]:3d}]  (0=Fracture)")
    print(f"Act 1 [{cm[1,0]:3d}  {cm[1,1]:3d}]  (1=No Fracture)")

    # Classification report
    print(f"\n📈 Detailed Classification Report:")
    class_names = ['Fracture', 'No Fracture']
    report = classification_report(ground_truth, predictions, target_names=class_names)
    print(report)

    # Success criteria
    target_accuracy = 0.90
    achieved_target = overall_accuracy >= target_accuracy
    
    print("\n" + "="*60)
    if achieved_target:
        print("🏆 🎉 SUCCESS! TARGET ACHIEVED! 🎉 🏆")
        print(f"✅ Achieved {overall_accuracy*100:.2f}% accuracy (Target: {target_accuracy*100:.0f}%+)")
        print("🚀 The hybrid system exceeds the 90% accuracy requirement!")
    else:
        print("❌ Target not met")
        print(f"📊 Achieved {overall_accuracy*100:.2f}% accuracy (Target: {target_accuracy*100:.0f}%+)")
        print(f"📈 Gap: {(target_accuracy - overall_accuracy)*100:.2f}%")

    # Save results
    results = {
        'model_type': 'final_hybrid_fracture_detector',
        'overall_accuracy': float(overall_accuracy),
        'fracture_accuracy': float(fracture_accuracy),
        'no_fracture_accuracy': float(no_fracture_accuracy),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'mean_confidence': float(np.mean(confidence_scores)),
        'agreement_rate': float(agreement_rate),
        'decision_reasons': decision_reasons,
        'total_samples': len(test_images),
        'confusion_matrix': cm.tolist(),
        'target_achieved': achieved_target,
        'target_accuracy': target_accuracy
    }
    
    with open('final_hybrid_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: final_hybrid_test_results.json")
    print("="*60)
    
    return achieved_target

if __name__ == "__main__":
    success = test_final_90plus_hybrid()
    if success:
        print("\n🏆 MISSION STATUS: ACCOMPLISHED! 🏆")
        print("🚀 Final hybrid system ready for deployment!")
    else:
        print("\n🔧 Further optimization may be needed")
    
    sys.exit(0 if success else 1)