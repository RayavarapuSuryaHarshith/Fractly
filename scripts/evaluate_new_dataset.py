"""
Quick dataset evaluation script for testing the hybrid model on a new dataset

Usage:
    python evaluate_new_dataset.py path/to/new/dataset/
    
Expected directory structure:
    new_dataset/
        fracture/
            image1.jpg
            image2.jpg
            ...
        no_fracture/
            image1.jpg  
            image2.jpg
            ...
"""
import os
import sys
from pathlib import Path
sys.path.append('src')

from src.final_hybrid_model import FinalHybridFractureDetector

def evaluate_dataset(dataset_path):
    """Evaluate the hybrid model on a new dataset"""
    dataset_path = Path(dataset_path)
    detector = FinalHybridFractureDetector()
    
    total = 0
    correct = 0
    
    for label_name, expected_label in [("fracture", 0), ("no_fracture", 1)]:
        label_dir = dataset_path / label_name
        if not label_dir.exists():
            print(f"Warning: {label_dir} not found")
            continue
            
        for img_path in label_dir.glob("*.jpg"):
            result = detector.predict(img_path.as_posix())
            predicted_fracture = result["is_fracture"]
            predicted_label = 0 if predicted_fracture else 1
            
            total += 1
            if predicted_label == expected_label:
                correct += 1
                
            print(f"✅ {img_path.name}: {result['decision_reason'][:50]}...")
    
    accuracy = correct / total if total > 0 else 0
    print(f"\n🎯 Results on New Dataset:")
    print(f"   Total images: {total}")
    print(f"   Correct: {correct}")
    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    if accuracy >= 0.90:
        print("✅ Excellent performance! Model works well on this dataset.")
    elif accuracy >= 0.80:
        print("⚠️  Good performance, but consider fine-tuning for better results.")
    elif accuracy >= 0.70:
        print("⚠️  Moderate performance. Fine-tuning recommended.")
    else:
        print("❌ Poor performance. Significant retraining needed.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate_new_dataset.py path/to/dataset/")
        sys.exit(1)
    
    evaluate_dataset(sys.argv[1])