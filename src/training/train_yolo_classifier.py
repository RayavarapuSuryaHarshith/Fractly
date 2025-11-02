"""
SIMPLIFIED YOLO FINE-TUNING FOR FRACTURE CLASSIFICATION
=======================================================

This script fine-tunes YOLOv11 for fracture classification rather than detection.
This is more appropriate since we have classification labels, not bounding boxes.
"""

import os
import torch
from ultralytics import YOLO
from pathlib import Path
import shutil

def setup_yolo_classification_dataset():
    """
    Create YOLO classification dataset from our fracture/non-fracture images
    """
    
    # Create dataset structure for classification
    dataset_dir = Path('new Dataset')
    
    # Create train and val directories with class subdirectories
    # Folders already created by split script; no need to create
    pass
    
    print("Using new Dataset for YOLO classification training...")
    
    # Just return the new dataset path
    return dataset_dir

def train_yolo_classification():
    """
    Train YOLOv11 for fracture classification
    """
    
    print("Fine-tuning YOLOv11 for fracture classification...")
    
    # Use new dataset directory for classification (expects folder, not YAML)
    import os
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../dataset/yolo_classification'))
    
    # Load YOLOv11 classification model
    model = YOLO('./weights/yolo11n-cls.pt')  # Use classification variant
    
    try:
        # Train the model
        results = model.train(
            data=dataset_dir,
            epochs=20,
            imgsz=224,
            batch=8,
            device='cpu',
            project='runs/fracture_classification',
            name='yolo_fracture_classifier_v1',
            save=True,
            patience=10,
            verbose=True
        )
        
        # Save the best model to our weights folder
        best_model_path = './weights/yolo11n_fracture_classifier.pt'
        os.makedirs('./weights', exist_ok=True)
        
        # Find and copy the best weights
        runs_dir = Path('runs/fracture_classification/yolo_fracture_classifier_v1/weights')
        if (runs_dir / 'best.pt').exists():
            shutil.copy2(runs_dir / 'best.pt', best_model_path)
            print(f"✅ Fine-tuned model saved to: {best_model_path}")
        else:
            print("❌ Best weights not found")
        
        return best_model_path
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        
        # Try to download the classification model first
        print("Downloading YOLOv11 classification model...")
        try:
            model = YOLO('yolo11n-cls.pt')  # This will download it
            model.export(format='pt')
            
            # Copy to our weights folder
            if os.path.exists('yolo11n-cls.pt'):
                shutil.copy2('yolo11n-cls.pt', './weights/yolo11n_fracture_classifier.pt')
                print(f"✅ Base classification model saved to: ./weights/yolo11n_fracture_classifier.pt")
                return './weights/yolo11n_fracture_classifier.pt'
                
        except Exception as e2:
            print(f"❌ Could not download base model: {e2}")
            
        return None

if __name__ == "__main__":
    model_path = train_yolo_classification()
    
    if model_path:
        print(f"\n🎯 YOLO Classification Fine-tuning Complete!")
        print(f"   Model saved: {model_path}")
        print(f"   Use this model for fracture classification in hybrid model")
    else:
        print(f"\n❌ Fine-tuning failed. Check error messages above.")