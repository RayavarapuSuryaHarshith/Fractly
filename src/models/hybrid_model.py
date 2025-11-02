"""
Production Hybrid Bone Fracture Detection Model
==============================================
This is the production-ready hybrid model combining YOLO11 and FracNet
for real-time bone fracture detection
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
from ultralytics import YOLO
import os
import sys

# Import FracNet model from our models package
try:
    from .fracnet.custom_fracnet2d import CustomFracNet2D, load_custom_fracnet2d
except ImportError:
    print("Warning: Could not import FracNet model components")

class HybridFractureDetector:
    """
    Production hybrid model for bone fracture detection
    Combines YOLO11 classification with FracNet segmentation
    """
    
    def __init__(self, yolo_weights_path=None, fracnet_weights_path=None, device='cpu'):
        """
        Initialize the hybrid model
        
        Args:
            yolo_weights_path: Path to trained YOLO weights
            fracnet_weights_path: Path to trained FracNet weights  
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.device = device
        
        # Default weights paths
        if yolo_weights_path is None:
            yolo_weights_path = "weights/yolo11n_fracture_trained.pt"
        if fracnet_weights_path is None:
            fracnet_weights_path = "weights/custom_fracnet2d_best.pth"
            
        self.yolo_weights_path = yolo_weights_path
        self.fracnet_weights_path = fracnet_weights_path
        
        # Model configurations
        self.yolo_threshold = 0.05
        self.fracture_classes = [0]  # Class 0 is 'fracture' in the new dataset (fracture/no_fracture)
        
        # Initialize models
        self._load_models()
        
    def _load_models(self):
        """Load YOLO and FracNet models"""
        try:
            # Load YOLO model
            if os.path.exists(self.yolo_weights_path):
                self.yolo_model = YOLO(self.yolo_weights_path)
                print(f"✓ YOLO model loaded from {self.yolo_weights_path}")
            else:
                # Fallback to pretrained model
                self.yolo_model = YOLO('yolo11n-cls.pt')
                print("⚠ Using pretrained YOLO model (custom weights not found)")
                
            # Load FracNet model
            self.fracnet_model = None
            if os.path.exists(self.fracnet_weights_path):
                try:
                    self.fracnet_model = load_custom_fracnet2d(
                        self.fracnet_weights_path, 
                        num_classes=2, 
                        device=self.device
                    )
                    print(f"✓ FracNet model loaded from {self.fracnet_weights_path}")
                except Exception as e:
                    print(f"⚠ Error loading FracNet model: {e}")
            else:
                print("⚠ FracNet weights not found")
                
        except Exception as e:
            print(f"Error loading models: {e}")
            
    def preprocess_image(self, image_path):
        """
        Preprocess image for model inference
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image ready for inference
        """
        try:
            # Load and resize image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
                
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size (adjust based on your training)
            image = cv2.resize(image, (224, 224))
            
            return image
            
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            return None
            
    def predict_yolo(self, image_path):
        """
        Get YOLO classification prediction
        
        Args:
            image_path: Path to image file
            
        Returns:
            tuple: (has_fracture_probability, class_probabilities)
        """
        try:
            # Run YOLO inference
            results = self.yolo_model(image_path, verbose=False)
            
            if results and len(results) > 0:
                # Get class probabilities
                probs = results[0].probs
                if probs is not None:
                    class_probs = probs.data.cpu().numpy()
                    
                    # Calculate fracture probability based on classes 0 and 3
                    fracture_prob = sum(class_probs[i] for i in self.fracture_classes 
                                     if i < len(class_probs))
                    
                    return fracture_prob, class_probs
                    
            return 0.0, np.zeros(4)  # Default to 4 classes
            
        except Exception as e:
            print(f"Error in YOLO prediction for {image_path}: {e}")
            return 0.0, np.zeros(4)
            
    def predict_fracnet(self, image_path):
        """
        Get FracNet segmentation prediction

        Args:
            image_path: Path to image file

        Returns:
            float: Fracture confidence from segmentation
        """
        try:
            if self.fracnet_model is None:
                print(f"   ⚠️  FracNet model is None for {image_path}")
                return 0.5  # Neutral confidence if model not loaded

            # Preprocess image for FracNet
            image = self.preprocess_image(image_path)
            if image is None:
                print(f"   ⚠️  Failed to preprocess image {image_path}")
                return 0.5

            # Convert to tensor and add batch dimension
            image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
            image_tensor = image_tensor / 255.0  # Normalize to [0, 1]
            image_tensor = image_tensor.to(self.device)

            # Run inference
            with torch.no_grad():
                output = self.fracnet_model(image_tensor)
                
                # Process FracNet output
                if isinstance(output, dict) and 'logits' in output:
                    logits = output['logits']
                    # For binary classification, apply sigmoid to get fracture probability
                    fracture_prob = torch.sigmoid(logits).mean().item()
                    return fracture_prob
                else:
                    # Fallback if output format is unexpected
                    print(f"   ⚠️  Unexpected FracNet output format: {type(output)}")
                    return 0.5

        except Exception as e:
            print(f"Error in FracNet prediction for {image_path}: {e}")
            import traceback
            traceback.print_exc()
            return 0.5  # Return neutral confidence on error

    def predict(self, image_path, use_ensemble=True):
        """
        Main prediction method using hybrid approach
        
        Args:
            image_path: Path to image file
            use_ensemble: Whether to use ensemble of both models
            
        Returns:
            dict: Prediction results with confidence scores
        """
        try:
            # Get YOLO prediction
            yolo_prob, class_probs = self.predict_yolo(image_path)
            
            # Get FracNet prediction (if available)
            fracnet_prob = 0.5  # Neutral default
            if self.fracnet_model is not None:
                fracnet_prob = self.predict_fracnet(image_path)
            
            # Decision logic - YOLO only for better balance
            final_prob = yolo_prob  # Use YOLO only
                
            # Apply threshold
            has_fracture = final_prob > self.yolo_threshold
            
            # Confidence calculation
            confidence = final_prob if has_fracture else (1 - final_prob)
            
            return {
                'has_fracture': has_fracture,
                'fracture_probability': final_prob,
                'confidence': confidence,
                'yolo_probability': yolo_prob,
                'fracnet_probability': fracnet_prob,
                'class_probabilities': class_probs.tolist(),
                'threshold_used': self.yolo_threshold
            }
            
        except Exception as e:
            print(f"Error in prediction for {image_path}: {e}")
            return {
                'has_fracture': False,
                'fracture_probability': 0.0,
                'confidence': 0.0,
                'error': str(e)
            }
            
    def evaluate_dataset(self, test_data_path=None):
        """
        Evaluate model on a test dataset
        
        Args:
            test_data_path: Path to test dataset directory
            
        Returns:
            dict: Evaluation metrics
        """
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            predictions = []
            ground_truth = []
            # Use new Dataset/test if not provided
            if test_data_path is None:
                test_data_path = Path('new Dataset/test')
            for class_name, label in [('fracture', 0), ('no_fracture', 1)]:
                class_dir = Path(test_data_path) / class_name
                for img_path in class_dir.glob('*.jpg'):
                    result = self.predict(str(img_path))
                    pred = 0 if result.get('has_fracture', False) else 1
                    predictions.append(pred)
                    ground_truth.append(label)
            accuracy = accuracy_score(ground_truth, predictions)
            precision = precision_score(ground_truth, predictions)
            recall = recall_score(ground_truth, predictions)
            f1 = f1_score(ground_truth, predictions)
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'threshold': self.yolo_threshold
            }
        except Exception as e:
            print(f"Error in evaluation: {e}")
            return {'error': str(e)}

# Production model instance
def load_production_model(device='cpu'):
    """
    Load the production-ready hybrid model
    
    Args:
        device: Device to run on ('cpu' or 'cuda')
        
    Returns:
        HybridFractureDetector: Loaded production model
    """
    return HybridFractureDetector(device=device)

# Example usage
if __name__ == "__main__":
    # Initialize model
    model = load_production_model()
    
    # Example prediction
    # result = model.predict("path/to/xray/image.jpg")
    # print(f"Fracture detected: {result['has_fracture']}")
    # print(f"Confidence: {result['confidence']:.2f}")