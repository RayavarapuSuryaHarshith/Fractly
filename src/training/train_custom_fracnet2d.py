"""
TRAINING SCRIPT FOR CUSTOM FRACNET 2D
=====================================

Training script specifically designed for the Custom FracNet 2D model.
This script handles data loading, training loop, validation, and model saving.

Features:
- Uses the modular CustomFracNet2D from custom_fracnet2d.py
- Enhanced data augmentation for X-ray images
- Learning rate scheduling and early stopping
- Model checkpointing and evaluation metrics
- Compatible with existing fracture dataset structure
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime

from src.models.fracnet.custom_fracnet2d import CustomFracNet2D, load_custom_fracnet2d

class FractureDataset(Dataset):
    """
    Dataset class optimized for Custom FracNet 2D training
from torch.utils.data import WeightedRandomSampler
    
    Handles fracture/non-fracture X-ray images with appropriate augmentation
    """
    
    def __init__(self, image_paths, labels, transform=None, image_size=224):
        self.image_size = image_size
        self.transform = transform
        # Default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])

        # Filter out unreadable images at initialization
        valid_image_paths = []
        valid_labels = []
        skipped = 0
        for img_path, label in zip(image_paths, labels):
            if self.is_image_readable(img_path):
                valid_image_paths.append(img_path)
                valid_labels.append(label)
            else:
                print(f"[Warning] Skipping unreadable image at init: {img_path}")
                skipped += 1
        if skipped > 0:
            print(f"[Warning] Skipped {skipped} unreadable images during dataset initialization.")
        if len(valid_image_paths) == 0:
            raise RuntimeError("No valid images found after filtering. Check your dataset for corrupt or missing files.")
        self.image_paths = valid_image_paths
        self.labels = valid_labels

    @staticmethod
    def is_image_readable(path):
        try:
            with Image.open(path) as img:
                img.verify()
            return True
        except Exception:
            return False
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return image, label
        except Exception as e:
            raise RuntimeError(f"[Error] Failed to load image at {image_path}: {e}")

def get_data_transforms(image_size, augment=True):
    """
    Returns train and validation transforms for the given image size.
    Args:
        image_size: Target image size
        augment: Whether to apply data augmentation
    Returns:
        train_transform, val_transform
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.2),
            transforms.RandomAffine(degrees=10, translate=(0.05,0.05), scale=(0.95,1.05), shear=5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.RandomApply([transforms.Lambda(lambda x: x + 0.01*torch.randn_like(x))], p=0.2),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

def load_fracture_data(data_dir):
    """
    Load fracture dataset from directory structure
    
    Expected structure:
    data_dir/
        fracture/
            *.jpg
        non_fracture/
            *.jpg
    """
    
    image_paths = []
    labels = []
    
    # Load fracture images (label = 1)
    fracture_dir = os.path.join(data_dir, 'fracture')
    if os.path.exists(fracture_dir):
        for img_file in os.listdir(fracture_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(fracture_dir, img_file))
                labels.append(1)
        print(f"Loaded {len([l for l in labels if l == 1])} fracture images from {fracture_dir}")
    
    # Load non-fracture images (label = 0)
    non_fracture_dir = os.path.join(data_dir, 'no_fracture')
    print(f"Checking non_fracture_dir: {non_fracture_dir}, exists: {os.path.exists(non_fracture_dir)}")
    if os.path.exists(non_fracture_dir):
        files = [f for f in os.listdir(non_fracture_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Found {len(files)} non_fracture image files")
        for img_file in files:
            image_paths.append(os.path.join(non_fracture_dir, img_file))
            labels.append(0)
        print(f"Loaded {len([l for l in labels if l == 0])} non-fracture images from {non_fracture_dir}")
    
    return image_paths, labels

def train_custom_fracnet2d():
    config = {
        'batch_size': 16,
        'learning_rate': 0.0005,  # Slightly higher learning rate with stronger regularization
        'num_epochs': 5,  # Use 5 epochs as requested
        'image_size': 224,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': './weights',
        'data_dir': r'dataset/yolo_classification',  # Use new dataset split for FracNet2D
        'early_stopping_patience': 12,  # More patience
        'use_class_balancing': True
    }
    """
    Main training function for Custom FracNet 2D
    """
    
    print("🚀 Training Custom FracNet 2D Model")
    print("=" * 50)
    
    # Training configuration

    print(f"Configuration: {config}")

    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)

    # Use fracture/non_fracture folders for train/val
    train_dir = os.path.join(config['data_dir'], 'train')
    val_dir = os.path.join(config['data_dir'], 'val')

    train_paths, train_labels = load_fracture_data(train_dir)
    val_paths, val_labels = load_fracture_data(val_dir)

    if len(train_paths) == 0 or len(val_paths) == 0:
        print("❌ No images found! Check data directory structure.")
        return

    print(f"Training images: {len(train_paths)}")
    print(f"Validation images: {len(val_paths)}")
    print(f"Fractures (train): {sum(train_labels)}, Non-fractures (train): {len(train_labels) - sum(train_labels)}")
    print(f"Fractures (val): {sum(val_labels)}, Non-fractures (val): {len(val_labels) - sum(val_labels)}")

    # Get transforms
    train_transform, val_transform = get_data_transforms(
        config['image_size'], augment=True
    )

    # Create datasets
    train_dataset = FractureDataset(train_paths, train_labels, train_transform)
    val_dataset = FractureDataset(val_paths, val_labels, val_transform)


    # Optional: Class balancing using WeightedRandomSampler
    if config.get('use_class_balancing', False):
        from torch.utils.data import WeightedRandomSampler
        class_sample_count = np.array([len([l for l in train_labels if l == 0]), len([l for l in train_labels if l == 1])])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in train_labels])
        samples_weight = torch.from_numpy(samples_weight).double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize model
    print(f"\n🔧 Initializing Custom FracNet 2D on {config['device']}")
    model = CustomFracNet2D(in_channels=3, num_classes=2)
    model.to(config['device'])
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.05)  # Increased weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'learning_rate': []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    
    print(f"\n🏋️ Starting training for {config['num_epochs']} epochs...")
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 30)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc='Training')
        for images, labels in train_pbar:
            images, labels = images.to(config['device']), labels.to(config['device'])
            
            optimizer.zero_grad()
            outputs = model(images)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * train_correct / train_total:.1f}%'
            })
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validation'):
                images, labels = images.to(config['device']), labels.to(config['device'])
                outputs = model(images)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rate'].append(current_lr)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.1f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.1f}%")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            
            # Save checkpoint
            checkpoint_path = os.path.join(config['save_dir'], 'custom_fracnet2d_best.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✅ New best model saved: {val_acc:.1f}% accuracy")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['early_stopping_patience']:
            print(f"\nEarly stopping triggered after {patience_counter} epochs without improvement")
            break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Final evaluation
    print(f"\n📊 Final Evaluation")
    print(f"Best Validation Accuracy: {best_val_acc:.1f}%")
    
    # Save final model
    final_path = os.path.join(config['save_dir'], 'custom_fracnet2d_final.pth')
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved: {final_path}")
    
    # Save training history
    history_path = os.path.join(config['save_dir'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved: {history_path}")
    
    print("\n🎯 Custom FracNet 2D training completed!")
    
    # Return the path to the best model weights
    best_weights_path = os.path.join(config['save_dir'], 'custom_fracnet2d_best.pth')
    return best_weights_path

if __name__ == "__main__":
    # Train Custom FracNet 2D
    weights_path = train_custom_fracnet2d()
    print(f"Weights saved at: {weights_path}")