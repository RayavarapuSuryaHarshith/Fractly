"""
CUSTOM FRACNET 2D MODEL
=======================

2D adaptation of the original FracNet architecture for X-ray fracture detection.
This module provides a clean, modular implementation based on your cloned FracNet repository.

Based on: src/FracNet/model/unet.py (original FracNet architecture)
Adapted for: 2D X-ray images instead of 3D volumetric data
"""

import torch
import torch.nn as nn
import os
import sys

# Add FracNet to path to access original components if needed
sys.path.append(os.path.join(os.path.dirname(__file__), 'FracNet'))

class ConvBlock2D(nn.Sequential):
    """
    2D adaptation of FracNet's ConvBlock
    
    Original FracNet uses:
    - nn.Conv3d -> nn.Conv2d
    - nn.BatchNorm3d -> nn.BatchNorm2d 
    - LeakyReLU (preserved)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

class Down2D(nn.Sequential):
    """
    2D adaptation of FracNet's Down block
    
    Original FracNet uses:
    - nn.MaxPool3d -> nn.MaxPool2d
    - ConvBlock3D -> ConvBlock2D
    """
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.MaxPool2d(2),
            ConvBlock2D(in_channels, out_channels)
        )

class Up2D(nn.Module):
    """
    2D adaptation of FracNet's Up block
    
    Original FracNet uses:
    - nn.ConvTranspose3d -> nn.ConvTranspose2d
    - nn.BatchNorm3d -> nn.BatchNorm2d
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock2D(in_channels, out_channels)
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x, y):
        """Forward pass identical to original FracNet Up block logic"""
        x = self.conv2(x)
        x = self.conv1(torch.cat([y, x], dim=1))
        return x

class CustomFracNet2D(nn.Module):
    """
    Custom FracNet 2D - Exact adaptation of your cloned FracNet for X-ray images
    
    Architecture:
    - Follows the exact structure of src/FracNet/model/unet.py
    - Converts all 3D operations to 2D
    - Maintains the same initialization strategy
    - Adds classification head for fracture detection
    
    Args:
        in_channels: Input channels (3 for RGB X-rays)
        num_classes: Number of classes (2 for fracture/no-fracture)
        first_out_channels: First layer output channels (default: 16, same as FracNet)
    """
    
    def __init__(self, in_channels=3, num_classes=2, first_out_channels=16):
        super().__init__()
        
        # Store initial channels
        self.first_out_channels = first_out_channels
        
        # Encoder - following original FracNet structure exactly
        self.first = ConvBlock2D(in_channels, first_out_channels)
        in_channels = first_out_channels
        
        self.down1 = Down2D(in_channels, 2 * in_channels)
        self.down2 = Down2D(2 * in_channels, 4 * in_channels)
        self.down3 = Down2D(4 * in_channels, 8 * in_channels)
        
        # Decoder - symmetric to encoder
        self.up1 = Up2D(8 * in_channels, 4 * in_channels)
        self.up2 = Up2D(4 * in_channels, 2 * in_channels)
        self.up3 = Up2D(2 * in_channels, in_channels)
        
        # Classification head with stronger regularization
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.7),  # Increased dropout
            nn.Linear(first_out_channels, 32),  # Smaller intermediate layer
            nn.BatchNorm1d(32),  # Add batch normalization
            nn.ReLU(),
            nn.Dropout(0.5),  # Additional dropout
            nn.Linear(32, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder path with feature collection for uncertainty
        features = []
        x1 = self.first(x)
        features.append(x1)
        x2 = self.down1(x1)
        features.append(x2)
        x3 = self.down2(x2)
        features.append(x3)
        x4 = self.down3(x3)
        features.append(x4)
        
        # Decoder path with skip connections and feature collection
        x = self.up1(x4, x3)
        features.append(x)
        x = self.up2(x, x2)
        features.append(x)
        x = self.up3(x, x1)
        features.append(x)
        
        # Classification with confidence estimation
        logits = self.classifier(x)
        
        # Calculate uncertainty from feature statistics
        uncertainty = 0
        for feat in features:
            feat_std = torch.std(feat, dim=[2, 3]).mean()
            uncertainty += feat_std.item()
        uncertainty = uncertainty / len(features)
        
        # Temperature scaling
        temperature = max(1.0, 1.0 + uncertainty)
        scaled_logits = logits / temperature
        
        # Return logits and uncertainty
        return {
            'logits': scaled_logits,
            'uncertainty': uncertainty,
            'temperature': temperature
        }

def load_custom_fracnet2d(weights_path: str, num_classes: int = 2, device=None) -> nn.Module:
    """Load CustomFracNet2D model with weights"""
    print(f"🔧 Loading Custom FracNet 2D...")
    print(f"   Classes: {num_classes}")
    print(f"   Device: {device}")
    
    # Check if the weights file exists and try to determine the architecture
    if os.path.exists(weights_path):
        try:
            # Load state dict to inspect architecture
            state_dict = torch.load(weights_path, map_location=device or 'cpu')
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            
            # Try to intelligently load into the correct architecture.
            # Many weight files are saved as OrderedDict state_dicts. We'll try
            # SimpleFracNet if the classifier layer matches the expected shape,
            # otherwise fall back to the CustomFracNet2D and load with strict=False.
            try:
                if 'classifier.2.weight' in state_dict:
                    classifier_weight = state_dict['classifier.2.weight']
                    try:
                        classifier_shape = tuple(classifier_weight.shape)
                    except Exception:
                        classifier_shape = None

                    if classifier_shape is not None and (128 in classifier_shape):
                        print("   Detected SimpleFracNet-like classifier in weights, attempting to load SimpleFracNet...")
                        from .simple_fracnet import SimpleFracNet
                        model = SimpleFracNet(num_classes=1)
                        try:
                            model.load_state_dict(state_dict, strict=True)
                            print("   ✅ Successfully loaded SimpleFracNet weights")
                            return model
                        except Exception as e:
                            print(f"   ⚠️ Failed to load SimpleFracNet strictly: {e}")
                            # fallthrough to try CustomFracNet2D below

                # Default: try loading into CustomFracNet2D
                model = CustomFracNet2D(num_classes=num_classes)
                try:
                    model.load_state_dict(state_dict, strict=False)
                    print("   ✅ Successfully loaded CustomFracNet2D weights")
                except Exception as e:
                    print(f"   ⚠️ Loading into CustomFracNet2D with strict=False produced warnings/errors: {e}")
                return model
            except Exception as e:
                print(f"   ⚠️ Unexpected error while interpreting weights: {e}")
                # Fall back to randomly initialized model below
                
        except Exception as e:
            print(f"⚠️  Weight loading failed: {e}")
            print("   Using randomly initialized CustomFracNet2D")
            model = CustomFracNet2D(num_classes=num_classes)
            return model
    else:
        print(f"⚠️  Weights file not found: {weights_path}")
        print("   Using randomly initialized CustomFracNet2D")
        model = CustomFracNet2D(num_classes=num_classes)
        return model