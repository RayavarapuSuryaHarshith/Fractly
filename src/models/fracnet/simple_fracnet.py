"""
SIMPLE FRACNET MODEL
===================

Simple and effective FracNet architecture for fracture detection.
This is the missing SimpleFracNet class that was causing loading issues.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class SimpleFracNet(nn.Module):
    """
    Simple FracNet model based on ResNet backbone for fracture detection.
    This matches the architecture that was saved in the weights file.
    """
    
    def __init__(self, num_classes=1):
        super(SimpleFracNet, self).__init__()
        
        # Use ResNet18 as backbone (offline version)
        self.backbone = models.resnet18(weights=None)  # Don't download pretrained weights
        
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Custom classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Feature extraction
        features = self.features(x)
        
        # Classification
        output = self.classifier(features)
        
        return output

def create_simple_fracnet(num_classes=1, pretrained=True):
    """Create SimpleFracNet model"""
    model = SimpleFracNet(num_classes=num_classes)
    return model