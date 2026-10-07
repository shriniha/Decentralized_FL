# models/cnn_model.py
import torch
import torch.nn as nn
from torchvision import models
import logging

class CNNModel(nn.Module):
    """
    ResNet-18 backbone (pre-trained on ImageNet).
    Final fully-connected layer replaced for binary skin-cancer classification.
    """
    def __init__(self, num_classes=2, pretrained=True, freeze_backbone=False):
        super(CNNModel, self).__init__()
        
        # Load pretrained ResNet-18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Remove the original classifier (avgpool + fc)
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])  # up to avgpool
        in_features = self.resnet.fc.in_features  # 512 for ResNet-18

        # Replace classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )

        # Optional: Freeze backbone (feature extraction mode)
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
            logging.info("ResNet backbone frozen. Only training classifier.")

    def forward(self, x):
        x = self.features(x)      # [B, 512, 1, 1]
        x = torch.flatten(x, 1)   # [B, 512]
        x = self.classifier(x)
        return x

    def get_trainable_params(self):
        """Useful for FedProx or parameter counting"""
        return list(self.classifier.parameters()) + \
               (list(self.features.parameters()) if not hasattr(self, 'freeze_backbone') else [])