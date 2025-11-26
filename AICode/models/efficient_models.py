"""
Efficient Pre-trained Models for Continual Learning
ResNet, EfficientNet, MobileNet adapted for Fashion-MNIST
"""
import torch
import torch.nn as nn
from torchvision import models


class ResNetContinual(nn.Module):
    """
    ResNet for continual learning
    Uses pre-trained ResNet18/34/50 with custom head
    """
    def __init__(self, num_classes=10, arch='resnet18', pretrained=True):
        super().__init__()
        
        # Load pre-trained ResNet
        if arch == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            hidden_dim = 512
        elif arch == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            hidden_dim = 512
        elif arch == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            hidden_dim = 2048
        else:
            raise ValueError(f"Unknown architecture: {arch}")
        
        # Adapt first conv layer for grayscale input
        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            1,  # Grayscale input
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        
        # Initialize with averaged RGB weights
        with torch.no_grad():
            self.backbone.conv1.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
        
        # Remove original classifier
        self.backbone.fc = nn.Identity()
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
    
    def forward(self, x):
        """Forward pass"""
        # Upsample if needed (ResNet expects larger images)
        if x.shape[-1] < 32:
            x = nn.functional.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
        
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x):
        """Extract features"""
        if x.shape[-1] < 32:
            x = nn.functional.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
        return self.backbone(x)


class MobileNetContinual(nn.Module):
    """
    MobileNetV2 for continual learning
    Lightweight and efficient for edge deployment
    """
    def __init__(self, num_classes=10, pretrained=True, width_mult=1.0):
        super().__init__()
        
        # Load pre-trained MobileNetV2
        self.backbone = models.mobilenet_v2(pretrained=pretrained, width_mult=width_mult)
        
        # Adapt first conv for grayscale
        original_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            1,  # Grayscale
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        
        # Initialize with averaged RGB weights
        with torch.no_grad():
            self.backbone.features[0][0].weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
        
        # Get hidden dimension
        hidden_dim = self.backbone.classifier[1].in_features
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Custom lightweight classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
    
    def forward(self, x):
        """Forward pass"""
        # MobileNet can handle smaller images better
        if x.shape[-1] < 28:
            x = nn.functional.interpolate(x, size=(28, 28), mode='bilinear', align_corners=False)
        elif x.shape[-1] == 28:
            # Keep 28x28 for efficiency
            pass
        
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


class EfficientNetContinual(nn.Module):
    """
    EfficientNet for continual learning
    State-of-the-art efficiency and accuracy
    """
    def __init__(self, num_classes=10, arch='efficientnet_b0', pretrained=True):
        super().__init__()
        
        # Load pre-trained EfficientNet
        if arch == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
        elif arch == 'efficientnet_b1':
            self.backbone = models.efficientnet_b1(pretrained=pretrained)
        elif arch == 'efficientnet_b2':
            self.backbone = models.efficientnet_b2(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown architecture: {arch}")
        
        # Adapt first conv for grayscale
        original_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            1,  # Grayscale
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        
        # Initialize with averaged RGB weights
        with torch.no_grad():
            self.backbone.features[0][0].weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
        
        # Get hidden dimension
        hidden_dim = self.backbone.classifier[1].in_features
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
    
    def forward(self, x):
        """Forward pass"""
        # EfficientNet expects larger images
        if x.shape[-1] < 32:
            x = nn.functional.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
        
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


def create_model(model_type='resnet18', num_classes=10, pretrained=True):
    """
    Factory function to create efficient models
    
    Args:
        model_type: 'resnet18', 'resnet34', 'resnet50', 'mobilenet', 'efficientnet_b0', etc.
        num_classes: Number of output classes
        pretrained: Use ImageNet pre-trained weights
    
    Returns:
        model: PyTorch model
    """
    if model_type.startswith('resnet'):
        model = ResNetContinual(num_classes, arch=model_type, pretrained=pretrained)
    elif model_type == 'mobilenet':
        model = MobileNetContinual(num_classes, pretrained=pretrained)
    elif model_type.startswith('efficientnet'):
        model = EfficientNetContinual(num_classes, arch=model_type, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model
