import torch
from torch import nn, Tensor
import torchvision


class EfficientNet(torch.nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False, freezed: bool = False, dropout: float = 0.5):
        super(EfficientNet, self).__init__()
    
        self.num_classes = num_classes

        torchvision_model = torchvision.models.efficientnet_b0(pretrained=pretrained)

        if freezed:
            for param in torchvision_model.parameters():
                param.requires_grad = False
                        
        self.backbone = torchvision_model.features
                
        self.avgpool = nn.AdaptiveAvgPool2d(1)
            
        self.emotion_classifier = torchvision_model.classifier
        self.emotion_classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
                
                
    def forward(self, x:Tensor)->Tensor:
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        emotion = self.emotion_classifier(x)
            
        return emotion