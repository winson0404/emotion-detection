import torch
from torch import nn, Tensor
import torchvision
from utils.constants import MOBILENETV3_SIZE

class AlexNet(torch.nn.Module):
	def __init__(self, num_classes: int, pretrained: bool = False, freezed: bool = False, dropout: float = 0.5):
		super(AlexNet, self).__init__()
		
		self.num_classes = num_classes
		
		torchvision_model = torchvision.models.alexnet(pretrained)
		
		if freezed:
			for param in torchvision_model.parameters():
				param.requires_grad = False
				
		self.backbone = torchvision_model.features
		
		self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
		
		self.gesture_classifier = nn.Sequential(
			nn.Dropout(p=dropout),
			nn.Linear(256 * 6 * 6, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(p=dropout),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),
			nn.Linear(4096, self.num_classes),
		)
		
		self.softmax = nn.Softmax(dim=1)
		
	def forward(self, x:Tensor)->Tensor:
		x = self.backbone(x)
		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		gesture = self.gesture_classifier(x)
		
		return gesture