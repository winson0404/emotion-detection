import torch
from torch import nn, Tensor
import torchvision
from utils.constants import MOBILENETV3_SIZE


class MobileNetV3(torch.nn.Module):
	
	def __init__(self, num_classes: int, model_size:MOBILENETV3_SIZE = MOBILENETV3_SIZE.SMALL, pretrained: bool = False, freezed: bool = False):
		super(MobileNetV3, self).__init__()
		
		
		self.num_classes = num_classes
		
		if model_size == MOBILENETV3_SIZE.SMALL:
			torchvision_model = torchvision.models.mobilenet_v3_small(pretrained)
			in_features = 576
			out_features = 1024
		else:
			torchvision_model = torchvision.models.mobilenet_v3_large(pretrained)
			in_features = 960
			out_features = 1280

		if freezed:
			for param in torchvision_model.parameters():
				param.requires_grad = False
				
		self.backbone = nn.Sequential(torchvision_model.features, torchvision_model.avgpool)
		
		self.emotion_classifier = nn.Sequential(
			nn.Linear(in_features=in_features, out_features=out_features),
			nn.Hardswish(),
			nn.Dropout(p=0.2, inplace=True),
			nn.Linear(in_features=out_features, out_features=self.num_classes),
		)
		
	def forward(self, x: Tensor)-> Tensor:
		x = self.backbone(x)
		x = x.view(x.size(0), -1)
		emotion = self.emotion_classifier(x)
		
		return emotion
	