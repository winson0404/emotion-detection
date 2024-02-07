import torch
from torch import nn, Tensor
import torchvision

class CustomNet(torch.nn.Module):
    def __init__(self, num_classes:int, dropout:float = 0.25):
        super(CustomNet,self).__init__()
        
        self.num_classes = num_classes
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),

            nn.Conv2d(64, 128, kernel_size=5, padding="same"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),

            nn.Conv2d(128, 512, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),

            nn.Conv2d(512, 512, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout)

        )

        self.head = nn.Sequential(
            nn.Linear(512 * 3 * 3, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),

            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),

            nn.Linear(512, self.num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x:Tensor)->Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        emotion = self.head(x)
        
        return emotion