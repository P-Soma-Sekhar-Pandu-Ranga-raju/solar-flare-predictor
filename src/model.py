import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class SDOModel(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        backbone.conv1 = nn.Conv2d(
            10, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        backbone.fc = nn.Identity()

        self.encoder = backbone

        self.regressor = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x: (B,4,10,256,256)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        feats = self.encoder(x)        # (B*T,512)
        feats = feats.view(B, T, 512)  # (B,4,512)

        feats = feats.mean(dim=1)      # temporal mean
        out = self.regressor(feats)

        return out.squeeze(1)
