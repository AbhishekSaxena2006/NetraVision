import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out, inplace=True)

class NetraVisionDRNet(nn.Module):
    """
    Deep Residual Convolutional Neural Network specifically configured for
    multi-class retinal fundus Diabetic Retinopathy classification.
    """
    def __init__(self, num_classes=5, dropout_rate=0.4):
        super(NetraVisionDRNet, self).__init__()
        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),  # 256 -> 128
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 128 -> 64
        )

        # Residual stages
        self.stage1 = nn.Sequential(
            ResidualBlock(32, 64, stride=1),
            ResidualBlock(64, 64, stride=1)
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),  # 64 -> 32
            ResidualBlock(128, 128, stride=1)
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),  # 32 -> 16
            ResidualBlock(256, 256, stride=1)
        )
        self.stage4 = nn.Sequential(
            ResidualBlock(256, 512, stride=2),  # 16 -> 8
            ResidualBlock(512, 512, stride=1)
        )

        # Global average pooling and classification head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits

def get_model(num_classes=5):
    return NetraVisionDRNet(num_classes=num_classes)
