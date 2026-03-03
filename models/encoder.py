import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Create ResidualBlock class for ResNet-like architecture
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += identity
        return F.relu(out)

# 2. Define min pooling layer
class MinPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size, stride, padding)

    def forward(self, x):
        return -self.maxpool(-x)


# 3. Define MelEncoder with configurable pooling type (max or min)
class MelEncoder(nn.Module):
    def __init__(self, pool_type="max"):
        super().__init__()

        if pool_type == "max":
            Pool = nn.MaxPool2d
        elif pool_type == "min":
            Pool = MinPool2d
        else:
            raise ValueError("pool_type must be 'max' or 'min'")

        self.block1 = ResidualBlock(1, 32)
        self.pool1 = Pool(2)

        self.block2 = ResidualBlock(32, 64)
        self.pool2 = Pool(2)

        self.block3 = ResidualBlock(64, 128)
        self.pool3 = Pool(2)

        self.block4 = ResidualBlock(128, 256)
        self.pool4 = Pool(2)

    def forward(self, x):
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.pool4(self.block4(x))

        # NO adaptive pooling
        x = torch.flatten(x, 1)

        return x
