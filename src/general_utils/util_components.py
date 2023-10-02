import torch.nn as nn


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels: # to control dimension of the first convolution
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            # without allocating any additional output
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class DownConv(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpConv(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.upsample_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)


class DoubleLinear(nn.Module):
    """(linear => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels: # to control dimension of the first linear
            mid_channels = out_channels
        self.double_linear = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.ReLU(),
            # without allocating any additional output
            nn.Linear(mid_channels, out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_linear(x)


class OutLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutLinear, self).__init__()
        self.conv = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)