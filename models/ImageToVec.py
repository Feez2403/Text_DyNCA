import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.down5 = DoubleConv(512, 1024)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(512*2, 256, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(256*2, 128, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(128*2, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(nn.functional.max_pool2d(x1, 2))
        x3 = self.down3(nn.functional.max_pool2d(x2, 2))
        x4 = self.down4(nn.functional.max_pool2d(x3, 2))
        x5 = self.down5(nn.functional.max_pool2d(x4, 2))
        x = self.up1(x5)
        x = self.up2(torch.cat([x, x4], dim=1))
        x = self.up3(torch.cat([x, x3], dim=1))
        x = self.up4(torch.cat([x, x2], dim=1))
        return x