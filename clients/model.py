# common/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, input_channels=1, n_classes=1):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.contract_block(input_channels, 32)
        self.enc2 = self.contract_block(32, 64)
        self.enc3 = self.contract_block(64, 128)

        # Bottleneck
        self.bottleneck = self.double_conv(128, 256)

        # Decoder
        self.upconv3 = self.up_block(256, 128)
        self.upconv2 = self.up_block(128, 64)
        self.upconv1 = self.up_block(64, 32)

        # Output Layer
        self.out_layer = nn.Conv2d(32, n_classes, kernel_size=1)

        # Initialize weights
        self.apply(self._init_weights)

    def double_conv(self, in_channels, out_channels):
        """Double convolution layers"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)  # Prevent overfitting
        )

    def contract_block(self, in_channels, out_channels):
        """Downsampling Path with Pooling"""
        return nn.Sequential(
            self.double_conv(in_channels, out_channels),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def up_block(self, in_channels, out_channels):
        """Upsampling Path with Skip Connection"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            self.double_conv(out_channels * 2, out_channels)  # Ensure correct channels after concatenation
        )

    def forward(self, x):
    # Encoder
        enc1 = self.enc1(x)  # (batch, 32, H, W)
        enc2 = self.enc2(enc1)  # (batch, 64, H/2, W/2)
        enc3 = self.enc3(enc2)  # (batch, 128, H/4, W/4)

        # Bottleneck
        bottleneck = self.bottleneck(enc3)  # (batch, 256, H/8, W/8)

        # Decoder with Corrected Skip Connections
        dec3 = self.upconv3[0](bottleneck)  # (batch, 128, H/4, W/4)
        dec3 = F.interpolate(dec3, size=enc3.shape[2:], mode="bilinear", align_corners=False)  
        dec3 = torch.cat((dec3, enc3), dim=1)  
        dec3 = self.upconv3[1](dec3)  

        dec2 = self.upconv2[0](dec3)  # (batch, 64, H/2, W/2)
        dec2 = F.interpolate(dec2, size=enc2.shape[2:], mode="bilinear", align_corners=False)  
        dec2 = torch.cat((dec2, enc2), dim=1)  
        dec2 = self.upconv2[1](dec2)  

        dec1 = self.upconv1[0](dec2)  # (batch, 32, H, W)
        dec1 = F.interpolate(dec1, size=enc1.shape[2:], mode="bilinear", align_corners=False)  
        dec1 = torch.cat((dec1, enc1), dim=1)  
        dec1 = self.upconv1[1](dec1)  

        # ✅ Output layer with size matching the input
        output = self.out_layer(dec1)
        output = F.interpolate(output, size=x.shape[2:], mode="bilinear", align_corners=False)  #  Fix size mismatch

        return output

    @staticmethod
    def _init_weights(m):
        """Improved Weight Initialization"""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)