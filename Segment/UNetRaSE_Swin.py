import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from RaSE_Swin import SwinUNETREncoder
class UNetRaSE_Swin(nn.Module):
    def __init__(self, num_classes=1, pretrained=True, checkpoint_path=None):
        super(UNetRaSE_Swin, self).__init__()
        
        # Load ResNet50 backbone
        self.backbone = SwinUNETREncoder(pretrained)

        self.pool = nn.MaxPool2d(2, 2)
        self.decoder5 = self.decoder_block(768, 384, 768)

        # Decoder layers
        self.decoder4 = self.decoder_block(1536, 192, 384)
        
        self.decoder3 = self.decoder_block(768, 96, 192)

        self.decoder2 = self.decoder_block(384, 48, 96)
        
        self.decoder1 = self.decoder_block(192, 48, 96)
        
        self.final_conv = nn.Conv2d(96, num_classes, kernel_size=1)
        for name, param in self.backbone.named_parameters():
            if "input_adapter" not in name:
                param.requires_grad_(False)
    def decoder_block(self, in_channels, mid_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1,enc2,enc3,enc4 = self.backbone(x)

        dec5 = self.pool(enc4)
        dec5 = self.decoder5(dec5)

        # Decoder with skip connections
        dec4 = F.interpolate(dec5,enc4.shape[2:],mode='bilinear')
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = F.interpolate(dec4,enc3.shape[2:],mode='bilinear')
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = F.interpolate(dec3,enc2.shape[2:],mode='bilinear')
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = F.interpolate(dec2,enc1.shape[2:],mode='bilinear')
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(F.interpolate(self.final_conv(dec1), x.shape[2:], mode='bilinear'))