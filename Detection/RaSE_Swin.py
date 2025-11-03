from monai.networks.nets import UNet
import torch
import torch.nn as nn
from typing import List, Tuple
from monai.networks.nets import SwinUNETR
class SwinUNETREncoder(torch.nn.Module):
    def __init__(self, pre_trained=False):
        super().__init__()
        # 创建原始UNet（保持in_channels=1）
       
        self.swinunetr_model = SwinUNETR(
            in_channels=1,
            out_channels=32,
            feature_size=48,
            drop_rate=0.0,
            spatial_dims=2,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=False,
            use_v2=True
        )
        if pre_trained:
            self.swinunetr_model.load_state_dict(
                torch.load('/home/jiayi/FM_downstream/Sotas/RaSE_2D_Swin.pth'))
        self.swinViT = self.swinunetr_model.swinViT

        
        # 添加3通道转1通道的适配层
        self.input_adapter = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=3,padding=1),
        )
    
    
    def forward(self, x):
        x = self.input_adapter(x)

        x = self.swinViT.patch_embed(x)
        x = self.swinViT.pos_drop(x)
        
        features = []
        
        # 2. 逐层处理
        for layer in [self.swinViT.layers1, self.swinViT.layers2, 
                     self.swinViT.layers3, self.swinViT.layers4]:
            # 正确调用BasicLayer的方式
            for basic_layer in layer:  # 遍历ModuleList中的每个BasicLayer
                x = basic_layer(x)
            features.append(x)
            
        return features