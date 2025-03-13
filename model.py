import torch
import torch.nn as nn
import torch.nn.functional as F

from config import train_config

class ResidualBlock(nn.Module):
    def __init__(self, conv_in_channels, conv_out_channels, conv_kernel_size, conv_stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=conv_in_channels, out_channels=conv_out_channels, kernel_size=conv_kernel_size, stride=conv_stride, padding=1)
        self.conv2 = nn.Conv2d(in_channels=conv_out_channels, out_channels=conv_out_channels, kernel_size=conv_kernel_size, stride=conv_stride, padding=1)
        self.batch_norm = nn.BatchNorm2d(num_features=conv_out_channels)
        self.prelu = nn.PReLU()
        
    def forward(self, x0):
        x1 = self.conv1(x0)
        x1 = self.batch_norm(x1)
        x1 = self.prelu(x1)
        x1 = self.conv2(x1)
        x1 = self.batch_norm(x1)
        # print(f'***x1: {x1.shape}, x0:{x0.shape}')
        x1 += x0 # skip connection
        return x1
        

class SRGAN_GEN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4)
        self.prelu1 = nn.PReLU()
        
        self.num_of_resblocks = train_config['num_of_res_blocks'] # 논문대로
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(64, 64, 3, 1) for _ in range(self.num_of_resblocks)
        ])
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.batch_norm = nn.BatchNorm2d(num_features=64) # batch norm한 다음에 skip-connection해야됨  
        
        self.upsample_block1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU()
        )
        
        self.upsample_block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU()
        )
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4)
        
        
    def forward(self, x0):
        # print(f'******x0: {x0.shape}')
        x0 = self.conv1(x0)
        x0 = self.prelu1(x0)
        
        for layer in self.residual_blocks:
            x1 = layer(x0) 
        
        x1 = self.conv2(x1)
        x1 = self.batch_norm(x1)
        x1 += x0
        # print(f'******x1: {x1.shape}')
        
        x1 = self.upsample_block1(x1)
        x1 = self.upsample_block2(x1)
        
        # print(f'******x1: {x1.shape}')
        x1 = self.conv3(x1)
        
        return x1

class SRGAN_DISC(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.2)
        
        # state size. (64) x 48 x 48
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # state size. (128) x 24 x 24
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # state size. (256) x 12 x 12
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.conv_block6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
         # state size. (512) x 6 x 6
        self.conv_block7 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.flat = nn.Flatten()
        self.ffn = nn.Sequential(
            nn.Linear(in_features = 512 * 6 * 6, out_features=1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=1024, out_features=1),
            nn.Sigmoid()
        )
        
    def forward(self, x0):
        # print(f'****x0: {x0.shape}')
        x1 = self.conv1(x0)
        x1 = self.lrelu1(x1)
        # print(f'****x1: {x1.shape}')
        x1 = self.conv_block1(x1)
        x1 = self.conv_block2(x1)
        # print(f'****x1: {x1.shape}')
        x1 = self.conv_block3(x1)
        x1 = self.conv_block4(x1)
        # print(f'****x1: {x1.shape}')
        x1 = self.conv_block5(x1)
        x1 = self.conv_block6(x1)
        # print(f'****x1: {x1.shape}')
        x1 = self.conv_block7(x1)
        # print(f'****x1: {x1.shape}')
        x1 = self.flat(x1)
        x1 = self.ffn(x1)
        
        return x1
        
        
        
        