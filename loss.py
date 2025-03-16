import torch
from torch import nn
from torchvision.models.vgg import vgg19

from config import train_config


class ContentLossMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss().to(train_config['device'])
        
    def forward(self, generated_fake_images, target_images):
        return self.mse_loss(generated_fake_images, target_images)



class ContentLossVGG(nn.Module):
    def __init__(self, feature_map_block_idx=4): # VGG 5, 4
        super().__init__()
        
        self.pretrained_vgg19_layers = nn.ModuleList([nn.Sequential()])
        self.mse_loss = nn.MSELoss().to(train_config['device'])
        
        # vgg19의 사전학습된 레이어들 가져오기
        vgg = vgg19(pretrained=True)
        
        
        # 논문대로 VGG54를 위해 4번째 feature map만 가져오기 위해 4번째 컨볼루션 블럭까지의 레이어들만 가져오기. VGG19 모델 참고
        layers = vgg.features
        conv_block_idx = 1
        
        for idx, layer in enumerate(layers):
            # feature_map_idx를 찾는 로직
            if isinstance(layer, nn.MaxPool2d):
                if conv_block_idx == feature_map_block_idx:
                    break
                conv_block_idx += 1
                self.pretrained_vgg19_layers.append(nn.Sequential())
            
            # 사전학습 레이어들 넣기
            self.pretrained_vgg19_layers[-1].add_module(str(idx), layer)
            
        for param in self.pretrained_vgg19_layers.parameters():
            param.requires_grad = False
            
        self.pretrained_vgg19_layers = self.pretrained_vgg19_layers.to(train_config['device'])
        # print(f'******pretrained_vgg19_layers len: {len(self.pretrained_vgg19_layers)}')
        # print(f'******pretrained_vgg19_layers 4th layer len: {len(self.pretrained_vgg19_layers[3])}')
        # for layer in self.pretrained_vgg19_layers[3]:
        #     print(f'*****layer: {layer}')
    
    def forward(self, generated_fake_images, target_images):
        for layer in self.pretrained_vgg19_layers:
            generated_fake_images = layer(generated_fake_images).to(train_config['device'])
            target_images = layer(target_images).to(train_config['device'])
        
        return self.mse_loss(generated_fake_images / 12.75, target_images / 12.75)
   
   
    
class AdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, generated_and_discriminated_images):
        # 모든 training samples들에 대해 더하여 (-) 붙임
        print(f'****dimension: {generated_and_discriminated_images.shape}')
        return (-1.) * torch.sum(torch.log(generated_and_discriminated_images), dim=0) # dim=0은 batch_size 차원이어야 함
    
    
    
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.content_loss = ContentLossVGG(4).to(train_config['device']) # ContentLossVGG 또는 ContentLossMSE 선택. 4는 "VGG19 4번째 컨볼루션 블럭 출력, 즉 5번째 풀링레이어 이전의 출력"
        self.adversarial_loss = AdversarialLoss().to(train_config['device'])
        
    def forward(self, generated_fake_images, generated_and_discriminated_images, target_images):
        '''
            upscaled_fake_images: G(I_LR)
            upscaled_and_discriminated_images: D(G(I_LR))
            target_images: I_HR
        '''
        return self.content_loss(generated_fake_images, target_images) + 0.001 * self.adversarial_loss(generated_and_discriminated_images)
    
    
class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCELoss()
        
    def forward(self, discriminated_fake, discriminated_real):
        real_loss = self.bce_loss(discriminated_real, torch.ones_like(discriminated_real))  # 실제 이미지에 대한 손실
        fake_loss = self.bce_loss(discriminated_fake, torch.zeros_like(discriminated_fake))  # 가짜 이미지에 대한 손실
        return real_loss + fake_loss