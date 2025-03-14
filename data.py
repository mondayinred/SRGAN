'''
    같은 훈련사진 폴더 위치에 있는 사진들을 train, valid로 나눠야돼서 transforms.Compose로 한꺼번에 하기 어려움
    그래서 __getitem__에서 적용
'''

from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import random_split
import torch
import torch.nn.functional as F
import os
import random
from PIL import Image
import numpy as np

from config import train_config

class UcsrTrainValidDataset:
    def __init__(self, lr_images_path, hr_images_path):
        lr_images_path_list = [os.path.join(lr_images_path, f) for f in os.listdir(lr_images_path)]
        hr_images_path_list = [os.path.join(hr_images_path, f) for f in os.listdir(hr_images_path)]
        
        #### train, valid 나누기 ####
        lr_images_path_list.sort()
        hr_images_path_list.sort()
        
        paired_list = list(zip(lr_images_path_list, hr_images_path_list))
        random.shuffle(paired_list)  
        lr_images_path_list, hr_images_path_list = zip(*paired_list)
        
        split_idx = int(len(lr_images_path_list) * train_config['train_ratio'])
        
        train_lr_list = list(lr_images_path_list[:split_idx])
        train_hr_list = list(hr_images_path_list[:split_idx])
        valid_lr_list = list(lr_images_path_list[split_idx:])
        valid_hr_list = list(hr_images_path_list[split_idx:])
    
        
        self.train_data = UcsrTrainDataset(train_lr_list, train_hr_list)
        print(self.train_data.__len__())
        self.valid_data = UcsrValidDataset(valid_lr_list, valid_hr_list)
        print(self.valid_data.__len__())
    
    def get_train(self):
        return self.train_data
    
    def get_valid(self):
        return self.valid_data 
    
class UcsrTrainDataset(Dataset):
    def __init__(self, train_lr_path_list, train_hr_path_list):
        self.train_lr_path_list = train_lr_path_list
        self.train_hr_path_list = train_hr_path_list
        print(f'*****train len: {len(self.train_lr_path_list)}')
    
    def __len__(self):
        return len(self.train_lr_path_list)
    
    def __getitem__(self, idx):
        lr_image = Image.open(self.train_lr_path_list[idx]).convert("RGB")
        lr_tensor = torch.as_tensor(np.array(lr_image) / 255.0, dtype=torch.float32).permute(2, 0, 1) #### 억까 심하네.. transforms.ToTensor()대신 쓰기
        # print(f'***{lr_tensor.shape}')
        lr_tensor = lr_tensor.unsqueeze(0)
        upscaled_lr_tensor = F.interpolate(lr_tensor, scale_factor=4, mode='bilinear', align_corners=False).squeeze(0)  # 4x upsampling
        # # print(f'***{upscaled_lr_tensor.shape}')
        
        hr_image = Image.open(self.train_hr_path_list[idx]).convert("RGB")
        hr_tensor = torch.as_tensor(np.array(hr_image) / 255.0, dtype=torch.float32).permute(2, 0, 1)
        # print(f'***{hr_tensor.shape}') 

        # 동일한 랜덤 crop 좌표 얻기
        i, j, h, w = transforms.RandomCrop.get_params(upscaled_lr_tensor, output_size=train_config['crop_size'])
        
        # 같은 위치에서 crop 적용
        lr_cropped = transforms.functional.crop(upscaled_lr_tensor, i, j, h, w)
        hr_cropped = transforms.functional.crop(hr_tensor, i, j, h, w)
        
        # lr_cropped의 크기를 1/4로 줄이기
        lr_cropped = F.interpolate(lr_cropped.unsqueeze(0), scale_factor=0.25, mode='bilinear', align_corners=False).squeeze(0)
        
        # print(f'***lr_cropped: {lr_cropped.dtype}, hr_cropped: {hr_cropped.dtype}')
        
        return lr_cropped, hr_cropped
    
class UcsrValidDataset(Dataset):
    def __init__(self, valid_lr_path_list, valid_hr_path_list):
        self.valid_lr_path_list = valid_lr_path_list
        self.valid_hr_path_list = valid_hr_path_list
        print(f'*****valid len: {len(self.valid_lr_path_list)}')
    
    def __len__(self):
        return len(self.valid_lr_path_list)
    
    def __getitem__(self, idx):
        lr_image = Image.open(self.valid_lr_path_list[idx]).convert("RGB")
        lr_tensor = torch.as_tensor(np.array(lr_image) / 255.0, dtype=torch.float32).permute(2, 0, 1) #### 억까 심하네.. transforms.ToTensor()대신 쓰기
        # print(f'***{lr_tensor.shape}')
        lr_tensor = lr_tensor.unsqueeze(0)
        upscaled_lr_tensor = F.interpolate(lr_tensor, scale_factor=4, mode='bilinear', align_corners=False).squeeze(0)  # 4x upsampling
        # # print(f'***{upscaled_lr_tensor.shape}')
        
        hr_image = Image.open(self.valid_hr_path_list[idx]).convert("RGB")
        hr_tensor = torch.as_tensor(np.array(hr_image) / 255.0, dtype=torch.float32).permute(2, 0, 1)
        # print(f'***{hr_tensor.shape}') 

        # 동일한 랜덤 crop 좌표 얻기
        i, j, h, w = transforms.RandomCrop.get_params(upscaled_lr_tensor, output_size=train_config['crop_size'])
        
        # 같은 위치에서 crop 적용
        lr_cropped = transforms.functional.crop(upscaled_lr_tensor, i, j, h, w)
        hr_cropped = transforms.functional.crop(hr_tensor, i, j, h, w)
        
        # lr_cropped의 크기를 1/4로 줄이기
        lr_cropped = F.interpolate(lr_cropped.unsqueeze(0), scale_factor=0.25, mode='bilinear', align_corners=False).squeeze(0)
        
        # print(f'***lr_cropped: {type(lr_cropped)}, hr_cropped: {type(hr_cropped)}')
        # print(f'***lr_cropped: {lr_cropped.shape}, hr_cropped: {hr_cropped.shape}')
        
        return lr_cropped, hr_cropped
    