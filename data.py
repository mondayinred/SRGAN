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
    def __init__(self, train_path, valid_path):
        train__images_path_list = [os.path.join(train_path, f) for f in os.listdir(train_path)]
        valid_images_path_list = [os.path.join(valid_path, f) for f in os.listdir(valid_path)]
        
        self.train_data = UcsrTrainDataset(train__images_path_list)
        self.valid_data = UcsrValidDataset(valid_images_path_list)
        print(f'train length: {self.train_data.__len__()}, valid length:{self.valid_data.__len__()}')
    
    def get_train(self):
        return self.train_data
    
    def get_valid(self):
        return self.valid_data 
    
class UcsrTrainDataset(Dataset):
    def __init__(self, train_path_list):
        self.train_path_list = train_path_list
    
    def __len__(self):
        return len(self.train_path_list)
    
    def __getitem__(self, idx):
        transform_to_lr = transforms.Compose([
            transforms.Resize(
                (train_config['crop_size'][0] // train_config['downsampling_factor'], 
                train_config['crop_size'][1] // train_config['downsampling_factor']), 
                interpolation=transforms.InterpolationMode.BICUBIC
                )  
        ])
        hr_image = Image.open(self.train_path_list[idx]).convert("RGB")
        lr_image = transform_to_lr(hr_image)
        hr_tensor = torch.as_tensor(np.array(hr_image) / 255.0, dtype=torch.float32).permute(2, 0, 1) #### 억까 심하네.. transforms.ToTensor()대신 쓰기
        lr_tensor = torch.as_tensor(np.array(lr_image) / 255.0, dtype=torch.float32).permute(2, 0, 1) 
        # print(f'***{lr_tensor.shape, hr_tensor.shape}')
        
        return lr_tensor, hr_tensor
    
class UcsrValidDataset(Dataset):
    def __init__(self, valid_path_list):
        self.valid_path_list = valid_path_list
    
    def __len__(self):
        return len(self.valid_path_list)
    
    def __getitem__(self, idx):
        hr_image = Image.open(self.valid_path_list[idx]).convert("RGB")
        transform_to_lr = transforms.Compose([
            transforms.Resize((
                hr_image.size[1] // train_config['downsampling_factor'], 
                hr_image.size[0] // train_config['downsampling_factor']), 
                interpolation=transforms.InterpolationMode.BICUBIC
                )  
        ])
        lr_image = transform_to_lr(hr_image)
        hr_tensor = torch.as_tensor(np.array(hr_image) / 255.0, dtype=torch.float32).permute(2, 0, 1) #### 억까 심하네.. transforms.ToTensor()대신 쓰기
        lr_tensor = torch.as_tensor(np.array(lr_image) / 255.0, dtype=torch.float32).permute(2, 0, 1) 
        # print(f'***{lr_tensor.shape, hr_tensor.shape}')
        
        return lr_tensor, hr_tensor
    