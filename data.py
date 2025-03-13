'''
    같은 훈련사진 폴더 위치에 있는 사진들을 train, valid로 나눠야돼서 transforms.Compose로 한꺼번에 하기 어려움
    그래서 __getitem__에서 적용
'''

from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import random_split
import os
import random
from PIL import Image

from config import train_config

class UcsrTrainValidDataset:
    def __init__(self, lr_images_path, hr_images_path, transform):
        lr_images_path_list = [os.path.join(lr_images_path, f) for f in os.listdir(lr_images_path)]
        hr_images_path_list = [os.path.join(hr_images_path, f) for f in os.listdir(hr_images_path)]
        
        #### train, valid 나누기 ####
        lr_images_path_list.sort()
        hr_images_path_list.sort()
        
        paired_list = list(zip(lr_images_path_list, hr_images_path_list))
        random.shuffle(paired_list)  
        lr_images_path_list, hr_images_path_list = zip(*paired_list)
        
        split_idx = int(len(lr_images_path_list) * (1 - train_config['train_ratio']))
        
        train_lr_list = list(lr_images_path_list[:split_idx])
        train_hr_list = list(hr_images_path_list[:split_idx])
        valid_lr_list = list(lr_images_path_list[split_idx:])
        valid_hr_list = list(hr_images_path_list[split_idx:])
        
        self.train_data = UcsrTrainDataset(train_lr_list, train_hr_list)
        self.valid_data = UcsrValidDataset(valid_lr_list, valid_hr_list)
    
    def get_train(self):
        return self.train_data
    
    def get_valid(self):
        return self.valid_data 
    
class UcsrTrainDataset(Dataset):
    def __init__(self, train_lr_path_list, train_hr_path_list):
        self.train_lr_path_list = train_lr_path_list
        self.train_hr_path_list = train_hr_path_list
    
    def __len__(self):
        return len(self.train_lr_path_list)
    
    def __getitem__(self, idx):
        lr_image = Image.open(self.train_lr_path_list[idx]).convert("RGB")
        hr_image = Image.open(self.train_hr_path_list[idx]).convert("RGB")

        # 동일한 랜덤 crop 좌표 얻기
        i, j, h, w = transforms.RandomCrop.get_params(lr_image, output_size=train_config['crop_size'])
        
        # 같은 위치에서 crop 적용
        lr_cropped = transforms.functional.crop(lr_image, i, j, h, w)
        hr_cropped = transforms.functional.crop(hr_image, i, j, h, w)
        
        lr_cropped = transforms.ToTensor()(lr_cropped)
        hr_cropped = transforms.ToTensor()(hr_cropped)
        
        return lr_cropped, hr_cropped
    
class UcsrValidDataset(Dataset):
    def __init__(self, valid_lr_path_list, valid_hr_path_list):
        self.valid_lr_path_list = valid_lr_path_list
        self.valid_hr_path_list = valid_hr_path_list
    
    def __len__(self):
        return len(self.valid_lr_path_list)
    
    def __getitem__(self, idx):
        lr_image = Image.open(self.valid_lr_path_list[idx]).convert("RGB")
        hr_image = Image.open(self.valid_hr_path_list[idx]).convert("RGB")

        # 동일한 랜덤 crop 좌표 얻기
        i, j, h, w = transforms.RandomCrop.get_params(lr_image, output_size=train_config['crop_size'])
        
        # 같은 위치에서 crop 적용
        lr_cropped = transforms.functional.crop(lr_image, i, j, h, w)
        hr_cropped = transforms.functional.crop(hr_image, i, j, h, w)
        
        lr_cropped = transforms.ToTensor()(lr_cropped)
        hr_cropped = transforms.ToTensor()(hr_cropped)
        
        return lr_cropped, hr_cropped
    