# LR 이미지는 [0, 1]로, HR 이미지는 [-1, 1]로 정규화해야됨
# LR 이미지는 HR이미지들에서 x1/4, bicubic해서 얻어내야 함 => UCSR 쓰면 상관없으니 이걸로

from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import random_split
import os


class UcsrTrainDataset(Dataset):
    def __init__(self, datapath, transform):
        lr_list = [os.path.join(datapath, f) for f in os.listdir(datapath)]
        for lr in 
    
    def __len__(self):
        return len(self.train_data)
    
    def __getitem__(self, index):
        return self.train_data[index] # 반환형은 torch.tensor
    
class UcsrValidDataset(Dataset):
    def __init__(self, data):
        self.valid_data = data
    
    def __len__(self):
        return len(self.valid_data)
    
    def __getitem__(self, index):
        return self.valid_data[index]
    
class UcsrTrainValidDataset:
    def __init__(self, datapath, transform):
        full_dataset = datasets.ImageFolder(root=datapath, transform=transform)
        train_size = int(0.8 * len(full_dataset))
        valid_size = len(full_dataset) - train_size
        train_data, valid_data = random_split(full_dataset, [train_size, valid_size])
        self.train_data = UcsrTrainDataset(train_data)
        self.valid_data = UcsrValidDataset(valid_data)
        
    def get_train(self):
        return self.train_data
    
    def get_valid(self):
        return self.valid_data
        
class UcsrTestDataset(Dataset):
    def __init__(self, datapath, transform):
        self.test_data = datasets.ImageFolder(root=datapath, transform=transform)
        
    def __len__(self):
        return len(self.test_data)
    
    def __getitem__(self, index):
        return self.test_data[index]