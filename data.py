# LR 이미지는 [0, 1]로, HR 이미지는 [-1, 1]로 정규화해야됨
# LR 이미지는 HR이미지들에서 x1/4, bicubic해서 얻어내야 함 => UCSR 쓰면 상관없으니 이걸로

from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import random_split



class UcsrTrainDataset(Dataset):
    def __init__(self, data):
        self.train_data = data
    
    def __len__(self):
        train_len = len(self.train_data)
        return train_len
    
    def __getitem__(self, index):
        return self.train_data[index]
    
class UcsrValidDataset(Dataset):
    def __init__(self, data):
        self.valid_data = data
    
    def __len__(self):
        valid_len = len(self.valid_data)
        return valid_len
    
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
    
    def get_test(self):
        return self.valid_data
        
class UcsrTestDataset(Dataset):
    def