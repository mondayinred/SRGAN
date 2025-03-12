import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

input_config = {
    'input_size' : (),
    'sampling_factor' : 4
}

train_config = {
    'num_epochs'  : 100,
    'batch_size' : 16,
    'learning_rate' : 0.0001,
    'saving_epoch_period' : 5,
    
    'preprocess_hr' : transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means=[0.5, 0.5], std=[0.5]), # [-1, 1]로 정규화
        transforms.GaussianBlur(kernel_size=5), # 가우시안 필터 적용
        transforms.Resize((input_config['input_size'][0] / input_config['sampling_factor'],  input_config['input_size'][1] / input_config['sampling_factor'])), # sampling_factor만큼 downsampling
    ]),
    
    'preprocess_lr' : transforms.Compose([
        transforms.ToTensor()
    ]),
    
    
        
}