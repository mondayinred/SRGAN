import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import torch

input_config = {
    'input_size' : (),
    'sampling_factor' : 4
}

train_config = {
    'num_epochs'  : 100,
    'batch_size' : 16,
    'learning_rate' : 0.0001,
    'saving_epoch_period' : 5,
    'device' : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    
    # 'preprocess_hr' : transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5], std=[0.5]), # [-1, 1]로 정규화
    #     transforms.GaussianBlur(kernel_size=5), # 가우시안 필터 적용
    #     transforms.Resize((input_config['input_size'][0] / input_config['sampling_factor'],  input_config['input_size'][1] / input_config['sampling_factor'])), # sampling_factor만큼 downsampling
    # ]),
    
    'train_lr_path' : "/home/lab/Datasets/UCSR_Datasets/Test/BSD100/LR_x4",
    'train_hr_path' : "/home/lab/Datasets/UCSR_Datasets/Test/BSD100/HR",
    'train_ratio' : 0.8,
    'crop_size' : (96, 96),
    'model_save_path' : '/home/kkm/work/graduate/SRGAN/model_parameters'
}