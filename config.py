import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import torch

train_config = {
    'num_epochs'  : 100,
    'num_epochs_srresnet' : 10000,
    'batch_size' : 16,
    'learning_rate' : 0.0001,
    'saving_epoch_period' : 50,
    'num_of_res_blocks' : 16,
    'downsampling_factor' : 4,
    'device' : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    
    # 'preprocess_hr' : transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5], std=[0.5]), # [-1, 1]로 정규화
    #     transforms.GaussianBlur(kernel_size=5), # 가우시안 필터 적용
    #     transforms.Resize((input_config['input_size'][0] / input_config['sampling_factor'],  input_config['input_size'][1] / input_config['sampling_factor'])), # sampling_factor만큼 downsampling
    # ]),
    
    'train_data_path' : "/home/lab/Datasets/UCSR_Datasets/Train/LR_HIFI/high",
    'train_ratio' : 0.8,
    'crop_size' : (96, 96),
    'srresnet_save_path' : '/home/lab/work/SRGAN/SRResnet_parameters',
    'model_save_path' : '/home/lab/work/SRGAN/model_parameters'
}