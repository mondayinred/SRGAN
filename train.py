import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
import os

from config import train_config, input_config
from model import SRGAN_GEN, SRGAN_DISC
from loss import PerceptualLoss, AdversarialLoss
from utils import psnr_srgan, ssim_srgan
from data import UcsrTrainDataset, UcsrTestDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_save_path = '/home/kkm/work/graduate/SRGAN/model_parameters'
train_path = "/home/lab/Datasets/UCSR_Datasets/Train"
test_path = "/home/lab/Datasets/UCSR_Datasets/Test"

if __name__ == "__main__":
    transform_train = transforms.Compose([
        transforms.ToTensor()     
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor()      
    ])
    train_data, val_data = UcsrTrainDataset(train_path, transform_train)
    test_data = UcsrTestDataset(test_path, transform_test)
    
    # 모델 및 옵티마이저, 손실함수
    generator = SRGAN_GEN().to(device)
    discriminator = SRGAN_DISC().to(device)
    optimizer_gen = optim.Adam(generator.parameters())
    optimizer_disc = optim.Adam(discriminator.parameters())
    criterion_gen = PerceptualLoss()
    criterion_disc = AdversarialLoss()
    
    for epoch in range(1, train_config['num_epochs']):
        #### train ####
        avg_train_loss_gen = 0
        avg_train_loss_disc = 0
        avg_train_psnr = 0
        avg_train_ssim = 0
        epoch_train_loss_gen = 0
        epoch_train_loss_disc = 0
        epoch_train_psnr = 0
        epoch_train_ssim = 0
        for input_train, target_train in tqdm(train_loader):
            generator.train()
            discriminator.train()
            
            # generator 업데이트    
            generated_image = generator(input_train)
            discriminated_output = discriminator(generated_image)
        
            optimizer_gen.zero_grad()
            loss_gen = criterion_gen(generated_image, discriminated_output, target_train)
            loss_gen.backward()
            optimizer_gen.step()
            
            # generator loss, psnr, ssim 축적
            epoch_train_loss_gen += loss_gen.item()
            epoch_train_psnr += psnr_srgan(generated_image, target_train)
            epoch_train_ssim += ssim_srgan(generated_image, target_train)
            
            # discriminator 업데이트
            generated_image = generator(input_train)
            discriminated_output = discriminator(generated_image.detach())
            
            optimizer_disc.zero_grad()
            loss_disc = criterion_disc(discriminated_output)
            loss_disc.backward()
            optimizer_disc.step()
            
            # discriminator loss 축적
            epoch_train_loss_disc += loss_disc.item()
            
        avg_train_loss_gen = epoch_train_loss_gen / len(train_loader.dataset)
        avg_train_loss_disc = epoch_train_loss_disc / len(train_loader.dataset)
        avg_train_psnr = epoch_train_psnr / len(train_loader.dataset)
        avg_train_ssim = epoch_train_ssim / len(train_loader.dataset)
        print(f'Epoch {epoch} / Generator Loss: {avg_train_loss_gen}, Discriminator Loss: {avg_train_loss_disc},\
            PSNR: {avg_train_psnr}, SSIM: {avg_train_ssim}')
            
        #### eval ####
        avg_val_psnr_gen = 0
        avg_val_ssim_gen = 0
        epoch_val_psnr_gen = 0
        epoch_val_ssim_gen = 0
        for input_val, target_val in tqdm(val_loader):
            generator.eval()
            generated_image = generator(input_val)
            epoch_val_psnr_gen += psnr_srgan(input_val, target_val)
            epoch_val_ssim_gen += ssim_srgan(input_val, target_val)
        
        avg_val_psnr_gen = epoch_val_psnr_gen / len(val_loader.dataset)
        avg_val_ssim_gen = epoch_val_ssim_gen / len(val_loader.dataset)
        print(f'Validation PSNR: {avg_val_psnr_gen}, SSIM: {avg_val_ssim_gen}')
        
        if (epoch % train_config['saving_epoch_period'] == 0):
            torch.save(generator.state_dict(), os.path.join(model_save_path, f'gen_epoch{epoch}'))
            torch.save(discriminator.state_dict(), os.path.join(model_save_path, f'disc_epoch{epoch}'))
            
            
            
        
        
            
            

            
            
    
    
    
    
    
    