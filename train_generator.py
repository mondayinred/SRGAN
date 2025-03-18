import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from config import train_config
from model import SRGAN_GEN
from loss import ContentLossMSE
from utils import psnr_srgan, ssim_srgan
from data import UcsrTrainValidDataset


if __name__ == "__main__":
    # 데이터 가져오기
    train_valid = UcsrTrainValidDataset(train_config['train_data_path'])
    train_data = train_valid.get_train()
    valid_data = train_valid.get_valid()
    
    train_loader = DataLoader(train_data, batch_size=train_config['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=train_config['batch_size'],  shuffle=False)
    
    # 모델 및 옵티마이저, 손실함수
    model = SRGAN_GEN().to(train_config['device'])
    optimizer = optim.Adam(model.parameters(), lr=train_config['learning_rate'])
    criterion = ContentLossMSE().to(train_config['device'])
    num_epochs = train_config['num_epochs_srresnet']
    
    # model.load_state_dict(torch.load("/home/lab/work/SRGAN/SRResnet_parameters/srresnet_epoch653_psnr17.54370880543264.pt", map_location=torch.device('cpu')))
    # model = model.to(train_config['device'])

    
    best_val_psnr = 0
    for epoch in range(num_epochs+1):
        #### train ####
        avg_train_loss= 0
        avg_train_psnr = 0
        avg_train_ssim = 0
        epoch_train_loss = 0
        epoch_train_psnr = 0
        epoch_train_ssim = 0
        
        count = 0
        for input_train, target_train in tqdm(train_loader):
            count += 1
            # print(f'input_train shape: {input_train.shape}, target_train type: {target_train.shape}')
            model.train()
            input_train = input_train.to(train_config['device'])
            target_train = target_train.to(train_config['device'])
            
            sr_train = model(input_train)
            optimizer.zero_grad()
            loss = criterion(sr_train, target_train)
            loss.backward()
            optimizer.step()
            
            # generator loss, psnr, ssim 축적
            epoch_train_loss += loss.item()
            epoch_train_psnr += psnr_srgan(sr_train, target_train)
            epoch_train_ssim += ssim_srgan(sr_train, target_train)
            

        #print(f'Total PSNR per batch: {epoch_train_psnr}')
        avg_train_loss = epoch_train_loss / count
        avg_train_psnr = epoch_train_psnr / count
        avg_train_ssim = epoch_train_ssim / count
        print(f'Epoch {epoch} / {num_epochs+1}::: Loss: {avg_train_loss}, PSNR: {avg_train_psnr}, SSIM: {avg_train_ssim}')
            
        #### eval ####
        avg_val_psnr = 0
        avg_val_ssim = 0
        epoch_val_psnr = 0
        epoch_val_ssim = 0
        model.eval()
        count = 0
        for input_val, target_val in tqdm(valid_loader):
            count += 1
            # print(f'input_valid shape: {input_val.shape}, target_valid type: {target_val.shape}')
            input_val = input_val.to(train_config['device'])
            target_val = target_val.to(train_config['device'])
            
            sr_valid = model(input_val).to(train_config['device'])
            epoch_val_psnr += psnr_srgan(sr_valid, target_val)
            epoch_val_ssim += ssim_srgan(sr_valid, target_val)
        
        avg_val_psnr = epoch_val_psnr / count
        avg_val_ssim = epoch_val_ssim / count
        print(f'Validation PSNR: {avg_val_psnr}, SSIM: {avg_val_ssim}')
        
        if (epoch % train_config['saving_epoch_period'] == 0):
            torch.save(model.state_dict(), os.path.join(train_config['srresnet_save_path'], f'srresnet_epoch{epoch}.pt'))
            
        if(avg_val_psnr > best_val_psnr):
            best_val_psnr = avg_val_psnr
            print(f"New best valid PSNR: {best_val_psnr}")
            torch.save(model.state_dict(), os.path.join(train_config['srresnet_save_path'], f'srresnet_epoch{epoch}_psnr{best_val_psnr}.pt'))