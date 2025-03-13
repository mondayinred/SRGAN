import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

from config import train_config

def psnr_srgan(input, target):
    input = input.cpu().detach()
    target = target.cpu().detach()
    print("Max value in input:", torch.max(input))
    print("Max value in target:", torch.max(target))
    assert torch.is_floating_point(input) and torch.is_floating_point(target), \
        "입력과 타겟은 부동 소수점 텐서여야 합니다."
    assert torch.max(input) <= 1.0 and torch.max(target) <= 1.0, \
        "입력과 타겟의 최대값은 1.0 이하이어야 합니다."
    mse_value = F.mse_loss(input, target)
    max_pixel_val = 1.0
    if mse_value <= 0:
        return 100
    return 10.0 * torch.log10(max_pixel_val ** 2 / mse_value)
    
def ssim_srgan(input, target):
    input = input.cpu().detach()
    target = target.cpu().detach()
    ssim = StructuralSimilarityIndexMeasure()
    return ssim(input, target)
    
    