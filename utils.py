import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure


def psnr_srgan(input, target):
    assert torch.is_floating_point(input) and torch.is_floating_point(target), \
        "입력과 타겟은 부동 소수점 텐서여야 합니다."
    assert torch.max(input) <= 1.0 and torch.max(target) <= 1.0, \
        "입력과 타겟의 최대값은 1.0 이하이어야 합니다."
    mse_value = F.mse_loss(input, target)
    max_pixel_val = 1.0
    return 10.0 * torch.log10(max_pixel_val ** 2 / torch.pow(mse_value, 2))
    
def ssim_srgan(input, target):
    ssim = StructuralSimilarityIndexMeasure()
    return ssim(input, target)
    
    