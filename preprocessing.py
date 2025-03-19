import os
from PIL import Image

from config import train_config

# 이미지 로드
image_src_path = '/home/lab/Datasets/UCSR_Datasets/Train/HR'
image_save_path = '/home/lab/Datasets/SRGAN_data/Train_patches4/'
patch_size = train_config['crop_size']
stride = int(train_config['crop_size'][0] / 4)  # stride 값 설정

patch_idx = 1  # 패치 번호
for f in os.listdir(image_src_path):
    image_path = os.path.join(image_src_path, f)
    image = Image.open(image_path)
    image = image.convert("RGB")  # RGB 변환 (필요한 경우)

    width, height = image.size
    patch_w, patch_h = patch_size

    for y in range(0, height - patch_h + 1, stride):
        for x in range(0, width - patch_w + 1, stride):
            patch = image.crop((x, y, x + patch_w, y + patch_h))
            patch.save(os.path.join(image_save_path, f"patch_{patch_idx}.png"))
            patch_idx += 1

    print(f"총 {patch_idx}개의 패치를 저장했습니다.")