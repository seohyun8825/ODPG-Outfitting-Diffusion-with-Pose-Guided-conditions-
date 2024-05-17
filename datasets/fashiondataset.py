import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

# util 모듈에서 필요한 함수들을 가져옵니다.

# 현재 파일의 위치를 기준으로 상대 경로를 설정합니다.
current_directory = Path(__file__).resolve().parent
util_path = current_directory.parent/'preprocess'/'openpose'/'annotator'/'openpose'
preprocess_path = current_directory.parent / 'preprocess' / 'humanparsing'
pose_utils_path = current_directory.parent
sys.path.append(str(util_path))
# sys.path에 경로를 추가합니다.
sys.path.append(str(preprocess_path))
sys.path.append(str(pose_utils_path))

from run_parsing import Parsing
from getmask import get_mask_location
from util import draw_bodypose
from pose_utils import cords_to_map, draw_pose_from_cords, load_pose_cords_from_strings

class FashionDataset(Dataset):
    def __init__(self, root_dir, phase='train', gt_img_size=(256, 256), pose_img_size=(256, 176)):
        self.root_dir = Path(root_dir)
        self.img_dir = self.root_dir / f"{phase}_highres"
        self.garment_dir = self.root_dir / f"{phase}_garment_highres"
        self.pairs = pd.read_csv(self.root_dir / f"fashion-resize-pairs-{phase}.csv")
        
        # CSV 파일 로드 시, 명확한 구분자와 함께 컬럼 이름을 정의합니다.
        self.annotations = pd.read_csv(self.root_dir / f"fashion-resize-annotation-{phase}.csv", delimiter=':', names=['name', 'keypoints_y', 'keypoints_x'])
        self.annotations.set_index('name', inplace=True)
        
        self.parsing_model = Parsing(gpu_id=0)  # Initialize parsing model

        self.transform = transforms.Compose([
            transforms.Resize(gt_img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]
        img_src_path = self.img_dir / row['from']
        img_tgt_path = self.img_dir / row['to']
        garment_img_path = self.garment_dir / row['garment']

        img_src = Image.open(img_src_path).convert('RGB')
        img_tgt = Image.open(img_tgt_path).convert('RGB')
        garment_img = Image.open(garment_img_path).convert('RGB')

        img_src = self.transform(img_src)
        img_tgt = self.transform(img_tgt)
        garment_img = self.transform(garment_img)

        pose_img_src, pose_map_src = self.get_pose_img(row['from'])
        pose_img_tgt, pose_map_tgt = self.get_pose_img(row['to'])

        masked_img_src, mask = self.generate_masked_image(img_src_path)

        return {
            "img_src": img_src,
            "img_tgt": img_tgt,
            "garment_img": garment_img,
            "pose_img_src": pose_img_src,
            "pose_img_tgt": pose_img_tgt,
            "pose_map_src": pose_map_src,
            "pose_map_tgt": pose_map_tgt,
            "masked_img_src": masked_img_src,
            "mask": mask
        }

    def get_pose_img(self, img_name):
        keypoints = self.annotations.loc[img_name]
        cords = load_pose_cords_from_strings(keypoints['keypoints_y'], keypoints['keypoints_x'])
        pose_img = draw_pose_from_cords(cords, (256, 176))
        pose_map = cords_to_map(cords, (256, 176))
        return torch.from_numpy(pose_img / 255.0).float(), torch.from_numpy(pose_map).float()

    def generate_masked_image(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)  # Transform the image to tensor
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension for ONNX

        # Ensure the tensor is on CPU and in the correct format
        if img_tensor.is_cuda:
            img_tensor = img_tensor.cpu()
        img_np = img_tensor.numpy()
        img_np = np.transpose(img_np, (0, 2, 3, 1))  # Convert from NCHW to NHWC

        parsed_image, face_mask = self.parsing_model(img_np)  # Call the parsing model
        mask, mask_gray = get_mask_location("hd", "upper_body", parsed_image, face_mask)  # Extract masks
        return mask_gray, mask


# 이미지 시각화 함수
import matplotlib.pyplot as plt

def show_images(images, titles=None, columns=3, figsize=(10, 10)):
    n = len(images)
    rows = (n + columns - 1) // columns
    plt.figure(figsize=figsize)
    for i, image in enumerate(images):
        ax = plt.subplot(rows, columns, i + 1)
        ax.imshow(image.permute(1, 2, 0).numpy() * 0.5 + 0.5)  # unnormalize
        ax.axis('off')
        if titles and i < len(titles):
            ax.set_title(titles[i])
    plt.tight_layout()
    plt.show()

# FashionDataset 인스턴스 생성
root_dir = 'C:/Users/user/Desktop/CFLD/CFLD/fashion'  # Windows 경로 형식으로 변경
dataset = FashionDataset(root_dir=root_dir, phase='train', gt_img_size=(256, 256), pose_img_size=(256, 176))

# 데이터셋에서 몇 개의 샘플을 가져와 시각화
num_samples_to_show = 6
imgs_to_show = []
titles = []

for i in range(num_samples_to_show):
    data = dataset[i]
    imgs_to_show.extend([
        data['img_src'], data['img_tgt'], data['garment_img'], data['masked_img_src']
    ])
    titles.extend([
        f"Source Image {i}", f"Target Image {i}", f"Garment Image {i}", f"Masked Source Image {i}"
    ])

show_images(imgs_to_show, titles=titles, columns=4, figsize=(15, 10))
