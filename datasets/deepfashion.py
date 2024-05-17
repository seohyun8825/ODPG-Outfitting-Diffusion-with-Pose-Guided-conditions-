"""
@author: Yanzuo Lu
@author: oliveryanzuolu@gmail.com
"""

import glob
import logging
import math
import os
import random
import sys
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
sys.path.append('C:\\Users\\user\\Desktop\\CFLD\\CFLD')
from pose_utils import transform_keypoints, draw_pose_from_cords, load_pose_cords_from_strings
logger = logging.getLogger()
import cv2

class PisTrainDeepFashion(Dataset):
    def __init__(self, root_dir, gt_img_size, pose_img_size, cond_img_size, min_scale,
                 log_aspect_ratio, pred_ratio, pred_ratio_var, psz):
        super().__init__()
        self.pose_img_size = pose_img_size
        self.cond_img_size = cond_img_size
        self.log_aspect_ratio = log_aspect_ratio
        self.pred_ratio = pred_ratio
        self.pred_ratio_var = pred_ratio_var
        self.psz = psz

        train_dir = os.path.join(root_dir, "train_highres")
        train_pairs_path = os.path.join(root_dir, "fashion-resize-pairs-train.csv")
        print(f"Root Directory: {root_dir}")
        print(f"Current Working Directory: {os.getcwd()}")

        if not os.path.exists(train_pairs_path):
            print(f"File does not exist: {train_pairs_path}")
        else:
            print(f"File exists: {train_pairs_path}")
            train_pairs = pd.read_csv(train_pairs_path)
        print(f"Checking file path: {train_pairs_path}")  
        train_pairs = pd.read_csv(train_pairs_path)

        self.img_items = self.process_dir(root_dir, train_pairs)
        self.annotation_file = pd.read_csv(os.path.join(root_dir, "fashion-resize-annotation-train.csv"), sep=':')
        self.annotation_file = self.annotation_file.set_index('name')

        self.transform_gt = transforms.Compose([
            transforms.Resize(gt_img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.transform_cond = transforms.Compose([
            transforms.Resize(cond_img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        aspect_ratio = cond_img_size[1] / cond_img_size[0]
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(cond_img_size, scale=(min_scale, 1.), ratio=(aspect_ratio * 3./4., aspect_ratio * 4./3.),
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]) if min_scale < 1.0 else transforms.Compose([
            transforms.Resize(cond_img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def process_dir(self, root_dir, csv_file):
        data = []
        for i in range(len(csv_file)):
            data.append((
                os.path.join(root_dir, "train_highres", csv_file.iloc[i]["from"]),
                os.path.join(root_dir, "train_highres", csv_file.iloc[i]["to"]),
                os.path.join(root_dir, "train_garment_highres", csv_file.iloc[i]["garment"])
            ))
        return data

    def get_pred_ratio(self):
        pred_ratio = []
        for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
            assert prm >= prv
            pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
            pred_ratio.append(pr)
        pred_ratio = random.choice(pred_ratio)
        return pred_ratio

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_path_from, img_path_to, img_path_garment = self.img_items[index]
        with open(img_path_from, 'rb') as f:
            img_from = Image.open(f).convert('RGB')
        with open(img_path_to, 'rb') as f:
            img_to = Image.open(f).convert('RGB')
        with open(img_path_garment, 'rb') as f:
            img_garment = Image.open(f).convert('RGB')

        original_size_from = img_from.size
        original_size_to = img_to.size

        img_src = self.transform_gt(img_from)
        img_tgt = self.transform_gt(img_to)
        img_cond = self.transform(img_from)
        img_garment = self.transform_gt(img_garment)

        pose_img_src = self.build_pose_img(img_path_from, original_size_from, self.transform_gt)
        pose_img_tgt = self.build_pose_img(img_path_to, original_size_to, self.transform_gt)

        mask = None
        if len(self.pred_ratio) > 0:
            H, W = self.cond_img_size[0] // self.psz, self.cond_img_size[1] // self.psz
            high = self.get_pred_ratio() * H * W

            mask = np.zeros((H, W), dtype=bool)
            mask_count = 0
            while mask_count < high:
                max_mask_patches = high - mask_count

                delta = 0
                for attempt in range(10):
                    low = (min(H, W) // 3) ** 2
                    target_area = random.uniform(low, max_mask_patches)
                    aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                    h = int(round(math.sqrt(target_area * aspect_ratio)))
                    w = int(round(math.sqrt(target_area / aspect_ratio)))
                    if w < W and h < H:
                        top = random.randint(0, H - h)
                        left = random.randint(0, W - w)

                        num_masked = mask[top: top + h, left: left + w].sum()
                        if 0 < h * w - num_masked <= max_mask_patches:
                            for i in range(top, top + h):
                                for j in range(left, left + w):
                                    if mask[i, j] == 0:
                                        mask[i, j] = 1
                                        delta += 1

                    if delta > 0:
                        break

                if delta == 0:
                    break
                else:
                    mask_count += delta

        return_dict = {
            "img_src": img_src,
            "img_tgt": img_tgt,
            "img_cond": img_cond,
            "pose_img_src": pose_img_src,
            "pose_img_tgt": pose_img_tgt,
            "img_garment": img_garment
        }
        return return_dict
    def build_pose_img(self, img_path, original_size, transform):
        img = Image.open(img_path).convert('RGB')
        transformed_img = transform(img)
        transformed_size = (transformed_img.shape[2], transformed_img.shape[1])  

        string = self.annotation_file.loc[os.path.basename(img_path)]
        keypoints = load_pose_cords_from_strings(string['keypoints_y'], string['keypoints_x'])

        if keypoints.size == 0:
            print("Warning: No valid keypoints for", img_path)
            return torch.zeros(3, transformed_size[1], transformed_size[0], dtype=torch.float32)

        keypoints = transform_keypoints(keypoints, original_size, transformed_size)

        pose_img = draw_pose_from_cords(keypoints, (transformed_size[1], transformed_size[0]), radius=3, draw_bones=True)
        pose_img_pil = Image.fromarray(pose_img)
        pose_img_transformed = transform(pose_img_pil)
        print(f"pose image shape: {pose_img_transformed.shape}")
        return pose_img_transformed

class PisTestDeepFashion(Dataset):
    def __init__(self, root_dir, gt_img_size, pose_img_size, cond_img_size, test_img_size):
        super().__init__()
        self.pose_img_size = pose_img_size
        root_dir = r"C:\Users\user\Desktop\CFLD\CFLD\fashion"
        test_pairs_path = os.path.join(root_dir, "fashion-resize-pairs-test.csv")
        print(f"Test file path: {test_pairs_path}")  # 경로 출력
        if not os.path.exists(test_pairs_path):
            print(f"Test file does not exist: {test_pairs_path}")
        else:
            print(f"Test file exists: {test_pairs_path}")
            test_pairs = pd.read_csv(test_pairs_path)

        # root_dir = os.path.join(root_dir, "DeepFashion")
        #test_pairs = os.path.join(root_dir, "fasion-resize-pairs-test.csv")
        #test_pairs = pd.read_csv(test_pairs)
        self.img_items = self.process_dir(root_dir, test_pairs)

        self.annotation_file = pd.read_csv(os.path.join(root_dir, "fashion-resize-annotation-test.csv"), sep=':')
        self.annotation_file = self.annotation_file.set_index('name')

        self.transform_gt = transforms.Compose([
            transforms.Resize(gt_img_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.transform_cond = transforms.Compose([
            transforms.Resize(cond_img_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize(test_img_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor()
        ])

    def process_dir(self, root_dir, csv_file):
        data = []
        for i in range(len(csv_file)):
            data.append((os.path.join(root_dir, "test_highres", csv_file.iloc[i]["from"]),
                         os.path.join(root_dir, "test_highres", csv_file.iloc[i]["to"])))
        return data

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_path_from, img_path_to = self.img_items[index]
        with open(img_path_from, 'rb') as f:
            img_from = Image.open(f).convert('RGB')
        with open(img_path_to, 'rb') as f:
            img_to = Image.open(f).convert('RGB')

        img_src = self.transform_gt(img_from) # for visualization
        img_tgt = self.transform_gt(img_to) # for visualization
        img_gt = self.transform_test(img_to) # for metrics, 3x256x176
        img_cond_from = self.transform_cond(img_from)

        pose_img_from = self.build_pose_img(img_path_from)
        pose_img_to = self.build_pose_img(img_path_to)

        return {
            "img_src": img_src,
            "img_tgt": img_tgt,
            "img_gt": img_gt,
            "img_cond_from": img_cond_from,
            "pose_img_from": pose_img_from,
            "pose_img_to": pose_img_to
        }

    def build_pose_img(self, img_path):
        string = self.annotation_file.loc[os.path.basename(img_path)]
        array = load_pose_cords_from_strings(string['keypoints_y'], string['keypoints_x'])
        pose_map = torch.tensor(cords_to_map(array, tuple(self.pose_img_size), (256, 176)).transpose(2, 0, 1), dtype=torch.float32)
        pose_img = torch.tensor(draw_pose_from_cords(array, tuple(self.pose_img_size), (256, 176)).transpose(2, 0, 1) / 255., dtype=torch.float32)
        pose_img = torch.cat([pose_img, pose_map], dim=0)
        return pose_img


class FidRealDeepFashion(Dataset):
    def __init__(self, root_dir, test_img_size):
        super().__init__()
        # root_dir = os.path.join(root_dir, "DeepFashion")
        train_dir = os.path.join(root_dir, "train_highres")
        self.img_items = self.process_dir(train_dir)

        self.transform_test = transforms.Compose([
            transforms.Resize(test_img_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor()
        ])

    def process_dir(self, root_dir):
        data = []
        img_paths = glob.glob(os.path.join(root_dir, '*.jpg'))
        for img_path in img_paths:
            data.append(img_path)
        return data

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_path = self.img_items[index]
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        return self.transform_test(img)
    

def print_dataset_samples(dataset, num_samples=5):
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        print(f"Sample {i+1}:")
        print(f"  Source Image Path: {dataset.img_items[i][0]}")
        print(f"  Target Image Path: {dataset.img_items[i][1]}")
        print(f"  Garment Image Path: {dataset.img_items[i][2]}")
        print(f"  Source Image Shape: {sample['img_src'].shape}")
        print(f"  Target Image Shape: {sample['img_tgt'].shape}")
        print(f"  Garment Image Shape: {sample['img_garment'].shape}")
        print(f"  Conditioned Image Shape: {sample['img_cond'].shape}")
        print("\n")

# Example usage
train_dataset = PisTrainDeepFashion(root_dir="C:\\Users\\user\\Desktop\\CFLD\\CFLD\\fashion", gt_img_size=(256, 176), pose_img_size=(256, 176),
                                    cond_img_size=(128, 88), min_scale=0.8, log_aspect_ratio=(-0.2, 0.2),
                                    pred_ratio=[0.1], pred_ratio_var=[0.05], psz=16)

print_dataset_samples(train_dataset)


import matplotlib.pyplot as plt
import random

def visualize_dataset_samples(dataset, num_samples=5):
    sample_indices = random.sample(range(len(dataset)), num_samples)
    scale_factor = 3  
    fig, axs = plt.subplots(num_samples, 6, figsize=(6 * 176 * scale_factor / 100, num_samples * 256 * scale_factor / 100))

    for idx, i in enumerate(sample_indices):
        sample = dataset[i]
        axs[idx, 0].imshow(sample['img_src'].permute(1, 2, 0) * 0.5 + 0.5, aspect='auto')
        axs[idx, 0].set_title("Source Image")
        axs[idx, 1].imshow(sample['pose_img_src'].permute(1, 2, 0) * 0.5 + 0.5, aspect='auto')
        axs[idx, 1].set_title("Pose Image Src")
        axs[idx, 2].imshow(sample['img_tgt'].permute(1, 2, 0) * 0.5 + 0.5, aspect='auto')
        axs[idx, 2].set_title("Target Image")
        axs[idx, 3].imshow(sample['pose_img_tgt'].permute(1, 2, 0) * 0.5 + 0.5, aspect='auto')
        axs[idx, 3].set_title("Pose Image Tgt")
        axs[idx, 4].imshow(sample['img_garment'].permute(1, 2, 0) * 0.5 + 0.5, aspect='auto')
        axs[idx, 4].set_title("Garment Image")
        axs[idx, 5].imshow(sample['img_cond'].permute(1, 2, 0) * 0.5 + 0.5, aspect='auto')
        axs[idx, 5].set_title("Conditioned Image")

        for ax in axs[idx]:
            ax.axis('off')

    plt.tight_layout()
    plt.show()


visualize_dataset_samples(train_dataset)
