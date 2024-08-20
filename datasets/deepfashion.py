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
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from pathlib import Path
import cv2
import re
import matplotlib.pyplot as plt 
sys.path.append('/home/user/Desktop/CFLD/CFLD/')

# Add the preprocess/humanparsing directory to sys.path
sys.path.insert(0, '/home/user/Desktop/CFLD/CFLD/preprocess/humanparsing')

# Import pose_utils and human parsing functions
from pose_utils import transform_keypoints, draw_pose_from_cords, load_pose_cords_from_strings, cords_to_map
from run_parsing import Parsing
from getmask import get_mask_location

logger = logging.getLogger()

# Initialize the parsing model
parsing_model = Parsing(gpu_id=0)  # GPU ID 설정

import torch.nn.functional as F

from PIL import ImageOps

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

        if not os.path.exists(train_pairs_path):
            print(f"File does not exist: {train_pairs_path}")
        else:
            train_pairs = pd.read_csv(train_pairs_path)

        self.img_items = self.process_dir(root_dir, train_pairs)
        self.annotation_file = pd.read_csv(os.path.join(root_dir, "fashion-resize-annotation-train.csv"), sep=':')
        self.annotation_file = self.annotation_file.set_index('name')

        self.transform_gt = transforms.Compose([
            transforms.Resize(gt_img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.transform_gt_1 = transforms.Compose([
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

        # Generate mask for img_src
        parsed_image_src, _ = parsing_model(img_from)
        keypoints_src = self.load_keypoints(img_path_from)
        mask_src, _ = get_mask_location("hd", "upper_body", parsed_image_src, keypoints_src)
        mask_src = mask_src.resize(original_size_from, Image.NEAREST)
        
        # Padding mask_src to match the original size
        mask_src = self.pad_to_match(mask_src, original_size_from)
        mask_src = self.transform_gt(mask_src.convert('RGB'))

        # Generate masked image for img_src
        masked_img_src = img_src * (1 - mask_src)

        # Generate mask for img_tgt
        parsed_image_tgt, _ = parsing_model(img_to)
        keypoints_tgt = self.load_keypoints(img_path_to)
        mask_tgt, _ = get_mask_location("hd", "upper_body", parsed_image_tgt, keypoints_tgt)
        mask_tgt = mask_tgt.resize(original_size_to, Image.NEAREST)
        
        # Padding mask_tgt to match the original size
        mask_tgt = self.pad_to_match(mask_tgt, original_size_to)
        mask_tgt = self.transform_gt(mask_tgt.convert('RGB'))

        # Generate masked image for img_tgt
        masked_img_tgt = img_tgt * (1 - mask_tgt)

        return_dict = {
            "img_src": img_src,
            "img_tgt": img_tgt,
            "img_cond": img_cond,
            "pose_img_src": pose_img_src,
            "pose_img_tgt": pose_img_tgt,
            "img_garment": img_garment,
            "masked_img_src": masked_img_src,
            "masked_img_tgt": masked_img_tgt
        }
        return return_dict

    def pad_to_match(self, img, target_size):
        """
        Pads the given image to match the target size.
        """
        width, height = img.size
        target_width, target_height = target_size
        padding = (
            (target_width - width) // 2,
            (target_height - height) // 2,
            (target_width - width + 1) // 2,
            (target_height - height + 1) // 2
        )
        return ImageOps.expand(img, padding)

    def build_pose_img(self, img_path, original_size, transform):
        img = Image.open(img_path).convert('RGB')
        transformed_img = transform(img)
        transformed_size = (transformed_img.shape[2], transformed_img.shape[1])

        string = self.annotation_file.loc[os.path.basename(img_path)]
        keypoints = load_pose_cords_from_strings(string['keypoints_y'], string['keypoints_x'])

        if keypoints.size == 0:
            print("Warning: No valid keypoints for", img_path)
            return torch.zeros(20, 256, 256, dtype=torch.float32)

        keypoints = transform_keypoints(keypoints, original_size, transformed_size)

        pose_map = cords_to_map(keypoints, (256, 256), transformed_size).transpose(2, 0, 1)
        pose_img = draw_pose_from_cords(keypoints, (256, 256))

        pose_map_tensor = torch.tensor(pose_map, dtype=torch.float32)
        pose_img_tensor = torch.tensor(pose_img.transpose(2, 0, 1), dtype=torch.float32)

        # Adjust number of channels in pose_map_tensor to be consistent
        num_keypoints = 17  # Assume 17 keypoints for consistency
        if pose_map_tensor.shape[0] < num_keypoints:
            pad_size = num_keypoints - pose_map_tensor.shape[0]
            pose_map_tensor = torch.nn.functional.pad(pose_map_tensor, (0, 0, 0, 0, 0, pad_size))
        elif pose_map_tensor.shape[0] > num_keypoints:
            pose_map_tensor = pose_map_tensor[:num_keypoints, :, :]

        combined_pose = torch.cat([pose_img_tensor, pose_map_tensor], dim=0)

        return combined_pose

    def load_keypoints(self, img_path):
        string = self.annotation_file.loc[os.path.basename(img_path)]
        keypoints_y = list(map(int, re.findall(r'\d+', string['keypoints_y'])))
        keypoints_x = list(map(int, re.findall(r'\d+', string['keypoints_x'])))
        keypoints = {
            'pose_keypoints_2d': list(zip(keypoints_x, keypoints_y))
        }
        return keypoints
class PisTestDeepFashion(Dataset):
    def __init__(self, root_dir, gt_img_size, pose_img_size, cond_img_size, test_img_size):
        super().__init__()
        self.pose_img_size = pose_img_size
        self.cond_img_size = cond_img_size

        test_pairs_path = os.path.join(root_dir, "fashion-resize-pairs-test.csv")
        if not os.path.exists(test_pairs_path):
            print(f"Test file does not exist: {test_pairs_path}")
        else:
            test_pairs = pd.read_csv(test_pairs_path)

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
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def process_dir(self, root_dir, csv_file):
        data = []
        for i in range(len(csv_file)):
            data.append((
                os.path.join(root_dir, "test_highres", csv_file.iloc[i]["from"]),
                os.path.join(root_dir, "test_highres", csv_file.iloc[i]["to"]),
                os.path.join(root_dir, "test_garment_highres", csv_file.iloc[i]["garment"])
            ))
        return data

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_path_from, img_path_to, _ = self.img_items[index]
        random_index = np.random.randint(0, len(self.img_items))
        _, _, random_img_path_garment = self.img_items[random_index]

        with open(img_path_from, 'rb') as f:
            img_from = Image.open(f).convert('RGB')
        with open(img_path_to, 'rb') as f:
            img_to = Image.open(f).convert('RGB')
        with open(random_img_path_garment, 'rb') as f:
            img_garment = Image.open(f).convert('RGB')

        print(f"Loading source image from: {img_path_from}")
        print(f"Loading target image from: {img_path_to}")
        print(f"Loading random garment image from: {random_img_path_garment}")

        original_size_from = img_from.size
        original_size_to = img_to.size

        img_src = self.transform_gt(img_from)
        img_tgt = self.transform_gt(img_to)
        img_gt = self.transform_test(img_to)
        img_cond_from = self.transform_cond(img_from)
        img_garment = self.transform_gt(img_garment)

        pose_img_from = self.build_pose_img(img_path_from, original_size_from, self.transform_gt)
        pose_img_to = self.build_pose_img(img_path_to, original_size_to, self.transform_gt)

        return {
            "img_src": img_src,
            "img_tgt": img_tgt,
            "img_gt": img_gt,
            "img_cond_from": img_cond_from,
            "pose_img_src": pose_img_from,
            "pose_img_tgt": pose_img_to,
            "img_garment": img_garment
        }

    def build_pose_img(self, img_path, original_size, transform):
        img = Image.open(img_path).convert('RGB')
        transformed_img = transform(img)
        transformed_size = (transformed_img.shape[2], transformed_img.shape[1])

        string = self.annotation_file.loc[os.path.basename(img_path)]
        keypoints = load_pose_cords_from_strings(string['keypoints_y'], string['keypoints_x'])

        if keypoints.size == 0:
            print("Warning: No valid keypoints for", img_path)
            return torch.zeros(20, 256, 256, dtype=torch.float32)

        keypoints = transform_keypoints(keypoints, original_size, transformed_size)

        pose_map = cords_to_map(keypoints, (256, 256), transformed_size).transpose(2, 0, 1)
        pose_img = draw_pose_from_cords(keypoints, (256, 256))

        pose_map_tensor = torch.tensor(pose_map, dtype=torch.float32)
        pose_img_tensor = torch.tensor(pose_img.transpose(2, 0, 1), dtype=torch.float32)

        num_keypoints = 17  # Assume 17 keypoints for consistency
        if pose_map_tensor.shape[0] < num_keypoints:
            pad_size = num_keypoints - pose_map_tensor.shape[0]
            pose_map_tensor = torch.nn.functional.pad(pose_map_tensor, (0, 0, 0, 0, 0, pad_size))
        elif pose_map_tensor.shape[0] > num_keypoints:
            pose_map_tensor = pose_map_tensor[:num_keypoints, :, :]

        combined_pose = torch.cat([pose_img_tensor, pose_map_tensor], dim=0)

        return combined_pose

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


def save_dataset_samples(dataset, num_samples=5, save_dir='samples', prefix='sample'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    sample_indices = random.sample(range(len(dataset)), num_samples)
    
    for idx, i in enumerate(sample_indices):
        sample = dataset[i]

        fig, axs = plt.subplots(1, 8, figsize=(24, 4))

        axs[0].imshow(sample['img_src'].permute(1, 2, 0).numpy() * 0.5 + 0.5)
        axs[0].set_title("Source Image")

        pose_img_src = sample['pose_img_src'][:3].permute(1, 2, 0).numpy()
        axs[1].imshow(pose_img_src * 0.5 + 0.5)
        axs[1].set_title("Pose Image Src")

        axs[2].imshow(sample['img_tgt'].permute(1, 2, 0).numpy() * 0.5 + 0.5)
        axs[2].set_title("Target Image")

        pose_img_tgt = sample['pose_img_tgt'][:3].permute(1, 2, 0).numpy()
        axs[3].imshow(pose_img_tgt * 0.5 + 0.5)
        axs[3].set_title("Pose Image Tgt")

        axs[4].imshow(sample['img_garment'].permute(1, 2, 0).numpy() * 0.5 + 0.5)
        axs[4].set_title("Garment Image")

        if 'img_cond' in sample:
            axs[5].imshow(sample['img_cond'].permute(1, 2, 0).numpy() * 0.5 + 0.5)
            axs[5].set_title("Conditioned Image")

        if 'masked_img_src' in sample:
            axs[6].imshow(sample['masked_img_src'].permute(1, 2, 0).numpy() * 0.5 + 0.5)
            axs[6].set_title("Masked Src Image")

        if 'masked_img_tgt' in sample:
            axs[7].imshow(sample['masked_img_tgt'].permute(1, 2, 0).numpy() * 0.5 + 0.5)
            axs[7].set_title("Masked Tgt Image")

        for ax in axs:
            ax.axis('off')

        print("HELLO")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}_{idx}.png"))
        plt.close(fig)

# Example usage
train_dataset = PisTrainDeepFashion(root_dir="/home/user/Desktop/CFLD/CFLD/fashion", gt_img_size=(256, 176), pose_img_size=(256, 176), cond_img_size=(128, 88), min_scale=0.8, log_aspect_ratio=(-0.2, 0.2), pred_ratio=[0.1], pred_ratio_var=[0.05], psz=16)

test_dataset = PisTestDeepFashion(root_dir="/home/user/Desktop/CFLD/CFLD/fashion", gt_img_size=(256, 176), pose_img_size=(256, 176), cond_img_size=(128, 88), test_img_size=(256, 176))

save_dataset_samples(train_dataset, num_samples=2, save_dir='samples', prefix='train')
save_dataset_samples(test_dataset, num_samples=2, save_dir='samples', prefix='test')