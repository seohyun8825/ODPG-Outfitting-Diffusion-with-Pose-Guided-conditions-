import pdb

import config
from pathlib import Path
import sys
import csv
PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import os

import cv2
import einops
import numpy as np
import random
import time
import json

# from pytorch_lightning import seed_everything
from preprocess.openpose.annotator.util import resize_image, HWC3
from preprocess.openpose.annotator.openpose import OpenposeDetector

import argparse
from PIL import Image
import torch
import pdb

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


class OpenPose:
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        torch.cuda.set_device(gpu_id)
        self.preprocessor = OpenposeDetector()

    def __call__(self, input_image, resolution=384):
        torch.cuda.set_device(self.gpu_id)
        if isinstance(input_image, Image.Image):
            # Resize the image to maintain the aspect ratio
            input_image = input_image.resize((resolution, int(input_image.height * resolution / input_image.width)) if input_image.width > input_image.height else (int(input_image.width * resolution / input_image.height), resolution))
            input_image = np.asarray(input_image)
        else:
            raise ValueError("Unsupported image type")
        
        with torch.no_grad():
            input_image = HWC3(input_image)
            input_image = resize_image(input_image, resolution)
            H, W, C = input_image.shape
            pose, detected_map = self.preprocessor(input_image, hand_and_face=False)

            # Handle the case where no bodies are detected
            if not pose['bodies']['subset']:
                return {"pose_keypoints_2d": [[-1, -1] for _ in range(18)]}  # Return default keypoints

            candidate = pose['bodies']['candidate']
            subset = pose['bodies']['subset'][0][:18]
            for i in range(18):
                if subset[i] == -1:
                    candidate.insert(i, [0, 0])
                    for j in range(i, 18):
                        if subset[j] != -1:
                            subset[j] += 1
                elif subset[i] != i:
                    candidate.pop(i)
                    for j in range(i, 18):
                        if subset[j] != -1:
                            subset[j] -= 1
            candidate = candidate[:18]
            for i in range(18):
                candidate[i][0] *= W
                candidate[i][1] *= H
            keypoints = {"pose_keypoints_2d": candidate}
            return keypoints

def process_images_and_save(lst_file, image_folder, output_csv):
    with open(lst_file, 'r') as file:
        image_names = file.read().splitlines()
    model = OpenPose(gpu_id=0) 
    with open(output_csv, 'w', newline='') as csvfile:

        csvfile.write("name:keypoints_y:keypoints_x\n")
        for image_name in image_names:
            image_path = image_folder / image_name
            if not image_path.exists():
                print(f"Image file does not exist: {image_path}")
                continue
            try:
                image = Image.open(image_path)
            except IOError:
                print(f"Failed to open image file: {image_path}")
                continue
            keypoints = model(image) 
            if keypoints:
                keypoints_y = '[' + ', '.join(map(str, [int(kp[1]) for kp in keypoints["pose_keypoints_2d"] if kp[1] != -1] + [-1 for kp in keypoints["pose_keypoints_2d"] if kp[1] == -1])) + ']'
                keypoints_x = '[' + ', '.join(map(str, [int(kp[0]) for kp in keypoints["pose_keypoints_2d"] if kp[0] != -1] + [-1 for kp in keypoints["pose_keypoints_2d"] if kp[0] == -1])) + ']'
            else:
                keypoints_y = keypoints_x = '[' + ', '.join(['-1'] * 18) + ']'
            

            csvfile.write(f"{image_name}:{keypoints_y}:{keypoints_x}\n")



if __name__ == '__main__':
    base_path = Path(__file__).resolve().parents[2] / 'fashion'
    process_images_and_save(base_path / 'new_test.lst', base_path / 'test', 'annotation-new-test.csv')
    process_images_and_save(base_path / 'new_train.lst', base_path / 'train', 'annotation-new-train.csv')
