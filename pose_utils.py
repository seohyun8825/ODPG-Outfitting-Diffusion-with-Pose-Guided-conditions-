import numpy as np
import torch
from PIL import Image, ImageDraw
import json
import cv2
import os

# BONES and COLORS definition
BONES = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
         [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
         [1,0], [0,14], [14,16], [0,15], [15,17],
         [2,17], [5,16]]

JOINT_COLORS = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44],
                [19,20], [21,22], [23,24], [25,26], [27,28], [29,30],
                [47,48], [49,50], [53,54], [51,52], [55,56],
                [37,38], [45,46]]

BONE_COLORS = [
    [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
    [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
    [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0],
    [100,100,100], [150,150,150]
]
def load_pose_cords_from_strings(y_str, x_str):
    y_cords = json.loads(y_str)
    x_cords = json.loads(x_str)
    cords = np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)
    valid_cords = [cord for cord in cords if cord[0] > 0 and cord[1] > 0]
    return np.array(valid_cords, dtype=np.float32) if valid_cords else np.empty((0, 2))

def transform_keypoints(keypoints, original_size, transformed_size):
    original_width, original_height = original_size
    transformed_width, transformed_height = transformed_size

    scale = min(transformed_width / original_width, transformed_height / original_height)
    new_width = original_width * scale
    new_height = original_height * scale
    offset_x = (transformed_width - new_width) / 2
    offset_y = (transformed_height - new_height) / 2 - (transformed_height - new_height) * 3.5 / 6 

    keypoints[:, 0] = keypoints[:, 0] * scale  + offset_y
    keypoints[:, 1] = keypoints[:, 1] * scale + offset_x

    return keypoints

def cords_to_map(coords, output_size, input_size):
    map = np.zeros((output_size[0], output_size[1], len(coords)))
    for i, (x, y) in enumerate(coords):
        x = int(x * output_size[1] / input_size[1])
        y = int(y * output_size[0] / input_size[0])
        if 0 <= x < output_size[1] and 0 <= y < output_size[0]:
            map[y, x, i] = 1
    return map

def draw_pose_from_cords(coords, output_size, radius=2, draw_bones=True):
    if coords.size == 0:
        return np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)

    pose_img = Image.new('RGB', output_size, (0, 0, 0))
    draw = ImageDraw.Draw(pose_img)
    coords = coords.astype(int)

    for (start_idx, end_idx), color in zip(BONES, BONE_COLORS):
        if start_idx >= len(coords) or end_idx >= len(coords):
            continue
        if all(0 <= coords[idx, 0] < output_size[0] and 0 <= coords[idx, 1] < output_size[1] for idx in [start_idx, end_idx]) and all(coords[idx, 0] > 0 and coords[idx, 1] > 0 for idx in [start_idx, end_idx]):
            draw.line([tuple(coords[start_idx][::-1]), tuple(coords[end_idx][::-1])], fill=tuple(color), width=2)

    for x, y in coords:
        if 0 <= x < output_size[0] and 0 <= y < output_size[1] and (x, y) != (0, 0):
            draw.ellipse((y-radius, x-radius, y+radius, x+radius), fill=(255, 0, 0))

    return np.array(pose_img)