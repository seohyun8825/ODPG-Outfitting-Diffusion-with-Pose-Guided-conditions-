import json
import logging
import cv2

import numpy as np

logger = logging.getLogger()

'''
BONES = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
         [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
         [1, 16], [3, 17], ]
'''
BONES = [[1,2],[0,1], [0,4], [2,3], [4,9], [5,6], [6,7], [7,12], [1,5], [8,9],
         [9,10], [1,11], [11,12], [12,13], [0,14], [14,16],
         [0,15], [15,17]]

JOINT_COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
                [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
                [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

BONE_COLORS = [[153, 0, 0], [153, 51, 0], [153, 102, 0], [153, 153, 0], [102, 153, 0], [51, 153, 0], [0, 153, 0], [0, 153, 51],
               [0, 153, 102], [0, 153, 153], [0, 102, 153], [0, 51, 153], [0, 0, 153], [51, 0, 153], [102, 0, 153],
               [153, 0, 153], [153, 0, 102]]*2


def load_pose_cords_from_strings(y_str, x_str):
    y_cords = json.loads(y_str)
    x_cords = json.loads(x_str)
    cords = np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)
    valid_cords = [cord for cord in cords if cord[0] > 0 and cord[1] > 0]
    return np.array(valid_cords, dtype=np.float32) if valid_cords else np.empty((0, 2))
def transform_keypoints(keypoints, original_size, transformed_size):
    original_width, original_height = original_size
    transformed_width, transformed_height = transformed_size

    # Calculate scale factors
    scale = min(transformed_width / original_width, transformed_height / original_height)

    # Calculate the new dimensions after scaling
    new_width = original_width * scale
    new_height = original_height * scale

    # Calculate the offsets to center the keypoints
    offset_x = (transformed_width - new_width) / 2
    offset_y = (transformed_height - new_height) / 2 - (transformed_height - new_height) * 3.5 / 6 

    # Apply scaling and offsets
    keypoints[:, 0] = keypoints[:, 0] * scale*0.7 + offset_y  # Y-axis scaling and offset
    keypoints[:, 1] = keypoints[:, 1] * scale + offset_x  # X-axis scaling and offset

    return keypoints


def draw_pose_from_cords(array, img_size, radius=2, draw_bones=True):
    if array.size == 0:
        return np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)

    colors = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)

    for i, (f, t) in enumerate(BONES):
        if f >= len(array) or t >= len(array) or array[f][0] <= 0 or array[f][1] <= 0 or array[t][0] <= 0 or array[t][1] <= 0:
            continue

        start_point = (int(array[f][1]), int(array[f][0]))
        end_point = (int(array[t][1]), int(array[t][0]))
        cv2.line(colors, start_point, end_point, BONE_COLORS[i], 2)

    for idx, joint in enumerate(array):
        if joint[0] <= 0 or joint[1] <= 0:
            continue

        center = (int(joint[1]), int(joint[0]))
        cv2.circle(colors, center, radius, JOINT_COLORS[idx], -1)
        cv2.putText(colors, str(idx), (center[0] + 3, center[1] + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    return colors