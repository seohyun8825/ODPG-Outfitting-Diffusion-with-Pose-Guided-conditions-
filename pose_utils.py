"""
@author: Yanzuo Lu
@author: oliveryanzuolu@gmail.com
"""

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
BONES = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], [8,9],
         [9,10], [1,11], [11,12], [12,13], [1,0], [0,14], [14,16],
         [0,15], [15,17]]

JOINT_COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
                [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
                [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

BONE_COLORS = [[153, 0, 0], [153, 51, 0], [153, 102, 0], [153, 153, 0], [102, 153, 0], [51, 153, 0], [0, 153, 0], [0, 153, 51],
               [0, 153, 102], [0, 153, 153], [0, 102, 153], [0, 51, 153], [0, 0, 153], [51, 0, 153], [102, 0, 153],
               [153, 0, 153], [153, 0, 102]]

def load_pose_cords_from_strings(y_str, x_str):
    y_cords = json.loads(y_str)
    x_cords = json.loads(x_str)
    cords = np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)
    # Remove invalid points more rigorously
    valid_cords = [cord for cord in cords if cord[0] > 0 and cord[1] > 0]
    return np.array(valid_cords, dtype=np.float32) if valid_cords else np.empty((0, 2))



def cords_to_map(cords, img_size, old_size):
    result = np.zeros(img_size + cords.shape[0:1], dtype='float32')
    
    # 각 키포인트에 대해 반복
    for i, point in enumerate(cords):
        if point[0] == -1 or point[1] == -1:
            continue  # 유효하지 않은 키포인트는 무시
        
        # 이미지 리사이즈 비율에 따라 키포인트 위치 조정
        x_scaled = (point[1] / old_size[1]) * img_size[1]
        y_scaled = (point[0] / old_size[0]) * img_size[0]

        # 맵 상에 키포인트 위치에 가우시안 분포를 적용
        xx, yy = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
        result[..., i] = np.exp(-((yy - y_scaled) ** 2 + (xx - x_scaled) ** 2) / (2 * 6 ** 2))

    return result
def draw_pose_from_cords(array, img_size, padding=23, radius=2, draw_bones=True):
    if array.size == 0:
        return np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)

    # 키포인트의 실제 사용 범위 계산
    min_x = np.min(array[:, 1])
    max_x = np.max(array[:, 1])
    min_y = np.min(array[:, 0])
    max_y = np.max(array[:, 0])

    # 실제 그려질 콘텐츠 너비와 높이
    content_width = max_x - min_x
    content_height = max_y - min_y

    # 각 축별 스케일링 비율 계산
    scale_x = (img_size[0] - 55) / content_width  # 가로에 대한 스케일
    scale_y = (img_size[1] ) / content_height  # 세로에 대한 스케일

    # 중앙에 위치시키기 위한 패딩 계산
    padding_x = (img_size[0] - (content_width * scale_x)) / 2
    padding_y = (img_size[1] - (content_height * scale_y)) / 2

    colors = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)

    for i, (f, t) in enumerate(BONES):
        if f >= len(array) or t >= len(array) or array[f][0] <= 0 or array[f][1] <= 0 or array[t][0] <= 0 or array[t][1] <= 0:
            continue

        start_point = (
            int((array[f][1] - min_x) * scale_x + padding_x),
            int((array[f][0] - min_y) * scale_y + padding_y)
        )
        end_point = (
            int((array[t][1] - min_x) * scale_x + padding_x),
            int((array[t][0] - min_y) * scale_y + padding_y)
        )
        cv2.line(colors, start_point, end_point, BONE_COLORS[i], radius)

    for i, joint in enumerate(array):
        if joint[0] <= 0 or joint[1] <= 0:
            continue

        center = (
            int((joint[1] - min_x) * scale_x + padding_x),
            int((joint[0] - min_y) * scale_y + padding_y)
        )
        cv2.circle(colors, center, radius, JOINT_COLORS[i], -1)

    return colors
