import csv
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch
import re
import sys

# Insert the path to the preprocessing modules
sys.path.insert(0, str(Path(__file__).resolve().parent / 'preprocess/humanparsing'))

from run_parsing import Parsing
from getmask import get_mask_location

transform_gt = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

parsing_model = Parsing(gpu_id=0)  # GPU ID 설정

def parse_keypoints(data):
    """정규 표현식을 이용하여 키포인트를 파싱합니다."""
    try:
        name, y_data, x_data = re.match(r'([^:]+):\[(.*?)\]:\[(.*?)\]', data).groups()

        keypoints_y = list(map(int, re.findall(r'\d+', y_data)))
        keypoints_x = list(map(int, re.findall(r'\d+', x_data)))
        return name, keypoints_y, keypoints_x
    except Exception as e:
        print(f"Error parsing data: {data} with error: {e}")
        return None, None, None

def process_parsing_and_mask(base_path, csv_file, image_folder, num_samples=5):
    base_path = Path(base_path)
    image_folder_path = base_path / image_folder
    csv_path = base_path / csv_file
    output_dir = base_path / image_folder
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # 첫 줄(헤더) 건너뛰기
        count = 0
        for row in reader:
            if count >= num_samples:
                break
            full_data = ''.join(row)
            name, keypoints_y, keypoints_x = parse_keypoints(full_data)
            if not name:
                continue

            image_path = image_folder_path / name
            if not image_path.exists():
                print(f"Image file does not exist: {image_path}")
                continue

            image = Image.open(image_path)
            keypoints = {
                'pose_keypoints_2d': list(zip(keypoints_x, keypoints_y))
            }

            parsed_image, face_mask = parsing_model(image)
            mask, mask_gray = get_mask_location("hd", "upper_body", parsed_image, keypoints)
            mask = mask.resize(image.size, Image.NEAREST).convert('RGB')

            mask_tensor = transform_gt(mask)
            image_tensor = transform_gt(image)
            overlay_tensor = image_tensor * (1 - mask_tensor)

            # Convert overlay_tensor to NumPy array, adjust values if necessary
            overlay_np = overlay_tensor.permute(1, 2, 0).numpy()
            overlay_np = (overlay_np * 0.5 + 0.5) * 255  # Scale to [0, 255]
            overlay_np = np.clip(overlay_np, 0, 255).astype(np.uint8)  # Clip and convert to uint8

            # Create PIL Image from NumPy array
            overlay_image = Image.fromarray(overlay_np)

            # Save overlay_image as PNG file
            overlay_image.save(output_dir / f'{row}_masked.png')

            mask.save(output_dir / f'{row}_masked.png')
            # parsed_image.save(output_dir / f'parsed_image_{count}.png')

            print(f"Processed and saved images for {name}")
            count += 1

if __name__ == "__main__":
    base_path = Path('/home/user/Desktop/CFLD/CFLD/')
    process_parsing_and_mask(base_path, 'fashion/fashion-resize-annotation-train.csv', 'fashion/train_highres')
