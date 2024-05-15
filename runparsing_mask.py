import csv
from pathlib import Path
from PIL import Image
import sys
import re 
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'preprocess/humanparsing'))

from run_parsing import Parsing
from getmask import get_mask_location


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
    output_dir = base_path / 'output'
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

            mask.save(output_dir / f'mask_{count}.png')
            mask_gray.save(output_dir / f'mask_gray_{count}.png')
            parsed_image.save(output_dir / f'parsed_image_{count}.png')

            print(f"Processed and saved images for {name}")
            count += 1

if __name__ == "__main__":
    base_path = Path('C:/Users/user/Desktop/CFLD/CFLD/fashion')
    process_parsing_and_mask(base_path, 'fashion-resize-annotation-train.csv', 'train_highres')