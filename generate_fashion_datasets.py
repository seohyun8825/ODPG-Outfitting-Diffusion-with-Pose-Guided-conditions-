import os
import shutil

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def format_path(path):
    path = path.lower().replace('\\', '/')
    path = path.replace('/', '')
    path = path.replace('_', '')
    return path

def make_dataset(dir):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    # 기본 이미지 폴더
    train_root = os.path.join(dir, 'train_highres')
    test_root = os.path.join(dir, 'test_highres')
    # 의류 이미지 전용 폴더
    train_garment_root = os.path.join(dir, 'train_garment_highres')
    test_garment_root = os.path.join(dir, 'test_garment_highres')

    for root in [train_root, test_root, train_garment_root, test_garment_root]:
        if not os.path.exists(root):
            os.makedirs(root)

    # 각 lst 파일로부터 이미지 매핑 정보 생성
    train_images, test_images, train_garments, test_garments = {}, {}, {}, {}

    # 일반 이미지 파일 목록
    with open(os.path.join(dir, 'train.lst'), 'r') as train_f:
        for line in train_f:
            original_line = line.strip()
            normalized_line = format_path(original_line)
            train_images[normalized_line] = original_line
    with open(os.path.join(dir, 'test.lst'), 'r') as test_f:
        for line in test_f:
            original_line = line.strip()
            normalized_line = format_path(original_line)
            test_images[normalized_line] = original_line

    # 의류 이미지 파일 목록
    with open(os.path.join(dir, 'train_garment.lst'), 'r') as train_garment_f:
        for line in train_garment_f:
            original_line = line.strip()
            normalized_line = format_path(original_line)
            train_garments[normalized_line] = original_line
    with open(os.path.join(dir, 'test_garment.lst'), 'r') as test_garment_f:
        for line in test_garment_f:
            original_line = line.strip()
            normalized_line = format_path(original_line)
            test_garments[normalized_line] = original_line

    # 이미지 파일 복사
    for root, _, fnames in os.walk(dir):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                formatted_path = format_path(path)
                # 일반 이미지 분류
                if formatted_path in train_images:
                    target_fname = train_images[formatted_path]
                    shutil.copy(path, os.path.join(train_root, target_fname))
                    print("Copying to train: ", target_fname)
                if formatted_path in test_images:
                    target_fname = test_images[formatted_path]
                    shutil.copy(path, os.path.join(test_root, target_fname))
                    print("Copying to test: ", target_fname)
                # 의류 이미지 분류
                if formatted_path in train_garments:
                    target_fname = train_garments[formatted_path]
                    shutil.copy(path, os.path.join(train_garment_root, target_fname))
                    print("Copying to train garment: ", target_fname)
                if formatted_path in test_garments:
                    target_fname = test_garments[formatted_path]
                    shutil.copy(path, os.path.join(test_garment_root, target_fname))
                    print("Copying to test garment: ", target_fname)

make_dataset('fashion')
