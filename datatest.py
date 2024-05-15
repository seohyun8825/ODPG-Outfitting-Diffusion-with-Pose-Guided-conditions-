import os
import csv
from itertools import permutations
import random

train_file = open('fashion-resize-pairs-train.csv', 'w', newline='')
test_file = open('fashion-resize-pairs-test.csv', 'w', newline='')

# 추가적인 파일 핸들러
train_lst_file = open('train.lst', 'w', newline='')
test_lst_file = open('test.lst', 'w', newline='')
train_garment_file = open('train_garment.lst', 'w', newline='')
test_garment_file = open('test_garment.lst', 'w', newline='')

train_writer = csv.writer(train_file)
test_writer = csv.writer(test_file)
train_lst_writer = csv.writer(train_lst_file)
test_lst_writer = csv.writer(test_lst_file)
train_garment_writer = csv.writer(train_garment_file)
test_garment_writer = csv.writer(test_garment_file)

train_writer.writerow(['from', 'to', 'garment'])
test_writer.writerow(['from', 'to', 'garment'])

train_set = set()
test_set = set()
train_garment_set = set()
test_garment_set = set()

for (path, dir, files) in os.walk("./fashion"):
    if dir == []:
        new_path = path[2:]
        path_in_list = new_path.split('\\')

        top_related_garments = ['Jackets_Vests', 'Shirts_Polos', 'Sweaters', 'Sweatshirts_Hoodies', 'Tees_Tanks', 'Blouses_Shirts', 'Cardigans', 'Graphic_Tees']
        if (path_in_list[1] not in ['MEN', 'WOMEN']) or (path_in_list[2] not in top_related_garments):
            continue
        path_in_list[3] = path_in_list[3][:2] + path_in_list[3][3:]
        separated_files = {}
        for file in files:
            file = file[:4] + file[5:]
            prefix = file[:2]
            if prefix not in separated_files:
                separated_files[prefix] = []
            separated_files[prefix].append(file)
        for file_list in separated_files.values():
            file_front_name_list = list(map(lambda filename: filename.split(".")[0][4:], file_list))
            if 'flat' not in file_front_name_list:
                continue
            file_front_without_flat = list([item for item in file_front_name_list if item != 'flat'])
            if len(file_front_without_flat) < 2:
                continue
            path_file_list = list(map(lambda filename: "".join(path_in_list) + filename, file_list))
            perm_input = [item for item in path_file_list if item[-8:-4] != 'flat']
            garment_element = [item for item in path_file_list if item[-8:-4] == 'flat'][0]
            perm_outputs = list(permutations(perm_input, 2))
            for perm_output in perm_outputs:
                if random.random() < 0.1:
                    test_writer.writerow(perm_output + (garment_element,))
                    test_set.update(perm_output)
                    test_garment_set.add(garment_element)
                else:
                    train_writer.writerow(perm_output + (garment_element,))
                    train_set.update(perm_output)
                    train_garment_set.add(garment_element)

# 중복 없이 각 lst 파일에 쓰기
for item in train_set:
    train_lst_writer.writerow([item])
for item in test_set:
    test_lst_writer.writerow([item])
for item in train_garment_set:
    train_garment_writer.writerow([item])
for item in test_garment_set:
    test_garment_writer.writerow([item])

# 파일 닫기
train_file.close()
test_file.close()
train_lst_file.close()
test_lst_file.close()
train_garment_file.close()
test_garment_file.close()
