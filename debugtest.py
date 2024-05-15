import csv
csv_path = 'C:/Users/user/Desktop/CFLD/CFLD/fashion/fashion-resize-annotation-train.csv'
with open(csv_path, 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    headers = next(reader)
    print("Headers:", headers)  # 헤더 출력하여 확인
    for i, row in enumerate(reader):
        if i < 5:  # 첫 5개 데이터만 출력
            print("Row", i, ":", row)
