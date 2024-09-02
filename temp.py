import os
import shutil

# 경로 설정
source_dir = '/mnt/hdd2/project/PatchModel/data/hyundae/val'
destination_dir = '/mnt/hdd2/project/PatchModel/data/hyundae/ann_dir/val'

# 대상 디렉토리 생성
os.makedirs(destination_dir, exist_ok=True)

# 파일 이동
for filename in os.listdir(source_dir):
    if filename.endswith('.json'):
        source_file = os.path.join(source_dir, filename)
        destination_file = os.path.join(destination_dir, filename)
        shutil.move(source_file, destination_file)

print("파일 이동 완료")