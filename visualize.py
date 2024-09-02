import os
import sys
from pathlib import Path
import mmcv
import cv2
import numpy as np
from mmseg.apis import init_model, inference_model
import shutil
import json
def open_directory(path):
    if sys.platform.startswith('darwin'):  # macOS
        os.system('open "{}"'.format(path))
    elif sys.platform.startswith('win'):  # Windows
        os.system('start "" "{}"'.format(path))
    elif sys.platform.startswith('linux'):  # Linux
        os.system('xdg-open "{}"'.format(path))
    else:
        print("Unsupported operating system.")

class MMSegWrapper:
    def __init__(self):
        self.model = None
        self.download_model()

    def download_model(self):
        config_file = '/media/spalab/sdd/kypark/PatchModel/configs/patchnet/patchnet_r34.py'
        checkpoint_file = '/media/spalab/sdd/kypark/PatchModel/work_dirs/hmc_5000/iter_40000.pth'
        self.model = init_model(config_file, checkpoint_file, device='cuda:0')

    def get_result(self, src_filename):
        if self.model:
            try:
                result_dict = {}
                ext = Path(src_filename).suffix
                save_dir = '/media/spalab/sdd/kypark/PatchModel/results'
                os.makedirs(save_dir, exist_ok=True)
                # img_dir = '/mnt/sdb/PatchModel/results_img'
                # os.makedirs(img_dir, exist_ok=True)
                # gt_dir = '/mnt/sdb/PatchModel/results_gt'
                # os.makedirs(gt_dir, exist_ok=True)
                dst_filename = os.path.join(save_dir, f'{Path(src_filename).stem}_result{ext}')
                if ext in ['.jpg', '.png', '.jpeg']:
                    img = mmcv.imread(src_filename)
                    original_size = img.shape[:2]
                    resized_img = cv2.resize(img, (512, 512))
                    result = inference_model(self.model, resized_img)
                    seg_map = result.pred_sem_seg.data.cpu().numpy().squeeze()
                    gt_map_path = src_filename.replace("img","ann").replace("png","json")
                    with open(gt_map_path, "r") as st_json:
                        gt_map = json.load(st_json)
                    gt_map = np.array(gt_map['patch'])
                    gt_map = gt_map[:,:,1]
                    # gt_map[gt_map == 2] = 1
                    print("Segmentation result shape (before resize):", seg_map.shape)

                    if seg_map.shape == (16, 16):
                        print("The segmentation result is 16x16.")
                    else:
                        print("The segmentation result is not 16x16.")

                    unique, counts = np.unique(seg_map, return_counts=True)
                    print("Unique values and their counts in the 16x16 result array:")
                    for u, c in zip(unique, counts):
                        print(f"Value: {u}, Count: {c}")

                    patch_height = original_size[0] // 16
                    patch_width = original_size[1] // 16

                    visualized_img = img.copy()
                    visualized_gt = img.copy()
                    color_map = {
                        1: (0, 0, 255, 128),  # 반투명한 빨간색
                        2: (0, 0, 255)  # 불투명한 빨간색
                    }
                    for i in range(16):
                        for j in range(16):
                            label = seg_map[i, j]
                            if label in color_map:
                                x_start = j * patch_width
                                y_start = i * patch_height
                                x_end = x_start + patch_width
                                y_end = y_start + patch_height

                                overlay = visualized_img[y_start:y_end, x_start:x_end].copy()
                                overlay[:, :, :3] = color_map[label][:3]

                                if label == 1:
                                    alpha = 0.5
                                    cv2.addWeighted(overlay, alpha, visualized_img[y_start:y_end, x_start:x_end], 1 - alpha, 0, visualized_img[y_start:y_end, x_start:x_end])
                                else:
                                    visualized_img[y_start:y_end, x_start:x_end] = overlay
                                    
                    for i in range(16):
                        for j in range(16):
                            label = gt_map[i, j]
                            if label in color_map:
                                x_start = j * patch_width
                                y_start = i * patch_height
                                x_end = x_start + patch_width
                                y_end = y_start + patch_height

                                overlay = visualized_gt[y_start:y_end, x_start:x_end].copy()
                                overlay[:, :, :3] = color_map[label][:3]
                                if label == 1:
                                    alpha = 0.5
                                    cv2.addWeighted(overlay, alpha, visualized_gt[y_start:y_end, x_start:x_end], 1 - alpha, 0, visualized_gt[y_start:y_end, x_start:x_end])
                                else:
                                    visualized_gt[y_start:y_end, x_start:x_end] = overlay
                    
                    # shutil.copy(src_filename, img_dir+'/'+dst_filename.split("/")[-1])
                    dst_filename_ = dst_filename.replace("results","results_gt")
                    visualize_raw_img = cv2.imread(src_filename)
                    new_img = np.concatenate((visualize_raw_img,visualized_gt,visualized_img),axis=1)
                    cv2.imwrite(dst_filename, new_img)
                    # cv2.imwrite(dst_filename_, visualized_gt)
                    # cv2.imwrite(dst_filename, visualized_img)
                    #open_directory(save_dir)  # 결과 디렉토리를 엽니다.
                    return dst_filename, result_dict
                else:
                    raise Exception('Unsupported file type.')
            except Exception as e:
                raise Exception(e)
        else:
            raise Exception('You have to call download_model first.')

if __name__ == "__main__":
    # 테스트할 이미지 경로
    img_path = '/media/spalab/sdd/kypark/PatchModel/data/hyundae/img_dir/val'
    
    wrapper = MMSegWrapper()
    for img_path in Path(img_path).glob('*'):
        if img_path.suffix in ['.jpg', '.png', '.jpeg']:
            dst_filename, result_dict = wrapper.get_result(str(img_path))
            print(f"Segmentation result saved at: {dst_filename}")
