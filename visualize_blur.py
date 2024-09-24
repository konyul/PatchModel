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
        config_file = 'configs/patchnet/0920/r34_1.py'
        checkpoint_file = 'work_dirs/hmc_5000_2nd/0923/dilation_1357_cls_weight_4/iter_40000.pth'
        self.model = init_model(config_file, checkpoint_file, device='cuda:0')

    def get_result(self, src_filename):
        if self.model:
            try:
                result_dict = {}
                ext = Path(src_filename).suffix
                save_dir = '/media/spalab/sdd/kypark/PatchModel/results_2nd'
                os.makedirs(save_dir, exist_ok=True)
                save_dir_raw = '/media/spalab/sdd/kypark/PatchModel/results_2nd_raw'
                os.makedirs(save_dir_raw, exist_ok=True)
                # img_dir = '/mnt/sdb/PatchModel/results_img'
                # os.makedirs(img_dir, exist_ok=True)
                # gt_dir = '/mnt/sdb/PatchModel/results_gt'
                # os.makedirs(gt_dir, exist_ok=True)
                dst_filename = os.path.join(save_dir, f'{Path(src_filename).stem}_result{ext}')
                dst_filename_raw = os.path.join(save_dir_raw, f'{Path(src_filename).stem}_result{ext}')
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
                    visualized_img_raw = img.copy()
                    visualized_gt = img.copy()
                    color_map = {
                        1: (255, 99, 71, 180),  # 토마토 레드 (반투명)
                        2: (25, 0, 130, 180) # 로열 블루 (반투명)
                        # 1: (0, 0, 255, 128),  # 반투명한 빨간색
                        # 2: (0, 0, 255)  # 불투명한 빨간색
                    }
                    
                    blurry_map = {
                        0: 'clean',
                        1: 'blur',
                        2: 'blockage'
                    }
                    def draw_grid(image, grid_size=16, color=(0, 255, 0), thickness=1):
                        # Open the image
                        h, w = image.shape[:2]
    
                        # Draw vertical lines
                        for x in range(0, w, w // grid_size):
                            cv2.line(image, (x, 0), (x, h), color, thickness)
                        
                        # Draw horizontal lines
                        for y in range(0, h, h // grid_size):
                            cv2.line(image, (0, y), (w, y), color, thickness)
                        return image
                    visualize_raw_img = cv2.imread(src_filename)
                    visualize_raw_img = draw_grid(visualize_raw_img)
                    visualized_img_raw = draw_grid(visualized_img_raw)
                    visualized_img = draw_grid(visualized_img)

                    for i in range(16):
                        for j in range(16):
                            label = seg_map[i, j]
                            label_gt = gt_map[i, j]
                            if (label_gt == 1 and label == 0) or (label_gt == 0 and label == 1):
                                label = 1
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
                            if (label_gt == 2 and label == 1) or (label_gt == 1 and label == 2):
                                label = 2
                                if label in color_map:
                                    x_start = j * patch_width
                                    y_start = i * patch_height
                                    x_end = x_start + patch_width
                                    y_end = y_start + patch_height

                                    overlay = visualized_img[y_start:y_end, x_start:x_end].copy()
                                    overlay[:, :, :3] = color_map[label][:3]

                                    if label == 2:
                                        alpha = 0.5
                                        cv2.addWeighted(overlay, alpha, visualized_img[y_start:y_end, x_start:x_end], 1 - alpha, 0, visualized_img[y_start:y_end, x_start:x_end])
                                    else:
                                        visualized_img[y_start:y_end, x_start:x_end] = overlay
                            
                            if label in color_map:
                                x_start = j * patch_width
                                y_start = i * patch_height
                                x_end = x_start + patch_width
                                y_end = y_start + patch_height

                                overlay = visualized_img_raw[y_start:y_end, x_start:x_end].copy()
                                overlay[:, :, :3] = color_map[label][:3]

                                
                                alpha = 0.5
                                cv2.addWeighted(overlay, alpha, visualized_img_raw[y_start:y_end, x_start:x_end], 1 - alpha, 0, visualized_img_raw[y_start:y_end, x_start:x_end])
                                
                    # for i in range(16):
                    #     for j in range(16):
                    #         label = gt_map[i, j]
                    #         if label in color_map:
                    #             x_start = j * patch_width
                    #             y_start = i * patch_height
                    #             x_end = x_start + patch_width
                    #             y_end = y_start + patch_height

                    #             overlay = visualized_gt[y_start:y_end, x_start:x_end].copy()
                    #             overlay[:, :, :3] = color_map[label][:3]
                    #             if label == 1:
                    #                 alpha = 0.5
                    #                 cv2.addWeighted(overlay, alpha, visualized_gt[y_start:y_end, x_start:x_end], 1 - alpha, 0, visualized_gt[y_start:y_end, x_start:x_end])
                    #             else:
                    #                 visualized_gt[y_start:y_end, x_start:x_end] = overlay
                    
                    # shutil.copy(src_filename, img_dir+'/'+dst_filename.split("/")[-1])
                    # dst_filename_ = dst_filename.replace("results","results_gt")
                    
                    
                    # new_img = np.concatenate((visualize_raw_img,visualized_gt,visualized_img),axis=1)
                    new_img = np.concatenate((visualize_raw_img,visualized_img),axis=1)
                    new_img_raw = np.concatenate((visualize_raw_img,visualized_img_raw),axis=1)
                    cv2.imwrite(dst_filename, new_img)
                    cv2.imwrite(dst_filename_raw, new_img_raw)
                    import pdb;pdb.set_trace()
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
    img_path = '/media/spalab/sdd/kypark/PatchModel/data/hyundae/img_dir/val_2nd'
    
    wrapper = MMSegWrapper()
    for img_path in Path(img_path).glob('*'):
        if img_path.suffix in ['.jpg', '.png', '.jpeg']:
            dst_filename, result_dict = wrapper.get_result(str(img_path))
            print(f"Segmentation result saved at: {dst_filename}")
