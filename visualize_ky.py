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
        config_file = 'configs/patchnet/dylee/r34.py'
        checkpoint_file = './work_dirs/hmc_5000_2nd/dilation_1357_cls_w_30_2/iter_72000.pth'
        self.model = init_model(config_file, checkpoint_file, device='cuda:0')

    def get_result(self, src_filename):
        if self.model:
            result_dict = {}
            ext = Path(src_filename).suffix
            save_dir = '/mnt/hdd2/project/PatchModel/data/human_anno/inf_result'
            os.makedirs(save_dir, exist_ok=True)
            save_dir_raw = '/mnt/hdd2/project/PatchModel/data/hyundae/img_dir/val_2nd'
            os.makedirs(save_dir_raw, exist_ok=True)
            dst_filename = os.path.join(save_dir, f'{Path(src_filename).stem}_result{ext}')
            
            if ext in ['.jpg', '.png', '.jpeg']:
                img = mmcv.imread(src_filename)
                original_size = img.shape[:2]
                resized_img = cv2.resize(img, (512, 512))
                result = inference_model(self.model, resized_img)
                seg_map = result.pred_sem_seg.data.cpu().numpy().squeeze()

                # Resize segmentation map back to original size
                seg_map_resized = cv2.resize(seg_map, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)

                # Prepare overlay on the image using the segmentation map
                color_map = {
                    1: (255, 191, 0),
                    2: (225, 105, 65) 
                }
                # def draw_grid(image, grid_size=16, color=(0, 255, 0), thickness=2):
                #     # Open the image
                #     h, w = image.shape[:2]

                #     # Draw vertical lines
                #     for x in range(0, w, w // grid_size):
                #         cv2.line(image, (x, 0), (x, h), color, thickness)
                    
                #     # Draw horizontal lines
                #     for y in range(0, h, h // grid_size):
                #         cv2.line(image, (0, y), (w, y), color, thickness)
                #     return image
                
                # img = draw_grid(img)
                overlay = img.copy()
                breakpoint()

                # Apply the segmentation map as an overlay
                for i in range(seg_map_resized.shape[0]):
                    for j in range(seg_map_resized.shape[1]):
                        label = seg_map_resized[i, j]
                        if label in color_map:
                            overlay[i, j] = color_map[label]

                # Blend the original image and the overlay
                alpha = 0.5  # Transparency factor
                cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

                # Save the final image with the segmentation map applied
                cv2.imwrite(dst_filename, img)

                return dst_filename
        else:
            raise Exception('You have to call download_model first.')

# if __name__ == "__main__":
#     # 테스트할 이미지 경로
#     #img_path = '/media/spalab/sdd/kypark/PatchModel/data/hyundae/img_dir/val_2nd'
#     img_path = '/mnt/hdd2/project/PatchModel/data/human_anno/images'
    
#     wrapper = MMSegWrapper()
#     for img_path in Path(img_path).glob('*'):
#         if img_path.suffix in ['.jpg', '.png', '.jpeg']:
#             dst_filename, result_dict = wrapper.get_result(str(img_path))
#             print(f"Segmentation result saved at: {dst_filename}")

if __name__ == "__main__":
    # 테스트할 이미지 경로
    img_path = '/mnt/hdd2/project/PatchModel/data/hyundae/img_dir/train_2nd/CMR_GT_Frame-N2207413-230206170557-ADAS_DRV3-RR_SD_CMR_LH-001-00000735.png'
    wrapper = MMSegWrapper()
    dst_filename = wrapper.get_result(str(img_path))
    print(f"Segmentation result saved at: {dst_filename}")