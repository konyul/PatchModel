
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
        config_file = './configs/patchnet/dylee/r34.py'
        checkpoint_file = './work_dirs/hmc_5000_2nd/dilation_1357_cls_w_30_2/iter_72000.pth'
        self.model = init_model(config_file, checkpoint_file, device='cuda:0')

    def get_result(self, src_filename):
        if self.model:
            try:
                ext = Path(src_filename).suffix
                save_dir = '/mnt/hdd2/project/PatchModel/data/human_anno/inference'
                os.makedirs(save_dir, exist_ok=True)
                dst_filename = os.path.join(save_dir, f'{Path(src_filename).stem}.npy')
                
                if ext in ['.jpg', '.png', '.jpeg']:
                    img = mmcv.imread(src_filename)
                    original_size = img.shape[:2]
                    resized_img = cv2.resize(img, (512, 512))
                    result = inference_model(self.model, resized_img)
                    seg_map = result.pred_sem_seg.data.cpu().numpy().squeeze()
                    print("Segmentation result shape (before resize):", seg_map.shape)
                    
                    # Save seg_map as an .npy file
                    np.save(dst_filename, seg_map)
                    print(f"Segmentation result saved at: {dst_filename}")
                    return dst_filename
                else:
                    raise Exception('Unsupported file type.')
            except Exception as e:
                raise Exception(e)
        else:
            raise Exception('You have to call download_model first.')

if __name__ == "__main__":
    # 테스트할 이미지 경로
    img_path = 'data/human_anno/images/CMR_GT_Frame-N2207413-230209153942-ADAS_DRV3-RR_SD_CMR_RH-001-00000675.png'
    wrapper = MMSegWrapper()
    dst_filename = wrapper.get_result(str(img_path))
    print(f"Segmentation result saved at: {dst_filename}")