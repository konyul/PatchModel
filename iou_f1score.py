import json
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

def calculate_f1_score(anno_patch, inf_patch):
    anno_flat = anno_patch.flatten()
    inf_flat = inf_patch.flatten()
    f1_per_class = f1_score(anno_flat, inf_flat, labels=[0, 1, 2], average=None)
    mean_f1 = np.mean(f1_per_class)  # 클래스별 F1-score의 평균
    return f1_per_class, mean_f1

def calculate_miou(anno_patch, inf_patch):
    anno_flat = anno_patch.flatten()
    inf_flat = inf_patch.flatten()
    conf_matrix = confusion_matrix(anno_flat, inf_flat, labels=[0, 1, 2])
    
    ious = []
    for i in range(3): 
        intersection = conf_matrix[i, i]
        union = np.sum(conf_matrix[i, :]) + np.sum(conf_matrix[:, i]) - intersection  # Union = TP + FP + FN
        if union == 0: 
            iou = 0
        else:
            iou = intersection / union
        ious.append(iou)
    miou = np.mean(ious)
    
    return ious, miou, conf_matrix

# person = ['dy', 'ky', 'dh', 'inf_1', 'inf_2']
person = ['inf_1', 'inf_2']
img_name = "CMR_GT_Frame-N2207413-230206170557-ADAS_DRV3-RR_SD_CMR_LH-001-00000735"
inf_file_path = f"/media/spalab/sdd/kypark/PatchModel/inference/{img_name}.npy"

inf_patch = np.load(inf_file_path)

for p in person:
    GT_file_path = f"/media/spalab/sdd/kypark/PatchModel/compare_iou_f1score/GT_1/{img_name}.json"
    
    with open(GT_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)   
    patch_array = np.array(data['patch'])
    anno_patch = patch_array[:, :, 1]
    
    print(f"\nPerson: {p}")
    f1_per_class, mean_f1 = calculate_f1_score(anno_patch, inf_patch)
    for i, f1 in enumerate(f1_per_class):
        print(f"Class {i} F1 Score: {round(f1,2)}")
    print(f"Mean F1 Score: {round(mean_f1,2)}")

    ious, miou, conf_matrix = calculate_miou(anno_patch, inf_patch)
    for i, iou in enumerate(ious):
        print(f"Class {i} IoU: {round(iou,2)}")
    print(f"Mean IoU: {round(miou,2)}")
    
    print(f"Confusion Matrix for {p}:")
    print(conf_matrix)
