CONFIG=patchnet_r34 #config name
CONFIG_PATH=patchnet

tools/dist_test.sh configs/patchnet/patchnet_r34.py ./work_dirs/hmc_5000_pretrained_conv3x3_dilated_x2/iter_36000.pth 1 --out ./work_dirs/pred_result.pkl
python tools/analysis_tools/confusion_matrix.py configs/patchnet/patchnet_r34.py ./work_dirs/pred_result.pkl ./work_dirs/hmc_5000_pretrained_conv3x3_dilated_x2/confusion_matrix