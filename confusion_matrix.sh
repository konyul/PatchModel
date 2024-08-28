CONFIG=patchnet_r34 #config name
CONFIG_PATH=patchnet

tools/dist_test.sh configs/patchnet/patchnet_r34.py ./work_dirs/hmc_5000/iter_40000.pth 1 --out ./work_dirs/pred_result.pkl
python tools/analysis_tools/confusion_matrix.py configs/patchnet/patchnet_r34.py ./work_dirs/pred_result.pkl ./work_dirs/hmc_5000/confusion_matrix