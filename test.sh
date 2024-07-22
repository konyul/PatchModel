CONFIG=patchnet_hyundae_r10_512x512_wo_aug_lr5e-5_bs8_iter80k_singlehead #config name
CONFIG_PATH=patchnet

PORT=29501 tools/dist_test.sh configs/$CONFIG_PATH/$CONFIG.py ./work_dir/$CONFIG_PATH/$CONFIG/iter_16000.pth 1 --out ./work_dir/$CONFIG_PATH/$CONFIG/pred_result.pkl
python tools/analysis_tools/confusion_matrix.py configs/$CONFIG_PATH/$CONFIG.py ./work_dir/$CONFIG_PATH/$CONFIG/pred_result.pkl ./work_dir/$CONFIG_PATH/$CONFIG/confusion_matrix