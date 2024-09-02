CONFIG=patchnet_hyundae_r18_512x512_wo_aug_lr5e-3_bs4_iter80k #config name
CONFIG_PATH=patchnet

for ITER in 40000
do
  PORT=29501 tools/dist_test.sh configs/$CONFIG_PATH/$CONFIG.py ./work_dir/$CONFIG_PATH/$CONFIG/iter_$ITER.pth 1 >> ./work_dir/$CONFIG_PATH/$CONFIG/iter_$ITER.txt
done

PORT=29501 tools/dist_test.sh configs/$CONFIG_PATH/$CONFIG.py ./work_dir/$CONFIG_PATH/$CONFIG/iter_16000.pth 1 --out ./work_dir/$CONFIG_PATH/$CONFIG/pred_result.pkl
# python tools/analysis_tools/confusion_matrix.py configs/$CONFIG_PATH/$CONFIG.py ./work_dir/$CONFIG_PATH/$CONFIG/pred_result.pkl ./work_dir/$CONFIG_PATH/$CONFIG/confusion_matrix