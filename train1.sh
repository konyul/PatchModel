# CONFIG=patchnet_r10_1xb4_cityscapes-512x512
CONFIG=patchnet_hyundae_r10_512x512_wo_aug_lr5e-3_bs4_iter80k_singlehead
CONFIG_PATH=patchnet


CUDA_LAUNCH_BLOCKING=1 tools/dist_train.sh configs/$CONFIG_PATH/$CONFIG.py 1 --work-dir work_dir/$CONFIG_PATH/$CONFIG
# PORT=29501 tools/dist_test.sh configs/$CONFIG_PATH/$CONFIG.py ./work_dir/$CONFIG_PATH/$CONFIG/iter_2000.pth 1 