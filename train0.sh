CONFIG=patchnet_ddrnet
CONFIG_PATH=patchnet
PORT=29501 tools/dist_train.sh configs/$CONFIG_PATH/$CONFIG.py 1 --work-dir work_dir/$CONFIG_PATH/$CONFIG

# CONFIG=patchnet_cxN_convnextv2_wo_pretrain
# CONFIG_PATH=patchnet
# PORT=29501 tools/dist_train.sh configs/$CONFIG_PATH/$CONFIG.py 4 --work-dir work_dir/$CONFIG_PATH/$CONFIG


# PORT=29500 tools/dist_test.sh configs/$CONFIG_PATH/$CONFIG.py ./work_dir/$CONFIG_PATH/$CONFIG/iter_2000.pth 1 

# CONFIG=patchnet_hyundae_r10_512x512_wo_aug_lr5e-5_bs8_iter80k_singlehead
# CONFIG_PATH=patchnet
# PORT=29502 CUDA_LAUNCH_BLOCKING=1 tools/dist_train.sh configs/$CONFIG_PATH/$CONFIG.py 1 --work-dir work_dir/$CONFIG_PATH/$CONFIG

# CONFIG=patchnet_hyundae_r10_512x512_wo_aug_lr5e-5_bs2_iter20k_singlehead
# CONFIG_PATH=patchnet
# PORT=29501 CUDA_LAUNCH_BLOCKING=1 tools/dist_train.sh configs/$CONFIG_PATH/$CONFIG.py 1 --work-dir work_dir/$CONFIG_PATH/$CONFIG

# CONFIG=patchnet_hyundae_r10_512x512_wo_aug_lr5e-3_bs2_iter20k_singlehead
# CONFIG_PATH=patchnet
# PORT=29501 CUDA_LAUNCH_BLOCKING=1 tools/dist_train.sh configs/$CONFIG_PATH/$CONFIG.py 1 --work-dir work_dir/$CONFIG_PATH/$CONFIG

