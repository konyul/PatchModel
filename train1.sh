# CONFIG=patchnet_cxN_convnextv2_wo_pretrain
# CONFIG=patchnet_cxNv2_convnextv2_wo_pretrain
# CONFIG=patchnet_ddrnet
# CONFIG=patchnet_r34_convnextv2_wo_pretrain
# CONFIG=patchnet_rx34_convnextv2_wo_pretrain
# CONFIG=patchnet_ixN_convnextv2_wo_pretrain
CONFIG=patchnet_CSPDark_convnextv2
CONFIG_PATH=patchnet
PORT=29504 CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 tools/dist_train.sh configs/$CONFIG_PATH/$CONFIG.py 1 --work-dir work_dir/$CONFIG_PATH/$CONFIG
# PORT=29504 CUDA_VISIBLE_DEVICES=0,1,2 CUDA_LAUNCH_BLOCKING=1 tools/dist_train.sh configs/$CONFIG_PATH/$CONFIG.py 3 --work-dir work_dir/$CONFIG_PATH/$CONFIG
# PORT=29500 tools/dist_test.sh configs/$CONFIG_PATH/$CONFIG.py ./work_dir/$CONFIG_PATH/$CONFIG/iter_4000.pth 1 
# PORT=29500 tools/dist_test.sh configs/$CONFIG_PATH/$CONFIG.py work_dir/patchnet/patchnet_hyundae_r10_512x512_wo_aug_lr5e-4_bs4_iter80k/iter_64000.pth 1 

# CONFIG=patchnet_hyundae_r10_512x512_wo_aug_lr5e-5_bs8_iter80k_singlehead
# CONFIG_PATH=patchnet
# PORT=29502 CUDA_LAUNCH_BLOCKING=1 tools/dist_train.sh configs/$CONFIG_PATH/$CONFIG.py 1 --work-dir work_dir/$CONFIG_PATH/$CONFIG

# CONFIG=patchnet_hyundae_r10_512x512_wo_aug_lr5e-5_bs2_iter20k_singlehead
# CONFIG_PATH=patchnet
# PORT=29501 CUDA_LAUNCH_BLOCKING=1 tools/dist_train.sh configs/$CONFIG_PATH/$CONFIG.py 1 --work-dir work_dir/$CONFIG_PATH/$CONFIG

# CONFIG=patchnet_hyundae_r10_512x512_wo_aug_lr5e-3_bs2_iter20k_singlehead
# CONFIG_PATH=patchnet
# PORT=29501 CUDA_LAUNCH_BLOCKING=1 tools/dist_train.sh configs/$CONFIG_PATH/$CONFIG.py 1 --work-dir work_dir/$CONFIG_PATH/$CONFIG

