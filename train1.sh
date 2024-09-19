# CONFIG=patchnet_cxN_convnextv2_wo_pretrain
# CONFIG=patchnet_cxNv2_convnextv2_wo_pretrain
# CONFIG=patchnet_ddrnet
# CONFIG=patchnet_r34_convnextv2_wo_pretrain
# CONFIG=patchnet_rx34_convnextv2_wo_pretrain
# CONFIG=patchnet_ixN_convnextv2_wo_pretrain
# CONFIG=patchnet_CSPDark_convnextv2
# CONFIG=patchnet_ddrnet_wo_pretrain
# CONFIG=patchnet_CSPDark_convnextv2_blurweight
# CONFIG=patchnet_CSPDark_s16_convnextv2
# CONFIG_PATH=patchnet
# PORT=29504 CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 tools/dist_train.sh configs/$CONFIG_PATH/$CONFIG.py 1 --work-dir work_dir/$CONFIG_PATH/$CONFIG
# PORT=29504 CUDA_VISIBLE_DEVICES=0,1,2 CUDA_LAUNCH_BLOCKING=1 tools/dist_train.sh configs/$CONFIG_PATH/$CONFIG.py 3 --work-dir work_dir/$CONFIG_PATH/$CONFIG
# PORT=29500 tools/dist_test.sh configs/$CONFIG_PATH/$CONFIG.py ./work_dir/$CONFIG_PATH/$CONFIG/iter_4000.pth 1 
# PORT=29500 tools/dist_test.sh configs/$CONFIG_PATH/$CONFIG.py work_dir/patchnet/patchnet_hyundae_r10_512x512_wo_aug_lr5e-4_bs4_iter80k/iter_64000.pth 1 




# CONFIG=patchnet_CSPDark_convnextv2
# CONFIG=patchnet_CSPDark_convnextv2_maxpool
# CONFIG=patchnet_ddrnet_wo_pretrain
# CONFIG=patchnet_CSPDark_convnextv2_downsample32
# CONFIG=patchnet_ddrnet_wo_pretrain_512
CONFIG=patchnet_CSPDark_s16_convnextv2
# CONFIG=patchnet_r34_s16_convnextv2_wo_pretrain
CONFIG_PATH=patchnet
# PORT=29504 CUDA_VISIBLE_DEVICES=0,1,2 CUDA_LAUNCH_BLOCKING=1 tools/dist_train.sh configs/$CONFIG_PATH/$CONFIG.py 3 --work-dir work_dir/$CONFIG_PATH/$CONFIG
# PORT=29500 tools/dist_test.sh configs/$CONFIG_PATH/$CONFIG.py ./work_dir/$CONFIG_PATH/patchnet_ddrnet_wo_pretrain_512/iter_40000.pth 1 
PORT=29501 CUDA_VISIBLE_DEVICES=2 tools/dist_test.sh configs/$CONFIG_PATH/$CONFIG.py ./work_dir/$CONFIG_PATH/$CONFIG/iter_40000.pth 1 

# CONFIG=patchnet_r101_convnextv2_wo_pretrain
# CONFIG_PATH=patchnet
# PORT=29504 CUDA_VISIBLE_DEVICES=0,1,2 CUDA_LAUNCH_BLOCKING=1 tools/dist_train.sh configs/$CONFIG_PATH/$CONFIG.py 3 --work-dir work_dir/$CONFIG_PATH/$CONFIG
# PORT=29504 CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 tools/dist_train.sh configs/$CONFIG_PATH/$CONFIG.py 1 --work-dir work_dir/$CONFIG_PATH/$CONFIG
