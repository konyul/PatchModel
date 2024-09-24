# CONFIG=patchnet_r10_1xb4_cityscapes-512x512
CONFIG=patchnet_r34_s16_FPN_convnextv2_wo_pretrain_blurweight
CONFIG_PATH=patchnet

PORT=29501 TORCH_DISTRIBUTED_DEBUG=DETAIL CUDA_VISIBLE_DEVICES=2,3 CUDA_LAUNCH_BLOCKING=3 tools/dist_train.sh configs/$CONFIG_PATH/$CONFIG.py 2 --work-dir work_dir/$CONFIG_PATH/$CONFIG
# PORT=29501 CUDA_VISIBLE_DEVICES=3 tools/dist_test.sh configs/$CONFIG_PATH/$CONFIG.py ./work_dir/$CONFIG_PATH/patchnet_ddrnet_wo_pretrain/iter_16000.pth 1 

