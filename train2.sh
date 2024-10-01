# configs/patchnet_midterm/r18_pre_x_aug_x.py
# CONFIG=r18_pre_o_aug_o
# CONFIG=r18_pre_o_aug_x
CONFIG=r18_pre_x_aug_x
# CONFIG=r34_pre_o_aug_o_dilation_o_wl_o
# CONFIG=r34_pre_o_aug_o_dilation_o_wl_x
# CONFIG=r34_pre_o_aug_o_dilation_x_wl_x
# CONFIG=r34_pre_o_aug_x_dilation_x_wl_x
# CONFIG=r34_pre_x_aug_x_dilation_x_wl_x

CONFIG_PATH=patchnet_midterm

PORT=29501 TORCH_DISTRIBUTED_DEBUG=DETAIL CUDA_VISIBLE_DEVICES=3 CUDA_LAUNCH_BLOCKING=3 tools/dist_train.sh configs/$CONFIG_PATH/$CONFIG.py 1 --work-dir work_dir/$CONFIG_PATH/$CONFIG
# PORT=29501 CUDA_VISIBLE_DEVICES=3 tools/dist_test.sh configs/$CONFIG_PATH/$CONFIG.py ./work_dir/$CONFIG_PATH/patchnet_ddrnet_wo_pretrain/iter_16000.pth 1 

