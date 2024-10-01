# CONFIG=patchnet_hyundae_r18_512x512_wo_aug_lr5e-3_bs4_iter80k #config name
# CONFIG_PATH=patchnet

# for ITER in 40000
# do
#   PORT=29501 tools/dist_test.sh ./configs/patchnet/dylee/r34.py ./work_dir/$CONFIG_PATH/$CONFIG/iter_$ITER.pth 1 >> ./work_dir/$CONFIG_PATH/$CONFIG/iter_$ITER.txt
# done

PORT=2951 tools/dist_test.sh ./configs/patchnet_midterm/oldver_r34_pre_o_aug_o.py ./work_dir/patchnet_midterm/oldver_r34_pre_o_aug_o/iter_8000.pth 1 
# python tools/analysis_tools/confusion_matrix.py configs/$CONFIG_PATH/$CONFIG.py ./work_dir/$CONFIG_PATH/$CONFIG/pred_result.pkl ./work_dir/$CONFIG_PATH/$CONFIG/confusion_matrix