#!/bin/sh
if [ $1 == 1 ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -W ignore train_wo_flow_fullbody.py \
        --outdir ./training-runs-wo_flow-full-fullbody_1122 \
        --data /datazy/Datasets/UPT_256 \
        --gpus 8 --cfg fashion \
        --cond true --batch 96 --l1_weight 40 \
        --vgg_weight 40 --use_noise_const_branch True \
        --workers 4 --contextual_weight 0 --pl_weight 0 \
        --mask_weight 50
fi
