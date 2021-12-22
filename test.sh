#!/bin/sh
if [ $1 == 1 ]; then
    CUDA_VISIBLE_DEVICES=0 python3 -W ignore test.py \
    --network /datazy/Codes/PASTA-GAN/PASTA-GAN_fullbody_model/network-snapshot-004000.pkl \
    --outdir /datazy/Datasets/pasta-gan_results/unpaired_results_fulltryonds \
    --dataroot /datazy/Datasets/PASTA_UPT_256 \
    --batchsize 16
elif [ $1 == 2 ]; then
    CUDA_VISIBLE_DEVICES=0 python3 -W ignore test_512.py \
    --network /datazy/Codes/PASTA-GAN/PASTA-GAN_fullbody_model_512/network-snapshot-005010.pkl \
    --outdir /datazy/Datasets/pasta-gan_results/unpaired_results_512 \
    --dataroot /datazy/Datasets/PASTA_UPT_512 \
    --batchsize 8
fi