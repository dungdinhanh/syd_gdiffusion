#!/bin/bash

MODEL_FLAGS="--image_size 128 --num_channels 256 --num_res_blocks 2 --resblock_updown True \
--use_new_attention_order True --use_scale_shift_norm True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 4e-4 --batch_size 16 --lr_anneal_steps 500000 --dropout 0.0 --attention_resolutions 32,16,8 \
 --num_heads 4 --learn_sigma True"

cmd="cd ../../"
echo ${cmd}
eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}


cmd="python scripts_hfai_gdiff/celeba64/image_train.py --data_dir path/to/imagenet --logdir runs/celeba64/celeba128_training_diff \
 $TRAIN_FLAGS $DIFFUSION_FLAGS $MODEL_FLAGS"
echo ${cmd}
eval ${cmd}