#!/bin/bash

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --image_size 64 \
 --learn_sigma True  --num_channels 192  --num_res_blocks 3 \
  --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine"
TRAIN_FLAGS="--lr 3e-4 --batch_size 32 --lr_anneal_steps 540000 --dropout 0.1 --attention_resolutions 32,16,8 \
 --num_head_channels 64"

cmd="cd .."
echo ${cmd}
eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}


cmd="python scripts_hfai_gdiff/image_train.py --data_dir path/to/imagenet --logdir runs/diffusion_training_8nodes/ \
 $TRAIN_FLAGS $DIFFUSION_FLAGS $MODEL_FLAGS"
echo ${cmd}
eval ${cmd}