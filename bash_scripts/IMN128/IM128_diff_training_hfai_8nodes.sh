#!/bin/bash

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --image_size 128 \
 --learn_sigma True  --num_channels 256  --num_res_blocks 3 \
  --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine"
TRAIN_FLAGS="--lr 3e-4 --batch_size 16 --lr_anneal_steps 1090000 --dropout 0.0 --attention_resolutions 32,16,8 \
 --num_heads 4 --save_interval 50000"
#total batch size = 32 * 8 * = 2048
cmd="cd ../../"
echo ${cmd}
eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}


cmd="python scripts_hfai_gdiff/image_train.py --data_dir path/to/imagenet --logdir runs/IM128/IM128_diffusion_unconditional/ \
 $TRAIN_FLAGS $DIFFUSION_FLAGS $MODEL_FLAGS"
echo ${cmd}
eval ${cmd}