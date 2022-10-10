#!/bin/bash

TRAIN_FLAGS="--iterations 300000 --anneal_lr True --batch_size 32 --lr 3e-4 --save_interval 10000 --weight_decay 0.05"
CLASSIFIER_FLAGS="--image_size 64 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 \
--classifier_width 128 --classifier_pool attention --classifier_resblock_updown True \
 --classifier_use_scale_shift_norm True"

cmd="cd .."
echo ${cmd}
eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}


cmd="python scripts_hfai_gdiff/classifier_train.py --data_dir path/to/imagenet $TRAIN_FLAGS $CLASSIFIER_FLAGS"
echo ${cmd}
eval ${cmd}