#!/bin/bash

TRAIN_FLAGS="--iterations 200000 --anneal_lr True --batch_size 32 --lr 3e-4 --save_interval 50000 --weight_decay 0.2"
CLASSIFIER_FLAGS="--image_size 256 --classifier_attention_resolutions 16,8,4 --classifier_depth 3 \
--classifier_width 128 --classifier_pool attention --classifier_resblock_updown True \
 --classifier_use_scale_shift_norm True --alphae 0.2"

#cmd="cd ../../"
#echo ${cmd}
#eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}


cmd="python scripts_hfai_local/celeba/classifier_localtrain_info_celeba_schm7.py --data_dir path/to/imagenet --logdir runs/celeba32/celeba32_info7/ \
 $TRAIN_FLAGS $CLASSIFIER_FLAGS"
echo ${cmd}
eval ${cmd}