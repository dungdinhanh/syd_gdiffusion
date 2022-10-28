#!/bin/bash

TRAIN_FLAGS="--iterations 200000 --anneal_lr True --batch_size 32 --lr 3e-4 --save_interval 50000 --weight_decay 0.2"
CLASSIFIER_FLAGS="--image_size 32 --classifier_attention_resolutions 16,8,4 --classifier_depth 3 \
--classifier_width 128 --classifier_pool attention --classifier_resblock_updown True \
 --classifier_use_scale_shift_norm True --alphae 0.2"

cmd="cd ../../"
echo ${cmd}
eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}

# later change name between 9 and 11
cmd="python scripts_hfai_gdiff/celeba32/classifier_train_info_celeba_schm11.py --data_dir path/to/imagenet \
    --logdir runs/celeba32info/celeba32_info11/ $TRAIN_FLAGS $CLASSIFIER_FLAGS"
echo ${cmd}
eval ${cmd}