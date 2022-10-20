#!/bin/bash

TRAIN_FLAGS="--iterations 150000 --anneal_lr True --batch_size 16 --lr 2e-4 --save_interval 10000 --weight_decay 0.2"
CLASSIFIER_FLAGS="--image_size 28 --classifier_attention_resolutions 14,7 --classifier_depth 1 \
--classifier_width 32 --classifier_pool attention --classifier_resblock_updown True \
 --classifier_use_scale_shift_norm True --alphae 0.12"

cmd="cd ../.."
echo ${cmd}
eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}


cmd="python scripts_hfai_gdiff/mnist/classifier_train_info_mnist_schm3.py --data_dir path/to/imagenet --logdir runs/mnist_tuneh/info_schm3_h012/ $TRAIN_FLAGS $CLASSIFIER_FLAGS"
echo ${cmd}
eval ${cmd}