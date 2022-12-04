#!/bin/bash

MODEL_FLAGS="--image_size 28 --num_channels 32 --num_res_blocks 2 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine "
TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --lr_anneal_steps 200000 --attention_resolutions 7,4 \
--dropout 0.1 --num_head_channels 8 --learn_sigma True"

cd_cmd="CUDA_VISIBLE_DEVICES=0"

cmd="cd ../../"
echo ${cmd}
#eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}


cmd="${cd_cmd} python scripts_hfai_local/mnist/image_train_mnist.py --data_dir path/to/imagenet --logdir ../output/runs/mnist/mnist_diff_sigma/ \
 $TRAIN_FLAGS $DIFFUSION_FLAGS $MODEL_FLAGS"
echo ${cmd}
eval ${cmd}