#!/bin/bash

SAMPLE_FLAGS="--batch_size 512 --num_samples 50000 --timestep_respacing 250"
#TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --schedule_sampler loss-second-moment"

MODEL_FLAGS="--attention_resolutions 16,8,4 --class_cond False --diffusion_steps 1000  \
--image_size 32 --noise_schedule cosine --num_channels 192 \
--num_head_channels 64 --num_res_blocks 3"

cmd="cd ../.."
echo ${cmd}
eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}

sample_model_path="runs/celeba32/celeba32_info9"

cmd="python scripts_hfai_gdiff/classifier_sample_infoq.py --logdir ${sample_model_path}/ref/ ${MODEL_FLAGS} \
--classifier_scale 5.0 --classifier_attention_resolutions 16,8,4 --classifier_path ${sample_model_path}/models/model199999.pt \
 --classifier_depth 3 --model_path runs/celeba32/celeba32_diff/models/ema_0.9999_200000.pt ${SAMPLE_FLAGS}"
echo ${cmd}
eval ${cmd}