#!/bin/bash

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 \
--learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 \
 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True "
SAMPLE_FLAGS="--batch_size 256 --num_samples 50000 --timestep_respacing 250"
#TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --schedule_sampler loss-second-moment"


cmd="python image_sample.py --model_path models/64x64_diffusion.pt  --logdir runs ${MODEL_FLAGS}  ${SAMPLE_FLAGS}"
echo ${cmd}
eval ${cmd}