#!/bin/bash

MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True "
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine --use_kl True"
SAMPLE_FLAGS="--batch_size 4 --num_samples 20 --timestep_respacing 250"
#TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --schedule_sampler loss-second-moment"


cmd="python image_sample.py --model_path models/cifar10_uncond_vlb_50M_500K.pt  --logdir runs ${MODEL_FLAGS} ${DIFFUSION_FLAGS} ${SAMPLE_FLAGS}"
echo ${cmd}
eval ${cmd}