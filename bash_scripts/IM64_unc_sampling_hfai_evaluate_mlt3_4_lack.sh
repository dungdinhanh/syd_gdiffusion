#!/bin/bash


MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 \
 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 \
  --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"


SAMPLE_FLAGS="--batch_size 8 --num_samples 10000 --timestep_respacing 250"

cmd="cd ../"
echo ${cmd}
eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}

pscales=("7")

for pscale in "${pscales[@]}"
do

cmd="python evaluations/evaluator_tolog.py reference/VIRTUAL_imagenet64_labeled.npz \
 runs/sampling2/IMN64/unconditional/scale0p${pscale}/reference/samples_50000x64x64x3.npz"
echo ${cmd}
eval ${cmd}


cmd="python evaluations/evaluator_tolog.py reference/VIRTUAL_imagenet64_labeled.npz \
 runs/sampling2/IMN64/unconditional_mlt2/scale0p${pscale}/reference/samples_50000x64x64x3.npz"
echo ${cmd}
eval ${cmd}

cmd="python evaluations/evaluator_tolog.py reference/VIRTUAL_imagenet64_labeled.npz \
 runs/sampling2/IMN64/unconditional_mlt2_cdiv/scale0p${pscale}/reference/samples_50000x64x64x3.npz"
echo ${cmd}
eval ${cmd}
done

cmd="python evaluations/evaluator_tolog.py reference/VIRTUAL_imagenet64_labeled.npz \
 runs/sampling2/IMN64/unconditional/scale1p0/reference/samples_50000x64x64x3.npz"
echo ${cmd}
eval ${cmd}





