#!/bin/bash


MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 \
 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 \
  --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"


SAMPLE_FLAGS="--batch_size 8 --num_samples 50000 --timestep_respacing 250"

cmd="cd ../../"
echo ${cmd}
eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}

scales=("0" "2" "4" "6" "8" "10")

for scale in "${scales[@]}"
do
cmd="python evaluations/evaluator_tolog.py reference/VIRTUAL_imagenet128_labeled.npz \
 runs/sampling/IMN128/conditional_mlt_divg/scale${scale}p0/reference/samples_10000x128x128x3.npz"
echo ${cmd}
eval ${cmd}
done

cmd="python evaluations/evaluator_tolog.py reference/VIRTUAL_imagenet128_labeled.npz \
 runs/sampling/IMN128/conditional_mlt_divg/scale0p5/reference/samples_10000x128x128x3.npz"
echo ${cmd}
eval ${cmd}


#cmd="python evaluations/evaluator_tolog.py reference/VIRTUAL_imagenet256_labeled.npz \
# runs/sampling/IMN256/conditional/scale10p0/reference/samples_50000x256x256x3.npz"
#echo ${cmd}
#eval ${cmd}
#
#cmd="python evaluations/evaluator_tolog.py reference/VIRTUAL_imagenet256_labeled.npz \
# runs/sampling/IMN256/conditional/scale10p0/reference/samples_50000x256x256x3.npz"
#echo ${cmd}
#eval ${cmd}

