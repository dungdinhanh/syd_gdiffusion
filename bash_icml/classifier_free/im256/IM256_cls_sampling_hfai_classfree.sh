#!/bin/bash


MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 \
 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 \
  --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"


SAMPLE_FLAGS="--batch_size 8 --num_samples 50000 --timestep_respacing 250"


cmd="cd ../../../"
echo ${cmd}
eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}

scales=( "1.0" "2.0" "4.0" "6.0")

for scale in "${scales[@]}"
do
cmd="python scripts_hfai_gdiff/classifier_free/classifier_free_sample.py $MODEL_FLAGS --cond_model_scale ${scale}  \
--cond_model_path models/256x256_diffusion.pt \
--model_path models/256x256_diffusion_uncond.pt  $SAMPLE_FLAGS \
 --logdir runs/sampling_icml_classifier_free/IMN256/normal/scale${scale}/ "
echo ${cmd}
eval ${cmd}
done


for scale in "${scales[@]}"
do
cmd="python evaluations/evaluator_tolog.py reference/VIRTUAL_imagenet256_labeled.npz \
 runs/sampling_icml_classifier_free/IMN256/normal/scale${scale}/reference/samples_50000x256x256x3.npz"
echo ${cmd}
eval ${cmd}
done

