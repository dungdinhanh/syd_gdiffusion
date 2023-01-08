#!/bin/bash


MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 \
 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 \
  --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"


SAMPLE_FLAGS="--batch_size 8 --num_samples 50000 --timestep_respacing 250"


cmd="cd ../../../"
echo ${cmd}
eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}

scales=( "6.0" "8.0" "10.0")

for scale in "${scales[@]}"
do
cmd="python  scripts_hfai_gdiff/multitask/classifier_sample_mlt_cdiv2.py  $MODEL_FLAGS --classifier_scale ${scale} \
 --classifier_path models/256x256_classifier.pt --model_path models/256x256_diffusion.pt $SAMPLE_FLAGS \
 --logdir runs/sampling_icml/IMN256/conditional/scale${scale}/"
echo ${cmd}
eval ${cmd}
done






#cmd="python scripts_hfai_gdiff/classifier_free_sample.py --logdir runs/classifier_pretrained/ ${MODEL_FLAGS} --classifier_scale 1.0 --classifier_path models/64x64_classifier.pt \
# --classifier_depth 4 --model_path models/64x64_diffusion.pt ${SAMPLE_FLAGS}"
