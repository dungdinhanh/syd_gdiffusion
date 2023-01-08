#!/bin/bash


MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 512 \
--learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 \
 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"


SAMPLE_FLAGS="--batch_size 4 --num_samples 50000 --timestep_respacing 250"


cmd="cd ../../../"
echo ${cmd}
eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}

scales=( "0.1" "0.5" "0.7" )

for scale in "${scales[@]}"
do
cmd="python  scripts_hfai_gdiff/multitask/classifier_sample_mlt2_largeres.py  $MODEL_FLAGS --classifier_scale ${scale} \
 --classifier_path models/512x512_classifier.pt --model_path models/512x512_diffusion.pt $SAMPLE_FLAGS \
 --logdir runs/sampling_icml/IMN512/conditional_cdiff/scale${scale}/ --half_save True"
echo ${cmd}
eval ${cmd}
done






#cmd="python scripts_hfai_gdiff/classifier_free_sample.py --logdir runs/classifier_pretrained/ ${MODEL_FLAGS} --classifier_scale 1.0 --classifier_path models/64x64_classifier.pt \
# --classifier_depth 4 --model_path models/64x64_diffusion.pt ${SAMPLE_FLAGS}"
