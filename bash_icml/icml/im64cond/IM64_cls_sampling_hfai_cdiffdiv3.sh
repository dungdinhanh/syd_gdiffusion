#!/bin/bash

SAMPLE_FLAGS="--batch_size 128 --num_samples 50000 --timestep_respacing 250"
#TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --schedule_sampler loss-second-moment"

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 \
 --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3\
 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"


#MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 \
# --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 \
#  --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
cmd="cd ../../../"
echo ${cmd}
eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}

scales=( "01" "03" "05" "07" "09" "001" "003" "005" "007" "009" )

for scale in "${scales[@]}"
do
cmd="python scripts_hfai_gdiff/multitask/classifier_sample_mlt_cdiffdiv.py $MODEL_FLAGS --classifier_scale 0.${scale}  \
--classifier_path models/64x64_classifier.pt --model_path models/64x64_diffusion.pt $SAMPLE_FLAGS \
 --logdir runs/sampling_icml/IMN64/conditional_cdiffdiv/scale0p${scale}/ --classifier_depth 4"
echo ${cmd}
eval ${cmd}
done


for scale in "${scales[@]}"
do
cmd="python evaluations/evaluator_tolog.py reference/VIRTUAL_imagenet64_labeled.npz \
 runs/sampling_icml/IMN64/conditional_cdiffdiv/scale0p${scale}/reference/samples_50000x64x64x3.npz"
echo ${cmd}
eval ${cmd}
done

