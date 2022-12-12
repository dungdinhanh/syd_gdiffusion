#!/bin/bash

SAMPLE_FLAGS="--batch_size 128 --num_samples 50000 --timestep_respacing 250"
#TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --schedule_sampler loss-second-moment"

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 64 \
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

#scales=("6" "8" "10")
pscales=("1" "2" "3" "4")

for pscale in "${pscales[@]}"
do
cmd="python scripts_hfai_gdiff/multitask/classifier_sample_mlt_cdiv.py $MODEL_FLAGS --classifier_scale 0.${pscale}  \
--classifier_path models/64x64_classifier.pt \
 --model_path runs/IM64/IM64_diffusion_training_unconditional/models/ema_0.9999_540000_final.pt \
 $SAMPLE_FLAGS --logdir runs/sampling2/IMN64/unconditional_mlt2_cdiv/scale0p${pscale}/ --classifier_depth 4"
echo ${cmd}
eval ${cmd}
done

