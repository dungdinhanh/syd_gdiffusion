#!/bin/bash

SAMPLE_FLAGS="--batch_size 128 --num_samples 50000 --timestep_respacing 250"


MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 64 \
 --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3\
 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"



cmd="cd ../../../"
echo ${cmd}
eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}

scales=( "0.5" "0.6" "0.7" "0.8" "0.9" )

for scale in "${scales[@]}"
do
cmd="python scripts_hfai_gdiff/classifier_free/classifier_free_sample.py $MODEL_FLAGS --cond_model_scale ${scale}  \
--cond_model_path models/64x64_diffusion.pt \
--model_path runs/IM64/IM64_diffusion_training_unconditional/models/ema_0.9999_540000_final.pt  $SAMPLE_FLAGS \
 --logdir runs/sampling_icml_classifier_free/IMN64/normal/scale${scale}/ "
echo ${cmd}
eval ${cmd}
done




for scale in "${scales[@]}"
do
cmd="python evaluations/evaluator_tolog.py reference/VIRTUAL_imagenet64_labeled.npz \
 runs/sampling_icml_classifier_free/IMN64/normal/scale${scale}/reference/samples_50000x64x64x3.npz"
echo ${cmd}
eval ${cmd}
done

