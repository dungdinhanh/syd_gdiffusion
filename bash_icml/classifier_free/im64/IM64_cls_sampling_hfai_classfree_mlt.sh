#!/bin/bash

SAMPLE_FLAGS="--batch_size 128 --num_samples 50000 --timestep_respacing 250"
#SAMPLE_FLAGS="--batch_size 5 --num_samples 50000 --timestep_respacing 250"
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

scales=( "1.0" "2.0" "3.0" "4.0")

for scale in "${scales[@]}"
do
cmd="python scripts_hfai_gdiff/classifier_free/classifier_free_sample_mlt.py $MODEL_FLAGS --cond_model_scale ${scale}  \
--cond_model_path models/64x64_diffusion.pt \
--model_path runs/IM64/IM64_diffusion_training_unconditional/models/ema_0.9999_540000_final.pt  $SAMPLE_FLAGS \
 --logdir runs/sampling_icml_classifier_free/IMN64/mlt/scale${scale}/ "
echo ${cmd}
eval ${cmd}
done

#cmd="CUDA_VISIBLE_DEVICES=0 python scripts_hfai_gdiff/classifier_free/classifier_free_sample_mlt.py $MODEL_FLAGS --cond_model_scale 0.2  \
#--cond_model_path  models/64x64_diffusion.pt  \
#--model_path ../output/diff_models/IM64/IM64_diffusion_training_unconditional/models/ema_0.9999_540000_final.pt  $SAMPLE_FLAGS \
# --logdir ../output/test_classifier_free/ "
#echo ${cmd}
#eval ${cmd}


for scale in "${scales[@]}"
do
cmd="python evaluations/evaluator_tolog.py reference/VIRTUAL_imagenet64_labeled.npz \
 runs/sampling_icml_classifier_free/IMN64/mlt/scale${scale}/reference/samples_50000x64x64x3.npz"
echo ${cmd}
eval ${cmd}
done

