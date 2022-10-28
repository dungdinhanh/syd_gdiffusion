#!/bin/bash

name_c="mnist_info_schm11"
cd_cmd="CUDA_VISIBLE_DEVICES=1"


cpath="../outputhfai/runs/mnist_classifier_training/models/model149999.pt"
#cpath="../outputhfai/local_analyse/twostages_transfer/${name_c}/models/model149999.pt"
#mpath="../outputhfai/runs/mnist_diffusion_training_1node/models/model200000.pt"
mpath="../output/runs/mnist/mnist_diff_sigma/models/ema_0.9999_200000.pt"
opath="../outputhfai/local_analyse/twostages_transfer/${name_c}/ref/"
name="sampling_dbg_ddim_s"

MODEL_FLAGS="--image_size 28 --num_channels 32 --num_res_blocks 2 --attention_resolutions 7,4 --diffusion_steps 1000 --dropout 0.1 --noise_schedule cosine
 --num_head_channels 8 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True --learn_sigma True"
#SAMPLE_FLAGS1="--batch_size 1001 --num_samples 10000 --timestep_respacing 250"
SAMPLE_FLAGS2="--batch_size 1001 --num_samples 10000 --timestep_respacing 250 --use_ddim True"

cmd="${cd_cmd} python scripts_hfai_local/classifier_sample_local.py --classifier_path ${cpath} \
--model_path ${mpath} --logdir ${opath}/${name}_ema4/ --classifier_scale 4.0 \
--classifier_attention_resolutions 14,7 --classifier_depth 1 \
--classifier_width 32 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True \
 $MODEL_FLAGS ${SAMPLE_FLAGS2}"
echo ${cmd}
eval ${cmd}

cmd="${cd_cmd} python scripts_hfai_local/classifier_sample_local.py --classifier_path ${cpath} \
--model_path ${mpath} --logdir ${opath}/${name}_ema5/ --classifier_scale 5.0 \
--classifier_attention_resolutions 14,7 --classifier_depth 1 \
--classifier_width 32 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True \
 $MODEL_FLAGS ${SAMPLE_FLAGS2}"
echo ${cmd}
eval ${cmd}


cmd="${cd_cmd} python scripts_hfai_local/classifier_sample_local.py --classifier_path ${cpath} \
--model_path ${mpath} --logdir ${opath}/${name}_ema7/ --classifier_scale 7.0 \
--classifier_attention_resolutions 14,7 --classifier_depth 1 \
--classifier_width 32 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True \
 $MODEL_FLAGS ${SAMPLE_FLAGS2}"
echo ${cmd}
eval ${cmd}

cmd="${cd_cmd} python scripts_hfai_local/classifier_sample_local.py --classifier_path ${cpath} \
--model_path ${mpath} --logdir ${opath}/${name}_ema10/ --classifier_scale 10.0 \
--classifier_attention_resolutions 14,7 --classifier_depth 1 \
--classifier_width 32 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True \
 $MODEL_FLAGS ${SAMPLE_FLAGS2}"
echo ${cmd}
eval ${cmd}


