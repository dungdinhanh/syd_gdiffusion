#!/bin/bash

SAMPLE_FLAGS="--batch_size 25 --num_samples 50000 --timestep_respacing 250"
#TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --schedule_sampler loss-second-moment"

MODEL_FLAGS="--attention_resolutions 16,8,4 --class_cond False --diffusion_steps 1000  \
--image_size 32 --noise_schedule cosine --num_channels 192 \
--num_head_channels 64 --num_res_blocks 3 --learn_sigma True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"

cmd="cd ../.."
echo ${cmd}
#eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}

sample_model_path="../output/runs/celeba32info/celeba32_info9"

cmd="python scripts_hfai_local/image_sample_infoq_celeba.py --logdir ${sample_model_path}/ref/sampling_s0 ${MODEL_FLAGS} \
--classifier_scale 0.0 --classifier_attention_resolutions 16,8,4 --classifier_path ${sample_model_path}/models/model199999.pt \
 --classifier_depth 3 --model_path ../output/runs/celeba32/celeba32_diff_dbg/models/ema_0.9999_300000_final.pt ${SAMPLE_FLAGS}"
echo ${cmd}
#eval ${cmd}

cmd="python scripts_hfai_local/classifier_sample_infoq_celeba_fixpath_grid.py --logdir ${sample_model_path}/ref/sampling_s5 ${MODEL_FLAGS} \
--classifier_scale 5.0 --classifier_attention_resolutions 16,8,4 --classifier_path ${sample_model_path}/models/model199999.pt \
 --classifier_depth 3 --model_path ../output/runs/celeba32/celeba32_diff_dbg/models/ema_0.9999_300000_final.pt ${SAMPLE_FLAGS}"
echo ${cmd}
#eval ${cmd}
#
cmd="python scripts_hfai_local/classifier_sample_infoq_celeba_fixpath_grid.py --logdir ${sample_model_path}/ref/sampling_s3 ${MODEL_FLAGS} \
--classifier_scale 3.0 --classifier_attention_resolutions 16,8,4 --classifier_path ${sample_model_path}/models/model199999.pt \
 --classifier_depth 3 --model_path ../output/runs/celeba32/celeba32_diff_dbg/models/ema_0.9999_300000_final.pt ${SAMPLE_FLAGS}"
echo ${cmd}
#eval ${cmd}
#
cmd="python scripts_hfai_local/classifier_sample_infoq_celeba_fixpath_grid.py --logdir ${sample_model_path}/ref/sampling_s7 ${MODEL_FLAGS} \
--classifier_scale 7.0 --classifier_attention_resolutions 16,8,4 --classifier_path ${sample_model_path}/models/model199999.pt \
 --classifier_depth 3 --model_path ../output/runs/celeba32/celeba32_diff_dbg/models/ema_0.9999_300000_final.pt ${SAMPLE_FLAGS}"
echo ${cmd}
eval ${cmd}
#
cmd="python scripts_hfai_local/classifier_sample_infoq_celeba_fixpath_grid.py --logdir ${sample_model_path}/ref/sampling_s10 ${MODEL_FLAGS} \
--classifier_scale 10.0 --classifier_attention_resolutions 16,8,4 --classifier_path ${sample_model_path}/models/model199999.pt \
 --classifier_depth 3 --model_path ../output/runs/celeba32/celeba32_diff_dbg/models/ema_0.9999_300000_final.pt ${SAMPLE_FLAGS}"
echo ${cmd}
eval ${cmd}