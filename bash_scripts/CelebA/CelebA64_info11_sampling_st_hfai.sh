#!/bin/bash

SAMPLE_FLAGS="--batch_size 100 --num_samples 50000 --timestep_respacing 250"
#TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --schedule_sampler loss-second-moment"

MODEL_FLAGS="--attention_resolutions 32,16,8  --diffusion_steps 1000 --dropout 0.1 \
--image_size 64 --learn_sigma True \
--noise_schedule cosine --num_channels 192 \
--num_head_channels 64 --num_res_blocks 3\
 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"



cmd="cd ../.."
echo ${cmd}
eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}

sample_model_path="runs/celeba64info/celeba64_info11"
diff_model_path="runs/celeba64/celeba64_training_diff/models/ema_0.9999_300000_final.pt"

CLS_FLAGS="--classifier_attention_resolutions 32,16,8 --classifier_depth 4 \
--classifier_width 128 --classifier_pool attention --classifier_resblock_updown True \
 --classifier_use_scale_shift_norm True"



cmd="python scripts_hfai_gdiff/classifier_sample_infoq.py --logdir ${sample_model_path}/ref/sampling_s3 ${MODEL_FLAGS} \
--model_path ${diff_model_path} --classifier_path ${sample_model_path}/models/model199999.pt ${CLS_FLAGS} \
 --classifier_scale 3.0  ${SAMPLE_FLAGS}"
echo ${cmd}
eval ${cmd}

cmd="python scripts_hfai_gdiff/classifier_sample_infoq.py --logdir ${sample_model_path}/ref/sampling_s5 ${MODEL_FLAGS} \
--model_path ${diff_model_path} --classifier_path ${sample_model_path}/models/model199999.pt ${CLS_FLAGS} \
 --classifier_scale 5.0  ${SAMPLE_FLAGS}"
echo ${cmd}
eval ${cmd}

cmd="python scripts_hfai_gdiff/classifier_sample_infoq.py --logdir ${sample_model_path}/ref/sampling_s7 ${MODEL_FLAGS} \
--model_path ${diff_model_path} --classifier_path ${sample_model_path}/models/model199999.pt ${CLS_FLAGS} \
 --classifier_scale 7.0  ${SAMPLE_FLAGS}"
echo ${cmd}
eval ${cmd}

cmd="python scripts_hfai_gdiff/classifier_sample_infoq.py --logdir ${sample_model_path}/ref/sampling_s10 ${MODEL_FLAGS} \
--model_path ${diff_model_path} --classifier_path ${sample_model_path}/models/model199999.pt ${CLS_FLAGS} \
 --classifier_scale 10.0  ${SAMPLE_FLAGS}"
echo ${cmd}
eval ${cmd}

