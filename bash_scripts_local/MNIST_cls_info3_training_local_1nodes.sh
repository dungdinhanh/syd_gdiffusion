#!/bin/bash

TRAIN_FLAGS="--iterations 200000 --anneal_lr True --batch_size 128 --lr 2e-4 --save_interval 10000 --weight_decay 0.2"
CLASSIFIER_FLAGS="--image_size 28 --classifier_attention_resolutions 14,7 --classifier_depth 1 \
--classifier_width 32 --classifier_pool attention --classifier_resblock_updown True \
 --classifier_use_scale_shift_norm True"

#cmd="cd .."
#echo ${cmd}
#eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}


cmd="python scripts_hfai_local/classifier_localtrain_info_mnist_schm3.py --data_dir path/to/imagenet --logdir ../outputhfai/local_runs_info/mnist_info3_classifier_training/ $TRAIN_FLAGS $CLASSIFIER_FLAGS"
echo ${cmd}
eval ${cmd}

MODEL_FLAGS="--image_size 28 --num_channels 32 --num_res_blocks 1 --attention_resolutions 7,4 --diffusion_steps 1000 --dropout 0.1 --noise_schedule cosine
 --num_head_channels 8"
#TRAIN_FLAGS="--lr 1e-4 --batch_size 16 --lr_anneal_steps 200000  --dropout 0.1 --num_head_channels 8"
SAMPLE_FLAGS="--batch_size 2048 --num_samples 10000 --timestep_respacing 250"

cpath="../outputhfai/local_runs_info/mnist_info3_classifier_training/models/model199999.pt"
mpath="../outputhfai/runs/mnist_diffusion_training_1node/models/model200000.pt"
opath="../outputhfai/local_runs_info"
namee="mnist_infosc3_sampling_1node"


cmd="python scripts_hfai_gdiff/classifier_sample_local.py --classifier_path ${cpath} \
--model_path ${mpath} \
--logdir ${opath}/${namee}_s1/ --classifier_scale 1.0 --classifier_attention_resolutions 14,7 --classifier_depth 1 \
--classifier_width 32 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True \
 $MODEL_FLAGS ${SAMPLE_FLAGS}"
echo ${cmd}
eval ${cmd}

cmd="python scripts_hfai_gdiff/classifier_sample_local.py --classifier_path ${cpath} \
--model_path ${mpath} \
--logdir ${opath}/${namee}_s2/ --classifier_scale 2.0 --classifier_attention_resolutions 14,7 --classifier_depth 1 \
--classifier_width 32 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True \
 $MODEL_FLAGS ${SAMPLE_FLAGS}"
echo ${cmd}
eval ${cmd}

cmd="python scripts_hfai_gdiff/classifier_sample_local.py --classifier_path ${cpath} \
--model_path ${mpath} \
--logdir ${opath}/${namee}_s3/ --classifier_scale 3.0 --classifier_attention_resolutions 14,7 --classifier_depth 1 \
--classifier_width 32 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True \
 $MODEL_FLAGS ${SAMPLE_FLAGS}"
echo ${cmd}
eval ${cmd}

cmd="python scripts_hfai_gdiff/classifier_sample_local.py --classifier_path ${cpath} \
--model_path ${mpath} \
--logdir ${opath}/${namee}_s4/ --classifier_scale 4.0 --classifier_attention_resolutions 14,7 --classifier_depth 1 \
--classifier_width 32 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True \
 $MODEL_FLAGS ${SAMPLE_FLAGS}"
echo ${cmd}
eval ${cmd}

cmd="python scripts_hfai_gdiff/classifier_sample_local.py --classifier_path ${cpath} \
--model_path ${mpath} \
--logdir ${opath}/${namee}_s5/ --classifier_scale 5.0 --classifier_attention_resolutions 14,7 --classifier_depth 1 \
--classifier_width 32 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True \
 $MODEL_FLAGS ${SAMPLE_FLAGS}"
echo ${cmd}
eval ${cmd}