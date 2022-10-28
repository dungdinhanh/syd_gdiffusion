#!/bin/bash

TRAIN_FLAGS="--iterations 150000 --anneal_lr True --batch_size 128 --lr 2e-4 --save_interval 10000 --weight_decay 0.2"
CLASSIFIER_FLAGS="--image_size 28 --classifier_attention_resolutions 14,7 --classifier_depth 1 \
--classifier_width 32 --classifier_pool attention --classifier_resblock_updown True \
 --classifier_use_scale_shift_norm True --alphae 0.2"




#cmd="cd ../.."
#echo ${cmd}
#eval ${cmd}
cd_cmd="CUDA_VISIBLE_DEVICES=0"

cmd="ls"
echo ${cmd}
eval ${cmd}

name_c="mnist_info_schm8"

cmd="${cd_cmd} python scripts_hfai_local/classifier_localtrain_info_mnist_schm8.py --data_dir path/to/imagenet \
--logdir ../outputhfai/local_analyse/twostages_transfer/${name_c}/ $TRAIN_FLAGS $CLASSIFIER_FLAGS"
echo ${cmd}
#eval ${cmd}

cpath="../outputhfai/local_analyse/twostages_transfer/${name_c}/models/model149999.pt"
mpath="../outputhfai/runs/mnist_diffusion_training_1node/models/model200000.pt"
opath="../outputhfai/local_analyse/twostages_transfer/${name_c}/ref/"
name="sampling_s"

MODEL_FLAGS="--image_size 28 --num_channels 32 --num_res_blocks 1 --attention_resolutions 7,4 --diffusion_steps 1000 --dropout 0.1 --noise_schedule cosine
 --num_head_channels 8"
SAMPLE_FLAGS="--batch_size 1001 --num_samples 10000 --timestep_respacing 250"

cmd="${cd_cmd} python scripts_hfai_local/classifier_sample_infoq_local.py --classifier_path ${cpath} \
--model_path ${mpath} \
--logdir ${opath}/${name}5/ --classifier_scale 5.0 --classifier_attention_resolutions 14,7 --classifier_depth 1 \
--classifier_width 32 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True \
 $MODEL_FLAGS ${SAMPLE_FLAGS}"
echo ${cmd}
eval ${cmd}