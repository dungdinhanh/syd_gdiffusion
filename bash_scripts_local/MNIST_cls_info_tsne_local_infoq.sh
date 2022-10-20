#!/bin/bash

#TRAIN_FLAGS="--iterations 200000 --anneal_lr True --batch_size 128 --lr 2e-4 --save_interval 10000 --weight_decay 0.2"
#CLASSIFIER_FLAGS="--image_size 28 --classifier_attention_resolutions 14,7 --classifier_depth 1 \
#--classifier_width 32 --classifier_pool attention --classifier_resblock_updown True \
# --classifier_use_scale_shift_norm True"

#cmd="cd .."
#echo ${cmd}
#eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}

cd_command="CUDA_VISIBLE_DEVICES=1"


#cmd="${cd_command} python scripts_hfai_local/classifier_localtrain_info_mnist_schm3.py --data_dir path/to/imagenet --logdir ../outputhfai/local_runs_info/mnist_info4_classifier_training/ $TRAIN_FLAGS $CLASSIFIER_FLAGS"
#echo ${cmd}
#eval ${cmd}

MODEL_FLAGS="--image_size 28 --num_channels 32 --num_res_blocks 1 --attention_resolutions 7,4 --diffusion_steps 1000 --dropout 0.1 --noise_schedule cosine
 --num_head_channels 8"
#TRAIN_FLAGS="--lr 1e-4 --batch_size 16 --lr_anneal_steps 200000  --dropout 0.1 --num_head_channels 8"
SAMPLE_FLAGS="--batch_size 1048 --num_samples 1000 --timestep_respacing 250"

cpath="../outputhfai/local_analyse/mnist_info/info_schm7/models/model149999.pt"
mpath="../outputhfai/runs/mnist_diffusion_training_1node/models/model200000.pt"
opath="../outputhfai/local_runs_info"
namee="mnist_infosc4_sampling_1node"


cmd="${cd_command} python scripts_hfai_local/classifier_tsne_local_infoq.py --classifier_path ${cpath} \
--model_path ${mpath}  --classifier_scale 1.0 --classifier_attention_resolutions 14,7 --classifier_depth 1 \
--classifier_width 32 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True \
 $MODEL_FLAGS ${SAMPLE_FLAGS}"
echo ${cmd}
eval ${cmd}


