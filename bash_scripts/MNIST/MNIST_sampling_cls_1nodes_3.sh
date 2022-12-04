#!/bin/bash

MODEL_FLAGS="--image_size 28 --num_channels 32 --num_res_blocks 1 --attention_resolutions 7,4 --diffusion_steps 1000 --dropout 0.1 --noise_schedule cosine
 --num_head_channels 8"
#TRAIN_FLAGS="--lr 1e-4 --batch_size 16 --lr_anneal_steps 200000  --dropout 0.1 --num_head_channels 8"
SAMPLE_FLAGS="--batch_size 2048 --num_samples 10000 --timestep_respacing 250"

#cmd="cd .."
#echo ${cmd}
#eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}

cpath="../outputhfai/runs/mnist_classifier_training/models/model149999.pt"
mpath="../outputhfai/runs/mnist_diffusion_training_1node/models/model200000.pt"
opath="../outputhfai/local_runs/"
npyname="reference/samples_10000x28x28x1.npz"

cmd="python scripts_hfai_gdiff/classifier_sample_local.py --classifier_path ${cpath} \
--model_path ${mpath} \
--logdir ${opath}/mnist_cls_sampling_1node_s0/ --classifier_scale 0.0 --classifier_attention_resolutions 14,7 --classifier_depth 1 \
--classifier_width 32 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True \
 $MODEL_FLAGS ${SAMPLE_FLAGS}"
echo ${cmd}
eval ${cmd}



cmd="python scripts_hfai_gdiff/npytoimgs.py --numpy ${opath}/mnist_cls_sampling_1node_s0/${npyname} "
echo ${cmd}
eval ${cmd}


