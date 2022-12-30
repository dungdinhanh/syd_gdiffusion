#!/bin/bash


MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 \
 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 \
  --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"


SAMPLE_FLAGS="--batch_size 8 --num_samples 10000 --timestep_respacing 250"

cmd="cd ../../"
echo ${cmd}
eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}


scales=("0.1" "0.5" "0.7"  "1.0" "2.0" "4.0" "6.0" "8.0" "10.0" )

for scale in "${scales[@]}"
do
cmd="python evaluations/evaluator_tolog.py reference/VIRTUAL_imagenet256_labeled.npz \
 runs/sampling_icml/IMN256/conditional_cdiff/scale${scale}/reference/samples_50000x256x256x3.npz"
echo ${cmd}
eval ${cmd}
done



#eval ${cmd}
#cmd="python scripts_hfai_gdiff/classifier_sample.py --logdir runs/classifier_pretrained/ ${MODEL_FLAGS} --classifier_scale 1.0 --classifier_path models/64x64_classifier.pt \
# --classifier_depth 4 --model_path models/64x64_diffusion.pt ${SAMPLE_FLAGS}"
