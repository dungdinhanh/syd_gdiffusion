#!/bin/bash


MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 128 \
--learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True\
 --use_fp16 True --use_scale_shift_norm True"

SAMPLE_FLAGS="--batch_size 32 --num_samples 10000 --timestep_respacing 250"



cmd="cd ../../"
echo ${cmd}
eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}

#scales=("0" "2" "4")
scales=("2")

for scale in "${scales[@]}"
do
cmd="python scripts_hfai_gdiff/multitask/classifier_sample_mlt_diversitygrad.py $MODEL_FLAGS --classifier_scale ${scale}.0 --classifier_path models/128x128_classifier.pt \
--model_path models/128x128_diffusion.pt $SAMPLE_FLAGS  --logdir runs/sampling/IMN128/conditional_mlt_divg/scale${scale}p0/"
echo ${cmd}
eval ${cmd}
done

cmd="python scripts_hfai_gdiff/multitask/classifier_sample_mlt_diversitygrad.py $MODEL_FLAGS --classifier_scale 0.5 --classifier_path models/128x128_classifier.pt \
--model_path models/128x128_diffusion.pt $SAMPLE_FLAGS  --logdir runs/sampling/IMN128/conditional_mlt_divg/scale0p5/"
#echo ${cmd}
#eval ${cmd}

#cmd="python scripts_hfai_gdiff/classifier_sample.py --logdir runs/classifier_pretrained/ ${MODEL_FLAGS} --classifier_scale 1.0 --classifier_path models/64x64_classifier.pt \
# --classifier_depth 4 --model_path models/64x64_diffusion.pt ${SAMPLE_FLAGS}"
