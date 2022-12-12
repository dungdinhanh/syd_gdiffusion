#!/bin/bash



hfai bash bash_scripts/mlt/im64unc/IM64_unc_cls_sampling_hfai_mlt.sh -- -n 1 --no_diff --name im64unc

hfai bash bash_scripts/mlt/im64unc/IM64_unc_cls_sampling_hfai_mlt2.sh -- -n 1 --no_diff --name im64unc2

#cmd="python scripts_hfai_gdiff/classifier_sample.py --logdir runs/sampling/IMN64/conditional/ \
# ${MODEL_FLAGS} --classifier_scale 1.0 --classifier_path models/64x64_classifier.pt \
# --classifier_depth 4 --model_path models/64x64_diffusion.pt ${SAMPLE_FLAGS}"
#echo ${cmd}
#eval ${cmd}