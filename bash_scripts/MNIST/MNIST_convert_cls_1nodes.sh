#!/bin/bash

#cmd="cd .."
#echo ${cmd}
#eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}

#cpath="../outputhfai/runs/mnist_classifier_training/models/model149999.pt"
#mpath="../outputhfai/runs/mnist_diffusion_training_1node/models/model200000.pt"
opath="../outputhfai/local_runs/"
npyname="reference/samples_10000x28x28x1.npz"

cmd="python scripts_hfai_gdiff/npytoimgs.py --numpy ${opath}/mnist_cls_sampling_1node_s5/${npyname} "
echo ${cmd}
eval ${cmd}


cmd="python scripts_hfai_gdiff/npytoimgs.py --numpy  ${opath}/mnist_cls_sampling_1node_s6/${npyname}"
echo ${cmd}
eval ${cmd}


cmd="python scripts_hfai_gdiff/npytoimgs.py --numpy  ${opath}/mnist_cls_sampling_1node_s7/${npyname}"
echo ${cmd}
eval ${cmd}

cmd="python scripts_hfai_gdiff/npytoimgs.py --numpy  ${opath}/mnist_cls_sampling_1node_s8/${npyname}"
echo ${cmd}
eval ${cmd}

cmd="python scripts_hfai_gdiff/npytoimgs.py --numpy  ${opath}/mnist_cls_sampling_1node_s9/${npyname}"
echo ${cmd}
eval ${cmd}