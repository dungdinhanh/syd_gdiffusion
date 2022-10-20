#!/bin/bash

cmd="bash bash_scripts_local/qnet/MNIST_sampling_cls_1nodes.sh info_schm7"
echo ${cmd}
#eval ${cmd}

cmd="bash bash_scripts_local/qnet/MNIST_sampling_cls_1nodes_s1.sh info_schm5_h02"
echo ${cmd}
eval ${cmd}

cmd="bash bash_scripts_local/qnet/MNIST_sampling_cls_1nodes_s1.sh info_schm6_h02"
echo ${cmd}
eval ${cmd}

