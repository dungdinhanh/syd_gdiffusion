#!/bin/bash

cmd="cd ../../../"
echo ${cmd}
eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}


cmd="python evaluations/evaluator_tolog.py reference/VIRTUAL_imagenet128_labeled.npz \
 reference/biggan_deep_trunc1_imagenet128.npz"
echo ${cmd}
eval ${cmd}


cmd="python evaluations/evaluator_tolog.py reference/VIRTUAL_imagenet256_labeled.npz \
 reference/biggan_deep_trunc1_imagenet256.npz"
echo ${cmd}
eval ${cmd}


cmd="python evaluations/evaluator_tolog.py reference/VIRTUAL_imagenet64_labeled.npz \
 reference/biggan_deep_imagenet64.npz"
echo ${cmd}
eval ${cmd}

cmd="python evaluations/evaluator_tolog.py reference/VIRTUAL_imagenet64_labeled.npz \
 reference/iddpm_imagenet64.npz"
echo ${cmd}
eval ${cmd}











