#!/bin/bash

infos=("7" "9" "11")
ss=("3" "5" "7" "10")

for info in "${infos[@]}"
do
for s in "${ss[@]}"
do
cmd="python evaluations/evaluator.py ../output/data/CelebA32/real_50000x32x32x3.npz \
../output/runs/celeba32info/celeba32_info${info}/ref/sampling_s${s}/reference/samples_50000x32x32x3.npz"
echo ${cmd}
eval ${cmd} &>> ../output/fid_log.txt
done
done

