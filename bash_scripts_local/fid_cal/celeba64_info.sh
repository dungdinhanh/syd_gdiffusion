#!/bin/bash

infos=("7" "9" "11")
ss=("3" "5" "7" "10")

for info in "${infos[@]}"
do
for s in "${ss[@]}"
do
cmd="python evaluations/evaluator.py ../output/data/CelebA64/real_50000x64x64x3.npz \
../output/runs/celeba64info/celeba64_info${info}/ref/sampling_s${s}/reference/samples_50000x64x64x3.npz"
echo ${cmd}
eval ${cmd}
done
done

