#!/bin/bash

scales=("0" "2" "4" "6" "8" "10")

for scale in "${scales[@]}"
do
file_model="runs/sampling/IMN128/conditional/scale${scale}p0/reference/score.csv"
cmd="hfai workspace download ${file_model}"
echo ${cmd}
#eval ${cmd}
done

cmd="hfai workspace download runs/sampling/IMN128/conditional/scale0p5/reference/score.csv"
echo ${cmd}
#eval ${cmd}

for scale in "${scales[@]}"
do
file_model="runs/sampling/IMN64/conditional/scale${scale}p0/reference/score.csv --force"
cmd="hfai workspace download ${file_model}"
echo ${cmd}
eval ${cmd}
done

cmd="hfai workspace download runs/sampling/IMN64/conditional/scale0p5/reference/score.csv --force"
echo ${cmd}
eval ${cmd}


scales=("0" "1" "10")

for scale in "${scales[@]}"
do
file_model="runs/sampling/IMN256/conditional/scale${scale}p0/reference/score.csv"
cmd="hfai workspace download ${file_model}"
echo ${cmd}
#eval ${cmd}
done


for scale in "${scales[@]}"
do
file_model="runs/sampling/IMN256/unconditional/scale${scale}p0/reference/score.csv"
cmd="hfai workspace download ${file_model}"
echo ${cmd}
eval ${cmd}
done

