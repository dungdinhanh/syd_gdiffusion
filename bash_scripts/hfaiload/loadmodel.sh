#!/bin/bash


folder=$1


if [ ! -z $2 ]
then
start=$2
else
  start=1
fi

if [ ! -z $3 ]
then
limit=$3
else
  limit=30
fi
workspace="gdiffusion"

step=$start

while [ $step -le $limit ]
do
# setup files
model=$(printf "model%02d" ${step})
file_model="runs/${folder}/models/${model}0000.pt"
optm=$(printf "opt%02d" ${step})
file_opt="runs/${folder}/models/${optm}0000.pt"
#
cmd="hfai workspace download ${file_model}"
echo ${cmd}
eval ${cmd}

cmd="hfai workspace download ${file_opt}"
echo ${cmd}
eval ${cmd}

cmd="hfai workspace remove ${workspace} -f ${file_model} --yes "
echo ${cmd}
eval ${cmd}

cmd="hfai workspace remove ${workspace} -f ${file_opt} --yes"
echo ${cmd}
eval ${cmd}

step=$(( $step + 1 ))

done

outside_folder="../outputhfai/runs/${folder}"
cmd="mkdir ${outside_folder}"
echo ${cmd}
eval ${cmd}

cmd="mv runs/${folder}/models/* ${outside_folder}"
echo ${cmd}
eval ${cmd}

