#!/bin/bash

hfai workspace push --no_diff -f

hfai workspace push --no_diff -f

hfai workspace push --no_diff -f

hfai workspace push --no_diff -f

hfai workspace push --no_diff -f

hfai workspace push --no_diff -f

#
hfai bash bash_icml/icml/cifar10_local/CIFAR10_cls_train_hfai.sh -- -n 1 --no_diff --name cifar10iddpmtraincls

hfai bash bash_icml/icml/eval_baselines/eval_baselines.sh -- -n 1 --no_diff --name evalbaselines






