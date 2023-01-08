#!/bin/bash

hfai workspace push --no_diff -f

hfai workspace push --no_diff -f



hfai bash bash_icml/classifier_free/im256/IM256_cls_sampling_hfai_classfree.sh -- -n 8 --no_diff --name im256clsfree1

hfai bash bash_icml/classifier_free/im256/IM256_cls_sampling_hfai_classfree2.sh -- -n 8 --no_diff --name im256clsfree2

hfai bash bash_icml/classifier_free/im256/IM256_cls_sampling_hfai_classfree3.sh -- -n 8 --no_diff --name im256clsfree3


#


