#!/bin/bash

hfai workspace push --no_diff -f

hfai workspace push --no_diff -f

hfai workspace push --no_diff -f

hfai workspace push --no_diff -f


#
hfai bash bash_icml/classifier_free/im64/IM64_cls_sampling_hfai_classfree.sh -- -n 4  --no_diff --name im64clsfree


hfai bash bash_icml/classifier_free/im64/IM64_cls_sampling_hfai_classfree2.sh -- -n 4  --no_diff --name im64clsfree2


hfai bash bash_icml/classifier_free/im64/IM64_cls_sampling_hfai_classfree3.sh -- -n 4  --no_diff --name im64clsfree3


hfai bash bash_icml/classifier_free/im64/IM64_cls_sampling_hfai_classfree_mlt.sh -- -n 4  --no_diff --name im64clsfreemlt


hfai bash bash_icml/classifier_free/im64/IM64_cls_sampling_hfai_classfree_mlt2.sh -- -n 4  --no_diff --name im64clsfreemlt2


hfai bash bash_icml/classifier_free/im64/IM64_cls_sampling_hfai_classfree_mlt3.sh -- -n 4  --no_diff --name im64clsfreemlt3

