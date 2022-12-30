#!/bin/bash

hfai workspace push --no_diff -f

#
#hfai bash bash_scripts2/icml/im64cond/IM64_cond_sampling_hfai_evaluate_mlt.sh -- -n 1 --no_diff --name evalim64condcdiv
#
#hfai bash bash_scripts2/icml/im64cond/IM64_cond_sampling_hfai_evaluate_mlt2.sh -- -n 1 --no_diff --name evalim64condcdiv2

hfai bash bash_icml/icml/im64cond/IM64_cls_sampling_hfai_cdiv.sh -- -n 2 --no_diff --name im64cdivcondr

hfai bash bash_icml/icml/im64cond/IM64_cls_sampling_hfai_cdiv2.sh -- -n 2 --no_diff --name im64cdivcondr2

hfai bash bash_icml/icml/im64cond/IM64_cls_sampling_hfai_cdiv3.sh -- -n 2 --no_diff --name im64cdivcondr3

