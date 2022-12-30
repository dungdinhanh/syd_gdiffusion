#!/bin/bash

hfai workspace push --no_diff -f

hfai workspace push --no_diff -f

#
hfai bash bash_icml/icml/im128cond/IM128_cls_sampling_hfai_cdiff.sh -- -n 2 --no_diff --name im128concdiff


hfai bash bash_icml/icml/im128cond/IM128_cls_sampling_hfai_cdiff2.sh -- -n 2 --no_diff --name im128concdiff2

hfai bash bash_icml/icml/im256cond/IM256_cls_sampling_hfai_cdiff.sh -- -n 4 --no_diff --name im256concdiff

hfai bash bash_icml/icml/im256cond/IM256_cls_sampling_hfai_cdiff2.sh -- -n 4 --no_diff --name im256concdiff2

hfai bash bash_icml/icml/im256cond/IM256_cls_sampling_hfai_cdiff3.sh -- -n 4 --no_diff --name im256concdiff3


hfai bash bash_icml/icml/im256unc/IM256_unc_cls_sampling_hfai_cdiv.sh -- -n 4 --no_diff --name im256unccdiv

hfai bash bash_icml/icml/im256unc/IM256_unc_cls_sampling_hfai_cdiv2.sh -- -n 4 --no_diff --name im256unccdiv2

hfai bash bash_icml/icml/im256unc/IM256_unc_cls_sampling_hfai_cdiv3.sh -- -n 4 --no_diff --name im256unccdiv3


