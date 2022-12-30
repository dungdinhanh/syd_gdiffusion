#!/bin/bash

hfai workspace push --no_diff -f

hfai workspace push --no_diff -f

hfai workspace push --no_diff -f

#
hfai bash bash_icml/icml/im256cond/IM256_cls_sampling_hfai.sh -- -n 4  --no_diff --name im256condcls

hfai bash bash_icml/icml/im256cond/IM256_cls_sampling_hfai2.sh -- -n 4  --no_diff --name im256condcls2

hfai bash bash_icml/icml/im256cond/IM256_cls_sampling_hfai3.sh -- -n 4  --no_diff --name im256condcls3

hfai bash bash_icml/icml/im256unc/IM256_unc_cls_sampling_hfai.sh -- -n 4 --no_diff --name im256unccls

hfai bash bash_icml/icml/im256unc/IM256_unc_cls_sampling_hfai2.sh -- -n 4 --no_diff --name im256unccls2

hfai bash bash_icml/icml/im256unc/IM256_unc_cls_sampling_hfai3.sh -- -n 4 --no_diff --name im256unccls3

hfai bash bash_icml/icml/im64cond/IM64_cls_sampling_hfai_cdiffdiv3.sh -- -n 2 --no_diff --name im64cdiffdiv

hfai bash bash_icml/icml/im128cond/IM128_cls_sampling_hfai.sh -- -n 4 --no_diff --name im128condnomlt


