#!/bin/bash

hfai workspace push --no_diff -f

hfai workspace push --no_diff -f

hfai bash bash_icml/icml/im128cond/IM128_cls_sampling_hfai_cdiv.sh -- -n 2 --no_diff --name icmlim128concdiv


hfai bash bash_icml/icml/im128cond/IM128_cls_sampling_hfai_cdiv2.sh -- -n 2 --no_diff --name icmlim128concdiv2


hfai bash bash_icml/icml/im256cond/IM256_cls_sampling_hfai_cdiv.sh -- -n 2 --no_diff --name icmlim256concdiv

hfai bash bash_icml/icml/im256cond/IM256_cls_sampling_hfai_cdiv2.sh -- -n 2 --no_diff --name icmlim256concdiv2

hfai bash bash_icml/icml/im256cond/IM256_cls_sampling_hfai_cdiv3.sh -- -n 2 --no_diff --name icmlim256concdiv3