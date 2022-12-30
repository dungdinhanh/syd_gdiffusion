#!/bin/bash

hfai workspace push --no_diff -f

hfai workspace push --no_diff -f

hfai workspace push --no_diff -f

hfai workspace push --no_diff -f

hfai workspace push --no_diff -f

#
hfai bash bash_icml/icml/im512/IM512_cls_sampling_hfai.sh -- -n 8  --no_diff --name im512condcls

hfai bash bash_icml/icml/im512/IM512_cls_sampling_hfai2.sh -- -n 8  --no_diff --name im512condcls2

hfai bash bash_icml/icml/im512/IM512_cls_sampling_hfai3.sh -- -n 8  --no_diff --name im512condcls3


hfai bash bash_icml/icml/im512/IM512_cls_sampling_hfai_cdiff.sh -- -n 8 --no_diff --name im512condcdiff

hfai bash bash_icml/icml/im512/IM512_cls_sampling_hfai_cdiff2.sh -- -n 8 --no_diff --name im512condcdiff2

hfai bash bash_icml/icml/im512/IM512_cls_sampling_hfai_cdiff3.sh -- -n 8 --no_diff --name im512condcdiff3







