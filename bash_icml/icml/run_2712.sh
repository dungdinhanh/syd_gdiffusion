#!/bin/bash

hfai workspace push --no_diff -f


hfai bash bash_scripts2/diversity/cdiv1wc/IM64_unc_sampling_hfai_evaluate_mlt.sh -- -n 1 --no_diff --name eval1wcim64unc

hfai bash bash_scripts2/diversity/cdiv1wc/IM64_unc_sampling_hfai_evaluate_mlt3_4.sh -- -n 1 --no_diff --name eval1wcim64unc2

hfai bash bash_scripts2/diversity/cdiv1wd/IM64_unc_sampling_hfai_evaluate_mlt.sh -- -n 1 --no_diff --name eval1wdim64unc

hfai bash bash_scripts2/diversity/cdiv1wd/IM64_unc_sampling_hfai_evaluate_mlt3_4.sh -- -n 1 --no_diff --name eval1wdim64unc2

hfai bash bash_scripts2/diversity/cdivonly/IM64_unc_sampling_hfai_evaluate_mlt.sh -- -n 1 --no_diff --name evalonlyim64unc

hfai bash bash_scripts2/diversity/cdivonly/IM64_unc_sampling_hfai_evaluate_mlt3_4.sh -- -n 1 --no_diff --name evalonlyim64unc2