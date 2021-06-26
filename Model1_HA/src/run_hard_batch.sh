#!/bin/bash

# for lang in mlt swa crh; do
for lang in mwf izh mlt; do
    #_d2k_o0_c_lcs-2 _d2k_o0_f_lcs-2 _d2k_o1_c_lcs-2 _d2k_o1_f_lcs-2 _d2k_o0_c_lcs-3 _d2k_o0_f_lcs-3 _d2k_o1_c_lcs-3 _d2k_o1_f_lcs-3
    for vers in 1; do
        
        a=$lang 
        echo "Preprocessing.."
        /bin/bash preprocessdata_all.sh $a        
        echo "Running hard attention for lang $a"
        /home/matt/venv/tf/bin/python hard_attention.py --dynet-mem 4096 --input=100 --hidden=100 --feat-input=20 --epochs=$1 --layers=2 --optimization=ADADELTA ../data/$a/$a.trn.r ../data/$a/$a.dev.r ../data/$a/$a.tst.r ../data/results/$a.results ../sigmorphon2016-base

    done
done
