#!/bin/bash

# for lang in mlt swa crh; do
for lang in mwf izh mlt; do
    epoch=$2
    vers=$1
    a=$lang$vers 
    echo "Running hard attention for lang $a"
    /home/matt/venv/tf/bin/python hard_attention.py --dynet-mem 4096 --input=100 --hidden=100 --feat-input=20 --epochs=$epoch --layers=2 --optimization=ADADELTA ../data/$a/$a.trn.r ../data/$a/$a.dev.r ../data/$a/$a.tst.r ../data/results/$a ../sigmorphon2016-base

    
done
