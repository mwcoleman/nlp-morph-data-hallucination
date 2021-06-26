#!/bin/bash

echo "Running hard attention for lang $1"
/home/matt/venv/tf/bin/python hard_attention.py --dynet-mem 8192 --input=100 --hidden=100 --feat-input=20 --epochs=$2 --layers=2 --optimization=ADADELTA ../data/$1/$1.trn.r ../data/$1/$1.dev.r ../data/$1/$1.tst.r ../data/results/$1_results .

