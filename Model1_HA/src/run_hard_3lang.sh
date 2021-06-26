#!/bin/bash

echo "Running hard attention for lang $1 $2 $3 $4"
/home/matt/venv/tf/bin/python hard_attention.py --dynet-mem 4096 --input=100 --hidden=100 --feat-input=20 --epochs=$4 --layers=2 --optimization=ADADELTA ../data/$1/$1.trn.r ../data/$1/$1.dev.r ../data/$1/$1.tst.r ../data/results/$1_results .
/home/matt/venv/tf/bin/python hard_attention.py --dynet-mem 4096 --input=100 --hidden=100 --feat-input=20 --epochs=$4 --layers=2 --optimization=ADADELTA ../data/$2/$2.trn.r ../data/$2/$2.dev.r ../data/$2/$2.tst.r ../data/results/$2_results .
/home/matt/venv/tf/bin/python hard_attention.py --dynet-mem 4096 --input=100 --hidden=100 --feat-input=20 --epochs=$4 --layers=2 --optimization=ADADELTA ../data/$3/$3.trn.r ../data/$3/$3.dev.r ../data/$3/$3.tst.r ../data/results/$3_results .

