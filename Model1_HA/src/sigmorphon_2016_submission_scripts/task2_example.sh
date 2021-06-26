#!/usr/bin/env bash

export PYTHONPATH=~/git/morphological-reinflection/
base_path=~/git/morphological-reinflection/src/

python $base_path/run_all_langs_generic.py --langs='russian' --script=sigmorphon_2016_submission_scripts/task2_ms2s.py \
--prefix=task2_ms2s --task=2 \
~/git/morphological-reinflection/src/ \
~/git/morphological-reinflection/results/results.txt \
~/git/sigmorphon2016/