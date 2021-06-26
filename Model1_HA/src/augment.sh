#!/bin/bash
for lng in izh mwf mlt; do
    /home/matt/venv/tf/bin/python augment.py ../data/$lng $lng --examples 10000
done
