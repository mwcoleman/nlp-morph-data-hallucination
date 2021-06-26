#!/bin/bash
for lng in izh mwf izh; do
    /home/matt/venv/tf/bin/python augment.py ../../task0-data/original $lng --examples 10000
done
