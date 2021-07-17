#!/bin/bash
: ${SCORING:="r2"}
: ${MODEL:="rf"}
python src/train_hparam_search.py --model ${MODEL} --scoring ${SCORING}