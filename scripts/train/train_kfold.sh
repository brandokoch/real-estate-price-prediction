#!/bin/bash
: ${METRIC:="r2_score"}
: ${MODEL:="baseline"}
python src/train_kfold.py --model ${MODEL} --metric ${METRIC}