#!/bin/bash
: ${METRIC:="r2_score"}
python src/train_kfold.py --model baseline --metric ${METRIC}
python src/train_kfold.py --model dt --metric ${METRIC}
python src/train_kfold.py --model rf --metric ${METRIC}
python src/train_kfold.py --model lin_reg --metric ${METRIC}
python src/train_kfold.py --model kn --metric ${METRIC}
python src/train_kfold.py --model svr --metric ${METRIC}
python src/train_kfold.py --model xgb --metric ${METRIC}
echo "Models available in models/{model_name}/ directory"