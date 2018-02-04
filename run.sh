#!/usr/bin/env bash

cd features
sh run.sh
cd ../model/
#cd model
#python xgboost_model.py
#python xgb_remove_feature.py
#python xgb_feature_select.py
#python xgboost_roof.py
python lightgbm_model.py
#python catboost_model.py
