#!/usr/bin/env bash

cd features
sh run.sh
cd ../model/
#cd model
python xgboost_model.py
#python xgboost_roof.py
#python lightgbm_model.py
#python catboost_model.py
