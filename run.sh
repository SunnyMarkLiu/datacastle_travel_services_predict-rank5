#!/usr/bin/env bash

cd features
#cd new_features
sh run.sh
cd ../model/
#cd ../new_model/
#cd model
#python xgboost_model.py
#python xgboost_roof.py
python lightgbm_model.py
