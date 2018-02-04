#!/usr/bin/env bash

cd features
#sh run.sh
cd ../model/

# single model
python xgboost_model.py
#python xgb_remove_feature.py
#python xgb_feature_select.py
#python xgboost_roof.py
#python lightgbm_model.py
#python catboost_model.py

# run out of fold
python xgboost_roof.py -f 5 -e 0.010 -w 20 -c 0.50 -d 10 -p 0.9  -s 0
python xgboost_roof.py -f 5 -e 0.011 -w 18 -c 0.61 -d 12 -p 0.8  -s 11
python xgboost_roof.py -f 5 -e 0.012 -w 17 -c 0.74 -d 9  -p 0.85 -s 43
python xgboost_roof.py -f 5 -e 0.013 -w 21 -c 0.55 -d 15 -p 0.91 -s 127
python xgboost_roof.py -f 5 -e 0.014 -w 22 -c 0.66 -d 10 -p 0.84 -s 223
python xgboost_roof.py -f 5 -e 0.015 -w 20 -c 0.64 -d 9  -p 0.81 -s 449
python xgboost_roof.py -f 5 -e 0.016 -w 19 -c 0.58 -d 12 -p 0.83 -s 523
python xgboost_roof.py -f 5 -e 0.017 -w 16 -c 0.49 -d 14 -p 0.75 -s 631
python xgboost_roof.py -f 5 -e 0.008 -w 15 -c 0.45 -d 15 -p 0.78 -s 977
python xgboost_roof.py -f 5 -e 0.009 -w 14 -c 0.52 -d 10 -p 0.76 -s 2017
