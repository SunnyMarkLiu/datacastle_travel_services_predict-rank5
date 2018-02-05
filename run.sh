#!/usr/bin/env bash

cd features
#sh run.sh
cd ../model/

# single model
#python xgboost_model.py
#python xgb_remove_feature.py
#python xgb_feature_select.py
#python lightgbm_model.py
#python catboost_model.py

######## stacking ########
# level 1, run out of fold
#python xgboost_roof.py -f 5 -e 0.010 -w 20 -c 0.50 -d 10 -p 0.9  -s 0
#python xgboost_roof.py -f 5 -e 0.011 -w 18 -c 0.61 -d 12 -p 0.8  -s 11
#python xgboost_roof.py -f 5 -e 0.012 -w 17 -c 0.74 -d 9  -p 0.85 -s 43
#python xgboost_roof.py -f 5 -e 0.013 -w 21 -c 0.55 -d 15 -p 0.91 -s 127
#python xgboost_roof.py -f 5 -e 0.014 -w 22 -c 0.66 -d 10 -p 0.84 -s 223
#python xgboost_roof.py -f 5 -e 0.015 -w 20 -c 0.64 -d 9  -p 0.81 -s 449
#python xgboost_roof.py -f 5 -e 0.016 -w 19 -c 0.58 -d 12 -p 0.83 -s 523
#python xgboost_roof.py -f 5 -e 0.017 -w 16 -c 0.49 -d 14 -p 0.75 -s 631
#python xgboost_roof.py -f 5 -e 0.008 -w 15 -c 0.45 -d 15 -p 0.78 -s 977
#python xgboost_roof.py -f 5 -e 0.009 -w 14 -c 0.52 -d 10 -p 0.76 -s 2017

#python lightgbm_roof.py --fl 5 --lr 0.010 --mw 1  --ff 0.9  --nl 64  --bf 0.7  --l1 0.5 --l2 0.5 --sd 3
#python lightgbm_roof.py --fl 5 --lr 0.015 --mw 2  --ff 0.85 --nl 64  --bf 0.8  --l1 0.5 --l2 0.5 --sd 7
#python lightgbm_roof.py --fl 5 --lr 0.013 --mw 4  --ff 0.91 --nl 128 --bf 0.75 --l1 0.5 --l2 0.5 --sd 47
#python lightgbm_roof.py --fl 5 --lr 0.018 --mw 3  --ff 0.86 --nl 64  --bf 0.78 --l1 0.5 --l2 0.5 --sd 123
#python lightgbm_roof.py --fl 5 --lr 0.020 --mw 6  --ff 0.87 --nl 32  --bf 0.66 --l1 0.5 --l2 0.5 --sd 227
#python lightgbm_roof.py --fl 5 --lr 0.008 --mw 8  --ff 0.8  --nl 64  --bf 0.90 --l1 0.5 --l2 0.5 --sd 451
#python lightgbm_roof.py --fl 5 --lr 0.010 --mw 10 --ff 0.5  --nl 128 --bf 0.67 --l1 0.5 --l2 0.5 --sd 551
#python lightgbm_roof.py --fl 5 --lr 0.022 --mw 5  --ff 0.93 --nl 64  --bf 0.71 --l1 0.5 --l2 0.5 --sd 1021
#python lightgbm_roof.py --fl 5 --lr 0.016 --mw 6  --ff 0.9  --nl 64  --bf 0.76 --l1 0.5 --l2 0.5 --sd 2017
#python lightgbm_roof.py --fl 5 --lr 0.025 --mw 9  --ff 0.88 --nl 128 --bf 0.6  --l1 0.5 --l2 0.5 --sd 521

# level 2, lr model
