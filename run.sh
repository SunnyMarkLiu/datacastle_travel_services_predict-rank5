#!/usr/bin/env bash

cd features
sh run.sh
cd ../model/

# single model
python xgboost_model.py
#python xgb_remove_feature.py
#python xgb_feature_select.py
#python huanglin_lightgbm_model.py
#python huang_lin_xgboost_model.py
#python sqg_lightgbm_model.py
#python sqg_xgboost_model.py
#python catboost_model.py

# stacking
######## level 1, run out of fold ########
python model_roof/lq_xgboost_roof.py -f 3 -e 0.010 -w 20 -c 0.50 -d 10 -p 0.9  -s 325
python model_roof/lq_xgboost_roof.py -f 3 -e 0.011 -w 18 -c 0.61 -d 12 -p 0.8  -s 12
python model_roof/lq_xgboost_roof.py -f 3 -e 0.012 -w 17 -c 0.74 -d 9  -p 0.85 -s 45
python model_roof/lq_xgboost_roof.py -f 3 -e 0.013 -w 21 -c 0.55 -d 15 -p 0.91 -s 147
python model_roof/lq_xgboost_roof.py -f 3 -e 0.014 -w 22 -c 0.66 -d 10 -p 0.84 -s 263
python model_roof/lq_xgboost_roof.py -f 3 -e 0.015 -w 20 -c 0.64 -d 9  -p 0.81 -s 479
python model_roof/lq_xgboost_roof.py -f 3 -e 0.016 -w 19 -c 0.58 -d 12 -p 0.83 -s 583
python model_roof/lq_xgboost_roof.py -f 3 -e 0.017 -w 16 -c 0.49 -d 14 -p 0.75 -s 61
python model_roof/lq_xgboost_roof.py -f 3 -e 0.008 -w 15 -c 0.45 -d 15 -p 0.78 -s 77
python model_roof/lq_xgboost_roof.py -f 3 -e 0.009 -w 14 -c 0.52 -d 10 -p 0.76 -s 217

python model_roof/lq_lightgbm_roof.py --fl 3 --lr 0.010 --mw 1  --ff 0.9  --nl 64  --bf 0.7  --l1 0.5 --l2 0.5 --sd 32
python model_roof/lq_lightgbm_roof.py --fl 3 --lr 0.015 --mw 2  --ff 0.85 --nl 64  --bf 0.8  --l1 0.5 --l2 0.5 --sd 74
python model_roof/lq_lightgbm_roof.py --fl 3 --lr 0.013 --mw 4  --ff 0.91 --nl 128 --bf 0.75 --l1 0.5 --l2 0.5 --sd 44
python model_roof/lq_lightgbm_roof.py --fl 3 --lr 0.018 --mw 3  --ff 0.86 --nl 64  --bf 0.78 --l1 0.5 --l2 0.5 --sd 13
python model_roof/lq_lightgbm_roof.py --fl 3 --lr 0.020 --mw 6  --ff 0.87 --nl 32  --bf 0.66 --l1 0.5 --l2 0.5 --sd 27
python model_roof/lq_lightgbm_roof.py --fl 3 --lr 0.008 --mw 8  --ff 0.8  --nl 64  --bf 0.90 --l1 0.5 --l2 0.5 --sd 51
python model_roof/lq_lightgbm_roof.py --fl 3 --lr 0.010 --mw 10 --ff 0.5  --nl 128 --bf 0.67 --l1 0.5 --l2 0.5 --sd 57
python model_roof/lq_lightgbm_roof.py --fl 3 --lr 0.022 --mw 5  --ff 0.93 --nl 64  --bf 0.71 --l1 0.5 --l2 0.5 --sd 121
python model_roof/lq_lightgbm_roof.py --fl 3 --lr 0.016 --mw 6  --ff 0.9  --nl 64  --bf 0.76 --l1 0.5 --l2 0.5 --sd 207
python model_roof/lq_lightgbm_roof.py --fl 3 --lr 0.025 --mw 9  --ff 0.88 --nl 128 --bf 0.6  --l1 0.5 --l2 0.5 --sd 520

python model_roof/lq_catboost_roof.py -f 5 -r 0.10 -l 5 -s 0.90 -d 8 -e 323
python model_roof/lq_catboost_roof.py -f 5 -r 0.09 -l 8 -s 0.91 -d 7 -e 142
python model_roof/lq_catboost_roof.py -f 5 -r 0.11 -l 7 -s 0.94 -d 6 -e 455
python model_roof/lq_catboost_roof.py -f 5 -r 0.13 -l 3 -s 0.85 -d 10 -e 1467
python model_roof/lq_catboost_roof.py -f 5 -r 0.14 -l 5 -s 0.86 -d 7 -e 2633
python model_roof/lq_catboost_roof.py -f 5 -r 0.11 -l 4 -s 0.94 -d 8 -e 4793
python model_roof/lq_catboost_roof.py -f 5 -r 0.10 -l 8 -s 0.88 -d 6 -e 5836
python model_roof/lq_catboost_roof.py -f 5 -r 0.11 -l 6 -s 0.89 -d 9 -e 614
python model_roof/lq_catboost_roof.py -f 5 -r 0.10 -l 5 -s 0.85 -d 8 -e 7766
python model_roof/lq_catboost_roof.py -f 5 -r 0.09 -l 4 -s 0.92 -d 10 -e 2174

python model_roof/lq_rf_roof.py -f 5 -n 1300 -l 1 -t 50 -s 10
python model_roof/lq_rf_roof.py -f 5 -n 1320 -l 2 -t 45 -s 0
python model_roof/lq_rf_roof.py -f 5 -n 1340 -l 3 -t 40 -s 43
python model_roof/lq_rf_roof.py -f 5 -n 1360 -l 4 -t 35 -s 127
python model_roof/lq_rf_roof.py -f 5 -n 1380 -l 5 -t 30 -s 223
python model_roof/lq_rf_roof.py -f 5 -n 1280 -l 10 -t 25 -s 449
python model_roof/lq_rf_roof.py -f 5 -n 1260 -l 9 -t 20 -s 523
python model_roof/lq_rf_roof.py -f 5 -n 1240 -l 8 -t 10 -s 631
python model_roof/lq_rf_roof.py -f 5 -n 1220 -l 7 -t 55 -s 977
python model_roof/lq_rf_roof.py -f 5 -n 3500 -l 6 -t 60 -s 2017

python model_roof/lq_et_roof.py -f 5 -n 3100 -l 1 -t 2 -s 10
python model_roof/lq_et_roof.py -f 5 -n 3120 -l 2 -t 4 -s 0
python model_roof/lq_et_roof.py -f 5 -n 3140 -l 3 -t 6 -s 45
python model_roof/lq_et_roof.py -f 5 -n 3160 -l 4 -t 8 -s 128
python model_roof/lq_et_roof.py -f 5 -n 3180 -l 5 -t 10 -s 257
python model_roof/lq_et_roof.py -f 5 -n 3080 -l 10 -t 12 -s 468
python model_roof/lq_et_roof.py -f 5 -n 3060 -l 9 -t 14 -s 579
python model_roof/lq_et_roof.py -f 5 -n 3040 -l 8 -t 16 -s 636
python model_roof/lq_lq_et_roof.py -f 5 -n 3020 -l 7 -t 18 -s 921
python model_roof/lq_lq_et_roof.py -f 5 -n 1300 -l 6 -t 20 -s 2017

### huang lin's datasets ###
python model_roof/huanglin_xgboost_roof.py -f 3 -e 0.010 -w 20 -c 0.50 -d 10 -p 0.9  -s 20
python model_roof/huanglin_xgboost_roof.py -f 3 -e 0.011 -w 18 -c 0.61 -d 12 -p 0.8  -s 311
python model_roof/huanglin_xgboost_roof.py -f 3 -e 0.012 -w 17 -c 0.74 -d 9  -p 0.85 -s 443
python model_roof/huanglin_xgboost_roof.py -f 3 -e 0.013 -w 21 -c 0.55 -d 15 -p 0.91 -s 5127
python model_roof/huanglin_xgboost_roof.py -f 3 -e 0.014 -w 22 -c 0.66 -d 10 -p 0.84 -s 6223
python model_roof/huanglin_xgboost_roof.py -f 3 -e 0.015 -w 20 -c 0.64 -d 9  -p 0.81 -s 7449
python model_roof/huanglin_xgboost_roof.py -f 3 -e 0.016 -w 19 -c 0.58 -d 12 -p 0.83 -s 8523
python model_roof/huanglin_xgboost_roof.py -f 3 -e 0.017 -w 16 -c 0.49 -d 14 -p 0.75 -s 4631
python model_roof/huanglin_xgboost_roof.py -f 3 -e 0.008 -w 15 -c 0.45 -d 15 -p 0.78 -s 6977
python model_roof/huanglin_xgboost_roof.py -f 3 -e 0.009 -w 14 -c 0.52 -d 10 -p 0.76 -s 72017

python model_roof/huanglin_lightgbm_roof.py --fl 3 --lr 0.025 --mw 2   --ff 0.30 --nl 30  --bf 0.50  --l1 0.5 --l2 1 --sd 53
python model_roof/huanglin_lightgbm_roof.py --fl 3 --lr 0.030 --mw 4   --ff 0.45 --nl 32  --bf 0.55  --l1 0.5 --l2 2 --sd 57
python model_roof/huanglin_lightgbm_roof.py --fl 3 --lr 0.010 --mw 5   --ff 0.30 --nl 34  --bf 0.56  --l1 0.5 --l2 3 --sd 447
python model_roof/huanglin_lightgbm_roof.py --fl 3 --lr 0.030 --mw 6   --ff 0.45 --nl 32  --bf 0.60  --l1 0.5 --l2 4 --sd 6123
python model_roof/huanglin_lightgbm_roof.py --fl 3 --lr 0.040 --mw 8   --ff 0.40 --nl 29  --bf 0.58  --l1 0.5 --l2 3 --sd 7227
python model_roof/huanglin_lightgbm_roof.py --fl 3 --lr 0.035 --mw 4   --ff 0.35 --nl 31  --bf 0.55  --l1 0.5 --l2 4 --sd 8451
python model_roof/huanglin_lightgbm_roof.py --fl 3 --lr 0.037 --mw 7   --ff 0.40 --nl 33  --bf 0.50  --l1 0.5 --l2 3 --sd 61021
python model_roof/huanglin_lightgbm_roof.py --fl 3 --lr 0.045 --mw 4   --ff 0.48 --nl 34  --bf 0.45  --l1 0.5 --l2 2 --sd 42017
python model_roof/huanglin_lightgbm_roof.py --fl 3 --lr 0.015 --mw 10  --ff 0.40 --nl 35  --bf 0.48  --l1 0.5 --l2 1 --sd 32018
python model_roof/huanglin_lightgbm_roof.py --fl 3 --lr 0.018 --mw 9   --ff 0.36 --nl 31  --bf 0.50  --l1 0.5 --l2 0.5 --sd 5521

python model_roof/huanglin_catboost_roof.py -f 5 -r 0.10 -l 5 -s 0.90 -d 8 -e 323
python model_roof/huanglin_catboost_roof.py -f 5 -r 0.09 -l 8 -s 0.91 -d 7 -e 142
python model_roof/huanglin_catboost_roof.py -f 5 -r 0.11 -l 7 -s 0.94 -d 6 -e 455
python model_roof/huanglin_catboost_roof.py -f 5 -r 0.13 -l 3 -s 0.85 -d 10 -e 1467
python model_roof/huanglin_catboost_roof.py -f 5 -r 0.14 -l 5 -s 0.86 -d 7 -e 2633
python model_roof/huanglin_catboost_roof.py -f 5 -r 0.11 -l 4 -s 0.94 -d 8 -e 4793
python model_roof/huanglin_catboost_roof.py -f 5 -r 0.10 -l 8 -s 0.88 -d 6 -e 5836
python model_roof/huanglin_catboost_roof.py -f 5 -r 0.11 -l 6 -s 0.89 -d 9 -e 614
python model_roof/huanglin_catboost_roof.py -f 5 -r 0.10 -l 5 -s 0.85 -d 8 -e 7766
python model_roof/huanglin_catboost_roof.py -f 5 -r 0.09 -l 4 -s 0.92 -d 10 -e 2174

### qian guo's datasets ###
python model_roof/qian_guo_xgboost_roof.py -f 3 -e 0.010 -w 20 -c 0.50 -d 10 -p 0.9  -s 70
python model_roof/qian_guo_xgboost_roof.py -f 3 -e 0.011 -w 18 -c 0.61 -d 12 -p 0.8  -s 511
python model_roof/qian_guo_xgboost_roof.py -f 3 -e 0.012 -w 17 -c 0.74 -d 9  -p 0.85 -s 463
python model_roof/qian_guo_xgboost_roof.py -f 3 -e 0.013 -w 21 -c 0.55 -d 15 -p 0.91 -s 1247
python model_roof/qian_guo_xgboost_roof.py -f 3 -e 0.014 -w 22 -c 0.66 -d 10 -p 0.84 -s 2233
python model_roof/qian_guo_xgboost_roof.py -f 3 -e 0.015 -w 20 -c 0.64 -d 9  -p 0.81 -s 4459
python model_roof/qian_guo_xgboost_roof.py -f 3 -e 0.016 -w 19 -c 0.58 -d 12 -p 0.83 -s 5263
python model_roof/qian_guo_xgboost_roof.py -f 3 -e 0.017 -w 16 -c 0.49 -d 14 -p 0.75 -s 6331
python model_roof/qian_guo_xgboost_roof.py -f 3 -e 0.008 -w 15 -c 0.45 -d 15 -p 0.78 -s 9787
python model_roof/qian_guo_xgboost_roof.py -f 3 -e 0.009 -w 14 -c 0.52 -d 10 -p 0.76 -s 20317

python model_roof/qian_guo_lightgbm_roof.py --fl 3 --lr 0.025 --ff 0.35 --nl 30 --bf 0.60  --l1 0 --l2 11.5 --sd 315
python model_roof/qian_guo_lightgbm_roof.py --fl 3 --lr 0.030 --ff 0.45 --nl 35 --bf 0.70  --l1 0 --l2 10.5 --sd 732
python model_roof/qian_guo_lightgbm_roof.py --fl 3 --lr 0.010 --ff 0.50 --nl 38 --bf 0.56  --l1 0 --l2 12.0 --sd 4731
python model_roof/qian_guo_lightgbm_roof.py --fl 3 --lr 0.020 --ff 0.55 --nl 32 --bf 0.68  --l1 0 --l2 11.5 --sd 12335
python model_roof/qian_guo_lightgbm_roof.py --fl 3 --lr 0.040 --ff 0.60 --nl 40 --bf 0.70  --l1 0 --l2 12.5 --sd 22746
python model_roof/qian_guo_lightgbm_roof.py --fl 3 --lr 0.035 --ff 0.55 --nl 45 --bf 0.55  --l1 0 --l2 9.50 --sd 45153
python model_roof/qian_guo_lightgbm_roof.py --fl 3 --lr 0.037 --ff 0.40 --nl 46 --bf 0.50  --l1 0 --l2 10.5 --sd 102157
python model_roof/qian_guo_lightgbm_roof.py --fl 3 --lr 0.045 --ff 0.68 --nl 50 --bf 0.45  --l1 0 --l2 11.5 --sd 201
python model_roof/qian_guo_lightgbm_roof.py --fl 3 --lr 0.015 --ff 0.70 --nl 38 --bf 0.78  --l1 0 --l2 13.5 --sd 20146
python model_roof/qian_guo_lightgbm_roof.py --fl 3 --lr 0.018 --ff 0.65 --nl 48 --bf 0.80  --l1 0 --l2 14.5 --sd 52100

python model_roof/qian_guo_catboost_roof.py -f 5 -r 0.10 -l 5 -s 0.90 -d 8 -e 323
python model_roof/qian_guo_catboost_roof.py -f 5 -r 0.09 -l 8 -s 0.91 -d 7 -e 142
python model_roof/qian_guo_catboost_roof.py -f 5 -r 0.11 -l 7 -s 0.94 -d 6 -e 455
python model_roof/qian_guo_catboost_roof.py -f 5 -r 0.13 -l 3 -s 0.85 -d 10 -e 1467
python model_roof/qian_guo_catboost_roof.py -f 5 -r 0.14 -l 5 -s 0.86 -d 7 -e 2633
python model_roof/qian_guo_catboost_roof.py -f 5 -r 0.11 -l 4 -s 0.94 -d 8 -e 4793
python model_roof/qian_guo_catboost_roof.py -f 5 -r 0.10 -l 8 -s 0.88 -d 6 -e 5836
python model_roof/qian_guo_catboost_roof.py -f 5 -r 0.11 -l 6 -s 0.89 -d 9 -e 614
python model_roof/qian_guo_catboost_roof.py -f 5 -r 0.10 -l 5 -s 0.85 -d 8 -e 7766
python model_roof/qian_guo_catboost_roof.py -f 5 -r 0.09 -l 4 -s 0.92 -d 10 -e 2174

######## level 2, 50 models，时间不够了，来不及跑第二层 stacking ！ ########
#python model_roof/level2_xgboost_roof.py -f 3 -e 0.010 -w 20 -c 0.50 -d 10 -p 0.9  -s 325
#python model_roof/level2_xgboost_roof.py -f 3 -e 0.011 -w 18 -c 0.61 -d 12 -p 0.8  -s 12
#python model_roof/level2_xgboost_roof.py -f 3 -e 0.012 -w 17 -c 0.74 -d 9  -p 0.85 -s 45
#python model_roof/level2_xgboost_roof.py -f 3 -e 0.013 -w 21 -c 0.55 -d 15 -p 0.91 -s 147
#python model_roof/level2_xgboost_roof.py -f 3 -e 0.014 -w 22 -c 0.66 -d 10 -p 0.84 -s 263
#python model_roof/level2_xgboost_roof.py -f 3 -e 0.015 -w 20 -c 0.64 -d 9  -p 0.81 -s 479
#python model_roof/level2_xgboost_roof.py -f 3 -e 0.016 -w 19 -c 0.58 -d 12 -p 0.83 -s 583
#python model_roof/level2_xgboost_roof.py -f 3 -e 0.017 -w 16 -c 0.49 -d 14 -p 0.75 -s 61
#python model_roof/level2_xgboost_roof.py -f 3 -e 0.008 -w 15 -c 0.45 -d 15 -p 0.78 -s 77
#python model_roof/level2_xgboost_roof.py -f 3 -e 0.009 -w 14 -c 0.52 -d 10 -p 0.76 -s 217
#
#python model_roof/level2_lightgbm_roof.py -f 3 -e 0.010 -w 20 -c 0.50 -d 10 -p 0.9  -s 325
#python model_roof/level2_lightgbm_roof.py -f 3 -e 0.011 -w 18 -c 0.61 -d 12 -p 0.8  -s 12
#python model_roof/level2_lightgbm_roof.py -f 3 -e 0.012 -w 17 -c 0.74 -d 9  -p 0.85 -s 45
#python model_roof/level2_lightgbm_roof.py -f 3 -e 0.013 -w 21 -c 0.55 -d 15 -p 0.91 -s 147
#python model_roof/level2_lightgbm_roof.py -f 3 -e 0.014 -w 22 -c 0.66 -d 10 -p 0.84 -s 263
#python model_roof/level2_lightgbm_roof.py -f 3 -e 0.015 -w 20 -c 0.64 -d 9  -p 0.81 -s 479
#python model_roof/level2_lightgbm_roof.py -f 3 -e 0.016 -w 19 -c 0.58 -d 12 -p 0.83 -s 583
#python model_roof/level2_lightgbm_roof.py -f 3 -e 0.017 -w 16 -c 0.49 -d 14 -p 0.75 -s 61
#python model_roof/level2_lightgbm_roof.py -f 3 -e 0.008 -w 15 -c 0.45 -d 15 -p 0.78 -s 77
#python model_roof/level2_lightgbm_roof.py -f 3 -e 0.009 -w 14 -c 0.52 -d 10 -p 0.76 -s 217
