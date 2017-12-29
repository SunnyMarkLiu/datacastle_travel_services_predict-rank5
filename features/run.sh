#!/usr/bin/env bash

python gen_user_features.py
python gen_user_action_features.py
#python gen_basic_action_features.py
python gen_order_history_features.py
python gen_comment_features.py
python gen_action_history_features.py
