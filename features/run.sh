#!/usr/bin/env bash

# baseline features 0.97210, done!
python gen_user_features.py
python gen_user_action_features.py
python gen_order_history_features.py
python gen_comment_features.py
python gen_action_history_features.py
python add_wxr_features.py

# new features
python add_sqg_features.py
python gen_advance_features.py
python gen_other_features.py
python gen_action_order_features.py
