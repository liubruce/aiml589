#!/usr/bin/env bash

python run.py specs/ppo_on_mujoco.py 'Hopper-v3' $1
python run.py specs/ppo_on_mujoco.py 'Walker2d-v3' $1
python run.py specs/ppo_on_mujoco.py 'Ant-v3' $1
python run.py specs/ppo_on_mujoco.py 'Humanoid-v3' $1
python run.py specs/ppo_on_mujoco.py 'HalfCheetah-v3' $1
#        # 'Walker2d-v3',
#        # 'Ant-v3',
#        # 'Humanoid-v3',
#        'HalfCheetah-v3'
#python run.py specs/td3_on_mujoco.py
#python run.py specs/sac_on_mujoco.py
#python run.py specs/sunrise_on_mujoco.py
#python run.py specs/ed2_on_mujoco.py
