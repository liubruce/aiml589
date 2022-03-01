#!/usr/bin/env bash

python run.py specs/ddpg_on_mujoco.py
python run.py specs/td3_on_mujoco.py
python run.py specs/sac_on_mujoco.py
python run.py specs/sunrise_on_mujoco.py
python run.py specs/ed2_on_mujoco.py
