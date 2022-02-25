## Training
To train the model(s) in the paper, run this command:

```train
python run.py <experiment_specification path>
```

Logger automatically stops training and evaluates current policy every `log_every` environment interactions. The data is printed to standard output and stored on drive.

We include specifications for our most important experiments.

| Path        | Description           |
| ------------- |:-------------:|
| specs/ed2_on_mujoco.py | Benchmark of our method | 
| specs/sac_on_mujoco.py | Benchmark of our implementation of SAC |
| specs/sunrise_on_mujoco.py | Benchmark of our implementation of SUNRISE |
| specc/sop_on_mujoco.py | Benchmark of our implementation of SOP |
| specc/ddpg_on_mujoco.py | Benchmark of our implementation of DDPG |
| specc/td3_on_mujoco.py | Benchmark of our implementation of TD3 |
