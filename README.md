## Training
To train the model(s) in the paper, run this command:

```train
python run.py <experiment_specification path> <problem> <seed>
```
for example

```train
python run.py ed2_torch_on_mujoco.py HalfCheetah-v3 42
```

Logger automatically stops training and evaluates current policy every `log_every` environment interactions. The data is printed to standard output and stored on drive.

We include specifications for our most important experiments.

| Path        | Description           |
| ------------- |:-------------:|
| specs/ed2_on_mujoco.py | Benchmark of ED2 | 
| specs/ed2_torch_on_mujoco.py | Benchmark of our implementation of ED2 pytorch version | 
| specs/sac_on_mujoco.py | Benchmark of ED2 implementation of SAC |
| specs/sunrise_on_mujoco.py | Benchmark of Spinup implementation of SUNRISE |
| specc/ppo_on_mujoco.py | Benchmark of Spinup implementation of ppo |
| specc/td3_on_mujoco.py | Benchmark of Spinup implementation of TD3 |

Five problems are solved in our each algorithm:

| Problem        | Description           |
| ------------- |:-------------:|
| Hopper-v3 | Hopper | 
| Walker2d-v3 | Walker | 
| Ant-v3 | Ant |
| Humanoid-v3 | Humanoid |
| HalfCheetah-v3 | HalfCheetah |
