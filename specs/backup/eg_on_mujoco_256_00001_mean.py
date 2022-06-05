base_config = {
    'agent': '@spinup.algos.pytorch.EG',
    'total_steps': 1_000_000,
    'num_test_episodes': 30,
    'use_noise_for_exploration': True,
    # 'update_every': 50,
    # 'update_after': 100,
    # 'log_every': 200,
    # 'train_intensity': 0.2,
    # 'log_every': 1100,
    'lamda_value': 0.00001,
    'ac_kwargs': {
        'hidden_sizes': [256, 256],
        # 'activation': 'relu',
        'prior_weight': 0.0
    },
    'save_freq': 1_000_000,
    'use_vote_policy': False,
    'save_path': './out/checkpoint',
    'logger_kwargs': {
        'exp_name': 'peg_distance_00001_noise_mean'
    }
}

params_grid = {
    'task': [
        'Hopper-v3',
        'Walker2d-v3',
        'Ant-v3',
        'Humanoid-v3',
    ],
    'seed': [42,
             7, 224444444, 11, 14,
             13, 5, 509758253, 777, 6051995,
             817604, 759621, 469592, 681422, 662896,
             680578, 50728, 680595, 650678, 984230,
             420115, 487860, 234662, 753671, 709357,
             755288, 109482, 626151, 459560, 629937
             ],
}
