import numpy as np
import scipy.signal

import torch
import torch.nn as nn
from torch.nn.functional import normalize

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

# def mlp(hidden_sizes, activation, trainable=True, name=None):
#     """Creates MLP with the specified parameters."""
#     return tf.keras.Sequential([
#         tf.keras.layers.Dense(size, activation=activation, trainable=trainable)
#         for size in hidden_sizes
#     ], name)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes)
        self.pi = mlp(pi_sizes, activation, nn.ReLU)
        # print('list(hidden_sizes)[len(hidden_sizes)-1] is ', list(hidden_sizes)[len(hidden_sizes)-1])
        self.last_layer = nn.Linear(list(hidden_sizes)[len(hidden_sizes)-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        mu = self.pi(obs)
        mu = self.last_layer(mu)
        # Normalize the actions
        # mu = normalize(mu, p=2.0, dim=0)

        m_tanh = nn.Tanh()
        return self.act_limit * m_tanh(mu)

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space,prior_weight, hidden_sizes=(256,256),
                 activation=nn.ReLU, ac_number=5):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]
        self.L_qf1, self.L_qf2, self.L_policy = [], [], []

        for _ in range(ac_number):

            # build policy and value functions
            self.L_policy.append(MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit))
            self.L_qf1.append(MLPQFunction(obs_dim, act_dim, hidden_sizes, activation))
            self.L_qf2.append(MLPQFunction(obs_dim, act_dim, hidden_sizes, activation))

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()
