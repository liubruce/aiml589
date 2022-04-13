import numpy as np
import torch
import torch.nn as nn

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit, act_noise):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes)
        self.pi = mlp(pi_sizes, activation, nn.ReLU)
        # print('list(hidden_sizes)[len(hidden_sizes)-1] is ', list(hidden_sizes)[len(hidden_sizes)-1])
        self.last_layer = nn.Linear(list(hidden_sizes)[len(hidden_sizes) - 1], act_dim)
        self._act_scale = act_limit
        self._act_noise = act_noise
        self.device = "cpu"

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        mu = self.pi(obs)
        mu = self.last_layer(mu)

        # Normalize the actions
        abs_mean = torch.abs(mu)
        K = torch.tensor(mu.size()[1]).to(self.device)
        Gs = torch.sum(abs_mean, dim=1).view(-1, 1)  ######
        Gs = Gs / K
        # Gs = Gs / beta
        ones = torch.ones(Gs.size()).to(self.device)
        Gs_mod1 = torch.where(Gs >= 1, Gs, ones)
        mu = mu / Gs_mod1

        # g = torch.mean(torch.abs(mu), dim=0)
        # print('The shape of g ', g.shape)
        # g = torch.maximum(g, torch.ones_like(g))
        # print('mu', mu.shape, g.shape)
        # mu = mu / g
        # print('before normalize mu ', mu.shape,mu)
        # mu = normalize(mu, p=1.0, dim=0)
        # print('after normalize mu ', mu.shape,mu)
        # print('mu', mu.shape)
        # Add the action noise
        pi = mu + self._act_noise * torch.empty(mu.shape).normal_()
        # Put the actions in the limit.
        mu = torch.tanh(mu) * self._act_scale
        pi = torch.tanh(pi) * self._act_scale
        # print('mu and pi', mu.shape, pi.shape)
        return mu, pi


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.

@torch.no_grad()
def init_weights(m):
    if type(m) == nn.Linear:
        # if index == 0:
        mu, sigma = 0, 0.01  # mean and standard deviation
        # s = torch.from_numpy(np.random.normal(mu, sigma, m.weight.size()).astype('float32'))
        # print(m.weight.dtype)
        m.weight.data.normal_(mu, sigma)
            # = torch.nn.Parameter(s, requires_grad=True)
        # bias = torch.from_numpy(np.random.normal(mu, sigma, m.bias.size()).astype('float32'))
        m.bias.data.normal_(mu, sigma)
            # print(m.weight)
            # print('init size', m.weight.size(), s.size())
            # init_params.append(s)
            # init_bias.append(bias)
        # else:
        #     if if_noise:
        #         noise = act_noise * torch.empty(m.weight.size()).normal_(mean=0, std=0.001)
        #         weight_noise = init_params[m_count] + noise
        #         # print('noise', noise.size(), weight_noise.size())
        #         m.weight = torch.nn.Parameter(weight_noise, requires_grad=True)
        #         bias_noise = act_noise * torch.empty(m.bias.size()).normal_(mean=0, std=0.001)
        #         bias_noise = init_bias[m_count] + bias_noise
        #         m.bias = torch.nn.Parameter(bias_noise, requires_grad=True)
        #     else:
        #         m.weight = torch.nn.Parameter(init_params[m_count], requires_grad=True)
        #         m.bias = torch.nn.Parameter(init_bias[m_count], requires_grad=True)
        #             # print(m.weight.dtype)


class EnsembleActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit, ac_number, act_noise):
        super(EnsembleActor, self).__init__()
        self.net_list = nn.ModuleList(
            [MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit, act_noise) for _ in range(ac_number)])
        # init_weights(self.net_list, act_noise, False)
        for index, actor in enumerate(self.net_list):
            actor.apply(init_weights)
            # print('The ' + str(index) + ' actor!!')
            # for p in actor.parameters():
            #     print(p)
        # exit(0)

    def _heads(self, obs_inputs):
        mus, pis = [], []
        index = 0
        for obs_input in torch.unbind(obs_inputs, dim=0):
            mu, pi = self.net_list[index](obs_input)
            mus.append(mu)
            pis.append(pi)
            index += 1
        return torch.stack(mus, dim=0), torch.stack(pis, dim=0)

    def forward(self, x, k=None):
        if k is not None:
            return self.net_list[k](x)
        else:
            net_heads = self._heads(x)
            return net_heads


class EnsembleCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, ac_number, prior_weight):
        super(EnsembleCritic, self).__init__()
        self.net_list = nn.ModuleList(
            [MLPQFunction(obs_dim, act_dim, hidden_sizes, activation) for _ in range(ac_number)])
        for index, critic in enumerate(self.net_list):
            critic.apply(init_weights)

    def _heads(self, obs_inputs, act_inputs):
        qs = []
        index = 0
        for obs_input, act_input in zip(torch.unbind(obs_inputs, dim=0),
                                                         torch.unbind(act_inputs, dim=0)):
            q = self.net_list[index](obs_input, act_input)
            qs.append(q)
            index += 1
        return torch.stack(qs, dim=0)

    def forward(self, x, act, k=None):
        if k is not None:
            return self.net_list[k](x, act[k])
        else:
            net_heads = self._heads(x, act)
            return net_heads


class MLPActorCriticFactory:
    """Factory of MLP stochastic actors and critics.

    Args:
        observation_space (gym.spaces.Box): A continuous observation space
          specification.
        action_space (gym.spaces.Box): A continuous action space
          specification.
        hidden_sizes (list): A hidden layers shape specification.
        activation (tf.function): A hidden layers activations specification.
        act_noise (float): Stddev for Gaussian exploration noise.
        ac_number (int): Number of the actor-critic models in the ensemble.
        prior_weight (float): Randomly initialized network output scaling.
    """

    def __init__(self, observation_space, action_space, hidden_sizes,
                 activation, prior_weight, act_noise, ac_number):
        self._obs_dim = observation_space.shape[0]
        self._act_dim = action_space.shape[0]
        self._act_scale = action_space.high[0]
        self._hidden_sizes = hidden_sizes
        self._activation = activation
        self._act_noise = act_noise
        self._ac_number = ac_number
        self._prior_weight = prior_weight


    def make_actor(self):
        """Constructs and returns the ensemble of actor models."""
        return EnsembleActor(self._obs_dim, self._act_dim, self._hidden_sizes, self._activation, self._act_scale, self._ac_number, self._act_noise)

    def make_critic(self):
        """Constructs and returns the ensemble of critic models."""
        return EnsembleCritic(self._obs_dim, self._act_dim, self._hidden_sizes, self._activation, self._ac_number, self._prior_weight)
