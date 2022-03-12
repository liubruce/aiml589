from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.ed2.core as core
from spinup.utils.logx import EpochLogger


class ReplayBuffer:
    """A simple FIFO experience replay buffer with ERE."""

    def __init__(self, obs_dim, act_dim, size, ac_number, max_ep_len,
                 init_ere_coeff):
        # self.obs1_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        # self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        # self.acts_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)

        self.ptr = 0
        self.size = 0

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_size = size
        self.ac_number = ac_number
        self.init_ere_coeff = init_ere_coeff
        self.ere_coeff = init_ere_coeff

        # Emphasize Recent Experience (ERE)
        imprv_span_step = self.max_size // 2
        imprv_span_ep = imprv_span_step / max_ep_len
        self.warmup_size = imprv_span_step + max_ep_len
        self.prev_ratio = 1 / imprv_span_ep
        self.recent_ratio = 1 / (imprv_span_ep * 0.1)
        self.prev_ep_ret = None
        self.recent_ep_ret = None
        self.max_imprv = 0

    def store(self, obs, act, rew, next_obs, done):
        """Store the transitions in the replay buffer."""
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size, most_recent):
        """Sample batch of (`most_recent`) experience."""
        # Emphasize Recent Experience (ERE)
        most_recent = max(most_recent, 5000)  # Value from the paper.
        # Guard for when buffer is not full yet.
        most_recent = min(most_recent, self.size)

        idxs = np.random.randint(
            self.size - most_recent, self.size, size=batch_size)
        # Shifts the range to the actual end of the buffer.
        idxs = (idxs + self.ptr) % self.size

        # obs_shape = [self.ac_number, batch_size, self.obs_dim]
        # act_shape = [self.ac_number, batch_size, self.act_dim]
        # rew_shape = [self.ac_number, batch_size]
        # print('obs_shape is ', obs_shape)
        # obs1 = np.broadcast_to(self.obs1_buf[idxs], obs_shape)
        # obs2 = np.broadcast_to(self.obs2_buf[idxs], obs_shape)
        # acts = np.broadcast_to(self.acts_buf[idxs], act_shape)
        # rews = np.broadcast_to(self.rews_buf[idxs], rew_shape)
        # done = np.broadcast_to(self.done_buf[idxs], rew_shape)
        # return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

        batch = dict(obs1=self.obs1_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     acts=self.acts_buf[idxs],
                     rews=self.rews_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


    def end_trajectory(self, ep_ret):
        """Bookkeeping at the end of the trajectory."""
        if self.prev_ep_ret is None and self.recent_ep_ret is None:
            self.prev_ep_ret = ep_ret
            self.recent_ep_ret = ep_ret
        else:
            self.prev_ep_ret = (self.prev_ratio * ep_ret +
                                (1 - self.prev_ratio) * self.prev_ep_ret)
            self.recent_ep_ret = (self.recent_ratio * ep_ret +
                                  (1 - self.recent_ratio) * self.recent_ep_ret)

        # Adapt ERE coeff.
        if self.size > self.warmup_size:
            recent_imprv = self.recent_ep_ret - self.prev_ep_ret
            self.max_imprv = max(self.max_imprv, recent_imprv)

            try:
                imprv_rate = max(recent_imprv / self.max_imprv, 0.)
            except ZeroDivisionError:
                imprv_rate = 0

            self.ere_coeff = self.init_ere_coeff * imprv_rate + (1 - imprv_rate)


def ed2(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, total_steps=1_000_000,
        replay_size=int(1e6), gamma=0.99,
        polyak=0.995, pi_lr=3e-4, q_lr=3e-4, batch_size=256, start_steps=10_000,
        update_after=1000, update_every=50, act_noise=0.29, target_noise=0.2,
        noise_clip=0.5, num_test_episodes=10, max_ep_len=1000,
        logger_kwargs=dict(), save_freq=10_000,  ac_number=5,
        init_ere_coeff=0.995,  train_intensity=1, use_noise_for_exploration=False,
        use_vote_policy=False,
        log_every=10_000,
        ):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            these should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``pi``       (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to TD3.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        act_noise (float): Stddev for Gaussian exploration noise added to 
            policy at training time. (At test time, no noise is added.)

        target_noise (float): Stddev for smoothing noise added to target 
            policy.

        noise_clip (float): Limit for absolute value of target policy 
            smoothing noise.

        policy_delay (int): Policy will only be updated once every 
            policy_delay times for each update of the Q-networks.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)



    pi_optimizers = []
    q_optimizers = []
    q_params = []
    for i in range(ac_number):
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in ac_targ.L_qf1[i].parameters():
            p.requires_grad = False
        for p in ac_targ.L_qf2[i].parameters():
            p.requires_grad = False
        for p in ac_targ.L_policy[i].parameters():
            p.requires_grad = False
        # List of parameters for both Q-networks (save this for convenience)
        q_param = itertools.chain(ac.L_qf1[i].parameters(), ac.L_qf2[i].parameters())
        # Set up optimizers for policy and q-function
        pi_optimizers.append(Adam(ac.L_policy[i].parameters(), lr=pi_lr))
        q_optimizers.append(Adam(q_param, lr=q_lr))
        q_params.append(q_param)
    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim,
                                 act_dim=act_dim,
                                 size=replay_size,
                                 ac_number=ac_number,
                                 max_ep_len=max_ep_len,
                                 init_ere_coeff=init_ere_coeff)
    # Count variables (protip: try to get a feel for how different size networks behave!)
    # var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    # logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)

    # Set up function for computing TD3 Q-losses
    def compute_loss_q(data,ac_index):
        o, a, r, o2, d = data['obs1'], data['acts'], data['rews'], data['obs2'], data['done']

        q1 = ac.L_qf1[ac_index](o,a)
        q2 = ac.L_qf2[ac_index](o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            pi_targ = ac.L_policy[ac_index](o2)

            # Target policy smoothing
            epsilon = torch.randn_like(pi_targ) * target_noise
            epsilon = torch.clamp(epsilon, -noise_clip, noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(a2, -act_limit, act_limit)

            # Target Q-values
            q1_pi_targ = ac_targ.L_qf1[ac_index](o2, a2)
            q2_pi_targ = ac_targ.L_qf2[ac_index](o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = (loss_q1 + loss_q2) * 0.5

        # Useful info for logging
        loss_info = dict(Q1Vals=q1.detach().numpy(),
                         Q2Vals=q2.detach().numpy())

        return loss_q, loss_info

    # Set up function for computing TD3 pi loss
    def compute_loss_pi(data,ac_index):
        o = data['obs1']
        q1_pi = ac.L_qf1[ac_index](o, ac.L_policy[ac_index](o))
        return -q1_pi.mean()



    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data):
        for i in range(ac_number):
            # First run one gradient descent step for Q1 and Q2
            q_optimizers[i].zero_grad()
            loss_q, loss_info = compute_loss_q(data,i)
            loss_q.backward()
            q_optimizers[i].step()

            # Record things
            logger.store(LossQ=loss_q.item(), **loss_info)

            # Possibly update pi and target networks

            # Freeze Q-networks so you don't waste computational effort
            # computing gradients for them during the policy learning step.
            for p in q_params[i]:
                p.requires_grad = False

            # Next run one gradient descent step for pi.
            pi_optimizers[i].zero_grad()
            loss_pi = compute_loss_pi(data, i)
            loss_pi.backward()
            pi_optimizers[i].step()

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in q_params[i]:
                p.requires_grad = True

            # Record things
            logger.store(LossPi=loss_pi.item())

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(ac.L_qf1[i].parameters(), ac_targ.L_qf1[i].parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)
                for p, p_targ in zip(ac.L_qf2[i].parameters(), ac_targ.L_qf2[i].parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, noise_scale):
        a = ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len, task_ret = test_env.reset(), False, 0, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                actions = []
                for k in range(ac_number):
                  actions.append(ac.L_policy[k](torch.as_tensor(o, dtype=torch.float32)))
                mean = torch.mean(torch.stack(actions), dim=0)
                o, r, d, _ = test_env.step(mean.detach().numpy())
                ep_ret += r
                ep_len += 1
                task_ret += info.get('reward_task', 0)
            logger.store(TestEpRet=ep_ret,
                         TestEpLen=ep_len,
                         TestTaskRet=task_ret,
                         TestTaskSolved=info.get('is_solved', False))
    def reset_episode(epoch):
        o, ep_ret, ep_len, task_ret = env.reset(), 0, 0, 0
        # tracer.on_episode_begin(env, o, epoch)
        actor_idx = np.random.choice(ac_number)  # Select policy
        return o, ep_ret, ep_len, task_ret, actor_idx

    def behavioural_policy(obs, ac_idx, use_noise):
        return ac.L_policy[ac_idx](torch.as_tensor(obs, dtype=torch.float32)).detach().numpy()



    # Prepare for interaction with environment
    # total_steps = steps_per_epoch * epochs
    start_time = time.time()

    o, ep_ret, ep_len, task_ret, actor_idx = reset_episode(epoch=0)

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy (with some noise, via act_noise). 
        if t > start_steps:
            # a = get_action(o, act_noise)
            a = behavioural_policy(o,
                                   actor_idx,
                                   use_noise_for_exploration)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, info = env.step(a)
        ep_ret += r
        ep_len += 1
        task_ret += info.get('reward_task', 0)
        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len,
                         TaskRet=task_ret,
                         TaskSolved=info.get('is_solved', False))
            replay_buffer.end_trajectory(ep_ret)
            o, ep_ret, ep_len, task_ret, actor_idx = reset_episode(
                epoch=(t + 1) // log_every)

        # Update handling
        if t >= update_after and t % update_every == 0:
            number_of_updates = int(update_every * train_intensity)
            for n in range(number_of_updates):
                most_recent = (
                    replay_buffer.max_size * replay_buffer.ere_coeff ** (
                        (n + 1) * 1000 / number_of_updates))
                batch = replay_buffer.sample_batch(batch_size,most_recent)
                update(data=batch)
                # metrics = dict(EREcoeff=replay_buffer.ere_coeff,
                #                LossPi=results['pi_loss'],
                #                LossQ1=results['q1_loss'],
                #                LossQ2=results['q2_loss'])
                # for idx, (q1, q2) in enumerate(
                #         zip(results['q1'], results['q2'])):
                #     metrics.update({
                #         f'Q1Vals_{idx + 1}': q1,
                #         f'Q2Vals_{idx + 1}': q2,
                #         f'QDiff_{idx + 1}': np.abs(q1 - q2),
                #     })
                # logger.store(**metrics)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Save model
            if ((t + 1) % save_freq == 0) or (t + 1 == total_steps):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='td3')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ed2(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
