import itertools
import numpy as np
import torch
from torch.optim import Adam
import time
import spinup.algos.pytorch.ed2.core as core
from spinup.utils.logx import EpochLogger
import torch.nn as nn


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

        obs_shape = [self.ac_number, batch_size, self.obs_dim]
        act_shape = [self.ac_number, batch_size, self.act_dim]
        rew_shape = [self.ac_number, batch_size]
        obs1 = np.broadcast_to(self.obs1_buf[idxs], obs_shape)
        obs2 = np.broadcast_to(self.obs2_buf[idxs], obs_shape)
        acts = np.broadcast_to(self.acts_buf[idxs], act_shape)
        rews = np.broadcast_to(self.rews_buf[idxs], rew_shape)
        done = np.broadcast_to(self.done_buf[idxs], rew_shape)

        return dict(obs1=torch.from_numpy(obs1),
                    obs2=torch.from_numpy(obs2),
                    acts=torch.from_numpy(acts),
                    rews=torch.from_numpy(rews),
                    done=torch.from_numpy(done))


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


def ed2(env_fn,
    actor_critic=core.MLPActorCriticFactory,
    ac_kwargs=None,
    ac_number=5,
    steps_per_epoch=4000,
    total_steps=1_000_000,
    replay_size=1_000_000,
    init_ere_coeff=0.995,
    gamma=0.99,
    polyak=0.995,
    lr=3e-4,
    batch_size=256,
    start_steps=10_000,
    update_after=1_000,
    update_every=50,
    train_intensity=1,
    act_noise=0.29,
    use_noise_for_exploration=False,
    use_vote_policy=False,
    max_ep_len=1_000,
    num_test_episodes=10,
    logger_kwargs=None,
    log_every=10_000,
    save_freq=10_000,
    save_path=None,
    trace_rate=None,
    seed=0,
        ):
    """Ensemble Deep Deterministic Policy Gradients.

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in `action_space` kwargs
            and returns actor and critic tf.keras.Model-s.

            Actor should take an observation in and output:
            ===========  ================  =====================================
            Symbol       Shape             Description
            ===========  ================  =====================================
            ``pi``       (batch, act_dim)  | Deterministically computes actions
                                           | from policy given states.
            ===========  ================  =====================================

            Critic should take an observation and action in and output:
            ===========  ================  =====================================
            Symbol       Shape             Description
            ===========  ================  =====================================
            ``q``        (batch,)          | Gives the current estimate of Q*
                                           | state and action in the input.
            ===========  ================  =====================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic
            function you provided to the agent.

        ac_number (int): Number of the actor-critic models in the ensemble.

        total_steps (int): Number of environment interactions to run and train
            the agent.

        replay_size (int): Maximum length of replay buffer.

        init_ere_coeff (float): How much emphasis we put on recent data.
            Always between 0 and 1, where 1 is uniform sampling.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.).

        lr (float): Learning rate (used for both policy and value learning).

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long
            you wait between updates, the ratio of env steps to gradient steps
            is locked to 1 / `train_intensity`.

        train_intensity (float): Number of gradient steps per each env step (see
            `update_every` docstring).

        act_noise (float): Stddev for Gaussian exploration noise added to
            policy at training time (for exploration and smoothing).

        use_noise_for_exploration (bool): If the noise should be added to the
            behaviour policy.

        use_vote_policy (bool): If true use vote_policy during evaluation
            instead of default mean_policy.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        log_every (int): Number of environment interactions that should elapse
            between dumping logs.

        save_freq (int): How often (in terms of environment iterations) to save
            the current policy and value function.

        save_path (str): The path specifying where to save the trained actor
            model. Setting the value to None turns off the saving.

        trace_rate (float): Fraction of episodes to trace, or None if traces
            shouldn't be saved.

        seed (int): Seed for random number generators.
    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # This implementation assumes all dimensions share the same bound!
    assert np.all(env.action_space.high == env.action_space.high[0])

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    # act_limit = env.action_space.high[0]

    ac_kwargs = ac_kwargs or {}
    ac_kwargs['observation_space'] = env.observation_space
    ac_kwargs['action_space'] = env.action_space
    ac_kwargs['act_noise'] = act_noise
    ac_kwargs['ac_number'] = ac_number
    ac_kwargs['activation'] = nn.ReLU


    # Create actor-critic module and target networks
    ac_factory = actor_critic(**ac_kwargs)
    actor = ac_factory.make_actor()
    critic1 = ac_factory.make_critic()
    critic2 = ac_factory.make_critic()

    critic_variables = itertools.chain(critic1.parameters(), critic2.parameters())

    # Target networks
    target_critic1 = ac_factory.make_critic()

    target_critic2 = ac_factory.make_critic()


    # Copy weights
    target_critic1.load_state_dict(critic1.state_dict())
    target_critic2.load_state_dict(critic2.state_dict())


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

    # Separate train ops for pi, q
    # optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    pi_optimizer = Adam(actor.parameters(), lr=lr)
    q_optimizer = Adam(critic_variables, lr=lr)

    def vote_evaluation_policy(obs):
        obs_actor = np.broadcast_to(obs, [ac_number, 1, *obs.shape])
        obs_critic = np.broadcast_to(obs, [ac_number, ac_number, *obs.shape])

        mu, _ = actor(obs_actor)
        # One action per batch.
        act = torch.reshape(mu, [1, ac_number, *mu.shape[2:]])
        # The same action for each component.
        act = np.broadcast_to(act, [ac_number, ac_number, *mu.shape[2:]])
        # Evaluate each action by all components.
        qs = critic1([obs_critic, act])
        # Average over ensemble.
        qs = torch.mean(qs, dim=0)

        return mu[torch.argmax(qs)][0]

    def mean_evaluation_policy(obs):
        obs_actor = np.broadcast_to(obs, [ac_number, 1, *obs.shape])
        # print('The shape of obs_actor is ', obs_actor.shape)
        mu, _ = actor(torch.from_numpy(obs_actor).float())
        return torch.mean(mu, dim=0)[0].detach().numpy()

    if use_vote_policy:
        evaluation_policy = vote_evaluation_policy
    else:
        evaluation_policy = mean_evaluation_policy

    def behavioural_policy(obs, ac_idx, use_noise):
        obs = np.broadcast_to(obs, [ac_number, 1, *obs.shape])
        mu, pi = actor(torch.from_numpy(obs).float())
        if use_noise:
            return pi[ac_idx, 0].detach().numpy()
        else:
            return mu[ac_idx, 0].detach().numpy()

    # Set up model saving
    logger.setup_pytorch_saver(actor)

    def learn_on_batch(obs1, obs2, acts, rews, done):
        mu, _ = actor(obs1)
        q_pi = critic1(obs1, mu)
        pi_loss = - torch.mean(q_pi)

        # Critic update.
        q1 = critic1(obs1, acts)
        q2 = critic2(obs1, acts)

        _, pi_next = actor(obs2)
        min_target_q = torch.min(
            target_critic1(obs2, pi_next),
            target_critic2(obs2, pi_next),
        )

        # Bellman backup for Q function.
        with torch.no_grad():
            q_backup = rews + gamma * (1 - done) * min_target_q
        q1_loss = ((q_backup - q1) ** 2).mean()
        q2_loss = ((q_backup - q2) ** 2).mean()
        value_loss = (q1_loss + q2_loss) * 0.5
        pi_optimizer.zero_grad()
        pi_loss.backward()
        pi_optimizer.step()

        q_optimizer.zero_grad()
        value_loss.backward()
        q_optimizer.step()

        with torch.no_grad():
            for p, p_targ in zip(critic1.parameters(), target_critic1.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)
            for p, p_targ in zip(critic2.parameters(), target_critic2.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

        return dict(
            pi_loss=pi_loss,
            q1_loss=q1_loss,
            q2_loss=q2_loss,
            q1=q1,
            q2=q2,
        )

    def test_agent():
        for _ in range(num_test_episodes):
            o, d, ep_ret, ep_len, task_ret = test_env.reset(), False, 0, 0, 0
            while not(d or (ep_len == max_ep_len)):
                o, r, d, info = test_env.step(
                    evaluation_policy(o))
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

    # Prepare for interaction with environment
    # total_steps = steps_per_epoch * epochs
    start_time = time.time()

    o, ep_ret, ep_len, task_ret, actor_idx = reset_episode(epoch=0)

    # Main loop: collect experience in env and update/log each epoch
    iter_time = time.time()
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
                results = learn_on_batch(**batch)
                metrics = dict(EREcoeff=replay_buffer.ere_coeff,
                               LossPi=results['pi_loss'].detach().numpy(),
                               LossQ1=results['q1_loss'].detach().numpy(),
                               LossQ2=results['q2_loss'].detach().numpy())
                for idx, (q1, q2) in enumerate(
                        zip(results['q1'], results['q2'])):
                    metrics.update({
                        f'Q1Vals_{idx + 1}': q1.detach().numpy(),
                        f'Q2Vals_{idx + 1}': q2.detach().numpy(),
                        f'QDiff_{idx + 1}': torch.abs(q1 - q2).detach().numpy(),
                    })
                logger.store(**metrics)

        # End of epoch handling
        if ((t + 1) % log_every == 0) or (t + 1 == total_steps):
            # Test the performance of the deterministic version of the agent.
            test_agent()
            epoch = (t + 1) // steps_per_epoch
            # Log info about epoch.
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TaskRet', average_only=True)
            logger.log_tabular('TestTaskRet', average_only=True)
            logger.log_tabular('TaskSolved', average_only=True)
            logger.log_tabular('TestTaskSolved', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t + 1)
            logger.log_tabular('EREcoeff', average_only=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ1', average_only=True)
            logger.log_tabular('LossQ2', average_only=True)
            for idx in range(ac_number):
                logger.log_tabular(f'Q1Vals_{idx + 1}', with_min_and_max=True)
                logger.log_tabular(f'Q2Vals_{idx + 1}', with_min_and_max=True)
                logger.log_tabular(f'QDiff_{idx + 1}', with_min_and_max=True)
            logger.log_tabular('StepsPerSecond',
                               log_every / (time.time() - iter_time))
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()

            iter_time = time.time()
        # Save model
        if ((t + 1) % save_freq == 0) or (t + 1 == total_steps):
            if ((t + 1) % save_freq == 0) or (t + 1 == total_steps):
                logger.save_state({'env': env}, None)
