"""Ensemble Deep Deterministic Policy Gradients algorithm implementation."""

import os
import os.path as osp
import random
import time

import numpy as np
import tensorflow as tf

from spinup.algos.tf2.ed2 import core
from spinup.utils import logx
from spinup.utils import tracing


class ReplayBuffer:
    """A simple FIFO experience replay buffer with ERE."""

    def __init__(self, obs_dim, act_dim, size, ac_number, max_ep_len,
                 init_ere_coeff):
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
        return dict(obs1=tf.convert_to_tensor(obs1),
                    obs2=tf.convert_to_tensor(obs2),
                    acts=tf.convert_to_tensor(acts),
                    rews=tf.convert_to_tensor(rews),
                    done=tf.convert_to_tensor(done))

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


def ed2(
    env_fn,
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
    pwd = os.getcwd()  # pylint: disable=possibly-unused-variable
    logger = logx.EpochLogger(**(logger_kwargs or {}))
    logger.save_config(locals())

    trace_dir = (osp.join(logger.output_dir, 'traces')
                 if trace_rate is not None else None)
    tracer = tracing.TraceCallback(trace_dir, trace_rate or 0.0)

    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    # This implementation assumes all dimensions share the same bound!
    assert np.all(env.action_space.high == env.action_space.high[0])

    # Share information about action space with policy architecture
    ac_kwargs = ac_kwargs or {}
    ac_kwargs['observation_space'] = env.observation_space
    ac_kwargs['action_space'] = env.action_space
    ac_kwargs['act_noise'] = act_noise
    ac_kwargs['ac_number'] = ac_number

    # Network
    ac_factory = actor_critic(**ac_kwargs)

    actor = ac_factory.make_actor()
    actor.build(input_shape=(None, obs_dim))

    critic1 = ac_factory.make_critic()
    critic1.build(input_shape=(None, obs_dim + act_dim))
    critic2 = ac_factory.make_critic()
    critic2.build(input_shape=(None, obs_dim + act_dim))

    critic_variables = critic1.trainable_variables + critic2.trainable_variables

    # Target networks
    target_critic1 = ac_factory.make_critic()
    target_critic1.build(input_shape=(None, obs_dim + act_dim))
    target_critic2 = ac_factory.make_critic()
    target_critic2.build(input_shape=(None, obs_dim + act_dim))

    # Copy weights
    target_critic1.set_weights(critic1.get_weights())
    target_critic2.set_weights(critic2.get_weights())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim,
                                 act_dim=act_dim,
                                 size=replay_size,
                                 ac_number=ac_number,
                                 max_ep_len=max_ep_len,
                                 init_ere_coeff=init_ere_coeff)

    # Separate train ops for pi, q
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    @tf.function
    def vote_evaluation_policy(obs):
        obs_actor = tf.broadcast_to(obs, [ac_number, 1, *obs.shape])
        obs_critic = tf.broadcast_to(obs, [ac_number, ac_number, *obs.shape])

        mu, _ = actor(obs_actor)
        # One action per batch.
        act = tf.reshape(mu, [1, ac_number, *mu.shape[2:]])
        # The same action for each component.
        act = tf.broadcast_to(act, [ac_number, ac_number, *mu.shape[2:]])
        # Evaluate each action by all components.
        qs = critic1([obs_critic, act])
        # Average over ensemble.
        qs = tf.reduce_mean(qs, axis=0)

        return mu[tf.math.argmax(qs)][0]

    @tf.function
    def mean_evaluation_policy(obs):
        obs_actor = tf.broadcast_to(obs, [ac_number, 1, *obs.shape])
        mu, _ = actor(obs_actor)
        return tf.reduce_mean(mu, axis=0)[0]

    if use_vote_policy:
        evaluation_policy = vote_evaluation_policy
    else:
        evaluation_policy = mean_evaluation_policy

    @tf.function
    def behavioural_policy(obs, ac_idx, use_noise):
        obs = tf.broadcast_to(obs, [ac_number, 1, *obs.shape])
        mu, pi = actor(obs)
        if use_noise:
            return pi[ac_idx, 0]
        else:
            return mu[ac_idx, 0]

    @tf.function
    def learn_on_batch(obs1, obs2, acts, rews, done):
        with tf.GradientTape(persistent=True) as g:
            # Actor update.
            mu, _ = actor(obs1)
            q_pi = critic1([obs1, mu])
            pi_loss = -tf.reduce_mean(q_pi)

            # Critic update.
            q1 = critic1([obs1, acts])
            q2 = critic2([obs1, acts])

            _, pi_next = actor(obs2)
            min_target_q = tf.minimum(
                target_critic1([obs2, pi_next]),
                target_critic2([obs2, pi_next]),
            )

            # Bellman backup for Q function.
            q_backup = tf.stop_gradient(
                rews + gamma * (1 - done) * min_target_q)
            q1_loss = tf.reduce_mean((q_backup - q1) ** 2)
            q2_loss = tf.reduce_mean((q_backup - q2) ** 2)
            value_loss = (q1_loss + q2_loss) * 0.5

        actor_gradients = g.gradient(pi_loss, actor.trainable_variables)
        optimizer.apply_gradients(
            zip(actor_gradients, actor.trainable_variables)
        )
        critic_gradients = g.gradient(value_loss, critic_variables)
        optimizer.apply_gradients(
            zip(critic_gradients, critic_variables))

        for v, target_v in zip(critic1.trainable_variables,
                               target_critic1.trainable_variables):
            target_v.assign(polyak * target_v + (1 - polyak) * v)
        for v, target_v in zip(critic2.trainable_variables,
                               target_critic2.trainable_variables):
            target_v.assign(polyak * target_v + (1 - polyak) * v)

        del g
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
            while not (d or (ep_len == max_ep_len)):
                o, r, d, info = test_env.step(
                    evaluation_policy(tf.convert_to_tensor(o)))
                ep_ret += r
                ep_len += 1
                task_ret += info.get('reward_task', 0)
            logger.store(TestEpRet=ep_ret,
                         TestEpLen=ep_len,
                         TestTaskRet=task_ret,
                         TestTaskSolved=info.get('is_solved', False))

    def reset_episode(epoch):
        o, ep_ret, ep_len, task_ret = env.reset(), 0, 0, 0
        tracer.on_episode_begin(env, o, epoch)
        actor_idx = np.random.choice(ac_number)  # Select policy
        return o, ep_ret, ep_len, task_ret, actor_idx

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len, task_ret, actor_idx = reset_episode(epoch=0)

    # Main loop: collect experience in env and update/log each epoch
    iter_time = time.time()
    for t in range(total_steps):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy (with some noise, via act_noise).
        if t > start_steps:
            a = behavioural_policy(tf.convert_to_tensor(o),
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
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Trace experience.
        info['actor_idx'] = actor_idx
        info['total_steps'] = t + 1
        tracer.on_real_step(info, a, o2, r, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling.
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret,
                         EpLen=ep_len,
                         TaskRet=task_ret,
                         TaskSolved=info.get('is_solved', False))
            replay_buffer.end_trajectory(ep_ret)
            tracer.on_episode_end()
            o, ep_ret, ep_len, task_ret, actor_idx = reset_episode(
                epoch=(t + 1) // log_every)

        # Update handling
        if (t + 1) >= update_after and (t + 1) % update_every == 0:
            number_of_updates = int(update_every * train_intensity)
            for n in range(number_of_updates):
                most_recent = (
                    replay_buffer.max_size * replay_buffer.ere_coeff ** (
                        (n + 1) * 1000 / number_of_updates))
                batch = replay_buffer.sample_batch(batch_size, most_recent)
                results = learn_on_batch(**batch)
                metrics = dict(EREcoeff=replay_buffer.ere_coeff,
                               LossPi=results['pi_loss'],
                               LossQ1=results['q1_loss'],
                               LossQ2=results['q2_loss'])
                for idx, (q1, q2) in enumerate(
                        zip(results['q1'], results['q2'])):
                    metrics.update({
                        f'Q1Vals_{idx + 1}': q1,
                        f'Q2Vals_{idx + 1}': q2,
                        f'QDiff_{idx + 1}': np.abs(q1 - q2),
                    })
                logger.store(**metrics)

        # End of epoch wrap-up
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
            if save_path is not None:
                tf.keras.models.save_model(actor, save_path)
