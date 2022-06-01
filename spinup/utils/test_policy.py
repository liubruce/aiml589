import time
import joblib
import os
import os.path as osp
import tensorflow as tf
import torch
from spinup.utils.logx import EpochLogger
from spinup.utils.logx import restore_tf_graph
import numpy as np

def load_policy_and_env(fpath, itr='last', deterministic=False, ensemble=False):
    """
    Load a policy from save, whether it's TF or PyTorch, along with RL env.

    Not exceptionally future-proof, but it will suffice for basic uses of the
    Spinning Up implementations.

    Checks to see if there's a tf1_save folder. If yes, assumes the model
    is tensorflow and loads it that way. Otherwise, loads as if there's a
    PyTorch save.
    """

    # determine if tf save or pytorch save
    if any(['checkpoint' in x for x in os.listdir(fpath)]):
        backend = 'tf1'
    else:
        backend = 'pytorch'

    # handle which epoch to load from
    if itr == 'last':
        # check filenames for epoch (AKA iteration) numbers, find maximum value

        if backend == 'tf1':
            saves = [int(x[8:]) for x in os.listdir(fpath) if 'tf1_save' in x and len(x) > 8]

        elif backend == 'pytorch':
            pytsave_path = osp.join(fpath, 'pyt_save')
            # Each file in this folder has naming convention 'modelXX.pt', where
            # 'XX' is either an integer or empty string. Empty string case
            # corresponds to len(x)==8, hence that case is excluded.
            saves = [int(x.split('.')[0][5:]) for x in os.listdir(pytsave_path) if len(x) > 8 and 'model' in x]

        itr = '%d' % max(saves) if len(saves) > 0 else ''

    else:
        assert isinstance(itr, int), \
            "Bad value provided for itr (needs to be int or 'last')."
        itr = '%d' % itr

    # load the get_action function
    print('backend is ', backend)
    if backend == 'tf1':
        get_action = load_tf_policy(fpath, itr, deterministic, ensemble)
    else:
        get_action = load_pytorch_policy(fpath, itr, deterministic, ensemble)

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'vars' + itr + '.pkl'))
        env = state['env']
    except:
        env = None

    return env, get_action


def load_tf_policy(fpath, itr, deterministic=False, ensemble=False):
    """ Load a tensorflow policy saved with Spinning Up Logger."""
    print('fpath is ', fpath, itr)
    fname = osp.join(fpath, 'checkpoint' + itr)
    print('\n\nLoading from %s.\n\n' % fname)

    # load the things!
    # sess = tf.Session()
    model = tf.keras.models.load_model(fname)
    # model = restore_tf_graph(sess, fname)

    # get the correct op for executing actions
    # if deterministic and 'mu' in model.keys():
    #     # 'deterministic' is only a valid option for SAC policies
    #     print('Using deterministic action op.')
    #     action_op = model['mu']
    # else:
    #     print('Using default action op.')
    #     action_op = model['pi']

    def get_action(x):
        obs = x
        obs_actor = tf.broadcast_to(obs, [5, 1, *obs.shape])
        mu, _ = model(obs_actor)
        return tf.reduce_mean(mu, axis=0)[0]

    return get_action


def load_pytorch_policy(fpath, itr, deterministic=False, ensemble= False):
    """ Load a pytorch policy saved with Spinning Up Logger."""

    fname = osp.join(fpath, 'pyt_save', 'model' + itr + '.pt')
    print('\n\nLoading from %s.\n\n' % fname)
    print('fname is ', fname)
    model = torch.load(fname)
    print('deterministic is', deterministic, ensemble)
    # make function for producing an action given a single state
    def get_action(x):
        with torch.no_grad():
            if ensemble:
                obs = x
                ac_number = 5
                obs_actor = np.broadcast_to(obs, [ac_number, 1, *obs.shape])
                # print('The shape of obs_actor is ', obs_actor.shape)
                mu, _ = model(torch.from_numpy(obs_actor).float())
                action = torch.mean(mu, dim=0)[0].detach().numpy()
            else:
                x = torch.as_tensor(x, dtype=torch.float32)
                action = model.act(x)
        # return action
            # x = torch.as_tensor(x, dtype=torch.float32)
            # action = model.net_list[0](x)
        return action

    return get_action


def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True):
    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    logger = EpochLogger()
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        a = get_action(o)
        o, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d' % (n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument('--ensemble', '-e', default='False')
    args = parser.parse_args()
    print('args.ensemble is ', args.ensemble)
    env, get_action = load_policy_and_env(args.fpath,
                                          args.itr if args.itr >= 0 else 'last',
                                          args.deterministic, eval(args.ensemble))
    run_policy(env, get_action, args.len, args.episodes, not (args.norender))