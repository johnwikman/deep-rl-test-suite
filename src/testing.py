"""
File for testing the Furuta pendulum swing-up task.

Last edit: 2022-03-16
By: dansah
"""

import custom_envs.furuta_swing_up_paper
import custom_envs.furuta_swing_up_paper_r

import spinup.utils.test_policy
import spinup.utils.plot

import torch.nn as nn

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

# Eager execution is required for deep copy. However, tf.placeholder() won't work with eager execution...
#tf.enable_eager_execution()
#import copy

# CPU-Only <- Can result in better performance
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Things to try:
# 1. Normalize reward
# 2. Early exit to avoid solutions that are obviously bad
# 3. Check whether to train all algorithms for the same amount of epochs.
#    Can choose whether to compare based on number of interactions or number of updates.

#########################
# High-level Parameters #
#########################
do_training = False
do_policy_test = True
do_plots = True

#######################
# Training parameters #
#######################
EPOCHS=5                                # The number of parameter updates to perform before stopping trainin. NOTE: Should no longer be used.
MIN_ENV_INTERACTIONS = EPOCHS * 4000    # The minimum number of interactions the agents should perform before stopping training.
use_tensorflow = True                   # Whether to use the Tensorflow versions of the algorithms (if available).
base_dir = '.\out\\'                    # The base directory for storing the output of the algorithms.

#from collections import OrderedDict

def make_env():
    """
    Creates a new Furuta Pendulum environment (swing-up).
    """
    return custom_envs.furuta_swing_up_paper.FurutaPendulumEnvPaper()

def make_env_r():
    """
    Creates a new Furuta Pendulum environment (swing-up),
    where one of the values in the observed state is the previous input to the environment.
    """
    return custom_envs.furuta_swing_up_paper_r.FurutaPendulumEnvPaperRecurrent()

def create_ac_kwargs(mlp_architecture=[64,64], activation_func=tf.nn.relu, arch_dict=dict(), output_dir="", slm_type=False):
    """
    Creates the ac_kwargs dictionary used by the algorithms (primarily the Spin Up ones).
    If slm_type is True, some extra key-value pairs are added that SLM algorithms use.
    """
    ac_kwargs = dict(hidden_sizes=mlp_architecture, activation=activation_func)
    if slm_type:
        ac_kwargs['activation_name'] = arch_dict['activation']
        ac_kwargs['rel_output_dir'] = output_dir.replace(base_dir, "")
    return ac_kwargs

def train_algorithm(alg_dict, arch_dict, max_ep_len=501, epochs=EPOCHS, seed=0): # NOTE: max_ep_len is actually 501.
    """
    Trains the given algorithm. The output is saved in the output directory
    returned by get_output_dir.
    Nothing is returned.
    """
    output_dir = get_output_dir(alg_dict['name'], arch_dict['name'])

    # Based on example from https://spinningup.openai.com/en/latest/user/running.html
    ac_kwargs = create_ac_kwargs(mlp_architecture=arch_dict['layers'], activation_func=get_activation_by_name(arch_dict['activation']))
    logger_kwargs = dict(output_dir=output_dir, exp_name='experiment_test0_' + output_dir)

    algorithm_fn = alg_dict['alg_fn']
    env_fn = alg_dict['env']

    steps_per_epoch = alg_dict['training_frequency']
    
    if (use_tensorflow and alg_dict['type'] == 'spinup') or alg_dict['type'] == 'baselines':
        with tf.Graph().as_default():
            algorithm_fn(env_fn=env_fn, ac_kwargs=ac_kwargs, max_ep_len=max_ep_len, steps_per_epoch=steps_per_epoch, 
                         epochs=epochs, logger_kwargs=logger_kwargs, seed=seed)
        #tf.get_default_session().close()
        #tf.reset_default_graph()
    else:
        if alg_dict['type'] == 'slm':
            ac_kwargs = create_ac_kwargs(mlp_architecture=arch_dict['layers'], activation_func=get_activation_by_name(arch_dict['activation']), 
                                         arch_dict=arch_dict, output_dir=output_dir, slm_type=True)
            ac_kwargs['min_env_interactions'] = MIN_ENV_INTERACTIONS
        algorithm_fn(env_fn=env_fn, ac_kwargs=ac_kwargs, max_ep_len=max_ep_len, steps_per_epoch=steps_per_epoch, 
                     epochs=epochs, logger_kwargs=logger_kwargs, seed=seed)

def evaluate_algorithm(alg_dict, arch_dict):
    """
    Evaluate a trained algorithm by applying it and rendering the result.
    """
    output_dir = get_output_dir(alg_dict['name'], arch_dict['name'])
    if alg_dict['type'] == 'spinup':
        itr = -1
        env, get_action = spinup.utils.test_policy.load_policy_and_env(output_dir,
                                                                       itr if itr >=0 else 'last',
                                                                       False) # Deterministic true/false. Only used by the SAC algorithm.
        spinup.utils.test_policy.run_policy(env, get_action, max_ep_len=500, num_episodes=2, render=True)
        env.close()
    elif alg_dict['type'] == 'baselines':
        model = alg_dict['alg_fn'](env_fn=alg_dict['env'], ac_kwargs=create_ac_kwargs(arch_dict['layers'], get_activation_by_name(arch_dict['activation'])), 
                                   load_path=output_dir)
        # Based on code from the file https://github.com/openai/baselines/blob/master/baselines/run.py
        from baselines import logger
        from baselines.common.vec_env.dummy_vec_env import VecEnv, DummyVecEnv
        from baselines.common import tf_util
        logger.log("Running trained model")
        env = DummyVecEnv(env_fns=[alg_dict['env']])
        obs = env.reset()

        state = model.initial_state if hasattr(model, 'initial_state') else None
        dones = np.zeros((1,))

        current_episode = 0
        num_episodes = 2
        episode_rew = np.zeros(env.num_envs) if isinstance(env, VecEnv) else np.zeros(1)
        while current_episode < num_episodes:
            if state is not None:
                actions, _, state, _ = model.step(obs,S=state, M=dones)
            else:
                actions, _, _, _ = model.step(obs)

            obs, rew, done, _ = env.step(actions)
            episode_rew += rew
            env.render()
            done_any = done.any() if isinstance(done, np.ndarray) else done
            if done_any:
                current_episode += 1
                for i in np.nonzero(done)[0]:
                    print('episode_rew={}'.format(episode_rew[i]))
                    episode_rew[i] = 0
        env.close()
        tf_util.get_session().close()
    elif alg_dict['type'] == 'slm':
        ac_kwargs = create_ac_kwargs(mlp_architecture=arch_dict['layers'], activation_func=get_activation_by_name(arch_dict['activation']), 
                                arch_dict=arch_dict, output_dir=output_dir, slm_type=True)
        ac_kwargs['min_env_interactions'] = 3*501 # 3 epochs
        alg_dict['alg_fn'](env_fn=alg_dict['env'], ac_kwargs=ac_kwargs, max_ep_len=501, steps_per_epoch=501, 
                           epochs=3, logger_kwargs=dict(), seed=0, mode='enjoy')
    else:
        raise NotImplementedError("No handler for algorithm type %s" % (alg_dict['type']))

def annonuce_message(message):
    """
    Prints a message in a fancy way.
    """
    print("------------------------------------")
    print("----> %s <----" % (message))
    print("------------------------------------")

def heads_up_message(message):
    """
    Prints a message in a fancy way.
    """
    print("----> %s <----" % (message))

def get_output_dir(alg_name, arch_name):
    """
    Returns the approriate output directory name relative to the
    project root for the given experiment.
    """
    return base_dir + alg_name + '\\' + arch_name + '\\'

def get_activation_by_name(activation_name):
    """
    Takes the activation function by name as a string, returning
    the appropriate activation function in Tensorflow or Pytorch.
    """
    if activation_name == "relu":
        if use_tensorflow:
            return tf.nn.relu
        else:
            return nn.ReLU
    elif activation_name == "tanh":
        if use_tensorflow:
            return tf.nn.tanh
        else:
            return nn.Tanh
    else:
        raise Exception("Invalid activation function name %s" % (activation_name))

def main():
    """
    Runns all of the experiments.
    """
    if use_tensorflow:
        from spinup import ddpg_tf1 as ddpg
        from spinup import ppo_tf1 as ppo
    else:
        from spinup import ddpg_pytorch as ddpg
        from spinup import ppo_pytorch as ppo
    from baselines.a2c.a2c import a2c as a2c
    from deps.SLM_Lab.dansah_custom.a2c import a2c as a2c_s
    from deps.SLM_Lab.dansah_custom.reinforce import reinforce
    all_algorithms = [
        {
            "name": "a2c_s", # The name of the algorithm. Must be unique, but could be any String without whitespace.
            "alg_fn": a2c_s, # Function that trains an agent using the algorithm. Should comply with the Spin Up API.
            "env": make_env, # Function that returns a new OpenAI Gym environment instance.
            "type": "slm", # Species the implementation type/origin of the algorithm.
            "training_frequency": 4000, # How often updates are performed. NOTE: Could be in terms of experiences or episodes; this depends on the algorithm.
        },
        {
            "name": "a2c",
            "alg_fn": a2c,
            "env" : make_env,
            "type": "baselines",
            "training_frequency": 4000,
        },
        {
            "name": "ddpg",
            "alg_fn": ddpg,
            "env": make_env,
            "type": "spinup",
            "training_frequency": 4000,
        },
        {
            "name": "ppo",
            "alg_fn": ppo,
            "env": make_env,
            "type": "spinup",
            "training_frequency": 4000,
        },
        {
            "name": "ddpg_r",
            "alg_fn": ddpg,
            "env": make_env_r,
            "type": "spinup",
            "training_frequency": 4000,
        },
        {
            "name": "reinforce",
            "alg_fn": reinforce,
            "env": make_env,
            "type": "slm",
            "training_frequency": 1,
        },
    ]
    all_architectures = [
        {
            "name": "64_relu",
            "layers": [64],
            "activation": "relu"
        },
        {
            "name": "64_64_relu",
            "layers": [64, 64],
            "activation": "relu"
        },
        {
            "name": "64_64_tanh",
            "layers": [64, 64],
            "activation": "tanh"
        },
        {
            "name": "256_128_relu",
            "layers": [256, 128],
            "activation": "relu"
        },
        {
            "name": "400_300_relu",
            "layers": [400, 300],
            "activation": "relu"
        },
    ]

    algorithms_to_use = ["reinforce"]#, "a2c_s", "ddpg", "ppo", "ddpg_r"]
    algorithms = []
    for alg_dict in all_algorithms:
        if alg_dict['name'] in algorithms_to_use:
            algorithms.append(alg_dict)

    architecture_to_use = ["64_64_relu", "256_128_relu"] # tanh does not work well; rather useless to try it.
    architectures = []
    for arch_dict in all_architectures:
        if arch_dict['name'] in architecture_to_use:
            architectures.append(arch_dict)

    if do_training:
        for alg_dict in algorithms:
            name = alg_dict['name']
            annonuce_message("Now training with %s" % (name))
            for arch_dict in architectures:
                heads_up_message("Using arch %s" % (arch_dict['name']))
                train_algorithm(alg_dict, arch_dict)

    if do_policy_test:
        for alg_dict in algorithms:
            name = alg_dict['name']
            annonuce_message("Now testing %s" % (name))
            for arch_dict in architectures:
                heads_up_message("Using arch %s" % (arch_dict['name']))
                if use_tensorflow:
                    with tf.Graph().as_default():
                        evaluate_algorithm(alg_dict, arch_dict)
                else:
                    evaluate_algorithm(alg_dict, arch_dict)

    if do_plots:
        dirs = []
        alg_names = []
        for alg_dict in algorithms:
            alg_name = alg_dict['name']
            for arch_dict in architectures:
                arch_name = arch_dict['name']
                dirs.append(get_output_dir(alg_name, arch_name))
                alg_names.append(alg_name + ' ' + arch_name)
        metrics = ['Performance']

        spinup.utils.plot.make_plots(dirs, legend=alg_names, xaxis='TotalEnvInteracts', values=metrics, count=False, smooth=1, select=None, exclude=None, estimator='mean')

if __name__ == "__main__":
    main()