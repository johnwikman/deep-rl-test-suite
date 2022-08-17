"""
File for testing the algorithms on various environments, such as
the Furuta pendulum swing-up ones.

NOTE: The word epoch is commonly used to refer to the number of
parameter updates performed throughout this code base.

Last edit: 2022-06-21
By: dansah
"""

import pathlib
import copy
from copy import deepcopy

import torch
import torch.nn as nn

import numpy as np
import os
import pickle
import random
import datetime
import logging
import sys

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())

LOG_FMT = logging.Formatter("%(asctime)s %(name)s:%(lineno)d [%(levelname)s]: %(message)s")

WORK_DIR = pathlib.Path().resolve()
BASE_DIR = os.path.join(WORK_DIR, "out")  # The base directory for storing the output of the algorithms.

MODEL_PATH = os.path.join(BASE_DIR, "ddpg_modified.pkl")
STATS_PATH = os.path.join(BASE_DIR, "ddpg_stats.pkl")
PLOT_PATH = os.path.join(BASE_DIR, "ddpg_mean_epret.svg")
HTML_PATH = os.path.join(BASE_DIR, "out.html")

def make_env_pbrs3():
    """
    Creates a Furuta Pendulum environment (swing-up) similar to PBRS V2,
    but with a big neative reward on early termination.
    """
    from custom_envs.furuta_swing_up_pbrs_v3 import FurutaPendulumEnvPBRS_V3
    return FurutaPendulumEnvPBRS_V3()


#########################
# Misc helper functions #
#########################
def get_output_dir(alg_name, arch_name, env_name, seed):
    """
    Returns the approriate output directory name relative to the
    project root for the given experiment.
    """
    return os.path.join(BASE_DIR + env_name, alg_name, arch_name, "seed" + str(seed) + os.sep)


def get_table_data_filepath():
    """
    Returns the relative filepath at which the
    table containing evaluation data for Furuta Pendulum
    environments should be stored.
    """
    return os.path.join(BASE_DIR, "eval_table")


def plot_training(statistics):
    import webbrowser
    import seaborn as sns
    import matplotlib.pyplot as plt
    from result_maker import make_html
    res_maker_dict = {
        "furuta_pbrs3": {
            "256_128_relu": {
                "diagrams": {"Performance": PLOT_PATH}
            }
        }
    }
    LOG.info("Analyzing metrics for environment furuta_pbrs3")

    plt.figure()
    sns.set(style="darkgrid", font_scale=1.5)
    sns.lineplot(data=statistics, x="TotalEnvInteracts", y="MeanEpRet", ci="sd")

    plt.legend(loc='best').set_draggable(True)

    if statistics["TotalEnvInteracts"].max() > 5e3:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plt.tight_layout(pad=0.5)

    plt.savefig(PLOT_PATH)

    LOG.info(f"generating {HTML_PATH}")
    make_html(HTML_PATH, res_maker_dict)
    LOG.info("opening in webbrowser")
    assert webbrowser.get().open("file://" + HTML_PATH)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(5, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, obs, act):
        qval = self.layers(torch.cat([obs, act], dim=-1))
        return torch.squeeze(qval, -1) # Critical to ensure q has right shape.


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )
        #self.scale = (action_space.high[0] - action_space.low[0]) / 2
        #self.mid = act_limit_lower + self.act_limit

    def forward(self, obs):
        return self.layers(obs).mul(200.0)

    def act(self, obs):
        with torch.no_grad():
            return self.forward(obs).numpy()


def new_implementation(seed=0, inter=0, train=False, plot=False, evaluate=False):
    """
    Heavily modified implementation, using a different flow.
    """
    LOG.info("Setting up environment")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = make_env_pbrs3()
    test_env = make_env_pbrs3()

    env.seed(seed)
    env.action_space.seed(seed)
    test_env.seed(seed)
    test_env.action_space.seed(seed)
    plot
    evaluate

    # Sanity check for constants later on
    assert env.MAX_TORQUE == 200.0
    assert len(env.observation_space.low) == 4
    assert len(env.action_space.low) == 1

    q = Critic()
    pi = Actor()

    q_opt = torch.optim.Adam(q.parameters(), lr=5e-4)
    pi_opt = torch.optim.Adam(pi.parameters(), lr=5e-4)

    def targ_maker():
        return (Critic(), Actor())

    max_ep_len = 501
    num_eval_episodes = 5
    statistics = None

    if train:
        LOG.info("Training")
        from deps.spinningup.dansah_modified.ddpg import ddpg as ddpg_custom
        statistics = ddpg_custom(env, test_env, q, pi, q_opt, pi_opt, targ_maker,
                                 max_ep_len=max_ep_len,
                                 steps_per_epoch=256,
                                 min_env_interactions=inter,
                                 log_frequency=2000,
                                 start_steps=10000)

        LOG.info(f"Saving model to {MODEL_PATH}")
        with open(MODEL_PATH, "wb+") as f:
            pickle.dump([q, pi], f)

        LOG.info(f"Saving statistics to {STATS_PATH}")
        with open(STATS_PATH, "wb+") as f:
            pickle.dump(statistics, f)

    if plot:
        LOG.info("Plotting")
        if statistics is None:
            LOG.info(f"Loading statistics from {STATS_PATH}")
            with open(STATS_PATH, "rb") as f:
                statistics = pickle.load(f)

        plot_training(statistics)

    if evaluate:
        LOG.info("Evaluating")
        from custom_envs.furuta_swing_up_eval import FurutaPendulumEnvEvalWrapper
        from deps.spinningup.dansah_modified import test_policy
        from deps.visualizer.visualizer import plot_animated

        if not train:
            LOG.info(f"Loading model from {MODEL_PATH}")
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
                #q = model[0]
                pi = model[1]

        def get_action(x):
            with torch.no_grad():
                x = torch.as_tensor(x, dtype=torch.float32)
                action = pi.act(x)
            return action

        #output_dir = get_output_dir("ddpg_modified", "256_128_relu", "furuta_pbrs3", seed)

        #env, get_action = test_policy.load_policy_and_env(output_dir,
        #                                                  'last',
        #                                                  deterministic=False,
        #                                                  force_disc=False)
        env = FurutaPendulumEnvEvalWrapper(env=env, seed=seed, qube2=False)
        env.collect_data()

        eval_data = []
        for ep in range(num_eval_episodes):
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            for _ in range(max_ep_len):
                a = get_action(o)
                o, r, d, _ = env.step(a)
                ep_ret += r
                ep_len += 1
                if d:
                    break

            eval_data.append(ep_ret)
            LOG.info(f"Episode {ep} | EpRet {ep_ret:.3f} | EpLen {ep_len}")

        env.reset()
        collected_data = env.get_data()
        env.close()
        independent_furuta_data = env.get_internal_rewards()

        assert collected_data is not None, "No data was collected for rendering!"
        name = "ddpg - 256_128_relu"
        best_episode_idx = np.argmax(independent_furuta_data) # Visualize the best episode
        plot_data = collected_data[best_episode_idx]
        plot_animated(phis=plot_data["phis"], thetas=plot_data["thetas"], l_arm=1.0, l_pendulum=1.0,
                      frame_rate=50, name=name, save_as=None)

        LOG.info(f"Independently defined evaluation data: {independent_furuta_data}")


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-s", "--seed", type=int, help="The seed to use", default=1993)
    argparser.add_argument("-i", "--inter", type=int, help="The number of interactions to target", default=500_000)
    argparser.add_argument("-p", "--plot", action="store_true", help="Produce plots")
    argparser.add_argument("-t", "--train", action="store_true", help="Perform training")
    argparser.add_argument("-e", "--evaluate", action="store_true", help="Perform evaluation")
    argparser.add_argument("-v", "--verbose", dest="verbosity", action="count", default=0, help="Increase verbosity of prints.")
    args = argparser.parse_args()

    # Setup the root logger
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(LOG_FMT)
    logging.getLogger().addHandler(stderr_handler)
    if args.verbosity >= 1:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    new_implementation(seed=args.seed, inter=args.inter, train=args.train, plot=args.plot, evaluate=args.evaluate)
