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

if not os.path.isdir(BASE_DIR):
    os.makedirs(BASE_DIR, mode=0o755)

MODEL_PATH = os.path.join(BASE_DIR, "ddpg_modified.pkl")
STATS_PATH = os.path.join(BASE_DIR, "ddpg_stats.pkl")
PLOT_PATH = os.path.join(BASE_DIR, "ddpg_mean_epret.svg")
HTML_PATH = os.path.join(BASE_DIR, "out.html")


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
        tx_obs = obs - torch.tensor([np.pi, 0.0, 0.0, 0.0]).float()
        qval = self.layers(torch.cat([tx_obs, act], dim=-1))
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

    def forward(self, obs):
        tx_obs = obs - torch.tensor([np.pi, 0.0, 0.0, 0.0]).float()
        return self.layers(tx_obs).mul(200.0)

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

    def make_env():
        from deeprl_utils.envs import FurutaPendulumEnv
        return FurutaPendulumEnv()

    env = make_env()
    test_env = make_env()

    env.seed(seed)
    env.action_space.seed(seed)
    test_env.seed(seed)
    test_env.action_space.seed(seed)

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
        from deeprl_utils.ddpg import ddpg as ddpg_custom
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
        import webbrowser
        import seaborn as sns
        import matplotlib.pyplot as plt
        from deeprl_utils.result_maker import make_html

        if statistics is None:
            LOG.info(f"Loading statistics from {STATS_PATH}")
            with open(STATS_PATH, "rb") as f:
                statistics = pickle.load(f)

        plt.figure()
        sns.set(style="darkgrid", font_scale=1.5)
        sns.lineplot(data=statistics, x="TotalEnvInteracts", y="MeanEpRet", ci="sd")
        plt.legend(loc='best').set_draggable(True)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.tight_layout(pad=0.5)
        plt.savefig(PLOT_PATH)

        LOG.info(f"generating {HTML_PATH}")
        html_dict = {"furuta_pbrs3": {"256_128_relu": {"diagrams": {"Performance": PLOT_PATH}}}}
        make_html(HTML_PATH, html_dict)
        assert webbrowser.get().open("file://" + HTML_PATH)

    if evaluate:
        LOG.info("Evaluating")
        from deeprl_utils.visualizer import plot_animated

        if not train:
            LOG.info(f"Loading model from {MODEL_PATH}")
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
                pi = model[1]

        def get_action(x):
            with torch.no_grad():
                x = torch.as_tensor(x, dtype=torch.float32)
                action = pi.act(x)
            return action

        env.reset()
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

        assert collected_data is not None, "No data was collected for rendering!"
        name = "ddpg - 256_128_relu"
        best_episode_idx = np.argmax([d["total_reward"] for d in collected_data]) # Visualize the best episode
        plot_data = collected_data[best_episode_idx]
        plot_animated(phis=plot_data["phis"], thetas=plot_data["thetas"], l_arm=1.0, l_pendulum=1.0,
                      frame_rate=50, name=name, save_as=None)

        LOG.info(f"Independently defined evaluation data: {[d['total_reward'] for d in collected_data]}")


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-s", "--seed", type=int, help="The seed to use", default=1993)
    argparser.add_argument("-i", "--inter", type=int, help="The number of interactions to target", default=800_000)
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
