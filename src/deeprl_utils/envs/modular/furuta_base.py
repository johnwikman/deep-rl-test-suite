"""
Based on the PBRSv3 environment

Last edit: 2022-05-03
By: dansah
"""

import gym
import logging
import math
import numpy as np
import random

from collections import namedtuple
from typing import Optional

from ipm_furuta import FurutaODE

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())

FurutaState = namedtuple("FurutaState", ["theta", "phi", "dtheta", "dphi"])


class FurutaPendulumEnv(gym.core.Env):
    def __init__(self,
                 max_torque=200.0,
                 dt=0.02,
                 discount=0.99,
                 wrap_angles=False):
        self.dt = 0.02
        self.max_torque = max_torque
        self.discount = discount
        self.ode_parameters = {
            "m": 1.0,
            "l": 0.5,
            "r": 1.0,
            "g": 9.81,
            "wrap_angles": wrap_angles,
        }

        # The reset action at the end sets the state
        self.state = {}

        # These are populated by the add_* functions
        self.terminations = {}
        self.initial_conditions = []
        self.sparse_rewards = {}
        self.dense_rewards = {}

        # Misc. OpenAI properties
        self.np_random = None
        self.is_discrete = False

        # Data collection
        self.__data = []
        self.__collect_data = False

        # test-suite variables
        self.MAX_TORQUE = self.max_torque
        self.MAX_EP_LEN = 505

        self.reset()

    @property
    def action_space(self):
        return gym.spaces.Box(
            low   = np.array([-float(self.max_torque)]),
            high  = np.array([float(self.max_torque)]),
            dtype = np.float32
        )

    @property
    def observation_space(self):
        return gym.spaces.Box(
            low   = -np.inf,
            high  = np.inf,
            shape = (4,),
            dtype = np.float32
        )

    def seed(self, seed=None):
        if seed is not None or self.np_random is None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)
        return np.array([seed])

    def collect_data(self, set_to=True):
        self.__collect_data = set_to

    def get_data(self):
        return self.__data

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.state = {
            "t": 0.0,
            "total_reward": 0.0,
            "ode": FurutaODE(wrap_angles=self.ode_parameters["wrap_angles"]),
        }

        state0 = FurutaState(theta=np.pi, phi=0, dtheta=0, dphi=0)
        if len(self.initial_conditions) > 0:
            weights = [ic["_weight"] for ic in self.initial_conditions]
            ic = random.choices(self.initial_conditions, weights=weights, k=1)[0]
            state0 = FurutaState(
                theta  = ic["theta"].sample(),
                phi    = ic["phi"].sample(),
                dtheta = ic["dtheta"].sample(),
                dphi   = ic["dphi"].sample()
            )

        self.state["ode"].init(
            theta0=state0.theta,
            phi0=state0.phi,
            dthetadt0=state0.dtheta,
            dphidt0=state0.dphi,
            m=self.ode_parameters["m"],
            l=self.ode_parameters["l"],
            r=self.ode_parameters["r"],
            g=self.ode_parameters["g"],
        )

        obs = self._get_current_obs()

        if self.__collect_data:
            self.__data.append({
                "phis": [obs.phi],
                "thetas": [obs.theta],
                "total_reward": 0.0,
            })

        return np.array(obs, dtype=np.float32)

    def step(self, action):
        """Take a step."""
        prev_obs = self._get_current_obs()

        LOG.debug("stepping")
        if abs(action[0]) > self.max_torque:
            LOG.warning(f"Maximum Torque exceeded, received value of {action[0]}")
        self.state["ode"].trans(action[0], self.dt)
        self.state["t"] += self.dt

        next_obs = self._get_current_obs()

        LOG.debug("computing rewards")
        sparse_rewards = 0.0
        for name, fn in self.sparse_rewards.items():
            sparse_rewards += fn(self, prev_obs, action, next_obs)

        dense_rewards = 0.0
        for name, fn in self.dense_rewards.items():
            dense_rewards += self.discount * fn(self, next_obs) - fn(self, prev_obs)

        step_reward = sparse_rewards + dense_rewards

        LOG.debug("checking for termination")
        term_name = None
        for name, terminfo in self.terminations.items():
            if terminfo["condition"](self, self.state["t"], next_obs):
                term_name = name

        reward = step_reward
        terminal = False
        info_dict = {}
        if term_name is not None:
            reward = self.terminations[term_name]["reward"]
            if self.terminations[term_name]["include_standard_reward"]:
                reward += step_reward
            terminal = True
            info_dict["rft"] = term_name

        self.state["total_reward"] += reward

        info_dict["total_reward"] = self.state["total_reward"]

        if self.__collect_data:
            self.__data[-1]["phis"].append(next_obs.phi)
            self.__data[-1]["thetas"].append(next_obs.theta)
            self.__data[-1]["total_reward"] = self.state["total_reward"]

        return (
            next_obs,
            reward,
            terminal,
            info_dict,
        )

    def _get_current_obs(self):
        s = self.state["ode"].output(ys=["theta", "phi", "dthetadt", "dphidt"])
        return FurutaState(theta=s["theta"], phi=s["phi"], dtheta=s["dthetadt"], dphi=s["dphidt"])

    def add_termination(self, name, condition, reward, include_standard_reward=False):
        """Adds a termination condition."""
        self.terminations[name] = {
            "condition": condition,
            "reward": reward,
            "include_standard_reward": include_standard_reward
        }

    def add_initial_condition(self, theta, phi, dtheta, dphi, _weight=1.0):
        """Adds an initial condition."""
        assert _weight > 0
        self.initial_conditions.append({
            "_weight": _weight,
            "theta": theta,
            "phi": phi,
            "dtheta": dtheta,
            "dphi": dphi
        })

    def add_sparse_reward(self, name, reward_function):
        """Adds a reward function to be called as R(s,a,s')."""
        self.sparse_rewards[name] = reward_function

    def add_dense_reward(self, name, phi_function):
        """Adds a PBRS reward function to be called as 0.99*Phi(s') - Phi(s)."""
        self.dense_rewards[name] = phi_function
