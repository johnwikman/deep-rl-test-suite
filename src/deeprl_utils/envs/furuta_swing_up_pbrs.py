"""
Based on the PBRSv3 environment

Last edit: 2022-05-03
By: dansah
"""

import gym
import math
import numpy as np

from gym import spaces
from gym.utils import seeding
from typing import Optional

from ipm_furuta import FurutaODE


def calc_reward(theta, dthetadt, phi, dphidt, dt=0.02):
    """
    Calculates a reward for the given state, such that the total
    reward for a trajectory is the number of seconds for which the 
    system was in a satisfactory state.
    NOTE: theta is pi when the vertical arm is upright. The angles should
    not be wrapped.
    """
    reward = dt * (abs(abs(theta) - np.pi) < np.pi / 4) * (abs(phi) < 2*np.pi) * (abs(dthetadt) < 2*np.pi/3)
    if dphidt is not None:
        reward *= (abs(dphidt) < 2*np.pi/3)
    return reward


class FurutaPendulumEnv(gym.core.Env):
    """
    OpenAI Gym wrapper for the Furuta pendulum environment.
    """
    # Optional
    metadata = {"render.modes": ["ansi", "human"]}
    #reward_range = (-float("inf"), float("inf"))
    #spec = None

    def __init__(self, wrap_angles=False):
        # Required
        self.MAX_TORQUE = 200 # Same as in "A Reinforcement Learning Controller for the Swing-Up of the Furuta Pendulum" by D. Guida et al. (2020)
        self.action_space = spaces.Box(low=np.array([-float(self.MAX_TORQUE)]), high=np.array([float(self.MAX_TORQUE)]), dtype=np.float16) # Experiment with np.float32 vs 16.

        self.internal_state = None
        self.START_THETA = math.pi # Radians
        self.TIME_LIMIT = 10.0 # Seconds
        self.DT = 0.02 # Time step size in seconds

        # The following parameters are from "A Reinforcement Learning Controller for the Swing-Up of the Furuta Pendulum" by D. Guida et al. (2020)
        self.r = 1.0 # meters, length of the horizontal arm
        self.l = 0.5 # meters, half the length of the vertical arm
        self.m = 1.0 # kg, mass of the arms

        # Misc
        self._collect_data = False
        self._data = []
        self.viewer = None
        self.np_random = None # Not needed in modern versions of OpenAI Gym

        self.wrap_angles = wrap_angles
        self.is_discrete = False # Used by SLM Lab
        self.seed()

        self.observation_space = spaces.Box(low=-float("inf"), high=float("inf"), shape=(4,), dtype=np.float16)

        # Override constants
        self.c1 = -1
        self.c_lim = -10000
        self.c2 = -5
        self.c_tau = -0.05
        self.c_dot_theta_2 = -0.5
        self.theta_2_min = np.pi
        self.dot_theta_2_min = 5
        self.c_balance = 50

        # Normalization constants
        self.max_phi = 2 * np.pi # c_lim is applied to the reward at > 2*pi, anything beyond is therefore doomed
        self.max_rot_speed = 5 * np.pi # a higher speed is unlikely to yield desirable results. not currently used.

        # From PBRSv2
        self.max_theta = 1.5 * np.pi
    
    def seed(self, seed=None): # Not needed in modern versions of OpenAI Gym
        if seed is not None or self.np_random is None:
            self.np_random, seed = seeding.np_random(seed)
        return np.array([seed])

    def collect_data(self):
        """
        Enable data collection. This will save the thetas and phis.
        """
        self._collect_data = True

    def get_data(self):
        """
        Returns the collected data (thetas and phis).
        The data has the following format:
        [
            {
                "phis": [phi1, ...],
                "thetas": [theta1, ...],
            },
            ...
        ]
        """
        return self._data

    def _internal_step(self, action):
        """
        Transitions the internal environment, without generating an
        observed state or a reward.
        """
        torque = action[0]
        if abs(torque) > self.MAX_TORQUE:
            print("Warning: Maximum Torque exceeded, received value of %d" % torque)
        self.internal_state["furuta_ode"].trans(torque, self.DT)
        if self._collect_data:
            new_state = self._get_internal_state()
            self._data[-1]["phis"].append(new_state[3])
            self._data[-1]["thetas"].append(new_state[0])

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (float): an action provided by the agent, which is the torque of the motor
        Returns:
            observation (numpy float array): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        self._old_internal_state = self._get_internal_state()
        self._internal_step(action)
        torque = action[0]

        internal_state = self._get_internal_state()
        theta = internal_state[0]
        dthetadt = internal_state[1]
        phi = internal_state[3]
        dphidt = internal_state[2]

        terminal = self._terminal_reached()

        # In the paper, theta_2 (here: theta) is 0 when the arm is hanging down vertically, and positive when rotating counter-clockwise.
        # Similarily, theta_1 (here: phi) is positive when rotating counter-clockwise.
        reward = self._calc_reward(theta_1=phi, theta_2=(theta - np.pi), dot_theta_2=dthetadt, tau_c=torque, dot_theta_1=dphidt)

        observed_state = self._get_observed_state_from_internal(internal_state)

        # Add information needed for Baselines (at least A2C) and SLM Lab.
        self.epinfo['r'] += reward
        self.epinfo['l'] += 1
        info_dict = {
            'total_reward': self.epinfo['r']
        }
        if terminal:
            info_dict['episode'] = self.epinfo
            info_dict['rft'] = 'bad_state' if self.non_timelimit_termination else 'timelimit' # Reason For Termination (of the environment).

        if self._collect_data:
            self._data[-1]["total_reward"] = self.epinfo['r']

        return observed_state, reward, terminal, info_dict

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Resets the environment to an initial state and returns an initial
        observation.
        NOTE: The seed parameter is ignored. Use the 'seed' method to seed.
        Returns:
            observation (object): the initial observation.
        """
        #super().reset(seed=seed) # Only works in modern versions of OpenAI Gym
        self.non_timelimit_termination = False
        self.swung_up = False

        # Reset the internal state.
        self.internal_state = {
            "furuta_ode": FurutaODE(wrap_angles=self.wrap_angles),
        }
        self.internal_state["furuta_ode"].init(theta0=self.START_THETA, m=self.m, l=self.l, r=self.r)
        self.epinfo = {'r': np.float16(0), 'l': np.int16(0)}

        new_state = self._get_internal_state()
        if self._collect_data:
            self._data.append({"phis": [new_state[3]], "thetas": [new_state[0]], "total_reward": 0.0})

        return self._get_observed_state_from_internal(new_state)

    def render(self, mode="human"):
        """
        Renders the environment.
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption. <- Not implemented
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
        """
        if mode == "ansi":
            # TODO: Improve
            internal_state = self._get_internal_state()
            theta = internal_state[0]
            angle_diff = np.min([theta, 2*np.pi - theta])
            return "Current angle diff: %s" % (angle_diff)
        elif mode == "human":
            if self.viewer is None:
                from gym.envs.classic_control import rendering

                offset = 1.2

                self.viewer = rendering.Viewer(600, 500)
                self.viewer.set_bounds(-2.64, 2.64, -2.2, 2.2)
                
                # Vertical arm
                vertical_arm = rendering.make_capsule(1, 0.2)
                vertical_arm.set_color(0.8, 0.3, 0.3)
                self.vertical_arm_transform = rendering.Transform(translation=(-offset,0))
                vertical_arm.add_attr(self.vertical_arm_transform)
                self.viewer.add_geom(vertical_arm)
                vertical_axle = rendering.make_circle(0.05)
                vertical_axle.set_color(0, 0, 0)
                vertical_axle.add_attr(rendering.Transform(translation=(-offset,0)))
                self.viewer.add_geom(vertical_axle)

                # Horizontal arm
                horizontal_arm = rendering.make_capsule(1, 0.2)
                horizontal_arm.set_color(0.3, 0.3, 0.8)
                self.horizontal_arm_transform = rendering.Transform(translation=(offset,0))
                horizontal_arm.add_attr(self.horizontal_arm_transform)
                self.viewer.add_geom(horizontal_arm)
                horizontal_axle = rendering.make_circle(0.05)
                horizontal_axle.set_color(0, 0, 0)
                horizontal_axle.add_attr(rendering.Transform(translation=(offset,0)))
                self.viewer.add_geom(horizontal_axle)

            
            state = self._get_internal_state()
            self.vertical_arm_transform.set_rotation(state[0] + np.pi / 2) # An angle of 0 means that it points downwards.
            self.horizontal_arm_transform.set_rotation(state[3] + np.pi / 2)

            return self.viewer.render(return_rgb_array=mode == "rgb_array")
        else:
            raise NotImplementedError

    def close(self):
        """
        Override close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def _get_internal_state(self):
        """
        Return the current internal state. The difference against the observed,
        is that Phi is included.
        """
        internal_state = self.internal_state["furuta_ode"].output(ys=["theta", "dthetadt", "dphidt", "phi"])
        return np.array([internal_state["theta"], internal_state["dthetadt"], internal_state["dphidt"], internal_state["phi"]])

    def _get_observed_state_from_internal(self, internal_state):
        """
        Return the current observed state based on the provided internal.
        Internal state should be of the form [theta, dthetadt, dphidt, phi].
        The oberseved state has the form [phi, dphidt, theta, dthetadt].
        """
        return np.array([internal_state[3], internal_state[2], internal_state[0] - np.pi, internal_state[1]])

    def _calc_reward(self, theta_1, theta_2, dot_theta_2, tau_c, dot_theta_1):
        """
        Calculates the reward.
        """
        if self.non_timelimit_termination:
            return -500 # semi-big negative reward

        def phi_func(theta, dthetadt, phi, dphidt):
            """
            NOTE: Theta should be pi when upright vertical.
            """
            phi_reward = (2*(3*np.pi - abs(abs(theta) - np.pi)) + (3*np.pi - abs(phi)) + \
                         np.maximum(-30, 3 - abs(dthetadt)) + np.maximum(-30, 3 - abs(dphidt))) / 10
            return phi_reward

        reward = calc_reward(theta=theta_2, dthetadt=dot_theta_2, phi=theta_1, dphidt=dot_theta_1, dt=10) # Sparse
        # PBRS, as relayed in "Reward Function Design in Reinforcement Learning" by J. Eschmann (2021) and
        # originally detailed in "Policy invariance under reward transformations: Theory and application to reward shaping"
        # by Andrew Y. Ng et al. (1999)
        # R'(s, a, s') = R(s, a, s') + F(s, s')
        # F(s, s') = gamma * Phi(s') - Phi(s)
        old_theta = self._old_internal_state[0]
        old_dthetadt = self._old_internal_state[1]
        old_phi = self._old_internal_state[3]
        old_dphidt = self._old_internal_state[2]
        outer_reward = 0.99 * phi_func(theta=theta_2, dthetadt=dot_theta_2, phi=theta_1, dphidt=dot_theta_1) - \
                       phi_func(theta=old_theta, dthetadt=old_dthetadt, phi=old_phi, dphidt=old_dphidt)

        return reward + outer_reward

    def _terminal_reached(self):
        """
        Returns true if a terminal state has been reached.
        """
        # Terminate if the vertical arm was swung up but then fell down.
        if not self.non_timelimit_termination:
            theta = abs(self.internal_state["furuta_ode"].output(ys=["theta"])["theta"] - np.pi)
            up_swung = theta > 2/3 * np.pi
            if self.swung_up and not up_swung:
                self.non_timelimit_termination = True
            self.swung_up = self.swung_up or up_swung

        vals = self.internal_state["furuta_ode"].output(ys=["t", "phi", "theta"]) # include "dthetadt", "dphidt" to regulate speed.
        abs_theta = abs(vals["theta"] - np.pi) # theta_2 is theta - np.pi.
        abs_phi = abs(vals["phi"])
        self.non_timelimit_termination = self.non_timelimit_termination or abs_theta > self.max_theta or abs_phi > self.max_phi
        time = vals["t"]
        return time >= self.TIME_LIMIT or self.non_timelimit_termination
