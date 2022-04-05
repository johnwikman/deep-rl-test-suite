from .reward_fns import *


REWARDS = {
    "Pendulum-v0": PendulumReward,
    "MountainCarContinuous-v0": MountainCarContinuousReward,
    "cartpole": CartpoleReward,
}