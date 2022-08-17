from copy import deepcopy
import numpy as np
import torch
import random
from torch.optim import Adam
import time
import logging
import pandas as pd

#from .logx import EpochLogger

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class ReplayMemory:
    def __init__(self, capacity):
        assert capacity > 0, "must have a positive capacity"
        self.__store = []
        self.__start = 0
        self.capacity = capacity

    def __len__(self):
        return len(self.__store)

    def add(self, e):
        """Adds an element to the replay memory"""
        if len(self.__store) < self.capacity:
            self.__store.append(e)
        else:
            self.__store[self.__start] = e
            self.__start = (self.__start + 1) % self.capacity

    def sample(self, k):
        """Returns a sample of k elements."""
        return random.choices(self.__store, k=k)


def ddpg(env, test_env, q, pi, q_optimizer, pi_optimizer, targ_maker, #q_targ, pi_targ,
         steps_per_epoch=4000, epochs=100, min_env_interactions=0,
         replay_size=int(1e6), gamma=0.99, polyak=0.995,
         batch_size=100, start_steps=10000, update_after=1000,
         act_noise=0.1, max_ep_len=1000, save_freq=-1,
         log_frequency=2000):

    # Set-up logging
    #logger = EpochLogger(**logger_kwargs)
    #logger.save_config(locals())

    statistics = pd.DataFrame(
        index=pd.TimedeltaIndex(data=[], name="ElapsedTime"),
        columns=["Epoch", "EpLen", "TotalEnvInteracts",
                 "AverageEpRet", "StdEpRet", "MaxEpRet", "MinEpRet",
                 "AverageQVals", "StdQVals", "MaxQVals", "MinQVals",
                 "LossPi", "LossQ"]
    )

    # Determine action space
    #env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    if env.is_discrete:
        act_dim = 1
        act_limit_lower = 0
        act_limit_upper = env.action_space.n - 1
    else:
        act_dim = env.action_space.shape[0]
        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        act_limit_lower = env.action_space.low[0]
        act_limit_upper = env.action_space.high[0]

    # Create actor-critic module and target networks
    q_targ, pi_targ = targ_maker()

    # Experience buffer
    replay_buffer = ReplayMemory(replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    LOG.info(f"Number of parameters: (pi: {count_vars(pi)} | q: {count_vars(q)})")

    # Set up function for computing DDPG Q-loss
    def compute_loss_q(data):
        #o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        o = torch.as_tensor(np.array([d[0] for d in data]), dtype=torch.float32)
        a = torch.as_tensor(np.array([d[1] for d in data]), dtype=torch.float32)
        r = torch.as_tensor(np.array([d[2] for d in data]), dtype=torch.float32)
        o2 = torch.as_tensor(np.array([d[3] for d in data]), dtype=torch.float32)
        d = torch.as_tensor(np.array([d[4] for d in data]), dtype=torch.float32)

        qval = q(o,a)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = q_targ(o2, pi_targ(o2))
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((qval - backup)**2).mean()

        # Useful info for logging
        loss_info = {"QVals": qval.detach().numpy()}

        return loss_q, loss_info

    # Set up function for computing DDPG pi loss
    def compute_loss_pi(data):
        #o = data['obs']
        o = torch.as_tensor(np.array([d[0] for d in data]), dtype=torch.float32)
        q_pi = q(o, pi(o))
        return -q_pi.mean()

    # Set up optimizers for policy and q-function
    #pi_optimizer = Adam(pi.parameters(), lr=pi_lr)
    #q_optimizer = Adam(q.parameters(), lr=q_lr)

    # Set up model saving

    def update(data):
        # First run one gradient descent step for Q.
        q_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Record things
        loss_info["LossQ"] = loss_q.item()
        loss_info["LossPi"] = loss_pi.item()

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for n, n_targ in [(q, q_targ), (pi, pi_targ)]:
                for p, p_targ in zip(n.parameters(), n_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

        return loss_info

    if min_env_interactions != 0: # Added by dansah
        epochs = int(np.ceil(min_env_interactions / steps_per_epoch))

    # Set save frequency. The final model is always saved.
    latest_saved_epoch = 0
    if save_freq < 1:
        save_freq = max(int(epochs / 5), 1)

    # Prepare for interaction with environment
    at_least_one_done = False
    prints_since_columnnames = 100
    latest_epoch = 0
    epoch = 0
    total_steps = steps_per_epoch * epochs
    start_time = pd.Timestamp.now()
    o, ep_ret, ep_len = env.reset(), 0, 0


    logstats = {
        "EpLen": [],
        "EpRet": [],
        "LossPi": [],
        "LossQ": [],
        "QVals": []
    }

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy (with some noise, via act_noise). 
        if t > start_steps:
            #a = get_action(o, act_noise)
            #def get_action(o, noise_scale):
            a = pi.act(torch.as_tensor(o, dtype=torch.float32))
            a += act_noise * np.random.randn(act_dim)
            if env.is_discrete:
                if len(a.shape) >= 1 and a.shape[0] == 1:
                    a = a[0]
                a = int(np.round(a))
            a = np.clip(a, act_limit_lower, act_limit_upper)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.add((o, a, r, o2, d))

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            at_least_one_done = True
            logstats["EpRet"].append(ep_ret)
            logstats["EpLen"].append(ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if (t+1) >= update_after and (t+1) % steps_per_epoch == 0:
            epoch += 1 # NOTE: Technically, the amount of updates is steps_per_epoch times larger than this value.
            for _ in range(steps_per_epoch):
                batch = replay_buffer.sample(batch_size)
                loss_info = update(data=batch)
                for k, v in loss_info.items():
                    logstats[k].append(v)

        # End of time step handling

        # Save model
        if latest_saved_epoch < epoch and (epoch % save_freq == 0 or (t+1) == total_steps):
            #logger.save_state({'env': env}, None)
            latest_saved_epoch = epoch
            LOG.info(f"Saved the model, at {t+1} steps. (NOT YET IMPLEMENTED...)")


        # Logging
        real_curr_t = t +1
        if real_curr_t % log_frequency == 0 and epoch > 0:
            assert latest_epoch != epoch
            assert at_least_one_done
            latest_epoch = epoch
            at_least_one_done = False

            epoch_stats = pd.DataFrame(
                index=pd.TimedeltaIndex(data=[pd.Timestamp.now() - start_time], name="ElapsedTime"),
                data={"Epoch": epoch, "TotalEnvInteracts": real_curr_t,
                      "MeanEpRet": np.mean(logstats["EpRet"]),
                      "StdEpRet": np.std(logstats["EpRet"]),
                      "MaxEpRet": np.max(logstats["EpRet"]),
                      "MinEpRet": np.min(logstats["EpRet"]),
                      "MeanQVals": np.mean(logstats["QVals"]),
                      "StdQVals": np.std(logstats["QVals"]),
                      "MaxQVals": np.max(logstats["QVals"]),
                      "MinQVals": np.min(logstats["QVals"]),
                      "MeanEpLen": np.mean(logstats["EpLen"]),
                      "MeanLossPi": np.mean(logstats["LossPi"]),
                      "MeanLossQ": np.mean(logstats["LossQ"])}
            )

            # Reset the aggregates
            for k in list(logstats.keys()):
                logstats[k] = []

            statistics = pd.concat([statistics, epoch_stats])

            _prev_fmt = pd.options.display.float_format
            pd.options.display.float_format = '{:,.2f}'.format

            # Log info about epoch
            lines = str(epoch_stats).split("\n")
            assert len(lines) == 3
            if prints_since_columnnames >= 32:
                LOG.info(lines[0])
                LOG.info(lines[1])
                prints_since_columnnames = 0
            LOG.info(lines[2])
            prints_since_columnnames += 1

            pd.options.display.float_format = _prev_fmt

    return statistics
