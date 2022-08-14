from copy import deepcopy
import numpy as np
import torch
import random
from torch.optim import Adam
import time
from .logx import EpochLogger


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


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
         act_noise=0.1, max_ep_len=1000,
         logger_kwargs=dict(), save_freq=-1):

    # Set-up logging
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

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
    #ac = actor_critic(env.observation_space, env.action_space, env.is_discrete, **ac_kwargs)
    #ac_targ = deepcopy(ac)
    #q_targ = deepcopy(q)
    #pi_targ = deepcopy(pi)
    q_targ, pi_targ = targ_maker()

    # Experience buffer
    replay_buffer = ReplayMemory(replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(count_vars(module) for module in [pi, q])
    logger.log('\nNumber of parameters: \t pi: %d, \t q: %d\n'%var_counts)

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
        loss_info = dict(QVals=qval.detach().numpy())

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
    logger.setup_pytorch_saver([q, pi])

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
        logger.store(LossQ=loss_q.item(), LossPi=loss_pi.item(), **loss_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for n, n_targ in [(q, q_targ), (pi, pi_targ)]:
                for p, p_targ in zip(n.parameters(), n_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, noise_scale):
        a = pi.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(act_dim)
        if env.is_discrete:
            if len(a.shape) >= 1 and a.shape[0] == 1:
                a = a[0]
            a = int(np.round(a))
        return np.clip(a, act_limit_lower, act_limit_upper)

    if min_env_interactions != 0: # Added by dansah
        epochs = int(np.ceil(min_env_interactions / steps_per_epoch))

    # Set save frequency. The final model is always saved.
    latest_saved_epoch = 0
    if save_freq < 1:
        save_freq = max(int(epochs / 5), 1)

    # Prepare for interaction with environment
    at_least_one_done = False
    latest_epoch = 0
    epoch = 0
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy (with some noise, via act_noise). 
        if t > start_steps:
            a = get_action(o, act_noise)
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
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if (t+1) >= update_after and (t+1) % steps_per_epoch == 0:
            epoch += 1 # NOTE: Technically, the amount of updates is steps_per_epoch times larger than this value.
            for _ in range(steps_per_epoch):
                batch = replay_buffer.sample(batch_size)
                update(data=batch)

        # End of time step handling

        # Save model
        if latest_saved_epoch < epoch and (epoch % save_freq == 0 or (t+1) == total_steps):
            logger.save_state({'env': env}, None)
            latest_saved_epoch = epoch
            print("NOTE: Saved the model, at %s steps." % (t+1))

        # Logging
        real_curr_t = t +1
        if real_curr_t % logger.log_frequency == 0 and epoch > 0:
            assert latest_epoch != epoch
            assert at_least_one_done
            latest_epoch = epoch
            at_least_one_done = False

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', real_curr_t)
            logger.log_tabular('QVals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()
