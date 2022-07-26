from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
from .logx import EpochLogger


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}



def ddpg(env, test_env, q, pi, steps_per_epoch=4000, epochs=100, min_env_interactions=0,
         replay_size=int(1e6), gamma=0.99, polyak=0.995, pi_lr=1e-3, q_lr=1e-3,
         batch_size=100, start_steps=10000, update_after=1000,
         act_noise=0.1, perform_eval=True, max_ep_len=1000,
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
    
    # Set-up evaluation
    if perform_eval:
        num_test_episodes = 5
    else:
        num_test_episodes = 0

    # Create actor-critic module and target networks
    #ac = actor_critic(env.observation_space, env.action_space, env.is_discrete, **ac_kwargs)
    #ac_targ = deepcopy(ac)
    q_targ = deepcopy(q)
    pi_targ = deepcopy(pi)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for targ in [q_targ, pi_targ]:
        for p in targ.parameters():
            p.requires_grad = False

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(count_vars(module) for module in [pi, q])
    logger.log('\nNumber of parameters: \t pi: %d, \t q: %d\n'%var_counts)

    # Set up function for computing DDPG Q-loss
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

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
        o = data['obs']
        q_pi = q(o, pi(o))
        return -q_pi.mean()

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(pi.parameters(), lr=pi_lr)
    q_optimizer = Adam(q.parameters(), lr=q_lr)

    # Set up model saving
    logger.setup_pytorch_saver([q, pi])

    def update(data):
        # First run one gradient descent step for Q.
        q_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort 
        # computing gradients for it during the policy learning step.
        for p in q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in q.parameters():
            p.requires_grad = True

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

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(o, 0))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

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
        replay_buffer.store(o, a, r, o2, d)

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
                batch = replay_buffer.sample_batch(batch_size)
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

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            if perform_eval:
                logger.log_tabular('TestEpRet', with_min_and_max=True)
                logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', real_curr_t)
            logger.log_tabular('QVals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()
