import torch
from torch.optim import Adam
from deps.pytorch_rl_il.rlil.agents import AIRL
from deps.pytorch_rl_il.dansah_custom.initializer import get_device, set_replay_buffer, get_replay_buffer
from .models import fc_reward, fc_v
from deps.pytorch_rl_il.rlil.approximation import Approximation, Discriminator, VNetwork
from deps.pytorch_rl_il.rlil.memory import ExperienceReplayBuffer, AirlWrapper


def airl(
        transitions=None,
        base_agent_fn=None,
        # Common settings
        discount_factor=0.98,
        # Adam optimizer settings
        lr_r=2e-4,
        lr_v=2e-4,
        # Training settings
        minibatch_size=512,
        update_frequency=1,
        # Replay Buffer settings
        replay_start_size=5000,
        replay_buffer_size=1e6
):
    """
    Adversarial Inverse Reinforcement Learning (AIRL) control preset

    Args:
        transitions:
            dictionary of transitions generated by cpprb.ReplayBuffer.get_all_transitions() 
        base_agent_fn (function):
            A function generated by a preset of an agent such as sac, td3, ddpg
            Currently, the base_agent_fn must be ppo preset.
        lr_r (float): Learning rate for the reward function network.
        lr_v (float): Learning rate for the value function network.
        update_frequency (int): Number of base_agent update per discriminator update.
        minibatch_size (int): Number of experiences to sample in each discriminator update.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        replay_buffer_size (int): Maximum number of experiences to store in the replay buffer.
    """
    def _airl(env):
        device = get_device()

        base_agent = base_agent_fn(env)

        reward_model = fc_reward(env).to(device)
        reward_optimizer = Adam(reward_model.parameters(),
                                lr=lr_r)
        reward_fn = Approximation(reward_model,
                                  reward_optimizer,
                                  name='airl_rew')

        value_model = fc_v(env).to(device)
        value_optimizer = Adam(value_model.parameters(),
                               lr=lr_v)
        value_fn = VNetwork(value_model,
                            value_optimizer,
                            name='airl_v')

        expert_replay_buffer = ExperienceReplayBuffer(1e7, env)
        if transitions is not None:
            samples = expert_replay_buffer.samples_from_cpprb(
                transitions, device="cpu")
            expert_replay_buffer.store(samples)

        replay_buffer = get_replay_buffer()
        replay_buffer = AirlWrapper(buffer=replay_buffer,
                                    expert_buffer=expert_replay_buffer,
                                    reward_fn=reward_fn,
                                    value_fn=value_fn,
                                    policy=base_agent.policy,
                                    feature_nw=base_agent.feature_nw,
                                    discount_factor=discount_factor)
        set_replay_buffer(replay_buffer)

        # replace base_agent's replay_buffer with gail_buffer
        base_agent.replay_buffer = replay_buffer

        return AIRL(
            base_agent=base_agent,
            minibatch_size=minibatch_size,
            replay_start_size=replay_start_size,
            update_frequency=update_frequency
        )
    return _airl


__all__ = ["airl"]
