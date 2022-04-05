import torch
from deps.pytorch_rl_il.dansah_custom.initializer import set_replay_buffer, get_replay_buffer
from deps.pytorch_rl_il.rlil.memory import ExperienceReplayBuffer, SqilWrapper


def sqil(
        transitions=None,
        base_agent_fn=None,
        # Replay Buffer settings
        replay_start_size=5000,
        replay_buffer_size=1e7
):
    """
    Soft Q Imitation Learning (SQIL) control preset

    Args:
        transitions:
            dictionary of transitions generated by cpprb.ReplayBuffer.get_all_transitions() 
        base_agent_fn (function):
            A function generated by a preset of an agent such as sac, td3, ddpg
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        replay_buffer_size (int): Maximum number of experiences to store in the replay buffer.
    """
    def _sqil(env):
        base_agent = base_agent_fn(env)
        expert_replay_buffer = ExperienceReplayBuffer(1e7, env)
        if transitions is not None:
            samples = expert_replay_buffer.samples_from_cpprb(
                transitions, device="cpu")
            expert_replay_buffer.store(samples)

        replay_buffer = get_replay_buffer()
        replay_buffer = SqilWrapper(replay_buffer,
                                    expert_replay_buffer)
        set_replay_buffer(replay_buffer)
        # replace base_agent's replay_buffer with gail_buffer
        base_agent.replay_buffer = replay_buffer

        return base_agent
    return _sqil


__all__ = ["gail"]
