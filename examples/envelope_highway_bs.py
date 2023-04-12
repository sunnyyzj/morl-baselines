import mo_gymnasium as mo_gym
from mo_gymnasium.utils import MORecordEpisodeStatistics

from morl_baselines.multi_policy.envelope.envelope import Envelope
# import gym
# from gym.wrappers.flatten_observation import *
from mo_gymnasium.utils import MORecordEpisodeStatistics
from gymnasium.wrappers import FlattenObservation

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from gymnasium.envs.registration import register

def main():
    env_id = 'mo-highwaybs-fast-v0'
    # register(
    #         id = env_id,
    #         entry_point = 'mo-highwaybs-fast-v0:mo_gymnasium.envs.highway.highway_bs:MOHighwayEnvFastBS',
    #         max_episode_steps = 500,
    #         )

    def make_env():
        #env = mo_gym.make("minecart-v0")
        # env = mo_gym.make("mo-highway-fast-v0")
        env = mo_gym.make(env_id)
        #mo-highway-fast-v0
        env = MORecordEpisodeStatistics(env, gamma=0.98)
        env = FlattenObservation(env)
        return env

    env = make_env()
    eval_env = make_env()
    eval_env = FlattenObservation(eval_env)
    # RecordVideo(make_env(), "videos/minecart/", episode_trigger=lambda e: e % 1000 == 0)

    agent = Envelope(
        env,
        max_grad_norm=0.1,
        learning_rate=3e-4,
        gamma=0.98,
        batch_size=512,#64
        net_arch=[64, 64,64],#256, 256
        buffer_size=int(2e6),
        initial_epsilon=1.0,
        final_epsilon=0.05,
        epsilon_decay_steps=50000,
        initial_homotopy_lambda=0.0,
        final_homotopy_lambda=1.0,
        homotopy_decay_steps=10000,
        learning_starts=100,
        envelope=True,
        gradient_updates=1,
        target_net_update_freq=1000,  # 1000,  # 500 reduce by gradient updates
        tau=1,
        log=True,
        project_name="MORL-Baselines",
        experiment_name="Envelope - highwaybs-v1",
    )

    agent.train(
        total_timesteps=100000,
        total_episodes=None,
        weight=None,
        eval_env=eval_env,
        eval_freq=100,
        reset_num_timesteps=False,
        reset_learning_starts=False,
    )

if __name__ == "__main__":
    main()
