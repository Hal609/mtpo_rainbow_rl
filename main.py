import numpy as np
import torch
import gymnasium as gym
from Agent import Agent
from MTPOPreprocessingCustom import MTPOPreprocessingCustom

from smb_env.mtpo_env import PunchOutEnv

def make_env(envs_create, framestack, headless=True):
    rom_path = (
        "/Users/hal/rainbow_nes_rl/smb_env/punch.nes"
    )
    print(f"Creating {envs_create} envs")

    def create_env():
        gym.register(
            id="gymnasium_env/mtpo-v5",
            entry_point=PunchOutEnv,
        )
        env = MTPOPreprocessingCustom(gym.make("gymnasium_env/mtpo-v5", rom_path=rom_path, headless=headless))

        return gym.wrappers.FrameStack(env, num_stack=framestack, lz4_compress=False)
    
    return gym.vector.AsyncVectorEnv(
        [lambda: create_env() for _ in range(envs_create)],
        context="spawn",  # Required for Windows
    )

def main():


    framestack = 4

    n_steps = 8000

    print("Currently Playing Game: MTPO")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device: " + str(device))

    env = make_env(1, framestack, headless=False)

    n_actions = env.action_space[0].n
    print(f"Env has {n_actions} actions")

    agent = Agent(n_actions=env.action_space[0].n, input_dims=[framestack, 84, 84], device=device, num_envs=1,
                  agent_name="MTPO", total_frames=n_steps)

    steps = 0
    episodes = 0
    observation, info = env.reset()
    processes = []

    while steps < n_steps:
        steps += 1
        action = agent.choose_action(observation)
        env.step_async(action)
        observation_, reward, done_, trun_, info = env.step_wait()
        done_ = np.logical_or(done_, trun_)

        if done_[0]:
            episodes += 1

        observation = observation_

    # wait for our evaluations to finish before we quit the program
    for process in processes:
        process.join()

    print("Evaluations finished, job completed successfully!")


if __name__ == '__main__':
    main()
