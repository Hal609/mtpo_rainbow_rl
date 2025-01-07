import torch
import numpy as np
from Agent import Agent
from boxing_gym import make_env

def main():
    framestack = 4

    n_steps = 8000

    print("Currently Playing Game: MTPO")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device: " + str(device))

    env = make_env()

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
