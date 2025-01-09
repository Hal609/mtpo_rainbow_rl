import os
import time
import torch
from copy import deepcopy
import numpy as np
import multiprocessing as mp
from boxing_gym import make_env
from Agent import Agent, choose_eval_action


def evaluate_agent(net_state_dict, network_creator, eval_envs, num_eval_episodes, agent_name, testing, game,
                   n_actions, device, index, framestack):

    eval_env = make_env(eval_envs, game, framestack=framestack, headless=False)
    evals = []
    eval_episodes = 0
    eval_scores = np.array([0 for i in range(eval_envs)])
    eval_observation, eval_info = eval_env.reset()

    eval_net = network_creator()

    state_dict_gpu = {k: v.to(device) for k, v in net_state_dict.items()}

    eval_net.load_state_dict(state_dict_gpu)

    if index <= 125:
        rng = 0.01
    else:
        rng = 0.0
    while eval_episodes < num_eval_episodes:

        eval_action = choose_eval_action(eval_observation, eval_net, n_actions, device, rng)
        eval_observation_, eval_reward, eval_done_, eval_trun_, eval_info = eval_env.step(eval_action)
        eval_done_ = np.logical_or(eval_done_, eval_trun_)

        for i in range(eval_envs):
            eval_scores[i] += eval_reward[i]
            if eval_done_[i]:
                eval_episodes += 1
                evals.append(eval_scores[i])
                eval_scores[i] = 0
                if eval_episodes >= num_eval_episodes:
                    break

        eval_observation = eval_observation_

    if not testing:
        fname = agent_name + "Evaluation.npy"
        data = np.load(fname)

        # Update the specified index in the 0th dimension
        data[index] = evals
        print("Evaluation " + str(index + 1) + "M Complete, average score:")
        print(np.mean(evals))

        # Save the updated array back to the file
        np.save(fname, data)


def main():
    game = "MTPO"
    envs = 1 # Number of parallel environments
    bs = 256 # Batch size
    rr = 1 # Replay ratio
    c = 500 # Replace target every c gradient steps
    lr = 1e-4 # Learning rate
    include_evals = True # Whether to include intermittent evaluations
    num_eval_episodes = 100 # Number of episodes to run to perform evaluation
    eval_envs = 10 # Number of environments which are created per evaluation
    analy = True # Analysis
    framestack = 4 # Number of frames stacked to form a single observation
    sticky = True
    repeat_probs = 0 if not sticky else 0.25
    nstep = 3
    munch_alpha = 0.9
    grad_clip = 10
    arch = "impala"
    discount = 0.997 # Discount factor gamma
    linear_size = 512 # Number of nodes in the linear portion of the model
    taus = 8
    model_size = 2 # Impala network scale factor
    arg_frames = 200000000 # Number of frames to run the environment for
    frames = arg_frames // 4
    ncos = 64
    vector = True # Whether environment is vectorised  
    per_alpha = 0.2 # Alpha for prioritised experience replay
    eps_steps = 2000000 # Number of steps to reduce epsilon to zero
    activation = "relu" # Activation function
    testing = False # Whether the environment is being tested

    if not vector:
        lr = 5e-5
        envs = 1
        bs = 16
        rr = 0.25

    lr_str = "{:e}".format(lr)
    lr_str = str(lr_str).replace(".", "").replace("0", "")
    frame_name = str(int(arg_frames / 1000000)) + "M"

    agent_name = "BTR_" + game + frame_name

    print("Agent Name:" + str(agent_name))

    if not testing:
        counter = 0
        while True:
            if counter == 0:
                new_dir_name = agent_name
            else:
                new_dir_name = f"{agent_name}_{counter}"
            if not os.path.exists(new_dir_name):
                break
            counter += 1
        os.mkdir(new_dir_name)
        print(f"Created directory: {new_dir_name}")
        os.chdir(new_dir_name)

    # Create blank evaluation file
    fname = agent_name + "Evaluation.npy"
    if not testing:
        np.save(fname, np.zeros((arg_frames // 1000000, num_eval_episodes)))

    if testing:
        num_envs = 4
        eval_envs = 2
        eval_every = 8000
        num_eval_episodes = 4
        n_steps = 8000
        bs = 32
    else:
        num_envs = envs
        n_steps = frames
        eval_every = 250000
    next_eval = eval_every

    print("Currently Playing Game: " + str(game))

    gpu = "0"
    device = torch.device('cuda:' + gpu if torch.cuda.is_available() else 'cpu')
    print("Device: " + str(device))

    env = make_env(num_envs, framestack=4, headless=False)
    n_actions = env.action_space[0].n
    print(f"Env has {n_actions} actions.")

    agent = Agent(n_actions=env.action_space[0].n, input_dims=[framestack, 84, 84], device=device, num_envs=num_envs,
                  agent_name=agent_name, total_frames=n_steps, testing=testing, batch_size=bs, rr=rr, lr=lr,
                  target_replace=c, discount=discount, taus=taus, model_size=model_size, linear_size=linear_size, 
                  ncos=ncos, replay_period=num_envs, analytics=analy, framestack=framestack, arch=arch, per_alpha=per_alpha,
                  eps_steps=eps_steps, activation=activation, n=nstep, munch_alpha=munch_alpha, grad_clip=grad_clip)

    scores_temp = []
    steps = 0
    last_steps = 0
    last_time = time.time()
    episodes = 0
    current_eval = 0
    scores_count = [0 for i in range(num_envs)]
    scores = []
    observation, info = env.reset()
    processes = []

    if testing:
        from torchsummary import summary
        summary(agent.net, (framestack, 84, 84))

    while steps < n_steps:
        steps += num_envs
        action = agent.choose_action(observation)
        env.step_async(action)
        agent.learn()
        observation_, reward, done_, trun_, info = env.step_wait()
        done_ = np.logical_or(done_, trun_)

        for i in range(num_envs):
            scores_count[i] += reward[i]
            if done_[i]:
                episodes += 1
                scores.append([scores_count[i], steps])
                scores_temp.append(scores_count[i])
                scores_count[i] = 0

        reward = np.clip(reward, -1., 1.)

        for stream in range(num_envs):
            terminal_in_buffer = done_[stream]
            agent.store_transition(observation[stream], action[stream], reward[stream], observation_[stream],
                                   terminal_in_buffer, stream=stream)

        observation = observation_

        if steps % 1200 == 0 and len(scores) > 0:
            avg_score = np.mean(scores_temp[-50:])
            if episodes % 1 == 0:
                print('{} {} avg score {:.2f} total_steps {:.0f} fps {:.2f} games {}'
                      .format(agent_name, game, avg_score, steps, (steps - last_steps) / (time.time() - last_time), episodes),
                      flush=True)
                last_steps = steps
                last_time = time.time()

        # Evaluation
        if steps >= next_eval or steps >= n_steps:
            print("Evaluating")

            # Save model
            if not testing:
                agent.save_model()

            fname = agent_name + "Experiment.npy"
            if not testing:
                np.save(fname, np.array(scores))

            if include_evals:

                # Wait for evaluations to finish before we start the next evaluation
                for process in processes:
                    process.join()

                agent.disable_noise(agent.net)
                net_state_dict = deepcopy({k: v.cpu() for k, v in agent.net.state_dict().items()})
                network_creator = deepcopy(agent.network_creator_fn)

                # Start evaluation in a separate process
                eval_process = mp.Process(target=evaluate_agent,
                                          args=(net_state_dict, network_creator, eval_envs, num_eval_episodes, agent_name, testing, game,
                                                n_actions, device, current_eval, framestack))
                eval_process.start()
                processes.append(eval_process)

                current_eval += 1

            next_eval += eval_every

    # wait for our evaluations to finish before we quit the program
    for process in processes:
        process.join()

    print("Evaluations finished, job completed successfully!")


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
