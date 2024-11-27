import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import random, datetime
import torch
import numpy as np
from pathlib import Path

import gymnasium as gym
from metrics import MetricLogger
from agent import Mario

from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, TransformObservation, FrameStackObservation
from smb_env_cynes import SuperMarioBrosEnv
# import gym_smb_cynes

from joypad_space import JoypadSpace

from metrics import MetricLogger

from wrappers import ResizeObservation, SkipFrame

# Initialize Super Mario environment
# env = SuperMarioBrosEnv(rom_path='super-mario-bros-rectangle.nes', headless=False)
gym.register(
    id="gymnasium_env/SMBEnv-v0",
    entry_point=SuperMarioBrosEnv,
)

env = gym.make("gymnasium_env/SMBEnv-v0", rom_path="super-mario-bros-rectangle.nes", headless=False)

# Limit the action-space to
#   0. walk right
#   1. jump right
env = JoypadSpace(
    env,
    [['right', 'B'],
    ['right', 'B', 'A']]
)

# # # Apply Wrappers to environment
env = SkipFrame(env, skip=4)
env = GrayscaleObservation(env, keep_dim=False)
env = ResizeObservation(env, (84, 84))
env = TransformObservation(env, lambda x: np.uint8(x / 255.), env.observation_space)
env = FrameStackObservation(env, stack_size=4)

env.reset()

save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

checkpoint =  None # Path('checkpoints\\2024-11-18T14-35-58\mario_net_33.chkpt')
mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)

logger = MetricLogger(save_dir)

episodes = 40000

### for Loop that train the model num_episodes times by playing the game
for e in range(episodes):

    state, _ = env.reset()

    # Play the game!
    while True:
        
        # 4. Run agent on the state
        action = mario.act(state)

        # 5. Agent performs action
        next_state, reward, done, truncated, info = env.step(action)

        # 6. Remember
        mario.cache(state, next_state, action, reward, done)

        # 7. Learn
        q, loss = mario.learn()

        # 8. Logging
        logger.log_step(reward, loss, q)

        # 9. Update state
        state = next_state

        # 10. Check if end of game
        if done or info['flag_get']:
            break

    logger.log_episode()

    if e % 20 == 0:
        print(f"Using {round(torch.cuda.memory_allocated()/1e9, 3)}GB of graphics memory")
        logger.record(
            episode=e,
            epsilon=mario.exploration_rate,
            step=mario.curr_step
        )
