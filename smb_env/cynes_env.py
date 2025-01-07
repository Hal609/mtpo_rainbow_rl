import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.spaces import Discrete
from collections import defaultdict
import numpy as np
from cynes.windowed import WindowedNES
from cynes import NES
from cynes import *
from ._rom import ROM

# height in pixels of the NES screen
SCREEN_HEIGHT = 240
# width in pixels of the NES screen
SCREEN_WIDTH = 256

SCREEN_SHAPE_24_BIT = SCREEN_HEIGHT, SCREEN_WIDTH, 3

def reverse_bits(n):
    result = 0
    for i in range(8):
        result <<= 1            # Shift result left by 1 bit
        result |= n & 1         # Copy the least significant bit of n to result
        n >>= 1                 # Shift n right by 1 bit
    return result

class NESRam:
    def __init__(self, nes):
        self.nes = nes

    def __getitem__(self, address):
        return self.nes[address]

    def __setitem__(self, address, value):
        self.nes[address] = value

class NESEnv(gym.Env):
    # relevant meta-data about the environment
    metadata = {
        'render_modes': ['rgb_array', 'human'],
        'video.frames_per_second': 60
    }

    # the legal range for rewards for this environment
    reward_range = (-float(30), float(30))

    # observation space for the environment is static across all instances
    # observation_space = Box(
    #     low=0,
    #     high=255,
    #     shape=SCREEN_SHAPE_24_BIT,
    #     dtype=np.uint8
    # )

    # action space is a bitmap of button press values for the 8 NES buttons
    # action_space = Discrete(256)

    def __init__(self, rom_path, headless=False):
        # create a ROM file from the ROM path
        rom = ROM(rom_path)
        # check that there is PRG ROM
        if rom.prg_rom_size == 0:
            raise ValueError('ROM has no PRG-ROM banks.')
        # ensure that there is no trainer
        if rom.has_trainer:
            raise ValueError('ROM has trainer. trainer is not supported.')
        # check the TV system
        if rom.is_pal:
            raise ValueError('ROM is PAL. PAL is not supported.')
        
        # store the ROM path
        self._rom_path = rom_path
        # setup a placeholder for a pointer to a backup state
        self._has_backup = False
        # setup a done flag
        self.done = True
        # setup the controllers, screen, and RAM buffers
        # self.controllers = [self._controller_buffer(port) for port in range(2)]
        # self.screen = self._screen_buffer()
        # self.ram = self._ram_buffer()

        self.headless = headless
        self.nes = NES(self._rom_path) if self.headless else WindowedNES(self._rom_path)
        self._ram = NESRam(self.nes)
        self.done = True

        # Define observation and action spaces
        self.observation_space = Box(low=0, high=255, shape=(240, 256, 3), dtype=np.uint8)
        self.action_space = Discrete(256)


    @property
    def ram(self):
        return self._ram
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # call the before reset callback
        self._will_reset()
        # reset the emulator
        if self._has_backup:
            self._restore()
        else:
            self.nes.reset()
        # call the after reset callback
        self._did_reset()
        self.done = False
        obs = self.nes.step(frames=1)  # Capture an initial frame
        obs = np.array(obs, dtype=np.uint8)  # Ensure it's a valid numpy array
        info = {}
        return obs, info

    def step(self, action):
        raise NotImplementedError

    def _backup(self):
        """Backup the current emulator state."""
        self._backup_state = self.nes.save()
        self._has_backup = True

    def _restore(self):
        """Restore the emulator state from backup."""
        self.nes.load(self._backup_state)

    def _frame_advance(self, action):
        """
        Advance the emulator by one frame with the given action.

        Args:
            action (int): The action to perform (controller input).

        """

        # Set the controller inputs
        self.nes.controller = action
        # Advance the emulator by one frame
        frame = self.nes.step(frames=1)
        # Update the current frame (observation)
        self.screen = frame

    # def close(self):
    #     # self.nes.close()
    #     if hasattr(self, 'window'):
    #         self.window.close()


