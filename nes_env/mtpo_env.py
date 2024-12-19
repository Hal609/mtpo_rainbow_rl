"""An OpenAI Gym environment for Mike Tyson's Punch Out!!"""
import gymnasium as gym
from .cynes_env import NESEnv
import numpy as np
import random
import time

NES_INPUT_RIGHT = 0x01
NES_INPUT_LEFT = 0x02
NES_INPUT_DOWN = 0x04
NES_INPUT_UP = 0x08
NES_INPUT_START = 0x10
NES_INPUT_SELECT = 0x20
NES_INPUT_B = 0x40
NES_INPUT_A = 0x80

class PunchOutEnv(NESEnv):
    """An environment for playing Mike Tyson's Punch Out!! with OpenAI Gym."""

    def __init__(self, rom_path, headless=True):
        """
        Initialize a new MTPO environment.

        Args:
            rom_path (str): The path to the ROM file.
            headless (bool): Whether to run the emulator in headless mode.

        """
        super().__init__(rom_path, headless=headless)

        # Existing initialization code...
        self.action_space = gym.spaces.Discrete(6)

        # Define the reduced action mapping
        self._action_map = [0, NES_INPUT_START, NES_INPUT_LEFT, NES_INPUT_B, NES_INPUT_B | NES_INPUT_UP, NES_INPUT_DOWN]  # Corresponds to Left and Left+Jump

        self.first_fight = 0 #random.randint(0, 4)

        # Initialize any additional variables
        self._time_last = 0
        self._mac_hp_last = 0
        self._opp_down_count_last = 0
        self._opp_hp_last = 0
        self._opp_id_last = 0

        self.was_hit = False

        self.last_time = time.time()
    
    def step(self, action):
        if self.done:
            raise ValueError('Cannot step in a done environment! Call `reset` first.')
        
        if self._time != 0 and not self._has_backup:
            self._backup()

        self.nes.controller = self._action_map[action]
        frame = self.nes.step(frames=1)

        obs = np.array(frame, dtype=np.uint8)
        reward = float(self.get_reward())
        # self.done = bool(self._get_done())

        if not self._in_fight:
            self.skip_between_rounds()

        # bound the reward in [min, max]
        if reward < self.reward_range[0]:
            reward = self.reward_range[0]
        elif reward > self.reward_range[1]:
            reward = self.reward_range[1]

        if (1/60 - (time.time() - self.last_time)) > 0:
            time.sleep(1/60 - (time.time() - self.last_time))
        self.last_time = time.time()

        return obs, reward, self.done, False, {}

    # MARK: Memory access

    def _read_mem_range(self, address, length):
        """
        Read a range of bytes where each byte is a 10's place figure.

        Args:
            address (int): The address to read from.
            length (int): The number of sequential bytes to read.

        Returns:
            int: The integer value of this 10's place representation.

        """
        return int(''.join(map(str, [self.ram[address + i] for i in range(length)])))
    
    @property
    def _in_fight(self):
        '''Return the current round number.'''
        return self.ram[0x0004] == 0xFF
    
    @property
    def _round(self):
        '''Return the current round number.'''
        return self.ram[0x0006]
    
    @property
    def _opp_id(self):
        '''Return the current fight id.'''
        return self.ram[0x0001]
    
    @property
    def _mac_health(self):
        '''Return the Mac's current HP'''
        return self.ram[0x0391]

    @property
    def _opp_health(self):
        '''Return the opponant's current HP'''
        return self.ram[0x0398]
    
    @property
    def _mac_down_count(self):
        '''Return the number of times Mac has been knocked down'''
        return self.ram[0x03D0]
    
    @property
    def _opp_down_count(self):
        '''Return the number of times opponant has been knocked down'''
        return self.ram[0x03D1]

    @property
    def _time(self):
        """Return the time left (0 to 999)."""
        # time is represented as a figure with 3 10's places
        return 60*self.ram[0x0302] + 10*self.ram[0x0304] + self.ram[0x0305]

    def skip_between_rounds(self):
        while (not self._in_fight) or self._time == self._time_last or self._time == 0:
            self._time_last = self._time
            start_time = time.time()
            self._frame_advance(0)
            if 1/60 - (time.time() - start_time) > 0:
                time.sleep(1/60 - (time.time() - start_time))
            start_time = time.time()
            self._frame_advance(0)
            if 1/60 - (time.time() - start_time) > 0:
                time.sleep(1/60 - (time.time() - start_time))
            start_time = time.time()
            self._frame_advance(NES_INPUT_START)
            if 1/60 - (time.time() - start_time) > 0:
                time.sleep(1/60 - (time.time() - start_time))
            start_time = time.time()
            self._frame_advance(NES_INPUT_START)
            if 1/60 - (time.time() - start_time) > 0:
                time.sleep(1/60 - (time.time() - start_time))

    # MARK: Reward Function

    @property
    def _health_penalty(self):
        """Return the """
        _reward = self._mac_health - self._mac_hp_last
        self.was_hit = _reward != 0
        self._mac_hp_last = self._mac_health
        return _reward
    
    @property
    def _hit_reward(self):
        """Return the reward based on left right movement between steps."""
        _reward = self._opp_hp_last - self._opp_health
        self._opp_hp_last = self._opp_health
        return _reward
    
    @property
    def _ko_reward(self):
        """Return the reward based on left right movement between steps."""
        _reward = self._opp_down_count - self._opp_down_count_last
        self._opp_down_count_last = self._opp_down_count
        return _reward
    
    @property
    def _next_opp_reward(self):
        """Return the reward based on left right movement between steps."""
        _reward = self._opp_id_last != self._opp_id
        self._opp_id_last = self._opp_id
        return int(_reward)

    @property
    def _time_penalty(self):
        """Return the penalty for the in-game clock ticking."""
        _reward = self._time_last - self._time
        self._time_last = self._time
        # time can only decrease, a positive reward results from a reset and
        # should default to 0 reward
        if _reward > 0:
            return 0

        return _reward

    @property
    def _death_penalty(self):
        """Return the penalty earned by dying."""
        if self._is_dying or self._is_dead:
            return -25

        return 0

    def _will_reset(self):
        """Handle and RAM hacking before a reset occurs."""
        self._time_last = 0
        self._mac_hp_last = 0
        self._opp_down_count_last = 0
        self._opp_hp_last = 0
        self._opp_id_last = 0
        self.ram[0x0001] = random.randint(0,5)

    def _did_reset(self):
        """Handle any RAM hacking after a reset occurs."""
        self._time_last = self._time
        self._mac_hp_last = self._mac_health
        self._opp_down_count_last = self._opp_down_count
        self._opp_hp_last = self._opp_health
        self._opp_id_last = self._opp_id


    def get_reward(self):
        """Return the reward after a step occurs."""
        return (15*self._next_opp_reward) + (self._time_penalty)*0.1 + (2*self._ko_reward) + self._hit_reward + self._health_penalty
