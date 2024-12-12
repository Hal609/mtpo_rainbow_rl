"""An OpenAI Gym environment for Super Mario Bros. and Lost Levels."""
from collections import defaultdict
import gymnasium as gym
from .cynes_env import NESEnv
import numpy as np
from enum import Flag
import random
import time

FIGHT_DICT = {
    0: "Glass Joe",
    1: "Von Kaiser",
    2: "Piston Honda (1)",
    3: "Don Flamenco (1)",
    4: "King Hippo",
    5: "Great Tiger",
    6: "Bald Bull (1)",
    7: "Piston Honda (2)",
    8: "Soda Popinski",
    9: "Bald Bull (2)",
    10: "Don Flamenco (2)",
    11: "Mr. Sandman",
    12: "Super Macho Man",
    13: "Mike Tyson"
}

NES_INPUT_RIGHT = 0x01
NES_INPUT_LEFT = 0x02
NES_INPUT_DOWN = 0x04
NES_INPUT_UP = 0x08
NES_INPUT_START = 0x10
NES_INPUT_SELECT = 0x20
NES_INPUT_B = 0x40
NES_INPUT_A = 0x80

FIGHT_IDS = [] # Don1=48,KH=35,Soda=28,Von&Tyson=32,PH2&Glass=0 

def reverse_bits(n):
    result = 0
    for i in range(8):
        result <<= 1            # Shift result left by 1 bit
        result |= n & 1         # Copy the least significant bit of n to result
        n >>= 1                 # Shift n right by 1 bit
    return result

class PunchOutEnv(NESEnv):
    """An environment for playing Super Mario Bros with OpenAI Gym."""

    def __init__(self, rom_path, headless=True):
        """
        Initialize a new Super Mario Bros environment.

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
        print(self.first_fight)

        # Initialize any additional variables
        self._time_last = 0
        self._mac_hp_last = 0
        self._opp_down_count_last = 0
        self._opp_hp_last = 0
        self._opp_id_last = 0

        self.was_hit = False

        # Reset the emulator
        # self.reset()
        # Skip the start screen
        # self.skip_start_screen()
        # Create a backup state to restore from on subsequent calls to reset
        # self._backup()

        self.last_time = time.time()
        print("Finished init")
    
    def step(self, action):
        if self.done:
            raise ValueError('Cannot step in a done environment! Call `reset` first.')
        
        if self._time != 0 and not self._has_backup:
            self._backup()

        self.nes.controller = self._action_map[action]
        frame = self.nes.step(frames=1)

        obs = np.array(frame, dtype=np.uint8)
        reward = float(self.get_reward())
        self.done = bool(self._get_done())
        truncated = False
        info = self._get_info()

        self._did_step(self.done)

        # bound the reward in [min, max]
        if reward < self.reward_range[0]:
            reward = self.reward_range[0]
        elif reward > self.reward_range[1]:
            reward = self.reward_range[1]

        if (1/60 - (time.time() - self.last_time)) > 0:
            time.sleep(1/60 - (time.time() - self.last_time))
        self.last_time = time.time()

        return obs, reward, self.done, truncated, info

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
    def _level(self):
        """Return the level of the game."""
        return self.ram[0x075f] * 4 + self.ram[0x075c]

    @property
    def _world(self):
        """Return the current world (1 to 8)."""
        return self.ram[0x075f] + 1

    @property
    def _stage(self):
        """Return the current stage (1 to 4)."""
        return self.ram[0x075c] + 1

    @property
    def _area(self):
        """Return the current area number (1 to 5)."""
        return self.ram[0x0760] + 1

    @property
    def _score(self):
        """Return the current player score (0 to 999990)."""
        # score is represented as a figure with 6 10's places
        return self._read_mem_range(0x07de, 6)

    @property
    def _time(self):
        """Return the time left (0 to 999)."""
        # time is represented as a figure with 3 10's places
        return 60*self.ram[0x0302] + 10*self.ram[0x0304] + self.ram[0x0305]


    # MARK: RAM Hacks
    def skip_start_screen(self):
        """Press and release start to skip the start screen."""
        # press and release the start button
        for i in range(300):
            self.ram[0x0001] = self.first_fight
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
        # Press start until the game starts
        while self._time == 0:
            # press and release the start button
            start_time = time.time()
            self._frame_advance(0)
            if 1/60 - (time.time() - start_time) > 0:
                time.sleep(1/60 - (time.time() - start_time))
            start_time = time.time()
        self._time_last = self._time


    def _skip_end_of_world(self):
        """Skip the cutscene that plays at the end of a world."""
        if self._is_world_over:
            # get the current game time to reference
            time = self._time
            # loop until the time is different
            while self._time == time:
                # frame advance with NOP
                self._frame_advance(0)

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

    # MARK: nes-py API calls

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

    def _did_step(self, done):
        """
        Handle any RAM hacking after a step occurs.

        Args:
            done: whether the done flag is set to true

        Returns:
            None

        """
        # if done flag is set a reset is incoming anyway, ignore any hacking
        if done:
            return
        if not self._in_fight:
            self.skip_between_rounds()

    def get_reward(self):
        """Return the reward after a step occurs."""
        return (15*self._next_opp_reward) + (self._time_penalty)*0.1 + (2*self._ko_reward) + self._hit_reward + self._health_penalty

    def _get_done(self):
        """Return True if the episode is over, False otherwise."""
        # if self.is_single_stage_env:
        # return self._is_dying or self._is_dead or self._flag_get
        return self.was_hit
        return self._mac_down_count > 0 or self._round > 1

    def _get_info(self):
        """Return the info after a step occurs"""
        return dict(
            
        )
