import math
from time import sleep
import pygame

from controllers.game_controller import GameController
from controllers.window import Window

from game.actions import Actions
from game.game_settings import GameSettings

from utils.board_constants import BOARD_ROWS, BOARD_COLUMNS

from enum import IntEnum

from gym.spaces import Discrete, Box, Dict, MultiDiscrete
from gym import Env

import numpy as np

class ScreenSizes(IntEnum):
    XXSMALL = 6,
    XSMALL = 8,
    SMALL = 10,
    MEDIUM = 12,
    LARGE = 14,
    XLARGE = 16,
    XXLARGE = 18

class TetrisEnv(Env):
    def __init__(self) -> None:
        self._game = GameController()
        self._window = None
        
        self.game_score = self._game.score
        
        low, high = self._game.get_board_value_bounds()

        self.action_space = Discrete(len(Actions))
        # self.observation_space = Dict(
        #     {
        #         "board": Box(low=low, high=high, shape=(BOARD_ROWS, BOARD_COLUMNS), dtype=np.int32),
        #         "queue": MultiDiscrete([high, high, high, high, high]),
        #         "hold": Discrete(high)
        #     }   
        # )
        self.observation_space = Box(low=low, high=high, shape=(BOARD_ROWS, BOARD_COLUMNS), dtype=np.int32)
        
    def _update_window(self):
        if (pygame.event.get(pygame.QUIT)):
            pygame.quit()
            
            # Delete window object
            self._window = None
            print("Stopped rendering window.")
        else:
            self._window.draw()
            
    def step(self, action, actions_per_second: int = 0):
        # Delay action - for analysing purposes
        if (actions_per_second > 0):
            sleep(1 / actions_per_second)
        
        previous_pieces_dropped = self._game.get_num_of_pieces_dropped()
        
        self._game.cycle_game_clock()
        self._game.perform_action(action)
        self.done, gained_score = self._game.run_logic()
        
        if (self._game.get_num_of_pieces_dropped() - previous_pieces_dropped):
            self.reward += self.occupied_spaces_to_average_height_ratio_reward()
        
        if self._window is not None:
            self._update_window()
            
        self.reward += gained_score
            
        info = {}
        
        return self.observation, self.reward, self.done, info
    
    def occupied_spaces_to_average_height_ratio_reward(self):
        average_height = self._game.get_average_board_height()
        occupied_spaces = self._game.get_occupied_spaces_on_board()
        
        nine_piece_row_reduction = math.floor(occupied_spaces / 10)
        reduced_occupied_spaces = occupied_spaces - nine_piece_row_reduction
        
        pieces_max_height_ratio = reduced_occupied_spaces / average_height
        
        return (1 - pieces_max_height_ratio) * 5
    
    def render(self, screen_size: ScreenSizes|int, show_fps: bool, show_score: bool):
        # Initial pygame setup
        pygame.display.init()
        pygame.font.init()
        
        # Create window Object
        self._window = Window(self._game, screen_size, show_fps, show_score)
        
    def seed(self, seed=None):
        GameSettings.seed = seed
        return GameSettings.seed   
        
    def reset(self):
        self._game.reset_game()
        
        self.game_score = self._game.score
        self.done = False
        self.reward = 0
        
        self.observation = self._game.get_board_state()
        
        return self.observation
    
    def close(self):
        print("Enviroment closed.")