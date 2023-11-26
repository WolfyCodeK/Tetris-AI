import math
from time import sleep
import pygame

from controllers.game_controller import GameController
from controllers.window import Window

from game.actions import Actions
from game.game_settings import GameSettings

import utils.board_constants as bc

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
        
        low, high = self._game.get_piece_value_bounds()

        self.action_space = Discrete(len(Actions))
        # self.observation_space = Dict(
        #     {
        #         "board": Box(low=low, high=high, shape=(BOARD_ROWS, BOARD_COLUMNS), dtype=np.int32),
        #         "queue": MultiDiscrete([high, high, high, high, high]),
        #         "hold": Discrete(high)
        #     }   
        # )
        size = (6 * bc.BOARD_COLUMNS) + 1
        self.observation_space = Box(low=low, high=high, shape=(size,), dtype=np.int32)
        # print(f"init: {self.observation_space}")
        
    def _update_window(self):
        if (pygame.event.get(pygame.QUIT)):
            pygame.quit()
            
            # Delete window object
            self._window = None
            print("Stopped rendering window.")
        else:
            if (pygame.display.get_active()):
                self._window.draw()
            
    def step(self, action, actions_per_second: int = 0):
        # Delay action - for analysing purposes
        if (actions_per_second > 0):
            sleep(1 / actions_per_second)
        
        last_num_of_pieces_dropped = self._game.get_num_of_pieces_dropped()
        prev_gaps = self._game.get_num_of_gaps()
        
        self._game.cycle_game_clock()
        
        self._game.perform_action(action)
        
        self.done, b2b = self._game.run_logic()
        
        if (self._game.get_num_of_pieces_dropped() - last_num_of_pieces_dropped) > 0:
            self.reward += self.add_reward(prev_gaps)
        
        if self._window is not None:
            self._update_window()
        
        self.game_steps += 1
        
        gaps = self._game.get_num_of_gaps()
        
        if (not self._game.piece_manager.board.check_all_previous_rows_filled()) or (gaps > 2) or (self._game.actions_per_piece > 20):
            self.done = True
            self.reward += b2b * 1000
            # print(self.reward)

        max_height = self._game.get_max_piece_height_on_board()
        current_piece_id = self._game.piece_manager.current_piece.id
        
        self.observation = self._game.get_board_state_range_removed(0, min(bc.BOARD_HEIGHT - max_height, bc.BOARD_HEIGHT - 6), 6)
        self.observation = self.observation.flatten()
        self.observation = np.insert(self.observation, 0, current_piece_id)
        # print(f"step: {len(self.observation)}")

        info = {}
        
        return self.observation, self.reward, self.done, info
    
    def add_reward(self, prev_gaps):
        occupied_spaces = self._game.get_occupied_spaces_on_board()
        max_height = self._game.get_max_piece_height_on_board()
        second_min_height = self._game.get_second_min_piece_height_on_board()
        board_height_difference = max_height - second_min_height

        nine_piece_row_reduction = math.floor(occupied_spaces / bc.BOARD_COLUMNS)
        reduced_occupied_spaces = occupied_spaces - nine_piece_row_reduction

        # ranges from 1-9 where 9 = best, 1 = worst
        # pieces_max_height_ratio = reduced_occupied_spaces / max_height

        diff = (5 - board_height_difference)
        
        if diff < 0:
            diff = 0

        gaps = 1
        
        if occupied_spaces > 4:
            gaps =  1 - (self._game.get_num_of_gaps() - prev_gaps)
            
            if gaps < 0:
                gaps = 0
                
        return occupied_spaces * diff * gaps
    
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
        
        current_piece_id = self._game.piece_manager.current_piece.id
        self.game_steps = 0
        self.game_score = self._game.score

        self.done = False
        self.reward = 0
        
        self.observation = self._game.get_board_state_range_removed(0, bc.BOARD_HEIGHT - 6, 6)
        self.observation = self.observation.flatten()
        self.observation = np.insert(self.observation, 0, current_piece_id)
        # print(f"reset: {len(self.observation)}")
        
        return self.observation
    
    def close(self):
        print("Enviroment closed.")