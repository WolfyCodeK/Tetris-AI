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
        
        self.observation_space = Dict({
            "additional": Box(low, high, shape=(len(self._get_additional_obs()),), dtype=np.uint32),
            "board": Box(low, high, shape=(bc.BOARD_ROWS, bc.BOARD_COLUMNS), dtype=np.uint32)
        })
        
        # self._board_view_range = bc.BOARD_ROWS
        # size = (self._board_view_range * bc.BOARD_COLUMNS) + len(self._get_additional_obs(reset=True))
        # self.observation_space = Box(low=low, high=high, shape=(size,), dtype=np.int32)
        
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
        
        self.done = self._game.run_logic()
        
        if (self._game.get_num_of_pieces_dropped() - last_num_of_pieces_dropped) > 0:
            self.reward += self.flat_stack_reward_method(prev_gaps)
        
        # if (self._game.get_num_of_pieces_dropped() - last_num_of_pieces_dropped) > 0:
        #     self.reward += self.line_clear_reward_method()
        
        if self._window is not None:
            self._update_window()
        
        self.game_steps += 1
        
        # gaps = self._game.get_num_of_gaps()
        
        # if (not self._game.piece_manager.board.check_all_previous_rows_filled()) or (gaps > 2) or (self._game.actions_per_piece > 20):
        #     self.done = True
        #     self.reward += b2b * 1000
        
        if (self._game.actions_per_piece > 20):
            self.done = True
        
        if self.done:
            self.reward += self._game.score * (self._game.b2b + 1)
        
        # self.observation = np.concatenate((
        #     self._get_additional_obs(), 
        #     self._get_board_obs(min(
        #         bc.BOARD_HEIGHT - self._game.get_max_piece_height_on_board(), 
        #         bc.BOARD_HEIGHT - self._board_view_range))
        #     )
        # )
        
        self.observation = self._get_observation()

        info = {}
        
        return self.observation, self.reward, self.done, info
    
    def flat_stack_reward_method(self, prev_gaps):
        occupied_spaces = self._game.get_occupied_spaces_on_board()
        board_height_difference = self._game.get_board_height_difference()

        # nine_piece_row_reduction = math.floor(occupied_spaces / bc.BOARD_COLUMNS)
        # reduced_occupied_spaces = occupied_spaces - nine_piece_row_reduction

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
                
        return diff * gaps
    
    def line_clear_reward_method(self):
        board_height_difference = self._game.get_board_height_difference()
        
        diff = (5 - board_height_difference)
        
        if diff < 0:
            diff = 0
        
        return diff
    
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
        
        # Reset values
        self.game_steps = 0
        self.game_score = self._game.score
        self.done = False
        self.reward = 0
        
        # self.observation = np.concatenate((
        #     self._get_additional_obs(reset = True), 
        #     self._get_board_obs(bc.BOARD_HEIGHT - self._board_view_range))
        # )
        
        self.observation = self._get_observation(reset=True)
        
        return self.observation
    
    def _get_observation(self, reset: bool = False):
        observation = self.observation_space.sample()
        
        observation["additional"] = self._get_additional_obs(reset)
        observation["board"] = self._game.get_board_state()
        
        return observation
    
    def _get_board_obs(self, height: int) -> np.ndarray:
        board_obs = self._game.get_board_state_range_removed(0, height, self._board_view_range)
        board_obs = board_obs.flatten()
        board_obs = np.where(board_obs == 0, 0, 1)
        
        return board_obs
    
    def _get_additional_obs(self, reset: bool = False) -> np.ndarray:
        current_piece_id = self._game.piece_manager.current_piece.id    
        held_piece = self._game.piece_manager.piece_holder.held_piece
        
        if held_piece != None:
            held_piece_id = held_piece.id
        else:
            held_piece_id = 0
        
        additional_obs = [
            self._game.get_board_height_difference(),
            self._game.get_max_piece_height_on_board(), 
            self._game.get_num_of_gaps(), 
            held_piece_id, 
            current_piece_id
        ]

        if reset:
            for i in range(len(additional_obs) - 1):
                additional_obs[i] = 0
                
        additional_obs = additional_obs + self._game.get_visible_piece_queue_id_list()
            
        return np.array(additional_obs)
    
    def close(self):
        print("Enviroment closed.")