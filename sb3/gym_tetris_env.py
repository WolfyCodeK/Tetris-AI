from time import sleep
import pygame

from pieces.piece_type_id import PieceTypeID

from controllers.game_controller import GameController
from controllers.window import Window

from game.actions import Actions
from game.game_settings import GameSettings

import utils.board_constants as bc

from enum import IntEnum

from gym.spaces import Discrete, Box, Dict, MultiDiscrete
from gym import Env

import numpy as np

from utils.screen_sizes import ScreenSizes

class TetrisEnv(Env):
    def __init__(self) -> None:
        self._game = GameController()
        self._window = None
        
        low, high = self._game.get_piece_value_bounds()

        self.action_space = Discrete(len(Actions))
        
        self._board_view_range = 6
        
        self.observation_space = Dict({
            "additional": Box(low, high, shape=(len(self._get_additional_obs()),), dtype=np.int16),
            "board": Box(low, high, shape=(self._board_view_range, bc.BOARD_COLUMNS), dtype=np.int8)
        })
        
    def _window_exists(self):
        return self._window is not None
        
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
        # Delay action - for debugging purposes
        if (actions_per_second > 0):
            sleep(1 / actions_per_second)
        
        last_num_of_pieces_dropped = self._game.get_num_of_pieces_dropped()
        prev_top_gaps = self._game.get_num_of_top_gaps()
        prev_full_gaps = self._game.get_num_of_full_gaps()
        prev_piece_id = self._game.piece_manager.current_piece.id 
        
        # Run game cycle methods
        self._game._cycle_game_clock()
        self._game._perform_action(action)
        self.done = self._game._run_logic()
        
        # Check if a piece was dropped
        if (self._game.get_num_of_pieces_dropped() - last_num_of_pieces_dropped) > 0:
            #self.reward += self.flat_stack_reward_method(prev_top_gaps, prev_full_gaps, prev_piece_id)
            #self.reward += self.line_clear_reward_method()
            self.reward += self._game.piece_manager.board.do_left_side_test() * 1000
        
        # Update the window if it is being rendered
        if self._window_exists():
            self._update_window()
        
        self.game_steps += 1
        
        full_gaps = self._game.get_num_of_full_gaps()
        
        if (not self._game.piece_manager.board.check_all_previous_rows_filled()) or (full_gaps > 0) or (self._game.piece_manager.actions_per_piece > 10) or (self._game.get_board_height_difference() >= 6):
            self.done = True
        
        # if self.done:
        #     self.reward += self._game.score * (self._game.b2b + 1)
        
        self.observation = self._get_observation()

        info = {}
        
        #print(f"Action Reward -> {self.reward}")
        #print(f"Step -> {self.game_steps}, obs ->{self.observation}")
        
        return self.observation, self.reward, self.done, info
    
    def flat_stack_reward_method(self, prev_top_gaps, prev_full_gaps, previous_piece_id):
        occupied_spaces = self._game.get_occupied_spaces_on_board()
        board_height_difference = self._game.get_board_height_difference()

        if board_height_difference < 6:
            diff = 1
        else: 
            diff = -20

        top_gaps = self._game.get_num_of_top_gaps() - prev_top_gaps
        full_gaps = self._game.get_num_of_full_gaps() - prev_full_gaps
        gaps_reward = 1
    
        if (full_gaps == 0):
            if top_gaps <= 0:
                gaps_reward = (1 - top_gaps)
            else:
                gaps_reward = -5 * top_gaps
                
            if occupied_spaces == 4 and (previous_piece_id == PieceTypeID.S_PIECE or previous_piece_id == PieceTypeID.Z_PIECE):
                gaps_reward = 1
                
        elif full_gaps > 0:
            gaps_reward = -20
        else:
            raise ValueError(f"Full gaps cannot be negative when calculating reward!: full_gap: {full_gaps}, prev_full_gaps: {prev_full_gaps}")
        
        if (gaps_reward < 0):
            return gaps_reward
        else:
            return diff + gaps_reward
    
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
        
        self.observation = self._get_observation(reset=True)
        
        return self.observation
    
    def _get_observation(self, reset: bool = False):
        observation = self.observation_space.sample()
        
        observation["additional"] = self._get_additional_obs(reset)
        observation["board"] = self._get_board_obs()
        
        return observation
    
    def _get_board_obs(self) -> np.ndarray:
        board_obs = self._game.get_board_state_range_removed(self._board_view_range)
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
            self._game.piece_manager.actions_per_piece,
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