import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

from controllers.game_controller import GameController
from controllers.window import Window
from game.actions import Actions
from game.game_settings import GameSettings
from pieces.piece_type_id import PieceTypeID
import utils.board_constants as bc
from utils.screen_sizes import ScreenSizes
import game.agent_actions as aa


class TetrisEnv(gym.Env):
    def __init__(self) -> None:
        # Init game controller
        self._game = GameController()

        # The indexs relating to the ID's of the tetris pieces i.e. (0-7)
        self.low, self.high = self._game.get_piece_value_bounds()
        
        # Reward consts
        self.LOSS_REWARD = -50
        
        self.MAX_BOARD_OBS_HEIGHT = 8
        
        # All available actions as described in the 'game\agent_actions.py' file
        self.action_space = spaces.Discrete(len(aa.movements))

        """
        Dictionary containing the tetris board information and any additional information about the game that the agent needs to know. e.g. the held piece, the pieces in the queue
        """
        self.observation_space = spaces.Dict({
            "board": spaces.Box(self.low, self.high, shape=(self.MAX_BOARD_OBS_HEIGHT, bc.BOARD_COLUMNS), dtype=np.int8),
            "additional": spaces.Box(self.low, self.high, shape=(len(self._get_additional_obs()),), dtype=np.int16)
        })
        
        self._window = None
        
    def _reward_calculation(self, prev_top_gaps, previous_piece_id) -> int:
        occupied_spaces = self._game.get_occupied_spaces_on_board()
        board_height_difference = self._game.get_board_height_difference()

        if board_height_difference < 8:
            diff = 8 - board_height_difference
        else: 
            diff = 0

        # Get the change in top gaps and full gaps
        top_gaps = self._game.get_num_of_top_gaps() - prev_top_gaps
            
        if occupied_spaces == 4 and (previous_piece_id == PieceTypeID.S_PIECE or previous_piece_id == PieceTypeID.Z_PIECE):
            gaps_reward = 5
        else:
            if top_gaps <= 0:
                gaps_reward = 5
            else:
                gaps_reward = -10
        
        if (gaps_reward < 0):
            return gaps_reward
        else:
            return diff + gaps_reward
        
    def seed(self, seed=None):
        GameSettings.seed = seed
        return GameSettings.seed   

    def step(self, action):
        # action_list = list(self.action_lists[action])
        action_list = list(aa.movements[action])
        
        # Add final hard drop at end of action list
        action_list.append(int(Actions.HARD_DROP))
        
        #TODO: add piece hold event
        
        prev_top_gaps = self._game.get_num_of_top_gaps()
        prev_piece_id = self._game.piece_manager.current_piece.id 
        
        for i in range(len(action_list)):
            terminated = self._game.run(action_list[i]) 
            
            if terminated:
                break
        
        # Calculate reward
        if (not self._game.piece_manager.board.check_all_previous_rows_filled()) or (self._game.get_num_of_full_gaps() > 0) or (self._game.get_board_height_difference() >= self.MAX_BOARD_OBS_HEIGHT):
            terminated = True    
            reward = self.LOSS_REWARD
        else:
            reward = self._reward_calculation(prev_top_gaps, prev_piece_id)
        
        # Update the window if it is being rendered
        if self._window_exists():
            self._update_window()
            
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, False, info
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        self._game.reset_game()
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def render(self, screen_size: ScreenSizes|int=ScreenSizes.MEDIUM, show_fps: bool=False, show_score: bool=False):
        # Initial pygame setup
        pygame.display.init()
        pygame.font.init()
        
        # Create window Object
        self._window = Window(self._game, screen_size, show_fps, show_score)
        
    def close(self):
        print("Enviroment closed.")
        
    def _set_action_space(self, size):
        self.action_space = spaces.Discrete(size)
        
    def _update_action_space(self, current_piece_id: int) -> list:
        if current_piece_id == int(PieceTypeID.I_PIECE):
            return aa.i_movements
        
        if current_piece_id == int(PieceTypeID.O_PIECE):
            return aa.o_movements
        
        if current_piece_id == int(PieceTypeID.S_PIECE) or int(PieceTypeID.Z_PIECE):
            return aa.sz_movements
        
        if current_piece_id == int(PieceTypeID.L_PIECE) or int(PieceTypeID.J_PIECE):
            return aa.lj_movements
        
        if current_piece_id == int(PieceTypeID.T_PIECE):
            return aa.t_movements
        
        if (current_piece_id <= self.low) or (current_piece_id > self.high):
            raise ValueError(f"Invalid piece id: {current_piece_id}")
        
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

    def _get_obs(self):
        return {"board": self._get_board_obs(), "additional": self._get_additional_obs()}
    
    def _get_info(self):
        pass 
    
    def _get_board_obs(self) -> np.ndarray:
        reduced_board = self._game.get_minimal_board_state()
        reduced_board = np.delete(reduced_board, obj=range(0, bc.BOARD_ROWS-self.MAX_BOARD_OBS_HEIGHT), axis=0)
        
        return np.where(reduced_board == 0, 0, 1)
    
    def _get_additional_obs(self) -> np.ndarray: 
        return np.array(
            [
                self._game.piece_manager.get_current_piece_id(),
                self._game.get_next_piece_id()
            ] 
        )