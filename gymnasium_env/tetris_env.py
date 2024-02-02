import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

from controllers.game_controller import GameController
from controllers.window import Window
from game.actions import Actions
import utils.board_constants as bc
from utils.screen_sizes import ScreenSizes


class TetrisEnv(gym.Env):
    def __init__(self, render_mode=None, window_size=ScreenSizes.MEDIUM) -> None:
        # Init game controller
        self._game = GameController()

        # The indexs relating to the ID's of the tetris pieces i.e. (0-7)
        low, high = self._game.get_piece_value_bounds()

        # All available actions as described in the 'game\actions.py' file
        self.action_space = spaces.Discrete(len(Actions))

        """
        Dictionary containing the tetris board information and any additional information about the game that the agent needs to know. e.g. the held piece, the pieces in the queue
        """
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low, high, shape=(bc.BOARD_ROWS, bc.BOARD_COLUMNS), dtype=np.int8),
            "additional": spaces.Box(low, high, shape=(len(self._get_additional_obs()),), dtype=np.int16)
        })
        
        self._window = None
        self.window_size = window_size
        
        self.clock = None

    def step(self, action):
        terminated = self._game.run(action) 
        
        observation = self._get_obs()
        info = self._get_info()

        reward = 0
        
        return observation, reward, terminated, False, info
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        self._game.reset_game()
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def render(self, screen_size: ScreenSizes|int, show_fps: bool, show_score: bool):
        # Initial pygame setup
        pygame.display.init()
        pygame.font.init()
        
        # Create window Object
        self._window = Window(self._game, screen_size, show_fps, show_score)
        
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

    def _get_obs(self, reset: bool=False):
        return {"board": self._get_board_obs(), "additional": self._get_additional_obs(reset)}
    
    def _get_info(self):
        pass 
    
    def _get_board_obs(self) -> np.ndarray:
        return np.where(self._game.get_minimal_board_state() == 0, 0, 1)
    
    def _get_additional_obs(self) -> np.ndarray: 
        return np.array(
            [
                self._game.piece_manager.get_actions_per_piece(),
                self._game.piece_manager.get_held_piece_id(), 
                self._game.piece_manager.get_current_piece_id()  
            ] 
            
            + self._game.get_visible_piece_queue_id_list()
        )
        
    def close(self):
        print("Enviroment closed.")