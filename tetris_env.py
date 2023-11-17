from time import sleep
import pygame

from controllers.game_controller import GameController
from controllers.window import Window

from game.actions import Actions
from game.game_settings import GameSettings

from utils.board_constants import BOARD_ROWS, BOARD_COLUMNS

from enum import IntEnum

from gym.spaces import Discrete, Box, Dict, MultiDiscrete
from gym.spaces import Discrete
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
        self.score = self._game.score
        
        low, high = self._game.get_board_value_bounds()
        
        self.action_space = Discrete(len(Actions))
        self.observation_space = Dict(
            {
                "board": Box(low=low, high=high, shape=(BOARD_ROWS, BOARD_COLUMNS), dtype=np.int32),
                "queue": MultiDiscrete([high, high, high, high, high]),
                "hold": Discrete(high)
            }   
        )
        
        self._window = None
        
    def _update_window(self):
        if (pygame.event.get(pygame.QUIT)):
            pygame.quit()
            
        self._window.draw()
            
    def step(self, action, actions_per_second: int = 0):
        # Delay action - for analysing purposes
        if (actions_per_second > 0):
            sleep(1 / actions_per_second)
        
        self._game.cycle_game_clock()
        self._game.perform_action(action)
        done = self._game.run_logic()
        self.score = self._game.score
        
        if self._window is not None:
            self._update_window()
        
        if done:
            self._game.reset()
        
        return done

    def render(self, screen_size: ScreenSizes|int, show_fps: bool, show_score: bool):
        # Initial pygame setup
        pygame.display.init()
        pygame.font.init()
        
        # Create window to be rendered
        self._window = Window(self._game, screen_size, show_fps, show_score)
        
    def seed(self, seed=None):
        GameSettings.seed = seed
        
        return GameSettings.seed   
        
    def reset(self):
        self._game.reset()
        return self.observation_space
    
    def close(self):
        print("Enviroment closed.")