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
    SCREEN_SIZE_TYPE_ERROR = "TetrisEnv.render() -> size must be of types int or str"
    SCREEN_SIZE_STRING_ERROR = f"TetrisEnv.render() -> size must be from list {ScreenSizes._member_names_}"
    
    render_window = False
    
    def __init__(self) -> None:
        self.game = GameController()
        
        low, high = self.game.get_board_value_bounds()
        
        self.action_space = Discrete(len(Actions))
        self.observation_space = Dict(
            {
                "board": Box(low=low, high=high, shape=(BOARD_ROWS, BOARD_COLUMNS), dtype=np.int32),
                "queue": MultiDiscrete([high, high, high, high, high]),
                "hold": Discrete(high)
            }   
        )
        
        self.window = None
        
    def _update_window(self):
        if (pygame.event.get(pygame.QUIT)):
            pygame.quit()
            
        self.window.draw()
            
    def step(self, action, actions_per_second: int = 0):
        # Delay action - for analysing purposes
        if (actions_per_second > 0):
            sleep(1 / actions_per_second)
        
        self.game.cycle_game_clock()
        self.game.perform_action(action)
        score, done = self.game.run_logic()
        
        if self.window is not None:
            self._update_window()
        
        if done:
            self.game.reset()
        
        return score, done

    def render(self, screen_size: ScreenSizes|int, show_fps: bool, show_score: bool):
        # Initial pygame setup
        pygame.display.init()
        pygame.font.init()
        
        # Verify that a valid screen size is being applied
        if (not (screen_size in ScreenSizes._value2member_map_)) and (not (type(screen_size) == int)):
            raise TypeError(GameSettings.SCREEN_SIZE_TYPE_ERROR)
        
        # After game settings have been configure, create window to be rendered
        self.window = Window(self.game, screen_size, show_fps, show_score)
        
        # Display first frame of window
        self._update_window()
        
    def seed(self, seed=None):
        GameSettings.seed = seed
        
        return GameSettings.seed   
        
    def reset(self):
        self.game.reset()
        return self.observation_space
    
    def close(self):
        print("Enviroment closed.")