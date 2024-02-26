import os
import time
import numpy as np

import gymnasium as gym
from gymnasium import spaces

from controllers.game_controller import GameController
from controllers.window import Window
from game.actions import Actions

from utils.screen_sizes import ScreenSizes
import utils.board_constants as bc
import utils.game_utils as gu

from game.agent_actions import movements

os.system('clear')
print("> Loading Enviroment...")

import pygame

class TestTetrisEnv(gym.Env):
    def __init__(self) -> None:    
        # Init game controller
        self._game = GameController()

        # The indexs relating to the ID's of the tetris pieces i.e. (0-7)
        self.low, self.high = gu.get_piece_value_bounds(self._game)
        
        # The first x number of pieces in the queue the agent can observe
        self.QUEUE_OBS_NUM = 5
        
        # The height the agent is allowed to place pieces above the lowest point of the stack
        self.MAX_BOARD_DIFF = 5
        
        # All available actions as described in the 'game\agent_actions.py' file
        self.action_space = spaces.Discrete(len(movements))

        """
        Dictionary containing the tetris board information and any additional information about the game that the agent needs to know. e.g. the held piece, the pieces in the queue
        """
        self.observation_space = spaces.Dict({
            "board": spaces.Box(self.low, self.high, shape=(bc.BOARD_COLUMNS,), dtype=np.int8),
            "additional": spaces.Box(self.low, self.high, shape=(len(self._get_additional_obs()),), dtype=np.int16)
        })
        
        self._window = None
        self.fps = 0
        
    def step(self, action):
        action_list = list(movements[action])
        
        # Add final hard drop at end of action list if not holding
        if action_list[0] != int(Actions.HOLD_PIECE):
            action_list.append(int(Actions.HARD_DROP))
        
        # Perform all actions in action list
        for i in range(len(action_list)):
            # Update the window at a human viewable speed if it is being rendered
            if self.playback:
                self._render_window_if_exists(playback=True)
            
            terminated = self._game.run(action_list[i]) 

        # Terminate for using the hold action more than once in a row
        if self._game.holds_performed_in_a_row > 1:
            terminated = True

        observation = self._get_obs()
        info = self._get_info()
        
        return observation, terminated, info
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        self._game.reset_game()
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def render(self, screen_size: ScreenSizes|int = ScreenSizes.MEDIUM, show_fps: bool = False, show_score: bool = False, show_queue: bool = True, playback = False, playback_aps = 10):
        """Renders the window and any additional information that needs to be shown.

        Args:
            screen_size (ScreenSizes | int, optional): The size of the screen. This can be a number. Defaults to ScreenSizes.MEDIUM.
            show_fps (bool, optional): If the fps counter should be rendered. Defaults to False.
            show_score (bool, optional): If the score should be rendered. Defaults to False.
            show_queue (bool, optional): If the piece queue should be rendered. Defaults to True.
            playback (bool, optional): If the game should be played back at a human viewable speed. Defaults to False.
            playback_aps (int, optional): The number of actions per second for the playback to show from the agent, where a higher number is a faster playback speed. Defaults to 10.
        """
        # Initial pygame setup
        pygame.display.init()   
        pygame.font.init()
        
        # Create window Object
        self._window = Window(self._game, screen_size, show_fps, show_score, show_queue)
        
        # Set playback options
        self.playback = playback
        self.playback_aps = playback_aps
        
    def close(self):
        print("Enviroment closed.")
        
    def _window_exists(self):
        return self._window is not None
        
    def _update_window(self):
        if (pygame.event.get(pygame.QUIT)):
            pygame.quit()
            
            # Delete window object
            self._window = None
            print("Stopped rendering window.")
        elif (pygame.display.get_active()):
                self._window.draw()
                self.fps = self._game.last_fps_recorded
                
    def _render_window_if_exists(self, playback: bool = False):
        if playback and self.playback_aps != 0:
            time.sleep(1 / self.playback_aps)
                
        if self._window_exists():
            self._update_window()

    def _get_obs(self):
        return {"board": self._get_board_obs(), "additional": self._get_additional_obs()}
    
    def _get_info(self):
        pass 
    
    def _get_board_obs(self) -> np.ndarray:
        return gu.get_relative_board_max_heights_excluding_well(self._game, self.MAX_BOARD_DIFF)
    
    def _get_additional_obs(self) -> np.ndarray: 
        return np.array(
            [
                int(gu.is_tetris_ready(self._game)),
                self._game.holds_performed_in_a_row,
                gu.get_held_piece_id(self._game),
                gu.get_current_piece_id(self._game)
                
            ] + gu.get_truncated_piece_queue(self._game, self.QUEUE_OBS_NUM)
        )