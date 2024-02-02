import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

from controllers.game_controller import GameController
from game.actions import Actions
import utils.board_constants as bc
from utils.screen_sizes import ScreenSizes


class TetrisEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

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
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        self._window = None
        self.window_size = window_size
        
        self.clock = None

    def step(self, action):
        # Run game cycle methods
        self._game.cycle_game_clock()
        self._game.perform_action(action)
        self.done = self._game.run_logic()    
        
    def _get_observation(self, reset: bool = False):
        observation = self.observation_space.sample()
        
        observation["additional"] = self._get_additional_obs(reset)
        observation["board"] = self._get_board_obs()
        
        return observation
    
    def _get_board_obs(self) -> np.ndarray:
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