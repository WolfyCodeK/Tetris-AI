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
        
        # Reward constants
        self.GAME_OVER_PUNISH = -50
        self.PIECE_PUNISH = -10
        self.LINE_CLEAR_REWARD = 25
        
        self.MAX_BOARD_OBS_HEIGHT = 6
        
        self.unstrict_piece_ids = [int(PieceTypeID.S_PIECE), int(PieceTypeID.Z_PIECE)]
        
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

    def step(self, action):
        # action_list = list(self.action_lists[action])
        action_list = list(aa.movements[action])
        
        # Add final hard drop at end of action list if not holding
        if action_list[0] != int(Actions.HOLD_PIECE):
            held_performed = False
            action_list.append(int(Actions.HARD_DROP))
        else:
            held_performed = True   
        
        #TODO: add piece hold event
        
        prev_action = self._game.previous_action
        prev_line_clears = self._game.lines_cleared
        prev_top_gaps = self._game.get_num_of_top_gaps()
        prev_full_gaps = self._game.get_num_of_full_gaps()
        
        for i in range(len(action_list)):
            terminated = self._game.run(action_list[i]) 
            
            if terminated:
                break
        
        # Calculate reward
        if (self._game.get_board_height_difference() > self.MAX_BOARD_OBS_HEIGHT):
            terminated = True    
            reward = self.GAME_OVER_PUNISH
        else:
            reward = self._overfit_reward_calculation(held_performed, prev_action, prev_full_gaps, prev_top_gaps, prev_line_clears)
            
        observation = self._get_obs()
        info = self._get_info()
        
        # Update the window if it is being rendered
        if self._window_exists():
            self._update_window()
        
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
        
    def _overfit_reward_calculation(self, held_performed, prev_action, prev_full_gaps, prev_top_gaps, prev_line_clears):
        board_height_difference = self._game.get_board_height_difference()
        top_gaps = self._game.get_num_of_top_gaps() - prev_top_gaps
        full_gaps = self._game.get_num_of_full_gaps() - prev_full_gaps

        lines_cleared = self._game.lines_cleared - prev_line_clears
        reward_gain = self.MAX_BOARD_OBS_HEIGHT - board_height_difference + (lines_cleared * self.LINE_CLEAR_REWARD)
        
        if top_gaps > 0 or full_gaps > 0:
            reward = self.PIECE_PUNISH * (top_gaps + full_gaps)
            
        elif board_height_difference <= self.MAX_BOARD_OBS_HEIGHT:
            reward = reward_gain
            
        if held_performed:
            if prev_action == int(Actions.HOLD_PIECE):
                reward = self.PIECE_PUNISH
            else:
                reward = 0

        return reward
        
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
                self._game.get_current_piece_id(),
                self._game.get_next_piece_id()
            ] 
        )