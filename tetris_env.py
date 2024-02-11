import time
import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

from controllers.game_controller import GameController
from controllers.window import Window
from game.actions import Actions
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
        self.GAME_OVER_PUNISH = -100
        self.PIECE_PUNISH = 4
        self.LINE_CLEAR_REWARD = 15
        self.REWARD_MULTIPLYER = 2
        
        self.MAX_BOARD_DIFF = 7
        
        # The first x number of pieces in the queue the agent can observe
        self.QUEUE_OBS_NUM = 5
        
        # All available actions as described in the 'game\agent_actions.py' file
        self.action_space = spaces.Discrete(len(aa.movements))

        """
        Dictionary containing the tetris board information and any additional information about the game that the agent needs to know. e.g. the held piece, the pieces in the queue
        """
        self.observation_space = spaces.Dict({
            "board": spaces.Box(self.low, self.high, shape=(bc.BOARD_COLUMNS,), dtype=np.int8),
            "additional": spaces.Box(self.low, self.high, shape=(len(self._get_additional_obs()),), dtype=np.int16)
        })
        
        self._window = None
        self.tick_speed = 0

    def step(self, action, playback: bool = False):
        prev_action = self._game.previous_action
        prev_lines_cleared = self._game.lines_cleared
        was_tetris_ready = self._game.is_tetris_ready()
        
        action_list = list(aa.movements[action])
            
        # Add final hard drop at end of action list if not holding
        if action_list[0] != int(Actions.HOLD_PIECE):
            held_performed = False
            action_list.append(int(Actions.HARD_DROP))
        else:
            held_performed = True   
        
        # Perform all actions in action list
        for i in range(len(action_list)):
            if playback and self._game.rendering_enabled:
                time.sleep(0.1)
                
                # Update the window if it is being rendered
                if self._window_exists():
                    self._update_window()
            
            self._game.take_admin_inputs()
            terminated = self._game.run(action_list[i]) 
            
            if terminated:
                reward = self.GAME_OVER_PUNISH
                break
        
        # Check how many lines were cleared after performing actions
        lines_cleared = self._game.lines_cleared - prev_lines_cleared
        
        if lines_cleared == 4:
            print(f"Tetris!: {action_list}")
        
        ###############################
        # Strict game over conditions #
        ###############################
        
        # Terminate if tetris ready and able to tetris but didn't
        if was_tetris_ready and self._game.is_tetris_ready():
            if (self._game.piece_manager.previous_piece == int(PieceTypeID.I_PIECE) or self._game.piece_manager.get_held_piece_id() == int(PieceTypeID.I_PIECE)):
                terminated = True    
                reward = self.GAME_OVER_PUNISH
                print(f"Failed Tetris: {action_list}")

        # Terminate if gap created on board
        if self._game.get_num_of_full_gaps() > 0 or self._game.get_num_of_top_gaps() > 0:
            terminated = True
            reward = self.GAME_OVER_PUNISH
        
        # Terminate if height difference violated of board well incorrectly filled
        if self._game.get_board_height_difference_with_well() > self.MAX_BOARD_DIFF:
            terminated = True
            reward = self.GAME_OVER_PUNISH    
            
        if not self._game.is_well_valid():
            terminated = True
            
            if self._game.piece_manager.previous_piece == int(PieceTypeID.I_PIECE):
                reward = 0
            else:
                reward = self.GAME_OVER_PUNISH    
                
        if held_performed and prev_action == int(Actions.HOLD_PIECE):
            reward = self.GAME_OVER_PUNISH * 10
            print("Held Twice!")

        ####################
        # Calculate reward #
        ####################
        
        if not terminated:
            reward = self._perfect_stacking_reward(lines_cleared)
        
        # Get observations 
        observation = self._get_obs()
        info = self._get_info()
        
        if not playback and self._game.rendering_enabled:
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
    
    def render(self, screen_size: ScreenSizes|int = ScreenSizes.MEDIUM, show_fps: bool = False, show_score: bool = False, show_queue: bool = True):
        # Initial pygame setup
        pygame.display.init()
        pygame.font.init()
        
        # Create window Object
        self._window = Window(self._game, screen_size, show_fps, show_score, show_queue)
        
    def close(self):
        print("Enviroment closed.")
        
    def _perfect_stacking_reward(self, lines_cleared):      
        relative_piece_height = self._game.piece_manager.placed_piece_max_height - self._game.get_min_piece_board_height()
        
        if lines_cleared == 4:
            lines_cleared_reward = self.REWARD_MULTIPLYER ** lines_cleared * self.LINE_CLEAR_REWARD
        else:
            lines_cleared_reward = 0
            
        reward = (bc.BOARD_ROWS - relative_piece_height) + lines_cleared_reward     

        return reward
        
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
                self.tick_speed = self._game.frames

    def _get_obs(self):
        return {"board": self._get_board_obs(), "additional": self._get_additional_obs()}
    
    def _get_info(self):
        pass 
    
    def _get_board_obs(self) -> np.ndarray:
        return np.array(self._game.get_board_peaks_list())
    
    def _get_additional_obs(self) -> np.ndarray: 
        gaps = self._game.get_num_of_full_gaps() + self._game.get_num_of_top_gaps()
        
        if gaps > 0:
            gaps = 1
        
        if self._game.is_tetris_ready():
            tetris_ready = True
        else:
            tetris_ready = False
            
        if self._game.previous_action == int(Actions.HOLD_PIECE):
            holds_used += 1
        else:
            holds_used = 0
        
        return np.array(
            [
                gaps,
                tetris_ready,
                holds_used,
                self._game.piece_manager.get_held_piece_id(),
                self._game.get_current_piece_id()
                
            ] + self._game.get_truncated_piece_queue(self.QUEUE_OBS_NUM)
        )