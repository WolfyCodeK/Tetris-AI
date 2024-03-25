import time
import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

from controllers.game_controller import GameController
from controllers.window import Window

from game.actions import Actions
from pieces.piece_type_id import PieceTypeID

from utils.screen_sizes import ScreenSizes
import utils.board_constants as bc
import utils.game_utils as gu

from game.agent_actions import movements


class TrainTetrisEnv(gym.Env):
    def __init__(self) -> None:
        # Init game controller
        self._game = GameController()

        # The indexs relating to the ID's of the tetris pieces i.e. (0-7)
        self.low, self.high = gu.get_piece_value_bounds(self._game)
        
        # Reward constants
        self.GAME_OVER_PUNISH = -100
        self.LINE_CLEAR_REWARD = 15
        self.TETRIS_REWARD = 1000
        self.BUMPINESS_REWARD = 10
        self.HOLD_PUNISH = -5
        
        # The height the agent is allowed to place pieces above the lowest point of the stack
        self.MAX_BOARD_DIFF = 5
        
        # The first x number of pieces in the queue the agent can observe
        self.QUEUE_OBS_NUM = 5
        
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
        
        self.playback = False
        self.admin_playback = False

    def step(self, action):
        prev_lines_cleared = self._game.lines_cleared
        prev_bumpiness = gu.get_bumpiness(self._game)
        
        action_list = list(movements[action])
            
        # Add final hard drop at end of action list if not holding
        if action_list[0] != int(Actions.HOLD_PIECE):
            held_performed = False
            action_list.append(int(Actions.HARD_DROP))
        else:
            held_performed = True   
        
        # Perform all actions in action list
        for i in range(len(action_list)):
            if self._window_exists():
                self._window.render_game, self.admin_playback = self._game.admin_render_toggle_input(pygame.event.get())
            
            # Update the window at a human viewable speed if it is being rendered
            if self.playback:
                self._render_window_if_exists(playback=True)
            elif self.admin_playback:
                self._render_window_if_exists(playback=True)
            
            terminated = self._game.run(action_list[i]) 
            
            if terminated:
                reward = self.GAME_OVER_PUNISH
                break
        
        # Check how many lines were cleared after performing actions
        lines_cleared = self._game.lines_cleared - prev_lines_cleared
        
        ###############################
        # Game termination conditions #
        ###############################
        
        # Terminate if a none tetris line clear occured
        if lines_cleared > 0 and lines_cleared < 4:
            terminated = True
            reward = self.GAME_OVER_PUNISH

        # Terminate if gap created on board
        if gu.does_board_have_gaps(self._game):
            terminated = True
            reward = self.GAME_OVER_PUNISH
        
        # Terminate if height difference violated of board well incorrectly filled
        if gu.has_exceeded_max_board_height_difference(self._game, self.MAX_BOARD_DIFF):
            terminated = True
            reward = self.GAME_OVER_PUNISH    
            
        # Termiante if pieces placed in well
        if gu.is_well_invalid(self._game):
            terminated = True
            
            # Make exception for I piece being placed in well
            if self._game.piece_manager.previous_piece == int(PieceTypeID.I_PIECE):
                reward = 0
            else:
                reward = self.GAME_OVER_PUNISH    
        
        # Terminate for using the hold action more than once in a row
        if self._game.holds_performed_in_a_row > 1:
            reward = self.GAME_OVER_PUNISH * 10
            terminated = True
            
        ################################
        # Get rewards and observations #
        ################################

        if not terminated:
            reward = self._perfect_stacking_reward(lines_cleared, prev_bumpiness, held_performed)

        observation = self._get_obs()
        info = self._get_info()
        
        # Update window at full speed if it is being rendered
        if not (self.admin_playback or self.playback):
            self._render_window_if_exists()
        
        return observation, reward, terminated, False, info
        
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
        
    def _perfect_stacking_reward(self, lines_cleared, prev_bumpiness, held_performed):
        # Reward agent for placing piece low down on the stack   
        piece_height_reward = bc.BOARD_ROWS - (gu.get_placed_piece_max_height(self._game) - gu.get_min_piece_height_on_board(self._game))
        
        # Reawrd agent for getting a tetris i.e. clearing 4 lines
        if lines_cleared == bc.TETRIS_LINE_CLEARS:
            lines_cleared_reward = self.TETRIS_REWARD
        else:
            lines_cleared_reward = 0
        
        # Reward agent for reducing or equaling the stack's bumpiness
        if prev_bumpiness >= gu.get_bumpiness(self._game):
            bumpiness_reward = self.BUMPINESS_REWARD 
        else:
            bumpiness_reward = -self.BUMPINESS_REWARD
        
        # Slightly punish holding so it becomes backup option for when all other moves are worse
        if not held_performed:
            reward = piece_height_reward + lines_cleared_reward + bumpiness_reward 
        else:
            reward = self.HOLD_PUNISH
        
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
                self.fps = self._game.last_fps_recorded
                
    def _render_window_if_exists(self, playback: bool = False):
        if playback:
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
        return np.array([
                int(gu.is_tetris_ready(self._game)),
                self._game.holds_performed_in_a_row,
                gu.get_held_piece_id(self._game),
                gu.get_current_piece_id(self._game)
                
            ] + gu.get_truncated_piece_queue(self._game, self.QUEUE_OBS_NUM)
        )