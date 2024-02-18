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


class TetrisEnv(gym.Env):
    def __init__(self) -> None:
        # Init game controller
        self._game = GameController()

        # The indexs relating to the ID's of the tetris pieces i.e. (0-7)
        self.low, self.high = gu.get_piece_value_bounds(self._game)
        
        # Reward constants
        self.GAME_OVER_PUNISH = -100
        self.LINE_CLEAR_REWARD = 15
        self.REWARD_MULTIPLYER = 2
        self.BUMPINESS_REWARD = 10
        
        # The height the agent is allowed to place pieces above the lowest point of the stack
        self.MAX_BOARD_DIFF = 5
        
        # The first x number of pieces in the queue the agent can observe
        self.QUEUE_OBS_NUM = 5
        
        # All available actions as described in the 'game\agent_actions.py' file
        self.action_space = spaces.Discrete(len(movements))
        
        self.tetris_ready = 0

        """
        Dictionary containing the tetris board information and any additional information about the game that the agent needs to know. e.g. the held piece, the pieces in the queue
        """
        self.observation_space = spaces.Dict({
            "board": spaces.Box(self.low, self.high, shape=(bc.BOARD_COLUMNS,), dtype=np.int8),
            "additional": spaces.Box(self.low, self.high, shape=(len(self._get_additional_obs()),), dtype=np.int16)
        })
        
        self._window = None
        self.fps = 0
        
        self.moves_after_tetris = 0
        self.after_tetris = False

    def step(self, action, playback: bool = False):
        prev_action = self._game.previous_action
        prev_lines_cleared = self._game.lines_cleared
        prev_bumpiness = gu.get_bumpiness(self._game)
        was_tetris_ready = gu.is_tetris_ready(self._game)
        
        action_list = list(movements[action])
            
        # Add final hard drop at end of action list if not holding
        if action_list[0] != int(Actions.HOLD_PIECE):
            held_performed = False
            action_list.append(int(Actions.HARD_DROP))
        else:
            held_performed = True   
        
        # Perform all actions in action list
        for i in range(len(action_list)):
            if playback:
                time.sleep(0.1)
                
                # Update the window if it is being rendered
                if self._window_exists():
                    self._update_window()

            self._window.render_game = self._game.admin_render_toggle_input(pygame.event.get())
            
            terminated = self._game.run(action_list[i]) 
            
            if terminated:
                reward = self.GAME_OVER_PUNISH
                break
            
        if gu.is_tetris_ready(self._game) and self._game.piece_manager.current_piece == int(PieceTypeID.I_PIECE):
            self.tetris_ready = 1    
        
        # Check how many lines were cleared after performing actions
        lines_cleared = self._game.lines_cleared - prev_lines_cleared
        
        if lines_cleared == 4:
            print(f"Tetris!: {action_list}")
            self.after_tetris = True
            self.tetris_ready = 1
            
        if self.after_tetris:
            self.moves_after_tetris += 1
            print(self.moves_after_tetris)
        
        ###############################
        # Strict game over conditions #
        ################################
        
        # Terminate if tetris ready and able to tetris but didn't
        if was_tetris_ready and gu.is_tetris_ready(self._game):
            if (self._game.piece_manager.previous_piece == int(PieceTypeID.I_PIECE) or gu.get_held_piece_id(self._game) == int(PieceTypeID.I_PIECE)):
                terminated = True    
                reward = self.GAME_OVER_PUNISH
                print(f"Failed Tetris: {action_list}")

        # Terminate if gap created on board
        if gu.get_num_of_full_gaps(self._game) > 0 or gu.get_num_of_top_gaps(self._game) > 0:
            terminated = True
            reward = self.GAME_OVER_PUNISH
        
        # Terminate if height difference violated of board well incorrectly filled
        if gu.get_board_height_difference_with_well(self._game) > self.MAX_BOARD_DIFF:
            terminated = True
            reward = self.GAME_OVER_PUNISH    
            
        # Termiante if pieces placed in well
        if not gu.is_well_valid(self._game):
            terminated = True
            
            if self._game.piece_manager.previous_piece == int(PieceTypeID.I_PIECE):
                reward = 0
            else:
                reward = self.GAME_OVER_PUNISH    
        
        # Terminate for using the hold action more than once in a row
        if held_performed and prev_action == int(Actions.HOLD_PIECE):
            reward = self.GAME_OVER_PUNISH
            terminated = True
            print(f"Held Twice! {self._game.holds_used_in_a_row}")

        if not terminated:
            reward = self._perfect_stacking_reward(lines_cleared, prev_bumpiness)

        if terminated:
            self.after_tetris = False
            self.moves_after_tetris = 0

        # Get observations 
        observation = self._get_obs()
        info = self._get_info()
        
        if not playback:
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
        
    def _perfect_stacking_reward(self, lines_cleared, prev_bumpiness):
        # Reward agent for placing piece low down on the stack   
        piece_height_reward = bc.BOARD_ROWS - (gu.get_placed_piece_max_height(self._game) - gu.get_min_piece_height_on_board(self._game))
        
        # Reawrd agent for getting a tetris i.e. clearing 4 lines
        if lines_cleared == bc.TETRIS_LINE_CLEARS:
            lines_cleared_reward = self.REWARD_MULTIPLYER ** lines_cleared * self.LINE_CLEAR_REWARD
        else:
            lines_cleared_reward = 0
        
        # Reward agent for reducing or equaling the stack's bumpiness
        if prev_bumpiness >= gu.get_bumpiness(self._game):
            bumpiness_reward = self.BUMPINESS_REWARD 
        else:
            bumpiness_reward = -self.BUMPINESS_REWARD
        
        reward = piece_height_reward + lines_cleared_reward + bumpiness_reward 
        
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

    def _get_obs(self):
        return {"board": self._get_board_obs(), "additional": self._get_additional_obs()}
    
    def _get_info(self):
        pass 
    
    def _get_board_obs(self) -> np.ndarray:
        board = np.array(gu.get_max_height_column_list(self._game))
        board = board - gu.get_min_gap_height_exluding_well(self._game)
        
        board = np.clip(board, a_min = 0, a_max = 20) 
        
        return board
    
    def _get_additional_obs(self) -> np.ndarray: 
        gaps = gu.get_num_of_full_gaps(self._game) + gu.get_num_of_top_gaps(self._game)
        
        if gaps > 0:
            gaps = 1

        return np.array(
            [
                gaps,
                self.tetris_ready,
                self._game.holds_used_in_a_row,
                gu.get_held_piece_id(self._game),
                gu.get_current_piece_id(self._game)
                
            ] + gu.get_truncated_piece_queue(self._game, self.QUEUE_OBS_NUM)
        )