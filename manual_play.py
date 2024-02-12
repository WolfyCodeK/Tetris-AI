import pygame
from controllers.game_controller import GameController
from controllers.window import Window
from game.actions import Actions
from pieces.piece_type_id import PieceTypeID
import utils.board_constants as bc
import numpy as np
from utils.screen_sizes import ScreenSizes

def _perfect_stacking_reward(held_performed, prev_action, lines_cleared):      
    if held_performed:
        if prev_action == int(Actions.HOLD_PIECE):
            reward = -100
        else:
            reward = 0
    else:
        relative_piece_height = game.piece_manager.placed_piece_max_height - game.get_min_piece_board_height()
        
        if lines_cleared > 0:
            lines_cleared_reward = 2 ** lines_cleared * 15
        else:
            lines_cleared_reward = 0
            
        reward = (bc.BOARD_ROWS - relative_piece_height) + lines_cleared_reward
            
    return reward

def _overfit_reward_calculation(self, held_performed, prev_action, prev_full_gaps, prev_top_gaps, prev_line_clears):
    board_height_difference = self._game.get_board_height_difference()
    
    min_height = self._game.get_second_lowest_gap()
    
    top_gaps = self._game.get_num_of_top_gaps() - prev_top_gaps
    full_gaps = self._game.get_num_of_full_gaps() - prev_full_gaps

    lines_cleared = self._game.lines_cleared - prev_line_clears
    
    if held_performed:
        if prev_action == int(Actions.HOLD_PIECE):
            reward = self.GAME_OVER_PUNISH
        else:
            reward = 0
    else:
        relative_piece_height = self._game.piece_manager.placed_piece_max_height - self._game.get_min_piece_board_height()
        reward_gain = ((self.BOARD_OBS_HEIGHT - relative_piece_height) * self.REWARD_MULTIPLYER) + (lines_cleared * self.LINE_CLEAR_REWARD)  
        
        if lines_cleared == 0:
            if full_gaps > 0:   
                reward = -int(2 ** relative_piece_height)
            elif top_gaps > 0:
                reward = -int(2 ** relative_piece_height)
            elif board_height_difference < self.BOARD_OBS_HEIGHT:
                reward = reward_gain
        else:
            reward = reward_gain
            
    if reward < self.GAME_OVER_PUNISH:
        reward = self.GAME_OVER_PUNISH
            
    return reward

def _get_obs():
    return {"board": _get_board_obs(), "additional": _get_additional_obs()}

def _get_board_obs() -> np.ndarray:
        board = np.array(game.get_board_peaks_list())
        board = board - game.get_min_gap_height_exluding_well()
        
        board = np.clip(board, a_min = 0, a_max = 20) 
        
        return board

def _get_additional_obs() -> np.ndarray: 
    gaps = game.get_num_of_full_gaps() + game.get_num_of_top_gaps()
        
    if gaps > 0:
        gaps = 1
    
    return np.array(
        [
            gaps,
            game.piece_manager.get_held_piece_id(),
            game.get_current_piece_id()
            
        ] + game.get_truncated_piece_queue(3)
    )

if __name__ == '__main__':    
    game = GameController()
    
    # Pygame intial setup
    pygame.display.init()
    pygame.font.init()
    
    window = Window(game, ScreenSizes.MEDIUM)
    
    pygame.display.set_caption("Tetris - Pygame")

    # Set window icon
    tetris_icon = pygame.image.load("res/tetris-icon.png")
    pygame.display.set_icon(tetris_icon)
    
    running = True
    
    reward = 0

    while running:
        # Check if user has quit the window
        if (pygame.event.get(pygame.QUIT)):
            running = False
            
        game._cycle_game_clock()    
            
        event_list = pygame.event.get()
        
        prev_action = game.previous_action
        prev_line_clears = game.lines_cleared
        was_tetris_ready = game.is_tetris_ready()
        
        game.take_player_inputs(event_list)
        done = game._run_logic()
        
        lines_cleared = game.lines_cleared - prev_line_clears
        
        held_performed = False
        dropped_piece = False
        
        for event in event_list:          
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    dropped_piece = True
                if event.key == pygame.K_LSHIFT:
                    held_performed = True
        
        if dropped_piece or held_performed:         
            # Calculate reward
            if game.get_num_of_full_gaps() > 0 or game.get_num_of_top_gaps() > 0 or game.get_board_height_difference_with_well() > 5 or (not game.is_well_valid()):
                done = True    
                reward = -100
            else:
                reward = _perfect_stacking_reward(held_performed, prev_action, lines_cleared)
                        
            observation = _get_obs()
            
            print(f"Reward: {reward}")
            print(f"Observation:\n{observation}")
            
            if was_tetris_ready and game.is_tetris_ready():
                if (game.piece_manager.previous_piece == int(PieceTypeID.I_PIECE) or game.piece_manager.get_held_piece_id() == int(PieceTypeID.I_PIECE)):
                    done = True  
        
        for event in event_list:     
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    done = True
                    
        window.draw()
        
        if done:
            game.reset_game()
            reward = 0          