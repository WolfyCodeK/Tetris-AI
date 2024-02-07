import pygame
from controllers.game_controller import GameController
from controllers.window import Window
from game.actions import Actions
from pieces.piece_type_id import PieceTypeID
import utils.board_constants as bc
import numpy as np
from utils.screen_sizes import ScreenSizes

def _perfect_stacking_reward(held_performed, prev_action, prev_full_gaps, prev_top_gaps, prev_line_clears):      
    if held_performed:
        if prev_action == int(Actions.HOLD_PIECE):
            reward = -100
        else:
            reward = 0
    else:
        relative_piece_height = game.piece_manager.placed_piece_max_height - game.get_min_piece_board_height()
        reward = (bc.BOARD_ROWS - relative_piece_height) + prev_line_clears
            
    return reward

def _overfit_reward_calculation(held_performed, prev_action, prev_full_gaps, prev_top_gaps, prev_line_clears):
    board_height_difference = game.get_board_height_difference()
    
    min_height = game.get_second_lowest_gap()
    
    top_gaps = game.get_num_of_top_gaps() - prev_top_gaps
    full_gaps = game.get_num_of_full_gaps() - prev_full_gaps

    lines_cleared = game.lines_cleared - prev_line_clears
    
    if held_performed:
        if prev_action == int(Actions.HOLD_PIECE):
            reward = -100
        else:
            reward = 0
    else:
        relative_piece_height = game.piece_manager.placed_piece_max_height - game.get_min_piece_board_height()
        reward_gain = ((7 - relative_piece_height) * 2) + (lines_cleared * 25)  
        
        if lines_cleared == 0:
            if full_gaps > 0:   
                reward = -int((full_gaps * 4) ** (relative_piece_height))
            elif top_gaps > 0:
                reward = -int((top_gaps * 2) ** (relative_piece_height))
            elif board_height_difference < 7:
                reward = reward_gain
        else:
            reward = reward_gain

    if reward < -100:
        reward = -100

    return reward

def _get_obs():
    return {"board": _get_board_obs(), "additional": _get_additional_obs()}

def _get_board_obs() -> np.ndarray:
    reduced_board = game.get_minimal_board_state()
    reduced_board = np.delete(reduced_board, obj=range(0, bc.BOARD_ROWS-6), axis=0)
    
    return np.where(reduced_board == 0, 0, 1)

def _get_additional_obs() -> np.ndarray: 
    return np.array(
        [
            game.get_current_piece_id(),
            game.get_next_piece_id()
        ] 
    )

if __name__ == '__main__':    
    game = GameController()
    
    # Pygame intial setup
    pygame.display.init()
    pygame.font.init()
    
    window = Window(game, screen_size=ScreenSizes.MEDIUM)
    
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
        prev_top_gaps = game.get_num_of_top_gaps()
        prev_full_gaps = game.get_num_of_full_gaps()
        
        game.take_player_inputs(event_list)
        done = game._run_logic()
        
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
            if (game.get_num_of_full_gaps() > 0 or game.get_num_of_top_gaps() > 0):
                done = True    
                reward = -100
            else:
                reward = _perfect_stacking_reward(held_performed, prev_action, prev_full_gaps, prev_top_gaps, prev_line_clears)
                        
            observation = _get_obs()
            
            print(f"Reward: {reward}")
            # print(f"Observation:\n{observation}")
        
        for event in event_list:     
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    done = True
                    
        window.draw()
        
        if done:
            game.reset_game()
            reward = 0          