import pygame
from controllers.game_controller import GameController
from controllers.window import Window
from game.actions import Actions
import utils.game_utils as gu
from pieces.piece_type_id import PieceTypeID
import utils.board_constants as bc
import numpy as np
from utils.screen_sizes import ScreenSizes

GAME_OVER_PUNISH = -100
MAX_BOARD_DIFF = 5
REWARD_MULTIPLYER = 2
LINE_CLEAR_REWARD = 15
QUEUE_OBS_NUM = 5

def _perfect_stacking_reward(lines_cleared):      
        relative_piece_height = game.piece_manager.placed_piece_max_height - gu.get_min_piece_height_on_board(game)
        
        if lines_cleared == 4:
            lines_cleared_reward = REWARD_MULTIPLYER ** lines_cleared * LINE_CLEAR_REWARD
        else:
            lines_cleared_reward = 0
            
        reward = (bc.BOARD_ROWS - relative_piece_height) + lines_cleared_reward     
        
        return reward

def _get_obs():
        return {"board": _get_board_obs(), "additional": _get_additional_obs()}
    
def _get_info():
    pass 

def _get_board_obs() -> np.ndarray:
    board = np.array(gu.get_max_height_column_list(game))
    board = board - gu.get_min_gap_height_exluding_well(game)
    
    board = np.clip(board, a_min = 0, a_max = 20) 
    
    return board

def _get_additional_obs() -> np.ndarray: 
    gaps = gu.get_num_of_full_gaps(game) + gu.get_num_of_top_gaps(game)
    
    if gaps > 0:
        gaps = 1
    
    if gu.is_tetris_ready(game):
        tetris_ready = 1
    else:
        tetris_ready = 0 

    return np.array(
        [
            gaps,
            tetris_ready,
            game.holds_used_in_a_row,
            gu.get_held_piece_id(game),
            gu.get_current_piece_id(game)
            
        ] + gu.get_truncated_piece_queue(game, QUEUE_OBS_NUM)
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
        prev_lines_cleared = game.lines_cleared
        was_tetris_ready = gu.is_tetris_ready(game)
        
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
            # Check how many lines were cleared after performing actions
            lines_cleared = game.lines_cleared - prev_lines_cleared
            
            if lines_cleared == 4:
                print(f"Tetris!")
            
            ###############################
            # Strict game over conditions #
            ###############################
            
            # Terminate if tetris ready and able to tetris but didn't
            if was_tetris_ready and gu.is_tetris_ready(game):
                if (game.piece_manager.previous_piece == int(PieceTypeID.I_PIECE) or gu.get_held_piece_id(game) == int(PieceTypeID.I_PIECE)):
                    done = True    
                    reward = GAME_OVER_PUNISH
                    print(f"Failed Tetris")

            # Terminate if gap created on board
            if gu.get_num_of_full_gaps(game) > 0 or gu.get_num_of_top_gaps(game) > 0:
                done = True
                reward = GAME_OVER_PUNISH
            
            # Terminate if height difference violated of board well incorrectly filled
            if gu.get_board_height_difference_with_well(game) > MAX_BOARD_DIFF:
                done = True
                reward = GAME_OVER_PUNISH    
                
            # Termiante if pieces placed in well
            if not gu.is_well_valid(game):
                done = True
                
                if game.piece_manager.previous_piece == int(PieceTypeID.I_PIECE):
                    reward = 0
                else:
                    reward = GAME_OVER_PUNISH    
            
            # Punish agent for using the hold action more than once in a row
            if held_performed and prev_action == int(Actions.HOLD_PIECE):
                reward = GAME_OVER_PUNISH * 10
                print("Held Twice!")

            if not done:
                reward = _perfect_stacking_reward(lines_cleared)

            # Get observations 
            observation = _get_obs()
            info = _get_info()
            
            print(f"Reward: {reward}")
            print(f"Observation:\n{observation}")
        
        for event in event_list:     
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    done = True
                    
        window.draw()
        
        if done:
            game.reset_game()
            reward = 0          