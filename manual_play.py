import math
import pygame
from controllers.game_controller import GameController
from pieces.piece_type_id import PieceTypeID
import utils.board_constants as bc 
from controllers.window import Window
from gym_tetris_env import ScreenSizes

def flat_stack_reward_method(prev_top_gaps, prev_full_gaps, previous_piece_id):
    occupied_spaces = game.get_occupied_spaces_on_board()
    max_height = game.get_max_piece_height_on_board()
    second_min_height = game.get_second_min_piece_height_on_board()
    board_height_difference = max_height - second_min_height

    nine_piece_row_reduction = math.floor(occupied_spaces / 10)
    reduced_occupied_spaces = occupied_spaces - nine_piece_row_reduction

    # ranges from 1-9 where 9 = best, 1 = worst
    # pieces_max_height_ratio = reduced_occupied_spaces / max_height

    if board_height_difference < 6:
        diff = 3
    else: 
        diff = 0

    top_gaps = game.get_num_of_top_gaps() - prev_top_gaps
    full_gaps = game.get_num_of_full_gaps() - prev_full_gaps
    gaps_multiplyer = 1
    
    if (full_gaps == 0):
        if top_gaps <= 0:
            gaps_multiplyer = (1 + (-top_gaps)) * 2
        else:
            gaps_multiplyer = (-top_gaps) * 2
            
        if occupied_spaces == 4 and (previous_piece_id == PieceTypeID.S_PIECE or previous_piece_id == PieceTypeID.Z_PIECE):
            gaps_multiplyer = 1
    elif full_gaps > 0:
        gaps_multiplyer = -4
    else:
        gaps_multiplyer = full_gaps * -2
    
    if (gaps_multiplyer < 0):
        return gaps_multiplyer
    else:
        return diff * gaps_multiplyer

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
            
        game.cycle_game_clock()
        
        previous_pieces_dropped = game.get_num_of_pieces_dropped()
        previous_piece_id = game.piece_manager.current_piece.id 
        prev_top_gaps = game.get_num_of_top_gaps()
        prev_full_gaps = game.get_num_of_full_gaps()
        
        event_list = pygame.event.get()
        
        game.take_player_inputs(event_list)
        done = game.run_logic()
        window.draw()

        if (game.get_num_of_pieces_dropped() - previous_pieces_dropped) > 0:
            reward += flat_stack_reward_method(prev_top_gaps, prev_full_gaps, previous_piece_id)
            # print(f"full: {game.get_num_of_full_gaps()}")
            # print(f"top: {game.get_num_of_top_gaps()}")

        for event in event_list:     
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    done = True
                else:
                    full_gaps = game.get_num_of_full_gaps()
                    
        full_gaps = game.get_num_of_full_gaps()
        
        if (not game.piece_manager.board.check_all_previous_rows_filled()) or (full_gaps > 0) or (game.piece_manager.actions_per_piece > 10):
            done = True
        
        if done:
            reward += game.score * (game.b2b + 1)
            game.reset_game()
            reward = 0          