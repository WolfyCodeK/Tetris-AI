import pygame
from controllers.game_controller import GameController
from controllers.window import Window
from pieces.piece_type_id import PieceTypeID
from utils.screen_sizes import ScreenSizes

def flat_stack_reward_method(prev_top_gaps, prev_full_gaps, previous_piece_id):
    occupied_spaces = game.get_occupied_spaces_on_board()
    board_height_difference = game.get_board_height_difference()

    if board_height_difference < 8:
        diff = 1
    else: 
        diff = -20

    top_gaps = game.get_num_of_top_gaps() - prev_top_gaps
    full_gaps = game.get_num_of_full_gaps() - prev_full_gaps
    gaps_reward = 1

    if (full_gaps == 0):
        if top_gaps <= 0:
            gaps_reward = (1 - top_gaps)
        else:
            gaps_reward = -5 * top_gaps
            
        if occupied_spaces == 4 and (previous_piece_id == PieceTypeID.S_PIECE or previous_piece_id == PieceTypeID.Z_PIECE):
            gaps_reward = 1
            
    elif full_gaps > 0:
        gaps_reward = -20
    else:
        raise ValueError(f"Full gaps cannot be negative when calculating reward!: full_gap: {full_gaps}, prev_full_gaps: {prev_full_gaps}")
    
    if (gaps_reward < 0):
        return gaps_reward
    else:
        return diff + gaps_reward

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
        
        prev_top_gaps = game.get_num_of_top_gaps()
        prev_full_gaps = game.get_num_of_full_gaps()
        prev_piece_id = game.piece_manager.current_piece.id 
        
        game.take_player_inputs(event_list)
        done = game._run_logic()
        
        # Calculate reward
        if (not game.piece_manager.board.check_all_previous_rows_filled()) or (game.get_num_of_full_gaps() > 0) or (game.get_board_height_difference() >= 8):
            done = True    
            reward = -50
        else:
            reward = flat_stack_reward_method(prev_top_gaps, prev_full_gaps, prev_piece_id)
        
        window.draw()

        for event in event_list:     
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    done = True
        
        if done:
            game.reset_game()
            print(reward)
            reward = 0          