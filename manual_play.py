import math
import pygame
from controllers.game_controller import GameController
import utils.board_constants as bc 
from controllers.window import Window
from tetris_env import ScreenSizes

def occupied_spaces_to_average_height_ratio_reward(prev_gaps, prev_height_difference):
    occupied_spaces = game.get_occupied_spaces_on_board()
    max_height = game.get_max_piece_height_on_board()
    second_min_height = game.get_second_min_piece_height_on_board()
    board_height_difference = max_height - second_min_height

    nine_piece_row_reduction = math.floor(occupied_spaces / 10)
    reduced_occupied_spaces = occupied_spaces - nine_piece_row_reduction

    # ranges from 1-9 where 9 = best, 1 = worst
    # pieces_max_height_ratio = reduced_occupied_spaces / max_height

    gaps_multiplyer = 1 - (game.get_num_of_gaps() - prev_gaps)
        
    if (gaps_multiplyer < 0):
        gaps_multiplyer = 0

    diff_reward = (5 - board_height_difference)
    
    if (diff_reward < 0):
        diff_reward = 0
    else:
        diff_reward = diff_reward ** 4
        
    if gaps_multiplyer < 1:
        gaps_multiplyer = 0
    
    reward = gaps_multiplyer * diff_reward

    return int(reward / 10)

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
    
    actions_over_limit = 0

    while running:
        # Check if user has quit the window
        if (pygame.event.get(pygame.QUIT)):
            running = False
            
        game.cycle_game_clock()
        
        previous_pieces_dropped = game.get_num_of_pieces_dropped()
        
        event_list = pygame.event.get()
        
        prev_gaps = game.get_num_of_gaps()
        
        max_height = game.get_max_piece_height_on_board()
        second_min_height = game.get_second_min_piece_height_on_board()
        prev_height_difference = max_height - second_min_height
        
        game.take_player_inputs(event_list)
        done, gained_score = game.run_logic()
        window.draw()

        if (game.get_num_of_pieces_dropped() - previous_pieces_dropped) > 0:
            reward += occupied_spaces_to_average_height_ratio_reward(prev_gaps, prev_height_difference)

        for event in event_list:     
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    done = True
                else:
                    gaps = game.get_num_of_gaps()
        
        if done:             
            print(reward)
            game.reset_game()
            reward = 0
        
        actions_over_limit = 0