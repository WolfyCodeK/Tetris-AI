import math
import pygame
from controllers.game_controller import GameController
import utils.board_constants as bc 
from controllers.window import Window
from tetris_env import ScreenSizes

def occupied_spaces_to_average_height_ratio_reward():
    occupied_spaces = game.get_occupied_spaces_on_board()
    max_height = game.get_max_piece_height_on_board()
    second_min_height = game.get_second_min_piece_height_on_board()
    board_height_difference = max_height - second_min_height

    nine_piece_row_reduction = math.floor(occupied_spaces / 10)
    reduced_occupied_spaces = occupied_spaces - nine_piece_row_reduction

    # ranges from 1-9 where 9 = best, 1 = worst
    # pieces_max_height_ratio = reduced_occupied_spaces / max_height

    adjusted_difference = (6 - board_height_difference)

    if adjusted_difference > 0 and occupied_spaces > bc.PIECE_COMPONENTS:
        reward = adjusted_difference ** 2
    else:
        reward = 0
        
    return reward

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
        
        gaps = game.get_num_of_gaps()
        
        game.take_player_inputs(event_list)
        done, gained_score = game.run_logic()
        window.draw()

        # if (game.get_num_of_pieces_dropped() - previous_pieces_dropped) > 0:
        #     reward += occupied_spaces_to_average_height_ratio_reward()

        for event in event_list:     
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    done = True
                else:
                    gaps = game.get_num_of_gaps()
        
        if done:   

            reward += 100 - gaps
            print(reward)
            game.reset_game()
            reward = 0
        
        actions_over_limit = 0