import math
import pygame
from controllers.game_controller import GameController
from controllers.window import Window
from tetris_env import ScreenSizes

def occupied_spaces_to_average_height_ratio_reward():
    max_height = game.get_max_piece_height_on_board()
    occupied_spaces = game.get_occupied_spaces_on_board()
    
    nine_piece_row_reduction = math.floor(occupied_spaces / 10)
    reduced_occupied_spaces = occupied_spaces - nine_piece_row_reduction
    
    pieces_max_height_ratio = reduced_occupied_spaces / max_height
    
    return (1 - pieces_max_height_ratio) * 5

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
        
        game.take_player_inputs(pygame.event.get())
        done, gained_score = game.run_logic()
        window.draw()
        
        if (game.get_num_of_pieces_dropped() - previous_pieces_dropped):
            reward += occupied_spaces_to_average_height_ratio_reward()

        if done:   
            game.reset_game()
            reward = 0
            
        reward += gained_score
        
        print(reward)