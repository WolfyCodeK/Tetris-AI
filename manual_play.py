import math
import pygame
from controllers.game_controller import GameController
from controllers.window import Window
from tetris_env import ScreenSizes
from pieces.piece_manager import PieceManager

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

        game.update_delta_time()
        game.increment_frames_passed()
        game.update_fps_counter()  
        game.take_player_inputs(pygame.event.get())
        done = game.run_logic()
        window.draw()
        
        if done:    
            reward += game.score
            reward += ((pieces_max_height_ratio - 5) * 5)
            print(reward)
            
            game.reset_game()
            reward = 0
        
        max_height = game.get_max_piece_height_on_board()
        
        if (max_height > 0):
            occupied_spaces = game.get_occupied_spaces_on_board()
            
            nine_piece_row_reduction = math.floor(occupied_spaces / 10)
            reduced_occupied_spaces = occupied_spaces - nine_piece_row_reduction
            
            # ranges from 1-9 where 9 = best, 1 = worst
            pieces_max_height_ratio = reduced_occupied_spaces / max_height