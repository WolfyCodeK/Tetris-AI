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
    
    while running:
        # Check if user has quit the window
        if (pygame.event.get(pygame.QUIT)):
            running = False

        game.update_delta_time()
        game.increment_frames_passed()
        game.update_fps_counter()  
        game.take_player_inputs(pygame.event.get())
        game.run_logic()
        window.draw()