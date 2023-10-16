import pygame
from game_states import GameStates
from piece_controller import PieceController
from game_controller import GameController
from window_controller import WindowController

# Pygame intial setup
pygame.init()
pygame.display.set_caption("Tetris - Pygame")

# Set window icon
tetris_icon = pygame.image.load("res/tetris-icon.png")
pygame.display.set_icon(tetris_icon)

# Create controllers for the game
p_controller = PieceController()
g_controller = GameController(p_controller)
w_controller = WindowController(g_controller)

game_state = GameStates.UPDATE_DELTA_TIME

running = True

while running:
    # Check if user has quit the window
    if (pygame.event.get(pygame.QUIT)):
        running = False
    
    match game_state:
        case GameStates.UPDATE_DELTA_TIME:
            g_controller.update_delta_time()
            g_controller.update_fps_counter()
            game_state = GameStates.TAKE_INPUTS
            
        case GameStates.TAKE_INPUTS:
            g_controller.take_player_inputs(pygame.event.get())
            game_state = GameStates.RUN_LOGIC
            
        case GameStates.RUN_LOGIC:
            g_controller.run_timed_game_logic()
            game_state = GameStates.DRAW_GAME
            
        case GameStates.DRAW_GAME:
            w_controller.draw()
            game_state = GameStates.UPDATE_DELTA_TIME
    
pygame.quit()