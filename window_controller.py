import pygame
import board_utils as bu
from game_controller import GameController

class WindowController():
    def __init__(self, g_controller: GameController) -> None:
        # Set Game Controller
        self.g_controller = g_controller
        
        # Set window and surface sizes
        self.window = pygame.display.set_mode((bu.SCR_WIDTH, bu.SCR_HEIGHT))
        self.board_surface = pygame.Surface((bu.SCR_WIDTH, bu.SCR_HEIGHT), pygame.SRCALPHA)

        # Load images from resources
        self.background_image = pygame.image.load("res/gradient_background_blue.jpg").convert()

    def draw(self):
        # Clear window and draw window background       
        self.window.fill(0)

        # Draw board background
        self.window.blit(self.board_surface, (0, 0))
        self.board_surface.fill(0)
        self.board_surface.fill((0, 0, 0, bu.BACKGROUND_ALPHA), pygame.Rect(bu.BOARD_LEFT_BUFFER, bu.BOARD_TOP_BUFFER, bu.BOARD_WIDTH, bu.BOARD_HEIGHT))

        # Draw board grids 
        #window.blit(grid_surface, (0, 0))   
        #grid_surface.fill(0) 
        bu.draw_grids(self.board_surface)

        # Draw all pieces
        #window.blit(piece_surface, (0, 0))
        #piece_surface.fill(0)
        self.g_controller.draw_pieces(self.board_surface)

        # Draw fps counter
        self.board_surface.blit(self.g_controller.fps_string, (bu.SCR_WIDTH - (bu.GRID_SIZE * 3), bu.GRID_SIZE / 2))

        # Draw score
        self.board_surface.blit(self.g_controller.score_string, ((bu.GRID_SIZE), bu.GRID_SIZE / 2))

        # Draw back 2 back counter
        self.board_surface.blit(self.g_controller.b2b_string, ((bu.GRID_SIZE), bu.GRID_SIZE * 2))

        # Update window
        pygame.display.flip()
