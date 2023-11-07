import pygame

import board.board_definitions as bd
import board.board_utils as bu
import game.game_settings as gs

from .logic_controller import LogicController


class WindowController():
    def __init__(self, g_controller: LogicController) -> None:
        # Set Game Controller
        self.g_controller = g_controller
        
        # Set window and surface sizes
        self.window = pygame.display.set_mode((bd.SCR_WIDTH, bd.SCR_HEIGHT))
        self.board_surface = pygame.Surface((bd.SCR_WIDTH, bd.SCR_HEIGHT), pygame.SRCALPHA)

        # Load images from resources
        self.background_image = pygame.image.load("res/gradient_background_blue.jpg").convert()

    def draw(self):
        """Draw all features to the screen.
        """
        # Clear window and draw window background       
        self.window.fill(0)

        # Draw board background
        self.window.blit(self.board_surface, (0, 0))
        self.board_surface.fill(0)
        self.board_surface.fill((0, 0, 0, bd.BACKGROUND_ALPHA), pygame.Rect(bd.BOARD_LEFT_BUFFER, bd.BOARD_TOP_BUFFER, bd.BOARD_PIXEL_WIDTH, bd.BOARD_PIXEL_HEIGHT))

        # Draw board grids 
        #window.blit(grid_surface, (0, 0))   
        #grid_surface.fill(0) 
        bu.draw_grids(self.board_surface)

        # Draw all pieces
        #window.blit(piece_surface, (0, 0))
        #piece_surface.fill(0)
        self.g_controller.draw_pieces(self.board_surface)

        # Draw fps counter
        if (gs.SHOW_FPS_COUNTER):
            self.board_surface.blit(self.g_controller.fps_string, (bd.SCR_WIDTH - (bd.GRID_SIZE * 3), bd.GRID_SIZE / 2))

        # Draw score
        self.board_surface.blit(self.g_controller.score_string, ((bd.GRID_SIZE), bd.GRID_SIZE / 2))

        # Draw back 2 back counter
        self.board_surface.blit(self.g_controller.b2b_string, ((bd.GRID_SIZE), bd.GRID_SIZE * 2))

        # Update window
        pygame.display.flip()
