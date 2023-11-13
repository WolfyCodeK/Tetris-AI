import pygame
import os

import board.board_definitions as bd
import board.board_utils as bu
from game.game_settings import GameSettings

from .logic_controller import LogicController


class WindowController():
    def __init__(self, l_controller: LogicController) -> None:
        # Set window position
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d, %d" %(100, 100)
        
        # Initial pygame setup
        pygame.display.init()
        pygame.font.init()
        
        pygame.display.set_caption("Tetris - Pygame")
        pygame.display.set_mode((1, 1))

        # Set window icon
        tetris_icon = pygame.image.load("res/tetris-icon.png")
        pygame.display.set_icon(tetris_icon)
        
        # Set logic controller
        self.l_controller = l_controller
        
        self.GRID_SIZE = self.l_controller.get_grid_size()
        self._init_board_size_values(self.GRID_SIZE)
        
        # Colour values
        self.fps_colour = (0, 255, 0)
        self.score_colour = (255, 255, 255)
        self.b2b_colour = (255, 0, 0)
        
        # Set fps string
        self.font = pygame.font.Font("freesansbold.ttf", self.l_controller.get_grid_size())
        self.fps_string = self.font.render(str("- - -"), True, self.fps_colour)
        
        # Set score string
        self.score_string = self.font.render(str(self.l_controller.score), True, (255, 255, 255))
        
        # Set back to back string
        self.b2b_string = self.font.render(str(self.l_controller.b2b), True, (255, 255, 255))
        
        # Set window and surface sizes
        self.window = pygame.display.set_mode((self.SCR_WIDTH, self.SCR_HEIGHT))
        self.board_surface = pygame.Surface((self.SCR_WIDTH, self.SCR_HEIGHT), pygame.SRCALPHA)

        # Load images from resources
        self.background_image = pygame.image.load("res/gradient_background_blue.jpg").convert()
    
    def _init_board_size_values(self, grid_size):
        # Board drawing helper value
        self.BOARD_RIGHT_BUFFER = 7 * grid_size
        self.BOARD_LEFT_BUFFER = 7 * grid_size
        self.BOARD_TOP_BUFFER = 4 * grid_size
        self.BOARD_BOTTOM_BUFFER = 4 * grid_size

        # Base board size in pixel
        self.BOARD_PIXEL_WIDTH = grid_size * bd.BOARD_COLUMNS
        self.BOARD_PIXEL_HEIGHT = grid_size * bd.BOARD_ROWS

        # Screen size
        self.SCR_WIDTH = self.BOARD_PIXEL_WIDTH + self.BOARD_RIGHT_BUFFER + self.BOARD_LEFT_BUFFER
        self.SCR_HEIGHT = self.BOARD_PIXEL_HEIGHT + self.BOARD_TOP_BUFFER + self.BOARD_BOTTOM_BUFFER

        # Board Coordinate Definitions
        self.TOP_LEFT_BOARD_CORNER = (self.BOARD_LEFT_BUFFER - 2, self.BOARD_TOP_BUFFER)
        self.TOP_RIGHT_BOARD_CORNER = (self.BOARD_LEFT_BUFFER + self.BOARD_PIXEL_WIDTH + 1, self.BOARD_TOP_BUFFER)
        self.BOTTOM_LEFT_BOARD_CORNER = (self.BOARD_LEFT_BUFFER - 2, self.BOARD_TOP_BUFFER + self.BOARD_PIXEL_HEIGHT + 1)
        self.BOTTOM_RIGHT_BOARD_CORNER = (self.BOARD_LEFT_BUFFER + self.BOARD_PIXEL_WIDTH + 1, self.BOARD_TOP_BUFFER + self.BOARD_PIXEL_HEIGHT + 1)
        
    def draw(self):
        """Draw all features to the screen.
        """
        # Clear window and draw window background       
        self.window.fill(0)

        # Draw board background
        self.window.blit(self.board_surface, (0, 0))
        self.board_surface.fill(0)
        self.board_surface.fill((0, 0, 0, bd.BACKGROUND_ALPHA), pygame.Rect(self.BOARD_LEFT_BUFFER, self.BOARD_TOP_BUFFER, self.BOARD_PIXEL_WIDTH, self.BOARD_PIXEL_HEIGHT))

        # Draw board grids 
        #window.blit(grid_surface, (0, 0))   
        #grid_surface.fill(0) 
        bu.draw_grids(self.board_surface)

        # Draw all pieces
        #window.blit(piece_surface, (0, 0))
        #piece_surface.fill(0)
        self.l_controller.draw_pieces(self.board_surface)

        # Draw fps counter
        if (GameSettings.show_fps_counter):
            fps_string = self.font.render(
                str(int(self.l_controller.last_fps_recorded)),
                True, 
                self.fps_colour
            )
            
            self.board_surface.blit(fps_string, (self.SCR_WIDTH - (self.GRID_SIZE * 3), self.GRID_SIZE / 2))
            
        # Draw score
        score_string = self.font.render(
            str(f"score:  {self.l_controller.score}"),
            True, 
            self.score_colour
        )
        
        # Draw back 2 back counter
        b2b_string = self.font.render(
            str(f"B2B:  {self.l_controller.b2b}"),
            True, 
            self.b2b_colour
        )

        # Draw score
        self.board_surface.blit(score_string, ((self.GRID_SIZE), self.GRID_SIZE / 2))

        # Draw back 2 back counter
        self.board_surface.blit(b2b_string, ((self.GRID_SIZE), self.GRID_SIZE * 2))

        # Update window
        pygame.display.flip()
