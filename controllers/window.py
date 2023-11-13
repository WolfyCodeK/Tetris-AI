import pygame
import os

from game.game_settings import GameSettings
import utils.board_constants as bc
import utils.window_utils as win_utils
from controllers.game_controller import GameController

class Window():
    def __init__(self, game: GameController) -> None:
        # Set window position
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d, %d" %(100, 100)
        
        pygame.display.set_caption("Tetris - Pygame")
        pygame.display.set_mode((1, 1))

        # Set window icon
        tetris_icon = pygame.image.load("res/tetris-icon.png")
        pygame.display.set_icon(tetris_icon)
        
        # Set logic controller
        self.game = game
        
        # Colour values
        self.fps_colour = (0, 255, 0)
        self.score_colour = (255, 255, 255)
        self.b2b_colour = (255, 0, 0)
        
        # Set fps string
        self.font = pygame.font.Font("freesansbold.ttf", win_utils.get_grid_size())
        self.fps_string = self.font.render(str("- - -"), True, self.fps_colour)
        
        # Set score string
        self.score_string = self.font.render(str(self.game.score), True, (255, 255, 255))
        
        # Set back to back string
        self.b2b_string = self.font.render(str(self.game.b2b), True, (255, 255, 255))
        
        # Set window and surface sizes
        self.scr_width, self.scr_height = win_utils.get_screen_sizes()
        self.window = pygame.display.set_mode((self.scr_width, self.scr_height))
        self.board_surface = pygame.Surface((self.scr_width, self.scr_height), pygame.SRCALPHA)

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
        _, left_buf, top_buf, _ = win_utils.get_board_buffers()
        self.board_surface.fill((0, 0, 0, bc.BACKGROUND_ALPHA), pygame.Rect(left_buf, top_buf, self.scr_width, self.scr_height))

        # Draw board grids 
        #window.blit(grid_surface, (0, 0))   
        #grid_surface.fill(0) 
        win_utils.draw_grids(self.board_surface)

        # Draw all pieces
        #window.blit(piece_surface, (0, 0))
        #piece_surface.fill(0)
        self.game.draw_pieces(self.board_surface)

        # Draw fps counter
        if GameSettings.show_fps_counter:
            fps_string = self.font.render(
                str(int(self.game.last_fps_recorded)),
                True, 
                self.fps_colour
            )
            
            self.board_surface.blit(fps_string, (self.scr_width - (win_utils.get_grid_size() * 3), win_utils.get_grid_size() / 2))
            
        # Draw score
        score_string = self.font.render(
            str(f"score:  {self.game.score}"),
            True, 
            self.score_colour
        )
        
        # Draw back 2 back counter
        b2b_string = self.font.render(
            str(f"B2B:  {self.game.b2b}"),
            True, 
            self.b2b_colour
        )

        # Draw score
        self.board_surface.blit(score_string, ((win_utils.get_grid_size()), win_utils.get_grid_size() / 2))

        # Draw back 2 back counter
        self.board_surface.blit(b2b_string, ((win_utils.get_grid_size()), win_utils.get_grid_size() * 2))

        # Update window
        pygame.display.flip()
