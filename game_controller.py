import pygame
import time
from board_utils import GRID_SIZE

class GameController():
    #Initialise time recording variables
    previous_time = time.time()
    total_time = 0
    fps_time = 0
    frames = 0
    
    def __init__(self, p_controller) -> None:
        self.p_controller = p_controller
    
        # Set fps string
        self.font = pygame.font.Font("freesansbold.ttf", GRID_SIZE)
        self.fps_string = self.font.render(str("- - -"), True, (0, 255, 0))
        
        # The speed at which the tetramino pieces fall
        self.drop_speed = 1
        self.drop_time = 1 / self.drop_speed
        
        # How long the tetramino can move around before without dropping before being deativated
        self.piece_deactivate_delay = self.drop_speed * 1000000
        self.piece_deactivate_timer = self.piece_deactivate_delay
        
        # Set score string
        self.score = 0
        self.score_string = self.font.render(str(self.score), True, (255, 255, 255))
        
        # Set back to back string
        self.b2b = 0
        self.b2b_string = self.font.render(str(self.b2b), True, (255, 255, 255))
        
    def set_drop_speed(self, speed):
        self.drop_speed = speed
        self.drop_time = 1 / speed
        
        self.piece_deactivate_delay = self.drop_speed
        self.piece_deactivate_timer = self.piece_deactivate_delay
        
    def update_delta_time(self):
        """Calculates the delta time.
        """
        self.delta_time = time.time() - self.previous_time
        self.previous_time = time.time()

        self.total_time += self.delta_time
        self.fps_time += self.delta_time
    
    def draw_pieces(self, surface):
        """Draws the peices to the parsed surface.

        Args:
            surface (Surface): The surface being drawn to.
        """
        self.p_controller.draw_ghost_pieces(surface)
        self.p_controller.draw_deactivated_pieces(surface)
        self.p_controller.draw_current_piece(surface)
        self.p_controller.draw_held_piece(surface)
        
    def clear_lines_and_add_score(self):
        lines_cleared = self.p_controller.perform_line_clears()
        
        # Award points
        if (lines_cleared == 1):
            self.score += 40
        
        if (lines_cleared == 2):
            self.score += 100
        
        if (lines_cleared == 3):
            self.score += 300
            
        if (lines_cleared == 4):
            self.score += 1200
            self.b2b += 1
        
        if (lines_cleared in [1, 2, 3]):
            self.b2b = 0
            
        # Update score
        self.score_string = self.font.render(
            str(f"score:  {self.score}"),
            True, 
            (255, 255, 255)
        )
        
        # Update back 2 back counter
        self.b2b_string = self.font.render(
            str(f"B2B:  {self.b2b}"),
            True, 
            (255, 0, 0)
        )
        
    def run_timed_game_logic(self):
        """Runs the logic for the movement of the pieces over time.
        """
        # Update fps counter every second
        if (self.fps_time >= 1):
            self.fps_string = self.font.render(
                str(int(self.frames)),
                True, 
                (0, 255, 0)
            )
            
            self.fps_time = self.fps_time - 1
            self.frames = 0
        
        # Attempt Drop current piece every set amount of time
        if (self.total_time > self.drop_time):
            if (self.p_controller.gravity_drop_piece()):
                # If delay timer was running then restart it
                if (self.piece_deactivate_timer < self.piece_deactivate_delay):
                    self.piece_deactivate_timer = self.piece_deactivate_delay
            else:
                self.piece_deactivate_timer -= 1
                
            # Create new piece and restart timer if piece needs deactivating
            if (self.piece_deactivate_timer < 0):
                self.new_piece_and_timer()

            # Cycle total time
            self.total_time = self.total_time - self.drop_time
            
        self.clear_lines_and_add_score()
        
        if (self.p_controller.check_game_over()):
            self.p_controller.restart_board()
            
    def new_piece_and_timer(self):
        self.p_controller.deactivate_piece()
        self.p_controller.new_piece()
        
        self.piece_deactivate_timer = self.piece_deactivate_delay