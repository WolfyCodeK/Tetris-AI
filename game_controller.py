import pygame
import time
import board_utils as bu 
from piece_controller import PieceController

class GameController():
    
    def __init__(self, p_controller: PieceController) -> None:
        # Set Piece Controller
        self.p_controller = p_controller
    
        # Initialise time recording variables
        self.previous_time = time.time()
        self.total_time = 0
        self.fps_time = 0
        self.frames = 0
    
        # Set fps string
        self.font = pygame.font.Font("freesansbold.ttf", bu.GRID_SIZE)
        self.fps_string = self.font.render(str("- - -"), True, (0, 255, 0))
        
        # The speed at which the tetramino pieces fall
        self.drop_speed = 1
        self.drop_time = 1 / self.drop_speed
        
        # Restrict how often player can input keys (temp solution)
        self.move_delay = 0
        self.rotate_delay = 0
        
        # How long the tetramino can move around before without dropping before being deativated
        self.piece_deactivate_delay = self.drop_speed
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
        if (lines_cleared != 0):
            if (lines_cleared == 1):
                self.score += 40
            
            if (lines_cleared == 2):
                self.score += 100
            
            if (lines_cleared == 3):
                self.score += 300
                
            if (lines_cleared == 4):
                self.score += 1200
                self.b2b += 1
            else:
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
    
    def reset_score(self):
        self.score = 0
        self.b2b = 0
        
    def take_player_inputs(self, event_list):
        # Take player input
        key = pygame.key.get_pressed()
        self.move_delay -= 1 
        self.rotate_delay -= 1
        
        if (key[pygame.K_RIGHT] == True) and (self.move_delay < 0):
            self.p_controller.shift_piece_horizontally(1)
            self.move_delay = 75 * 32 / bu.GRID_SIZE
            
        if key[pygame.K_LEFT] == True and (self.move_delay < 0):
            self.p_controller.shift_piece_horizontally(-1)
            self.move_delay = 75 * 32 / bu.GRID_SIZE
            
        if key[pygame.K_x] == True and (self.rotate_delay < 0):
            self.p_controller.rotate_piece(1)
            self.rotate_delay = 120 * 32 / bu.GRID_SIZE
            
        if key[pygame.K_z] == True and (self.rotate_delay < 0):
            self.p_controller.rotate_piece(-1)
            self.rotate_delay = 120 * 32 / bu.GRID_SIZE
        
        # DEBUG EVENTS
        if key[pygame.K_a] == True:
            self.p_controller.set_drop_speed(20)
        
        if key[pygame.K_s] == True:
            self.p_controller.set_drop_speed(1)
        
        for event in event_list:          
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.p_controller.hard_drop_piece()
                    self.new_piece_and_timer()
                    
                if event.key == pygame.K_DOWN:
                    self.p_controller.hard_drop_piece()
                    
                if event.key == pygame.K_r:
                    self.p_controller.restart_board()
                    self.reset_score()
                    
                if event.key == pygame.K_LSHIFT:
                    self.p_controller.hold_piece()
    
    def update_fps_counter(self):
        # Update fps counter every second
        if (self.fps_time >= 1):
            self.fps_string = self.font.render(
                str(int(self.frames)),
                True, 
                (0, 255, 0)
            )
            
            self.fps_time = self.fps_time - 1
            self.frames = 0
    
    def run_timed_game_logic(self):
        """Runs the logic for the movement of the pieces over time.
        """
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