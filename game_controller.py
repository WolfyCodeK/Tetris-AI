import pygame
import time

class GameController():
    previous_time = time.time()
    total_time = 0
    frames = 0
    fps_update_delay = 0
    
    drop_speed = 2
    drop_time = 1 / drop_speed
    
    piece_deactivate_delay = drop_speed
    piece_deactivate_timer = piece_deactivate_delay
    
    def __init__(self, p_controller) -> None:
        self.p_controller = p_controller
        
        # Fps counter font
        self.font = pygame.font.Font("freesansbold.ttf", 32)
        
        # Set in-game fps
        self.fps_string = self.font.render(str("- - -"), True, (0, 255, 0))
        pass
    
    def update_delta_time(self):
        self.delta_time = time.time() - self.previous_time
        self.previous_time = time.time()

        self.total_time += self.delta_time
    
    def draw_pieces(self, board_surface):
        self.p_controller.draw_deactivated_pieces(board_surface)
        self.p_controller.draw_current_piece(board_surface)
    
    def run_game_loop(self):
        # Drop current piece and update fps counter
        if (self.total_time > self.drop_time):
            print(f"(x: {self.p_controller.current_piece.x_pos}, y: x: {self.p_controller.current_piece.y_pos})")
            if (self.fps_update_delay >= self.drop_speed):
                self.fps_string = self.font.render(
                    str(int(self.frames * self.drop_speed)),
                    True, 
                    (0, 255, 0)
                )
                
                self.fps_update_delay = 0
            
            if (self.p_controller.gravity_drop_piece()):
                # If delay timer was running then restart it
                if (self.piece_deactivate_timer < self.piece_deactivate_delay):
                    self.piece_deactivate_timer = self.piece_deactivate_delay
            else:
                self.piece_deactivate_timer -= 1
                
            if (self.piece_deactivate_timer < 0):
                self.p_controller.deactivate_piece()
                self.p_controller.new_piece()
                
                self.piece_deactivate_timer = self.piece_deactivate_delay

            self.total_time = self.total_time - self.drop_time
            self.frames = 0
            self.fps_update_delay += 1