import time

from game.game_settings import GameSettings
from game.actions import Actions

from pieces.piece_controller import PieceController
from pieces.piece_type_id import PieceTypeID

class GameController():
    
    def __init__(self) -> None:
        # Set Piece Controller
        self.p_controller = PieceController()
        
        # The seed being used for all random number generators
        self.seed = 0
    
        # Initialise time recording variables
        self.previous_time = time.time()
        self.total_time = 0
        self.fps_time = 0
        self.frames = 0
        self.last_fps_recorded = 0
        
        # Scores
        self.score = 0
        self.b2b = 0
        
        # The speed at which the tetramino pieces fall
        self.drop_speed = GameSettings.drop_speed
        self.drop_time = 1 / self.drop_speed
        
        # Restrict how often player can input keys (temp solution)
        self.move_delay = 0
        self.rotate_delay = 0
        
        # How long the tetramino can move around before without dropping before being deativated
        self.piece_deactivate_delay = self.drop_speed
        self.piece_deactivate_timer = self.piece_deactivate_delay
        
    def increment_frames_passed(self):
        """Increase number of frames that have passed by 1.
        """
        self.frames += 1
        
    def set_drop_speed(self, speed: int):
        """Set the drop speed of the pieces.

        Args:
            speed (int): How fast to drop the pieces.
        """
        self.drop_speed = speed
        self.drop_time = 1 / speed
        
        self.piece_deactivate_delay = self.drop_speed
        self.piece_deactivate_timer = self.piece_deactivate_delay
    
    def cycle_game_clock(self):
        self.update_delta_time()
        self.increment_frames_passed()
        self.update_fps_counter()
        
    def update_delta_time(self):
        """Updates to the latest delta time value.
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
        self.p_controller.draw_board_pieces(surface)
        self.p_controller.draw_ghost_pieces(surface)
        self.p_controller.draw_current_piece(surface)
        self.p_controller.draw_held_piece(surface)
        self.p_controller.draw_queued_pieces(surface)
        
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
                
        return self.score
    
    def reset_scores(self):
        self.score = 0
        self.b2b = 0
        
    def get_board_state(self):
        return self.p_controller.board.get_minimal_board_state()
    
    def get_board_value_bounds(self):
        return self.p_controller.board.EMPTY_PIECE_ID, len(PieceTypeID)
        
    
    def perform_action(self, action: int):
        match(action):
            case Actions.MOVE_LEFT:
                self.p_controller.shift_piece_horizontally(-1)
                
            case Actions.MOVE_RIGHT:
                self.p_controller.shift_piece_horizontally(1)
                
            case Actions.ROTATE_CLOCKWISE:
                self.p_controller.rotate_piece(clockwise=True)
                
            case Actions.ROTATE_ANTICLOCKWISE:
                self.p_controller.rotate_piece(clockwise=False)
                
            case Actions.SOFT_DROP:
                self.p_controller.hard_drop_piece()
                
            case Actions.HARD_DROP:
                self.p_controller.hard_drop_piece()
                self.new_piece_and_timer()
                
            case Actions.HOLD_PIECE:
                self.p_controller.hold_piece()
            
            case _:
                raise ValueError(f"ERROR: perform_action(action) - action '{action}' is invalid")
    
    def update_fps_counter(self):
        """Update the fps counter with the current number of frames.
        """
        if (self.fps_time >= 1):
            self.fps_time -= 1
            self.last_fps_recorded = self.frames
            self.frames = 0
    
    def run_logic(self):
        """Runs the logic for the movement of the pieces over time.
        """
        # Attempt Drop current piece every set amount of time
        if (self.total_time > self.drop_time):
            if (self.p_controller.gravity_drop_piece()):
                # print(f"fps -> {self.last_fps_recorded}")
                # If delay timer was running then restart it
                if (self.piece_deactivate_timer < self.piece_deactivate_delay):
                    self.piece_deactivate_timer = self.piece_deactivate_delay
            else:
                self.piece_deactivate_timer -= 1
                
            # Create new piece and restart timer if piece has been touching ground for too long
            if (self.piece_deactivate_timer < 0):
                self.new_piece_and_timer()

            # Cycle total time
            self.total_time = self.total_time - self.drop_time
            
        return self.p_controller.check_game_over(), self.score
    
    def reset_game(self):
        self.p_controller.reset_board_and_pieces()
        self.reset_scores()
        
    def new_piece_and_timer(self):
        self.p_controller.deactivate_piece()
        self.p_controller.next_piece()
        
        self.piece_deactivate_timer = self.piece_deactivate_delay