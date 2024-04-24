import time

import pygame
import utils.window_utils as win_utils

from game.game_settings import GameSettings
from game.actions import Actions

from pieces.piece_manager import PieceManager
from pieces.piece_type_id import PieceTypeID

class GameController():
    
    def __init__(self) -> None:
        # Set Piece Controller
        self.piece_manager = PieceManager()
    
        # Initialise time recording variables
        self.previous_time = time.time()
        self.total_time = 0
        self.fps_time = 0
        self.frames = 0
        self.last_fps_recorded = 0
        
        self.lines_cleared = 0
        self.holds_performed_in_a_row = 0
        self.previous_action = None
        
        self.render_game = True
        self.playback = False
        
        # Scores
        self.score = 0
        self.b2b = 0
        self.max_b2b = 0
        
        # The speed at which the tetramino pieces fall
        self.drop_speed = GameSettings.drop_speed
        self.drop_time = 1 / self.drop_speed
        
        # Restrict how often player can input keys (temp solution)
        self.move_delay = 0
        self.rotate_delay = 0
        
        # How long the tetramino can move around before without dropping before being deativated
        self.piece_deactivate_delay = self.drop_speed
        self._reset_piece_timers()
        
    def run(self, action: int) -> bool:
        """Take in an action and run the game for a single cycle.

        Args:
            action (int): The action performed.

        Returns:
            bool: True if the game has finished after this cycle has run.
        """
        self._cycle_game_clock()
        self._perform_action(action)
        
        return self._run_logic()    
    
    def draw_pieces(self, surface, draw_queue: bool = True):
        """Draws the peices to the parsed surface.

        Args:
            surface (Surface): The surface being drawn to.
        """
        self.piece_manager.draw_board_pieces(surface)
        self.piece_manager.draw_ghost_pieces(surface)
        self.piece_manager.draw_current_piece(surface)
        self.piece_manager.draw_held_piece(surface)
        
        if draw_queue:
            self.piece_manager.draw_queued_pieces(surface)
        
    def reset_game(self):
        self.piece_manager.reset()
        self._reset_scores()
        
        self.lines_cleared = 0
        self.previous_action = None
        
    def reset_b2b(self):
        if self.max_b2b < self.b2b:
            self.max_b2b = self.b2b
            
        self.b2b = 0
        
    def _increment_frames_passed(self):
        """Increase number of frames that have passed by 1.
        """
        self.frames += 1
        
    def _set_drop_speed(self, speed: int):
        """Set the drop speed of the pieces.

        Args:
            speed (int): How fast to drop the pieces.
        """
        self.drop_speed = speed
        self.drop_time = 1 / speed
        
        self.piece_deactivate_delay = self.drop_speed
        self.piece_deactivate_timer = self.piece_deactivate_delay * 3
    
    def _cycle_game_clock(self):
        self._update_delta_time()
        self._increment_frames_passed()
        self._update_fps_counter()
        
    def _update_delta_time(self):
        """Updates to the latest delta time value.
        """
        self.delta_time = time.time() - self.previous_time
        self.previous_time = time.time()

        self.total_time += self.delta_time
        self.fps_time += self.delta_time
        
    def _clear_lines_and_add_score(self):
        new_lines_cleared = self.piece_manager.perform_line_clears()
        self.lines_cleared += new_lines_cleared
        
        # Award points
        match new_lines_cleared:
            case 1:
                self.score += 100
                self.reset_b2b()
            
            case 2:
                self.score += 300
                self.reset_b2b()
            
            case 3:
                # T-spin triple
                if (self.piece_manager.previous_piece.id == PieceTypeID.T_PIECE):
                    self.score += 1600
                    self.b2b += 1
                else:
                    self.score += 500
                    self.reset_b2b()
                
            case 4:
                if self.b2b > 1:
                    self.score += 1200
                else:
                    self.score += 800
                    
                self.b2b += 1
    
    def _reset_scores(self):
        self.score = 0
        self.b2b = 0
        self.max_b2b = 0
        
    def _reset_piece_timers(self):
        self.piece_deactivate_timer = self.piece_deactivate_delay
        self.piece_global_deactivate_timer = self.piece_deactivate_delay * 3
    
    def _perform_action(self, action: int):
        match(action):
            case Actions.MOVE_LEFT:
                self.piece_manager.shift_piece_horizontally(-1)
                
            case Actions.MOVE_RIGHT:
                self.piece_manager.shift_piece_horizontally(1)
                
            case Actions.ROTATE_CLOCKWISE:
                self.piece_manager.rotate_piece(clockwise=True)
                
            case Actions.ROTATE_ANTICLOCKWISE:
                self.piece_manager.rotate_piece(clockwise=False)
                
            case Actions.SOFT_DROP:
                self.piece_manager.hard_drop_piece()
                
            case Actions.HARD_DROP:
                self.piece_manager.hard_drop_piece()
                self._new_piece_and_timer()
                
            case Actions.HOLD_PIECE:
                self.piece_manager.hold_piece()
            
            case _:
                raise ValueError(f"ERROR: perform_action(action) - action '{action}' is invalid")
            
        if action == int(Actions.HOLD_PIECE):
            self.holds_performed_in_a_row += 1
        else:
            self.holds_performed_in_a_row = 0
        
        self.previous_action = action
        self.piece_manager.actions_per_piece += 1
    
    def _update_fps_counter(self):
        """Update the fps counter with the current number of frames.
        """
        if (self.fps_time >= 1):
            self.fps_time -= 1
            self.last_fps_recorded = self.frames
            self.frames = 0
    
    def _run_logic(self) -> bool:
        """
        Runs the logic for the movement of the pieces over time.
        
        Returns: 
            bool: True if the game is finished after the logic has been run and False otherwise
        """
        # Attempt Drop current piece every set amount of time
        if self.total_time > self.drop_time:
            
            if self.piece_manager.gravity_drop_piece():

                # If delay timer was running then restart it
                if (self.piece_deactivate_timer < self.piece_deactivate_delay):
                    self.piece_deactivate_timer = self.piece_deactivate_delay
                    
            else:
                # Decrement piece timers
                self.piece_deactivate_timer -= 1
                self.piece_global_deactivate_timer -= 1
                
            # Create new piece and restart timer if piece has been touching ground for too long
            if self.piece_deactivate_timer < 0 or self.piece_global_deactivate_timer < 0:
                self._new_piece_and_timer()

            # Cycle total time
            self.total_time = self.total_time - self.drop_time
            
        self._clear_lines_and_add_score()    
            
        return self._check_game_over()
    
    def _check_game_over(self) -> bool:
        """Checks if there is a game over.

        Returns:
            bool: True if the game has ended, false otherwise
        """
        return self.piece_manager.board.check_game_over()
        
    def _new_piece_and_timer(self):
        self.piece_manager.deactivate_piece()
        self.piece_manager.next_piece()
        
        self._reset_piece_timers()

    def admin_render_toggle_input(self, event_list):
        for event in event_list:          
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    print("ENABLE RENDERING")
                    self.render_game = True
                    
                if event.key == pygame.K_DOWN:
                    print("DISABLED RENDERING")
                    self.render_game = False
                    
                if event.key == pygame.K_RIGHT:
                    print("SLOW PLAYBACK DISABLED")
                    self.playback = False
                    
                if event.key == pygame.K_LEFT:
                    print("SLOW PLAYBACK ENABLED")
                    self.playback = True
            
        return self.render_game, self.playback
    
    # DEBUGGING FUNCTION FOR MANUAL PLAY - CAN SAFELY BE DELETED
    def take_player_inputs(self, event_list):
        # Take player input
        key = pygame.key.get_pressed()
        self.move_delay -= 1 
        self.rotate_delay -= 1
        
        if (key[pygame.K_RIGHT] == True) and (self.move_delay < 0):
            self.piece_manager.shift_piece_horizontally(1)
            self.move_delay = 75 * 32 / win_utils.get_grid_size()
            
        if key[pygame.K_LEFT] == True and (self.move_delay < 0):
            self.piece_manager.shift_piece_horizontally(-1)
            self.move_delay = 75 * 32 / win_utils.get_grid_size()
            
        if key[pygame.K_x] == True and (self.rotate_delay < 0):
            self.piece_manager.rotate_piece(clockwise=True)
            self.rotate_delay = 120 * 32 / win_utils.get_grid_size()
            
        if key[pygame.K_z] == True and (self.rotate_delay < 0):
            self.piece_manager.rotate_piece(clockwise=False)
            self.rotate_delay = 120 * 32 / win_utils.get_grid_size()
        
        # DEBUG EVENTS
        if key[pygame.K_a] == True:
            self._set_drop_speed(20)
        
        if key[pygame.K_s] == True:
            self._set_drop_speed(GameSettings.drop_speed)
        
        for event in event_list:          
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.piece_manager.hard_drop_piece()
                    self._new_piece_and_timer()
                    
                if event.key == pygame.K_DOWN:
                    self.piece_manager.hard_drop_piece()
                    
                if event.key == pygame.K_r:
                    self.reset_game()
                    
                if event.key == pygame.K_LSHIFT:
                    self.piece_manager.hold_piece()