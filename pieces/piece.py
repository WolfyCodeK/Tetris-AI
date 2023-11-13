from numpy import ndarray, array_equal

import utils.board_constants as bc
import utils.window_utils as win_utils
from abc import ABC, abstractmethod

class Piece(ABC):
    GHOST_PIECE_COLOUR = (50, 50, 50, 225)
    
    NUM_OF_KICK_OPTIONS = 4
    NUM_OF_ROTATION_STATES = 4
    KICK_OPTIONS = list(range(0, NUM_OF_KICK_OPTIONS))
    
    def __init__(self, id: chr, x: int, colour: tuple, shape: ndarray) -> None:
        self.START_X_POS = x
        self.START_Y_POS = bc.MAX_PIECE_LENGTH
        self.x_pos = self.START_X_POS
        self.y_pos = self.START_Y_POS
        
        self.id = id
        self.colour = colour
        self.shape = shape
        
        self.previous_pos = (self.x_pos, self.y_pos)
        
        self.rotation_state = 0
        self.rotating_clockwise = True
        
        self.previous_shape = shape.copy()
        self.minos = shape.copy()
        
        self.update_minos()
    
    @abstractmethod
    def rotate(self, clockwise: bool):
        pass
    
    @abstractmethod
    def kick(self, kick_index:int, clockwise: bool):
        pass
    
    def _set_x_pos(self, x: int) -> None:
        self.x_pos = x
        self.update_minos()
    
    def _set_y_pos(self, y: int) -> None:
        self.y_pos = y
        self.update_minos()    
    
    def draw(self, surface):
        for i in range(len(self.minos)):
            win_utils.draw_rect(self.minos[i][0], self.minos[i][1], self.colour, surface)

    def draw_ghost(self, surface, max_height):
        for i in range(len(self.minos)):
            win_utils.draw_rect(self.minos[i][0], self.minos[i][1] + max_height, self.GHOST_PIECE_COLOUR, surface)
        
    def increment_rotation_state(self, rotation_state) -> int:
        # Mirror clockwise transformations by incrementing rotation state - uses modulo operator to wrap around value    
        rotation_state = (rotation_state + self.NUM_OF_ROTATION_STATES + 1) % self.NUM_OF_ROTATION_STATES
        
        return rotation_state    
    
    def decrement_rotation_state(self, rotation_state) -> int:
        # Mirror clockwise transformations by incrementing rotation state - uses modulo operator to wrap around value    
        rotation_state = (rotation_state + self.NUM_OF_ROTATION_STATES - 1) % self.NUM_OF_ROTATION_STATES
        
        return rotation_state  
    
    def update_minos(self):      
        for i in range(len(self.shape)):
            self.minos[i][0] = self.shape[i][0] + self.x_pos
            self.minos[i][1] = self.shape[i][1] + self.y_pos    
        
    def transform(self, x: int, y: int) -> None:
        # Save previous position in case tranformation needs reverting
        self.previous_pos = (self.x_pos, self.y_pos)
        
        self._set_x_pos(self.x_pos + x)
        self._set_y_pos(self.y_pos + y)
        
    def reset_piece(self):
        self.shape = self.DEFAULT_SHAPE.copy()
        self._set_x_pos(self.START_X_POS)
        self._set_y_pos(self.START_Y_POS)
        
    def revert_kick(self):
        self._set_x_pos(self.previous_pos[0])
        self._set_y_pos(self.previous_pos[1])
        
    def revert_rotation(self) -> bool:
        if (not array_equal(self.shape, self.previous_shape.copy())):
            self.shape = self.previous_shape.copy()
            self.rotating_clockwise = not self.rotating_clockwise
            self.update_minos()
            return True
        else:
            return False