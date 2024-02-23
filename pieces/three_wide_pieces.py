import numpy as np
from pieces.piece_lookup_tables import (
    THREE_WIDE_PIECE_ROTATION_TABLE, 
    THREE_WIDE_PIECE_KICK_TABLE
    )
from .piece import Piece

class ThreeWidePiece(Piece):
    def __init__(self, id: chr, x: int, colour: tuple, kick_priority: dict, shape: np.ndarray) -> None:
        super().__init__(id, x, colour, shape)
        self.kick_priority = kick_priority
    
    def _is_side_square(self, x: int, y: int) -> bool:
        return (not x) ^ (not y)
    
    def _rotate_from_table(self, clockwise: bool, shape: np.ndarray, state: int, piece_index: int) -> np.ndarray:
        piece_num = 0
        
        if (shape[piece_index][0] == 0 and shape[piece_index][1] == -1) or (shape[piece_index][0] == -1 and shape[piece_index][1] == -1): 
            piece_num = 0
            
        if (shape[piece_index][0] == 0 and shape[piece_index][1] == 1) or (shape[piece_index][0] == 1 and shape[piece_index][1] == -1): 
            piece_num = 1
            
        if (shape[piece_index][0] == -1 and shape[piece_index][1] == 0) or (shape[piece_index][0] == 1 and shape[piece_index][1] == 1): 
            piece_num = 2
            
        if (shape[piece_index][0] == 1 and shape[piece_index][1] == 0) or (shape[piece_index][0] == -1 and shape[piece_index][1] == 1): 
            piece_num = 3
        
        if not clockwise:
            state += 2
        
        x_adjust = THREE_WIDE_PIECE_ROTATION_TABLE[state][piece_num][0]
        y_adjust = THREE_WIDE_PIECE_ROTATION_TABLE[state][piece_num][1]
        
        if (shape[piece_index][0] == 0 and shape[piece_index][1] == 0):
            x_adjust = 0
            y_adjust = 0
        
        shape[piece_index][0] = shape[piece_index][0] + x_adjust
        shape[piece_index][1] = shape[piece_index][1] + y_adjust
        
        return shape
    
    def get_shape_after_rotation(self, clockwise: bool) -> tuple:
        self.previous_shape = self.shape.copy()
        new_shape = self.shape.copy()
        
        for i in range(len(new_shape)):
            if (self._is_side_square(new_shape[i][0], new_shape[i][1])):
                new_shape = self._rotate_from_table(clockwise, new_shape, 0, i)
            else:
                new_shape = self._rotate_from_table(clockwise, new_shape, 1, i)  
        
        rotation_state = self.cycle_rotation_state(self.rotation_state, clockwise)
        
        return new_shape.copy(), rotation_state
    
    def set_minos_from_shape(self, clockwise: bool, shape: np.ndarray):
        self.shape = shape
        self.minos = self.convert_to_absolute_shape(self.shape)
        
        self.rotating_clockwise = clockwise
        self.rotation_state = self.cycle_rotation_state(self.rotation_state, clockwise)
        
    def cycle_rotation_state(self, rotation_state: int, clockwise: bool) -> int:
        if clockwise:
            rotation_state = self.increment_rotation_state(rotation_state)
        elif not clockwise:
            rotation_state = self.decrement_rotation_state(rotation_state)
            
        return rotation_state
        
    def get_kick_priority(self) -> dict:
        return self.kick_priority
    
    def _kick_from_table(self, shape: np.ndarray, clockwise: bool, rotation_state: int, kick_index: int) -> tuple:
        if (clockwise):
            invert_transformation = 1
        else:
            invert_transformation = -1
            
        new_x_pos = invert_transformation * THREE_WIDE_PIECE_KICK_TABLE[rotation_state][kick_index][0]
        new_y_pos = invert_transformation * THREE_WIDE_PIECE_KICK_TABLE[rotation_state][kick_index][1]
            
        return self.convert_to_absolute_shape(shape.copy(), new_x_pos, new_y_pos), new_x_pos, new_y_pos
        
    def get_minos_after_kick(self, shape: np.ndarray, kick_index: int, clockwise: bool, rotation_state: int) -> tuple:
        relative_rotation_state = rotation_state
        
        if not clockwise:
            relative_rotation_state = self.increment_rotation_state(relative_rotation_state)
        
        return self._kick_from_table(shape, clockwise, relative_rotation_state, kick_index)