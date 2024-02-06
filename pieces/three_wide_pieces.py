import numpy as np
from pieces.piece_lookup_tables import (
    THREE_WIDE_PIECE_ROTATION_TABLE, 
    THREE_WIDE_PIECE_KICK_TABLE
    )

from .piece import Piece
from .piece_type_id import PieceTypeID

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
        
class ZPiece(ThreeWidePiece):
    ID = PieceTypeID.Z_PIECE
    START_BOARD_X = 4
    COLOUR = (255,85,82)
    DEFAULT_SHAPE = np.array([[0, 0], [1, 0], [0, -1], [-1, -1]])
    KICK_PRIORITY = {
        0: [0, 1, 2, 3],
        1: [2, 0, 1, 3],
        2: [1, 0, 2, 3],
        3: [3, 0, 1, 2]
    }
    
    def __init__(self) -> None:
        super().__init__(self.ID, self.START_BOARD_X, self.COLOUR, self.KICK_PRIORITY, self.DEFAULT_SHAPE.copy())

class LPiece(ThreeWidePiece):
    ID = PieceTypeID.L_PIECE
    START_BOARD_X = 4
    COLOUR = (255,159,122)
    DEFAULT_SHAPE = np.array([[0, 0], [-1, 0], [1, 0], [1, -1]])
    KICK_PRIORITY = {
        0: [1, 0, 2, 3],
        1: [0, 3, 1, 2],
        2: [1, 0, 2, 3],
        3: [3, 0, 1, 2]
    }
    
    def __init__(self) -> None:
        super().__init__(self.ID, self.START_BOARD_X, self.COLOUR, self.KICK_PRIORITY, self.DEFAULT_SHAPE.copy())
        
class SPiece(ThreeWidePiece):
    ID = PieceTypeID.S_PIECE
    START_BOARD_X = 4
    COLOUR = (82,255,97)
    DEFAULT_SHAPE = np.array([[0, 0], [-1, 0], [0, -1], [1, -1]])
    KICK_PRIORITY = {
        0: [0, 1, 2, 3],
        1: [3, 0, 1, 2],
        2: [1, 0, 2, 3],
        3: [2, 0, 1, 3]
    }
    
    def __init__(self) -> None:
        super().__init__(self.ID, self.START_BOARD_X, self.COLOUR, self.KICK_PRIORITY, self.DEFAULT_SHAPE.copy())
        
class JPiece(ThreeWidePiece):
    ID = PieceTypeID.J_PIECE
    START_BOARD_X = 4
    COLOUR = (62,101,255)
    DEFAULT_SHAPE = np.array([[0, 0], [-1, 0], [1, 0], [-1, -1]])
    KICK_PRIORITY = {
        0: [1, 0, 2, 3],
        1: [0, 2, 1, 3],
        2: [1, 0, 2, 3],
        3: [3, 0, 1, 2]
    }
    
    def __init__(self) -> None:
        super().__init__(self.ID, self.START_BOARD_X, self.COLOUR, self.KICK_PRIORITY, self.DEFAULT_SHAPE.copy())
        
class TPiece(ThreeWidePiece):
    ID = PieceTypeID.T_PIECE
    START_BOARD_X = 4
    COLOUR = (255,100,167)
    DEFAULT_SHAPE = np.array([[0, 0], [-1, 0], [1, 0], [0, -1]])
    KICK_PRIORITY = {
        0: [1, 0, 2, 3],
        1: [3, 0, 1, 2],
        2: [1, 0, 2, 3],
        3: [3, 2, 1, 0]
    }
    
    def __init__(self) -> None:
        super().__init__(self.ID, self.START_BOARD_X, self.COLOUR, self.KICK_PRIORITY, self.DEFAULT_SHAPE.copy())
    
    def get_minos_after_kick(self, shape: np.ndarray, kick_index, clockwise, rotation_state: int) -> tuple:
        relative_rotation_state = rotation_state
        
        if not clockwise:
            relative_rotation_state = self.increment_rotation_state(relative_rotation_state)

        # If illegal kick is being attempted, do nothing
        if (relative_rotation_state == 1) and (kick_index == 2):
            return None, 0, 0
        elif (relative_rotation_state == 3) and (kick_index == 1):
            return None, 0, 0
        else:
            return self._kick_from_table(shape, clockwise, relative_rotation_state, kick_index)