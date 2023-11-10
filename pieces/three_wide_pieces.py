from numpy import array_equal, ndarray, array
from pieces.piece_lookup_tables import (
    THREE_WIDE_PIECE_ROTATION_TABLE, 
    THREE_WIDE_PIECE_KICK_TABLE
    )

from .piece import Piece


class ThreeWidePiece(Piece):
    def __init__(self, id: chr, x: int, colour: tuple, kick_priority: dict, shape: ndarray) -> None:
        super().__init__(id, x, colour, shape)
        self.kick_priority = kick_priority
    
    def _is_side_square(self, x: int, y: int) -> bool:
        return (not x) ^ (not y)
    
    def _rotate_from_table(self, clockwise: bool, shape: ndarray, state: int, piece_index: int) -> ndarray:
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
    
    def rotate(self, clockwise: bool) -> None:
        self.previous_shape = self.shape.copy()
        
        self.rotating_clockwise = clockwise
        
        for i in range(len(self.shape)):
            if (self._is_side_square(self.shape[i][0], self.shape[i][1])):
                self.shape = self._rotate_from_table(clockwise, self.shape, 0, i)
            else:
                self.shape = self._rotate_from_table(clockwise, self.shape, 1, i)
        
        self.rotation_state = self.cycle_rotation_state(self.rotation_state, clockwise)  
        self.update_minos()
        
    def cycle_rotation_state(self, rotation_state: int, clockwise: bool) -> int:
        if clockwise:
            rotation_state = self.increment_rotation_state(rotation_state)
        elif not clockwise:
            rotation_state = self.decrement_rotation_state(rotation_state)
            
        return rotation_state
        
    def get_kick_priority(self) -> dict:
        return self.kick_priority
    
    def _kick_from_table(self, clockwise: bool, rotation_state: int, kick_index: int) -> None:
        if (clockwise):
            invert_transformation = 1
        else:
            invert_transformation = -1
            
        self.transform(
            invert_transformation * THREE_WIDE_PIECE_KICK_TABLE[rotation_state][kick_index][0], 
            invert_transformation * THREE_WIDE_PIECE_KICK_TABLE[rotation_state][kick_index][1]
        )
        
    def kick(self, kick_index: int, clockwise: bool):
        relative_rotation_state = self.rotation_state
        
        if not clockwise:
            relative_rotation_state = self.increment_rotation_state(relative_rotation_state)

        self._kick_from_table(clockwise, relative_rotation_state, kick_index) 
        
    def revert_rotation(self) -> bool:
        if (not array_equal(self.shape, self.previous_shape.copy())):
            self.shape = self.previous_shape.copy()
            self.rotating_clockwise = not self.rotating_clockwise
            self.rotation_state = self.cycle_rotation_state(self.rotation_state, self.rotating_clockwise)
            return True
        else:
            return False
        
class ZPiece(ThreeWidePiece):
    ID = 'Z'
    START_BOARD_X = 4
    COLOUR = (255,85,82)
    DEFAULT_SHAPE = array([[0, 0], [1, 0], [0, -1], [-1, -1]])
    KICK_PRIORITY = {
        0: [0, 1, 2, 3],
        1: [2, 0, 1, 3],
        2: [1, 0, 2, 3],
        3: [3, 0, 1, 2]
    }
    
    def __init__(self) -> None:
        super().__init__(self.ID, self.START_BOARD_X, self.COLOUR, self.KICK_PRIORITY, self.DEFAULT_SHAPE.copy())

class LPiece(ThreeWidePiece):
    ID = 'L'
    START_BOARD_X = 4
    COLOUR = (255,159,122)
    DEFAULT_SHAPE = array([[0, 0], [-1, 0], [1, 0], [1, -1]])
    KICK_PRIORITY = {
        0: [1, 0, 2, 3],
        1: [0, 3, 1, 2],
        2: [1, 0, 2, 3],
        3: [3, 0, 1, 2]
    }
    
    def __init__(self) -> None:
        super().__init__(self.ID, self.START_BOARD_X, self.COLOUR, self.KICK_PRIORITY, self.DEFAULT_SHAPE.copy())
        
class SPiece(ThreeWidePiece):
    ID = 'S'
    START_BOARD_X = 4
    COLOUR = (82,255,97)
    DEFAULT_SHAPE = array([[0, 0], [-1, 0], [0, -1], [1, -1]])
    KICK_PRIORITY = {
        0: [0, 1, 2, 3],
        1: [3, 0, 1, 2],
        2: [1, 0, 2, 3],
        3: [2, 0, 1, 3]
    }
    
    def __init__(self) -> None:
        super().__init__(self.ID, self.START_BOARD_X, self.COLOUR, self.KICK_PRIORITY, self.DEFAULT_SHAPE.copy())
        
class JPiece(ThreeWidePiece):
    ID = 'J'
    START_BOARD_X = 4
    COLOUR = (62,101,255)
    DEFAULT_SHAPE = array([[0, 0], [-1, 0], [1, 0], [-1, -1]])
    KICK_PRIORITY = {
        0: [1, 0, 2, 3],
        1: [0, 2, 1, 3],
        2: [1, 0, 2, 3],
        3: [3, 0, 1, 2]
    }
    
    def __init__(self) -> None:
        super().__init__(self.ID, self.START_BOARD_X, self.COLOUR, self.KICK_PRIORITY, self.DEFAULT_SHAPE.copy())
        
class TPiece(ThreeWidePiece):
    ID = 'T'
    START_BOARD_X = 4
    COLOUR = (255,100,167)
    DEFAULT_SHAPE = array([[0, 0], [-1, 0], [1, 0], [0, -1]])
    KICK_PRIORITY = {
        0: [1, 0, 2, 3],
        1: [3, 0, 1, 2],
        2: [1, 0, 2, 3],
        3: [3, 2, 1, 0]
    }
    
    def __init__(self) -> None:
        super().__init__(self.ID, self.START_BOARD_X, self.COLOUR, self.KICK_PRIORITY, self.DEFAULT_SHAPE.copy())
    
    def kick(self, kick_index, clockwise):
        relative_rotation_state = self.rotation_state
        
        if not clockwise:
            relative_rotation_state = self.increment_rotation_state(relative_rotation_state)

        # If illegal kick is being attempted, do nothing
        if (relative_rotation_state == 1) and (kick_index == 2):
            pass
        elif (relative_rotation_state == 3) and (kick_index == 1):
            pass
        else:
            self._kick_from_table(clockwise, relative_rotation_state, kick_index)