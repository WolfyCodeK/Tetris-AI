from numpy import ndarray, array
from pieces.piece_lookup_tables import (
    SYMMETRICAL_PIECE_ROTATION_TABLE, 
    SYMMETRICAL_PIECE_KICK_TABLE
    )

from .piece import Piece


class SymmetricalPiece(Piece):
    def __init__(self, pid: chr, x: int, colour: tuple, kick_priority: dict, shape: ndarray) -> None:
        super().__init__(pid, x, colour, shape)
        self.kick_priority = kick_priority
    
    def _is_side_square(self, x: int, y: int) -> bool:
        return (not x) ^ (not y)
    
    def _adjust_symmetrical_piece(self, clockwise: bool, shape: ndarray, state: int, i: int):
        piece_num = 0
        
        if (shape[i][0] == 0 and shape[i][1] == -1) or (shape[i][0] == -1 and shape[i][1] == -1): 
            piece_num = 0
            
        if (shape[i][0] == 0 and shape[i][1] == 1) or (shape[i][0] == 1 and shape[i][1] == -1): 
            piece_num = 1
            
        if (shape[i][0] == -1 and shape[i][1] == 0) or (shape[i][0] == 1 and shape[i][1] == 1): 
            piece_num = 2
            
        if (shape[i][0] == 1 and shape[i][1] == 0) or (shape[i][0] == -1 and shape[i][1] == 1): 
            piece_num = 3
        
        if not clockwise:
            state += 2
        
        x_adjust = SYMMETRICAL_PIECE_ROTATION_TABLE[state][piece_num][0]
        y_adjust = SYMMETRICAL_PIECE_ROTATION_TABLE[state][piece_num][1]
        
        if (shape[i][0] == 0 and shape[i][1] == 0):
            x_adjust = 0
            y_adjust = 0
        
        shape[i][0] = shape[i][0] + x_adjust
        shape[i][1] = shape[i][1] + y_adjust
        
        return shape
    
    def rotate(self, clockwise: bool):
        self.previous_shape = self.shape.copy()
        
        if clockwise:
            self.rotation_direction = 1
        else:
            self.rotation_direction = -1
        
        for i in range(len(self.shape)):
            if (self._is_side_square(self.shape[i][0], self.shape[i][1])):
                self.shape = self._adjust_symmetrical_piece(clockwise, self.shape, 0, i)
            else:
                self.shape = self._adjust_symmetrical_piece(clockwise, self.shape, 1, i)
        
        self.update_rotation_state()  
        self.update_minos()
        
    def update_rotation_state(self):
        if self.rotation_direction == 1:
            if (self.rotation_state == 3):
                self.rotation_state = 0
            else:
                self.rotation_state += 1
        elif self.rotation_direction == -1:
            if (self.rotation_state == 0):
                self.rotation_state = 3
            else:
                self.rotation_state -= 1
        
    def get_kick_priority(self):
        return self.kick_priority
        
    def kick(self, kick_index, clockwise):
        relative_rot_state = self.rotation_state
        
        if clockwise:
            rot = 1
        else:
            # Invert horizonal transformations
            rot = -1
            
            # Mirror clockwise transformations
            if (self.rotation_state in [0, 1, 2]):
                relative_rot_state = relative_rot_state + 1
                
            if (self.rotation_state == 3):
                relative_rot_state = 0

        self.transform(
            rot * SYMMETRICAL_PIECE_KICK_TABLE[relative_rot_state][kick_index][0], 
            rot * SYMMETRICAL_PIECE_KICK_TABLE[relative_rot_state][kick_index][1]
        )
        
    def revert_rotation(self):
        self.shape = self.previous_shape.copy()
        self.rotation_direction = -self.rotation_direction
        self.update_rotation_state()
        
class ZPiece(SymmetricalPiece):
    PID = 'Z'
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
        super().__init__(self.PID, self.START_BOARD_X, self.COLOUR, self.KICK_PRIORITY, self.DEFAULT_SHAPE.copy())

class LPiece(SymmetricalPiece):
    PID = 'L'
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
        super().__init__(self.PID, self.START_BOARD_X, self.COLOUR, self.KICK_PRIORITY, self.DEFAULT_SHAPE.copy())
        
class SPiece(SymmetricalPiece):
    PID = 'S'
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
        super().__init__(self.PID, self.START_BOARD_X, self.COLOUR, self.KICK_PRIORITY, self.DEFAULT_SHAPE.copy())
        
class JPiece(SymmetricalPiece):
    PID = 'J'
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
        super().__init__(self.PID, self.START_BOARD_X, self.COLOUR, self.KICK_PRIORITY, self.DEFAULT_SHAPE.copy())
        
class TPiece(SymmetricalPiece):
    PID = 'T'
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
        super().__init__(self.PID, self.START_BOARD_X, self.COLOUR, self.KICK_PRIORITY, self.DEFAULT_SHAPE.copy())
    
    def kick(self, kick_index, clockwise):
        relative_rot_state = self.rotation_state
        
        if clockwise:
            rot = 1
        else:
            # Invert horizonal transformations
            rot = -1
            
            # Mirror clockwise transformations
            if (self.rotation_state in [0, 1, 2]):
                relative_rot_state = relative_rot_state + 1
                
            if (self.rotation_state == 3):
                relative_rot_state = 0

        # If illegal kick is being attempted, do nothing
        if (relative_rot_state == 1) and (kick_index == 2):
            pass
        elif (relative_rot_state == 3) and (kick_index == 1):
            pass
        else:
            self.transform(
                rot * SYMMETRICAL_PIECE_KICK_TABLE[relative_rot_state][kick_index][0], 
                rot * SYMMETRICAL_PIECE_KICK_TABLE[relative_rot_state][kick_index][1]
            )