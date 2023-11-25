import numpy as np
from game.game_exceptions import ShapeStateMissing
from pieces.piece_lookup_tables import IPIECE_ROTATION_TABLE, IPIECE_KICK_TABLE
from .piece import Piece
from .piece_type_id import PieceTypeID

class IPiece(Piece):
    ID = PieceTypeID.I_PIECE
    START_BOARD_X = 4
    COLOUR = (122,161,255)
    DEFAULT_SHAPE = np.array([[0, 0], [-1, 0], [1, 0], [2, 0]])
    CLOCKWISE_KICK_PRIORITY = {
        0: [2, 0, 1, 3],
        1: [2, 0, 1, 3],
        2: [3, 0, 1, 2],
        3: [0, 1, 2, 3]
    }
    
    ANTI_CLOCKWISE_KICK_PRIORITY = {
        0: [0, 1, 2, 3],
        1: [2, 0, 1, 3],
        2: [2, 0, 1, 3],
        3: [3, 0, 1, 2]
    }
    
    def __init__(self) -> None:
        super().__init__(self.ID, self.START_BOARD_X, self.COLOUR, self.DEFAULT_SHAPE.copy())
    
    def _rotate_from_table(self, clockwise: bool, shape: np.ndarray, state: int, i: int):
        if state in [0, 2]:
            j = 0
        else:
            j = 1
        
        if not clockwise:
            state += 4
        
        piece_num = shape[i][j] + 1
        
        shape[i][0] = shape[i][0] + IPIECE_ROTATION_TABLE[state][piece_num][0]
        shape[i][1] = shape[i][1] + IPIECE_ROTATION_TABLE[state][piece_num][1]
        
        return shape
    
    def get_shape_after_rotation(self, clockwise: bool) -> tuple:
        """Gets shape after rotation the piece in either the clockwise or anticlockwise direction by 90 degrees.

        Args:
            clockwise (bool): True if piece should be rotated clockwisae
            
        Returns:
            ndarray: The resulting shape
        """
        self.previous_shape = self.shape.copy()
        new_shape = self.shape.copy()
        
        if (new_shape[0][0] == 0 and new_shape[0][1] == 0): # STATE 0
            for i in range(len(new_shape)):
                new_shape = self._rotate_from_table(clockwise, new_shape, 0, i)
                    
        elif (new_shape[0][0] == 1 and new_shape[0][1] == 0): # STATE 1
            for i in range(len(new_shape)):
                new_shape = self._rotate_from_table(clockwise, new_shape, 1, i)
                
        elif (new_shape[0][0] == 1 and new_shape[0][1] == 1): # STATE 2
            for i in range(len(new_shape)):
                new_shape = self._rotate_from_table(clockwise, new_shape, 2, i)
                
        elif (new_shape[0][0] == 0 and new_shape[0][1] == 1): # STATE 3
            for i in range(len(new_shape)):
                new_shape = self._rotate_from_table(clockwise, new_shape, 3, i)
        else:
            raise ShapeStateMissing(self.id, new_shape)

        return new_shape.copy(), self.rotation_state
    
    def set_minos_from_shape(self, clockwise: bool, shape: np.ndarray):
        self.shape = shape.copy()
        self.minos = self.convert_to_absolute_shape(self.shape)
        self.rotating_clockwise = clockwise
    
    def get_kick_priority(self):
        if self.rotating_clockwise:
            return self.CLOCKWISE_KICK_PRIORITY
        elif not self.rotating_clockwise:
            return self.ANTI_CLOCKWISE_KICK_PRIORITY 
        
    def get_minos_after_kick(self, shape: np.ndarray, kick_index: int, clockwise: bool, rotation_state: int):
        relative_rotation_state = rotation_state
        
        if not clockwise:
            relative_rotation_state += 4
        
        new_x_pos = IPIECE_KICK_TABLE[relative_rotation_state][kick_index][0]
        new_y_pos = IPIECE_KICK_TABLE[relative_rotation_state][kick_index][1]
        
        return self.convert_to_absolute_shape(shape.copy(), new_x_pos, new_y_pos), new_x_pos, new_y_pos