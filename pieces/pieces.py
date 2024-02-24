import numpy as np
from pieces.piece_lookup_tables import IPIECE_ROTATION_TABLE, IPIECE_KICK_TABLE
from game.game_exceptions import ShapeStateMissing
from .piece import Piece
from .piece_type_id import PieceTypeID
from .piece_colours import PieceColours
from .three_wide_pieces import ThreeWidePiece
        
class ZPiece(ThreeWidePiece):
    ID = PieceTypeID.Z_PIECE
    START_BOARD_X = 4
    COLOUR = PieceColours.Z_PIECE_COLOUR.value
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
    COLOUR = PieceColours.L_PIECE_COLOUR.value
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
    COLOUR = PieceColours.S_PIECE_COLOUR.value
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
    COLOUR = PieceColours.J_PIECE_COLOUR.value
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
    COLOUR = PieceColours.T_PIECE_COLOUR.value
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
        
        
class OPiece(Piece):
    ID = PieceTypeID.O_PIECE
    START_BOARD_X = 5
    COLOUR = PieceColours.O_PIECE_COLOUR.value
    DEFAULT_SHAPE = np.array([[0, 0], [-1, 0], [-1, -1], [0, -1]])
    
    def __init__(self) -> None:
        super().__init__(self.ID, self.START_BOARD_X, self.COLOUR, self.DEFAULT_SHAPE.copy())
    
    # O Piece cannot rotate or kick using SRS#
    def get_kick_priority(self) -> dict[int, list[int]]:
        return super().get_kick_priority()
    
    def get_shape_after_rotation(self, clockwise: bool) -> tuple:
        return super().get_shape_after_rotation(clockwise)
    
    def get_minos_after_kick(self, shape: np.ndarray, kick_index: int, clockwise: bool) -> tuple:
        return super().get_minos_after_kick(shape, kick_index, clockwise)
    
    def set_minos_from_shape(self, clockwise: bool, shape: IPIECE_ROTATION_TABLE):
        return super().set_minos_from_shape(clockwise, shape)
        
class IPiece(Piece):
    ID = PieceTypeID.I_PIECE
    START_BOARD_X = 4
    COLOUR = PieceColours.I_PIECE_COLOUR.value
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

        return new_shape, self.rotation_state
    
    def set_minos_from_shape(self, clockwise: bool, shape: np.ndarray):
        self.shape = shape.copy()
        self.minos = self.convert_to_absolute_shape(self.shape)
        self.rotating_clockwise = clockwise
    
    def get_kick_priority(self):
        if self.rotating_clockwise:
            return self.CLOCKWISE_KICK_PRIORITY
        else:
            return self.ANTI_CLOCKWISE_KICK_PRIORITY 
        
    def get_minos_after_kick(self, shape: np.ndarray, kick_index: int, clockwise: bool, rotation_state: int):
        relative_rotation_state = rotation_state
        
        if not clockwise:
            relative_rotation_state += 4
        
        new_x_pos = IPIECE_KICK_TABLE[relative_rotation_state][kick_index][0]
        new_y_pos = IPIECE_KICK_TABLE[relative_rotation_state][kick_index][1]
        
        return self.convert_to_absolute_shape(shape, new_x_pos, new_y_pos), new_x_pos, new_y_pos