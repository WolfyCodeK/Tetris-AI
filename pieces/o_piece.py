import numpy as np

from .piece import Piece
from .piece_type_id import PieceTypeID

class OPiece(Piece):
    ID = PieceTypeID.O_PIECE
    START_BOARD_X = 5
    COLOUR = (255,255,102)
    DEFAULT_SHAPE = np.array([[0, 0], [-1, 0], [-1, -1], [0, -1]])
    
    def __init__(self) -> None:
        super().__init__(self.ID, self.START_BOARD_X, self.COLOUR, self.DEFAULT_SHAPE.copy())
    
    # O Piece cannot rotate or kick using SRS
    
    def get_shape_after_rotation(self, clockwise: bool):
        return self.shape, self.rotation_state
    
    def get_minos_after_kick(self, shape: np.ndarray, kick_index: int, clockwise: bool) -> tuple:
        return super().get_minos_after_kick(shape, kick_index, clockwise)
    
    def set_minos_from_shape(self, clockwise: bool, shape: np.ndarray):
        self.shape = shape
        self.minos = self.convert_to_absolute_shape(self.shape)