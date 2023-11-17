from numpy import array

from .piece import Piece
from .piece_type_id import PieceTypeID

class OPiece(Piece):
    ID = PieceTypeID.O_PIECE
    START_BOARD_X = 5
    COLOUR = (255,255,102)
    DEFAULT_SHAPE = array([[0, 0], [-1, 0], [-1, -1], [0, -1]])
    
    def __init__(self) -> None:
        super().__init__(self.ID, self.START_BOARD_X, self.COLOUR, self.DEFAULT_SHAPE.copy())
    
    # O Piece cannot rotate or kick using SRS
        
    def rotate(self, clockwise: bool):
        return super().rotate(clockwise)
    
    def kick(self, kick_index: int, clockwise: bool):
        return super().kick(kick_index, clockwise)