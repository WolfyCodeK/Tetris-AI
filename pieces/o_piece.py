from numpy import array

from .piece import Piece

class OPiece(Piece):
    PID = 'O'
    START_BOARD_X = 5
    COLOUR = (255,255,102)
    DEFAULT_SHAPE = array([[0, 0], [-1, 0], [-1, -1], [0, -1]])
    
    def __init__(self) -> None:
        super().__init__(self.PID, self.START_BOARD_X, self.COLOUR, self.DEFAULT_SHAPE.copy())
        
    def rotate(self, clockwise: bool):
        return super().rotate(clockwise)