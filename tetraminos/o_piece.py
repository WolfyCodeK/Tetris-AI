from numpy import array

from .tetramino import Tetramino

class OPiece(Tetramino):
    PID = 'O'
    START_BOARD_X = 5
    COLOUR = (255,255,102)
    SHAPE = array([[0, 0], [-1, 0], [-1, -1], [0, -1]])
    
    def __init__(self) -> None:
        super().__init__(self.PID, self.START_BOARD_X, self.COLOUR, self.SHAPE)
        
    def rotate(self, clockwise: bool):
        # OPiece cannot rotate using SRS system
        pass