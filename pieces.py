import numpy as np
from tetramino import Tetramino

class ZPiece(Tetramino):
    PID = 'Z'
    START_BOARD_X = 4
    COLOUR = (255, 0, 0)
    
    def __init__(self) -> None:
        super().__init__(
            self.PID,
            self.START_BOARD_X, 
            self.COLOUR,
            np.array([[0, 0], [1, 0], [0, -1], [-1, -1]])
        )
        
class LPiece(Tetramino):
    PID = 'L'
    START_BOARD_X = 4
    COLOUR = (255,120,0)
    
    def __init__(self) -> None:
        super().__init__(
            self.PID,
            self.START_BOARD_X, 
            self.COLOUR,
            np.array([[0, 0], [-1, 0], [1, 0], [1, -1]])
        )
        
class SPiece(Tetramino):
    PID = 'S'
    START_BOARD_X = 4
    COLOUR = (0, 255, 0)
    
    def __init__(self) -> None:
        super().__init__(
            self.PID,
            self.START_BOARD_X, 
            self.COLOUR,
            np.array([[0, 0], [-1, 0], [0, -1], [1, -1]])
        )
        
class JPiece(Tetramino):
    PID = 'J'
    START_BOARD_X = 4
    COLOUR = (0,0,255)
    
    def __init__(self) -> None:
        super().__init__(
            self.PID,
            self.START_BOARD_X, 
            self.COLOUR,
            np.array([[0, 0], [-1, 0], [1, 0], [-1, -1]])
        )
        
class TPiece(Tetramino):
    PID = 'T'
    START_BOARD_X = 4
    COLOUR = (255,20,147)
    
    def __init__(self) -> None:
        super().__init__(
            self.PID,
            self.START_BOARD_X, 
            self.COLOUR,
            np.array([[0, 0], [-1, 0], [1, 0], [0, -1]])
        )
