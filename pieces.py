import numpy as np
from tetramino import Tetramino

class ZPiece(Tetramino):
    PID = 'Z'
    START_BOARD_X = 4
    COLOUR = (255,85,82)
    
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
    COLOUR = (255,159,122)
    
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
    COLOUR = (82,255,97)
    
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
    COLOUR = (62,101,255)
    
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
    COLOUR = (255,100,167)
    
    def __init__(self) -> None:
        super().__init__(
            self.PID,
            self.START_BOARD_X, 
            self.COLOUR,
            np.array([[0, 0], [-1, 0], [1, 0], [0, -1]])
        )

class IPiece(Tetramino):
    PID = 'I'
    START_BOARD_X = 4
    COLOUR = (122,161,255)
    
    def __init__(self) -> None:
        super().__init__(
            self.PID,
            self.START_BOARD_X, 
            self.COLOUR,
            np.array([[0, 0], [-1, 0], [1, 0], [2, 0]])
        )
        
class OPiece(Tetramino):
    PID = 'O'
    START_BOARD_X = 5
    COLOUR = (255,255,102)
    
    def __init__(self) -> None:
        super().__init__(
            self.PID,
            self.START_BOARD_X, 
            self.COLOUR,
            np.array([[0, 0], [-1, 0], [-1, -1], [0, -1]])
        )