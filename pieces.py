import numpy as np
from tetramino import Tetramino
import tetramino_features as tf

class ZPiece(Tetramino):
    def __init__(self) -> None:
        super().__init__(
            tf.Z_PIECE_PID,
            tf.Z_PIECE_START_X, 
            tf.Z_PIECE_START_Y, 
            tf.Z_PIECE_COLOUR
        )
        
        self.update_occupying_squares(self.x_pos, self.y_pos)
    
    def update_occupying_squares(self, x, y):    
        self.occupying_squares = np.array([[x, y], [x + 1, y], [x, y - 1], [x - 1, y - 1]])
        
class LPiece(Tetramino):
    def __init__(self) -> None:
        super().__init__(
            tf.L_PIECE_PID,
            tf.L_PIECE_START_X, 
            tf.L_PIECE_START_Y, 
            tf.L_PIECE_COLOUR
        )
        
        self.update_occupying_squares(self.x_pos, self.y_pos)
    
    def update_occupying_squares(self, x, y):    
        self.occupying_squares = np.array([[x, y], [x - 1, y], [x + 1, y], [x + 1, y - 1]])
        
class SPiece(Tetramino):
    def __init__(self) -> None:
        super().__init__(
            tf.S_PIECE_PID,
            tf.S_PIECE_START_X, 
            tf.S_PIECE_START_Y, 
            tf.S_PIECE_COLOUR
        )
        
        self.update_occupying_squares(self.x_pos, self.y_pos)
    
    def update_occupying_squares(self, x, y):    
        self.occupying_squares = np.array([[x, y], [x - 1, y], [x, y - 1], [x + 1, y - 1]])
        
class JPiece(Tetramino):
    def __init__(self) -> None:
        super().__init__(
            tf.J_PIECE_PID,
            tf.J_PIECE_START_X, 
            tf.J_PIECE_START_Y, 
            tf.J_PIECE_COLOUR
        )
        
        self.update_occupying_squares(self.x_pos, self.y_pos)
    
    def update_occupying_squares(self, x, y):    
        self.occupying_squares = np.array([[x, y], [x - 1, y], [x + 1, y], [x -1, y - 1]])