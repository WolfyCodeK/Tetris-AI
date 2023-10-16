import board_utils as bu
import numpy as np
import rotation_transformations as rt

class Tetramino:
    GHOST_PIECE_ALPHA = 225
    
    def __init__(self, pid: chr, x: int, colour: tuple, shape: np.ndarray) -> None:
        self.START_X_POS = x + bu.BOARD_STATE_WIDTH_BUFFER
        self.START_Y_POS = bu.MAX_PIECE_LENGTH
        
        self.pid = pid
        self.x_pos = self.START_X_POS
        self.y_pos = self.START_Y_POS
        self.colour = colour
        self.shape = shape
        
        self.DEFAULT_SHAPE = shape.copy()
        self.minos = shape.copy()
        
        self.update_minos()
        
        self.active = True
    
    def draw(self, surface):
        for i in range(len(self.minos)):
            bu.draw_rect(self.minos[i][0], self.minos[i][1], self.colour, surface)
            
    def draw_ghost_pieces(self, surface, max_height):
        for i in range(len(self.minos)):
            bu.draw_rect(self.minos[i][0], self.minos[i][1] + max_height, (50, 50, 50, self.GHOST_PIECE_ALPHA), surface)
    
    def update_minos(self):      
        for i in range(len(self.shape)):
            self.minos[i][0] = self.shape[i][0] + self.x_pos
            self.minos[i][1] = self.shape[i][1] + self.y_pos
    
    def set_x_pos(self, x: int) -> None:
        self.x_pos = x
        self.update_minos()
    
    def set_y_pos(self, y: int) -> None:
        self.y_pos = y
        self.update_minos()
        
    def reset_pos(self):
        self.set_x_pos(self.START_X_POS)
        self.set_y_pos(self.START_Y_POS)
        
    def reset_shape(self):
        self.shape = self.DEFAULT_SHAPE.copy()
        self.update_minos()
        
    def rotate_piece(self, clockwise: bool, is_IPiece):
        if (clockwise):
            self.shape = rt.rotate_clockwise(self.shape, is_IPiece)
        else:
            self.shape = rt.rotate_anticlockwise(self.shape, is_IPiece)
        
        self.update_minos()