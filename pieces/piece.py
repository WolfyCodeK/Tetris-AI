from numpy import ndarray, array_equal

from game.game_exceptions import RevertRotationError
import board.board_definitions as bd
import board.board_utils as bu
from abc import ABC, abstractmethod

class Piece(ABC):
    GHOST_PIECE_ALPHA = 225
    
    def __init__(self, pid: chr, x: int, colour: tuple, shape: ndarray) -> None:
        self.START_X_POS = x
        self.START_Y_POS = bd.MAX_PIECE_LENGTH
        
        self.pid = pid
        self.x_pos = self.START_X_POS
        self.y_pos = self.START_Y_POS
        self.colour = colour
        self.shape = shape
        
        self.kick_options = list(range(0, 4))
        self.previous_pos = (self.x_pos, self.y_pos)
        
        self.rotation_state = 0
        self.rotating_clockwise = True
        
        self.previous_shape = shape.copy()
        self.minos = shape.copy()
        
        self.update_minos()
        
        self.active = True
    
    def draw(self, surface):
        for i in range(len(self.minos)):
            bu.draw_rect(self.minos[i][0], self.minos[i][1], self.colour, surface)

    def draw_ghost(self, surface, max_height):
        for i in range(len(self.minos)):
            bu.draw_rect(self.minos[i][0], self.minos[i][1] + max_height, (50, 50, 50, self.GHOST_PIECE_ALPHA), surface)
            
    @abstractmethod
    def rotate(self, clockwise: bool):
        pass
    
    @abstractmethod
    def kick(self, kick_index:int, clockwise: bool):
        pass
        
    def revert_rotation(self) -> bool:
        if (not array_equal(self.shape, self.previous_shape.copy())):
            self.shape = self.previous_shape.copy()
            self.rotating_clockwise = not self.rotating_clockwise
            return True
        else:
            return False
    
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
        
    def transform(self, x: int, y: int) -> None:
        self.set_x_pos(self.x_pos + x)
        self.set_y_pos(self.y_pos + y)
        
    def reset_pos(self):
        self.set_x_pos(self.START_X_POS)
        self.set_y_pos(self.START_Y_POS)
        
    def reset_shape(self):
        self.shape = self.DEFAULT_SHAPE.copy()
        self.update_minos() 
        
    def save_previous_pos(self):
        self.previous_pos = (self.x_pos, self.y_pos)
        
    def revert_kick(self):
        self.set_x_pos(self.previous_pos[0])
        self.set_y_pos(self.previous_pos[1])