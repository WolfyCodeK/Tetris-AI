import board_utils as bu
import numpy as np

class Tetramino:
    def __init__(self, pid: chr, x: int, colour: tuple, shape: np.ndarray, large_rotation: bool = False) -> None:
        self.pid = pid
        self.x_pos = x
        self.y_pos = bu.MAX_PIECE_LENGTH
        self.colour = colour
        self.shape = shape
        self.occupying_squares = shape.copy()
        
        self.update_occupying_squares()
        
        self.active = True
    
    def draw(self, surface):
        for i in range(len(self.occupying_squares)):
            bu.draw_rect(self.occupying_squares[i][0], self.occupying_squares[i][1], self.colour, surface)
            
    def draw_ghost_pieces(self, surface, max_height):
        for i in range(len(self.occupying_squares)):
            bu.draw_rect(self.occupying_squares[i][0], self.occupying_squares[i][1] + max_height, (50, 50, 50, 200), surface)
    
    def update_occupying_squares(self):      
        for i in range(len(self.shape)):
            self.occupying_squares[i][0] = self.shape[i][0] + self.x_pos
            self.occupying_squares[i][1] = self.shape[i][1] + self.y_pos
    
    def set_x_pos(self, x: int) -> None:
        self.x_pos = x
        self.update_occupying_squares()
    
    def set_y_pos(self, y: int) -> None:
        self.y_pos = y
        self.update_occupying_squares()
    
    def rotate_anticlockwise(self, is_IPiece = False):
        if (not is_IPiece):
            for i in range(len(self.shape)):
                if (self.is_side_square(self.shape[i][0], self.shape[i][1])):
                    if (self.shape[i][1] == -1):
                        self.shape[i][0] = self.shape[i][0] - 1
                        self.shape[i][1] = self.shape[i][1] + 1
                    elif (self.shape[i][1] == 1):
                        self.shape[i][0] = self.shape[i][0] + 1
                        self.shape[i][1] = self.shape[i][1] - 1  
                    elif (self.shape[i][0] == -1):
                        self.shape[i][0] = self.shape[i][0] + 1
                        self.shape[i][1] = self.shape[i][1] + 1 
                    elif (self.shape[i][0] == 1):
                        self.shape[i][0] = self.shape[i][0] - 1
                        self.shape[i][1] = self.shape[i][1] - 1
                else:
                    if (self.shape[i][0] == -1) and ((self.shape[i][1] == -1)):
                        self.shape[i][1] = self.shape[i][1] + 2
                    elif (self.shape[i][0] == 1) and ((self.shape[i][1] == -1)):
                        self.shape[i][0] = self.shape[i][0] - 2
                    elif (self.shape[i][0] == 1) and ((self.shape[i][1] == 1)):
                        self.shape[i][1] = self.shape[i][1] - 2
                    elif (self.shape[i][0] == -1) and ((self.shape[i][1] == 1)):
                        self.shape[i][0] = self.shape[i][0] + 2
        else:
            if (self.shape[0][0] == 0 and self.shape[0][1] == 0):
                # STATE 1
                for i in range(len(self.shape)):
                    if (self.shape[i][0] == -1):
                        self.shape[i][0] = self.shape[i][0] + 1
                        self.shape[i][1] = self.shape[i][1] + 2
                    elif (self.shape[i][0] == 0):
                        self.shape[i][1] = self.shape[i][1] + 1
                    elif (self.shape[i][0] == 1):
                        self.shape[i][0] = self.shape[i][0] - 1
                    elif (self.shape[i][0] == 2):
                        self.shape[i][0] = self.shape[i][0] - 2
                        self.shape[i][1] = self.shape[i][1] - 1
            elif (self.shape[0][0] == 1 and self.shape[0][1] == 0):
                # STATE 2
                for i in range(len(self.shape)):
                    if (self.shape[i][1] == -1):
                        self.shape[i][0] = self.shape[i][0] - 2
                        self.shape[i][1] = self.shape[i][1] + 1
                    elif (self.shape[i][1] == 0):
                        self.shape[i][0] = self.shape[i][0] - 1
                    elif (self.shape[i][1] == 1):
                        self.shape[i][1] = self.shape[i][1] - 1
                    elif (self.shape[i][1] == 2):
                        self.shape[i][0] = self.shape[i][0] + 1
                        self.shape[i][1] = self.shape[i][1] - 2
            elif (self.shape[0][0] == 1 and self.shape[0][1] == 1):
                # STATE 3
                for i in range(len(self.shape)):
                    if (self.shape[i][0] == -1):
                        self.shape[i][0] = self.shape[i][0] + 2
                        self.shape[i][1] = self.shape[i][1] + 1
                    elif (self.shape[i][0] == 0):
                        self.shape[i][0] = self.shape[i][0] + 1
                    elif (self.shape[i][0] == 1):
                        self.shape[i][1] = self.shape[i][1] - 1
                    elif (self.shape[i][0] == 2):
                        self.shape[i][0] = self.shape[i][0] - 1
                        self.shape[i][1] = self.shape[i][1] - 2
            elif (self.shape[0][0] == 0 and self.shape[0][1] == 1):
                # STATE 4
                for i in range(len(self.shape)):
                    if (self.shape[i][1] == -1):
                        self.shape[i][0] = self.shape[i][0] - 1
                        self.shape[i][1] = self.shape[i][1] + 2
                    elif (self.shape[i][1] == 0):
                        self.shape[i][1] = self.shape[i][1] + 1
                    elif (self.shape[i][1] == 1):
                        self.shape[i][0] = self.shape[i][0] + 1
                    elif (self.shape[i][1] == 2):
                        self.shape[i][0] = self.shape[i][0] + 2
                        self.shape[i][1] = self.shape[i][1] - 1
                    
        self.update_occupying_squares()
    
    def rotate_clockwise(self, is_IPiece = False):
        if (not is_IPiece):
            for i in range(len(self.shape)):
                if (self.is_side_square(self.shape[i][0], self.shape[i][1])):
                    if (self.shape[i][1] == -1):
                        self.shape[i][0] = self.shape[i][0] + 1
                        self.shape[i][1] = self.shape[i][1] + 1
                    elif (self.shape[i][1] == 1):
                        self.shape[i][0] = self.shape[i][0] - 1
                        self.shape[i][1] = self.shape[i][1] - 1  
                    elif (self.shape[i][0] == -1):
                        self.shape[i][0] = self.shape[i][0] + 1
                        self.shape[i][1] = self.shape[i][1] - 1 
                    elif (self.shape[i][0] == 1):
                        self.shape[i][0] = self.shape[i][0] - 1
                        self.shape[i][1] = self.shape[i][1] + 1
                else:
                    if (self.shape[i][0] == -1) and ((self.shape[i][1] == -1)):
                        self.shape[i][0] = self.shape[i][0] + 2
                    elif (self.shape[i][0] == 1) and ((self.shape[i][1] == -1)):
                        self.shape[i][1] = self.shape[i][1] + 2
                    elif (self.shape[i][0] == 1) and ((self.shape[i][1] == 1)):
                        self.shape[i][0] = self.shape[i][0] - 2
                    elif (self.shape[i][0] == -1) and ((self.shape[i][1] == 1)):
                        self.shape[i][1] = self.shape[i][1] - 2
        else:
            if (self.shape[0][0] == 0 and self.shape[0][1] == 0):
                # STATE 1
                for i in range(len(self.shape)):
                    if (self.shape[i][0] == -1):
                        self.shape[i][0] = self.shape[i][0] + 2
                        self.shape[i][1] = self.shape[i][1] - 1
                    elif (self.shape[i][0] == 0):
                        self.shape[i][0] = self.shape[i][0] + 1
                    elif (self.shape[i][0] == 1):
                        self.shape[i][1] = self.shape[i][1] + 1
                    elif (self.shape[i][0] == 2):
                        self.shape[i][0] = self.shape[i][0] - 1
                        self.shape[i][1] = self.shape[i][1] + 2
            elif (self.shape[0][0] == 1 and self.shape[0][1] == 0):
                # STATE 2
                for i in range(len(self.shape)):
                    if (self.shape[i][1] == -1):
                        self.shape[i][0] = self.shape[i][0] + 1
                        self.shape[i][1] = self.shape[i][1] + 2
                    elif (self.shape[i][1] == 0):
                        self.shape[i][1] = self.shape[i][1] + 1
                    elif (self.shape[i][1] == 1):
                        self.shape[i][0] = self.shape[i][0] - 1
                    elif (self.shape[i][1] == 2):
                        self.shape[i][0] = self.shape[i][0] - 2
                        self.shape[i][1] = self.shape[i][1] - 1
            elif (self.shape[0][0] == 1 and self.shape[0][1] == 1):
                # STATE 3
                for i in range(len(self.shape)):
                    if (self.shape[i][0] == -1):
                        self.shape[i][0] = self.shape[i][0] + 1
                        self.shape[i][1] = self.shape[i][1] - 2
                    elif (self.shape[i][0] == 0):
                        self.shape[i][1] = self.shape[i][1] - 1
                    elif (self.shape[i][0] == 1):
                        self.shape[i][0] = self.shape[i][0] - 1
                    elif (self.shape[i][0] == 2):
                        self.shape[i][0] = self.shape[i][0] - 2
                        self.shape[i][1] = self.shape[i][1] + 1
            elif (self.shape[0][0] == 0 and self.shape[0][1] == 1):
                # STATE 4
                for i in range(len(self.shape)):
                    if (self.shape[i][1] == -1):
                        self.shape[i][0] = self.shape[i][0] + 2
                        self.shape[i][1] = self.shape[i][1] + 1
                    elif (self.shape[i][1] == 0):
                        self.shape[i][0] = self.shape[i][0] + 1
                    elif (self.shape[i][1] == 1):
                        self.shape[i][1] = self.shape[i][1] - 1
                    elif (self.shape[i][1] == 2):
                        self.shape[i][0] = self.shape[i][0] - 1
                        self.shape[i][1] = self.shape[i][1] - 2
                    
        self.update_occupying_squares()
        
    def is_side_square(self, x, y) -> bool:
        return (not x) ^ (not y)